package org.apache.pinot.perf;

import java.lang.foreign.Arena;
import java.util.concurrent.locks.ReentrantReadWriteLock;


/// Combines Direction A's correctness architecture (shard by key hash, so a key always routes to the
/// same shard regardless of which thread produced it -- no "local trim before merge" risk) with
/// Direction B's off-heap storage per shard (GC-pressure benefit, measured in OffHeapGroupTable's own
/// verification). Optionally also ports Direction A's top1Share adaptive-capacity mechanism (see
/// AdaptiveConcurrentIndexedTable in ShardedIndexedTable.java) onto these off-heap shards, to address
/// the memory-ceiling problem (numShards * trimSize) this design would otherwise inherit from Direction
/// A's non-adaptive sharding.
///
/// Concurrency: OffHeapGroupTable has NO internal thread-safety (unlike ConcurrentIndexedTable's
/// ConcurrentHashMap, which safely handles concurrent upserts to different keys on its own). Every
/// upsert to a shard therefore needs full exclusive locking against every other access to that same
/// shard -- there is no fine-grained "read lock for the common case" available the way
/// ConcurrentIndexedTable gets from ConcurrentHashMap. This version uses one ReentrantReadWriteLock per
/// shard and takes the WRITE lock for every upsert, not just for trim/resize. This is a real, deliberate
/// simplification for a first correctness-focused prototype -- it means more serialization per shard
/// than Direction A's ConcurrentIndexedTable shards have, and needs to be measured, not assumed, before
/// claiming a performance verdict against either direction alone. One upside of this simplification: the
/// adaptive-capacity signal tracking below can use plain double fields instead of the lock-free
/// DoubleAdder/DoubleAccumulator AdaptiveConcurrentIndexedTable needed -- every upsert already holds the
/// shard's exclusive write lock, so there is no separate concurrency problem left to solve for the signal.
///
/// Arena lifecycle: one Arena.ofShared() (NOT ofConfined() -- confined arenas restrict access to their
/// creating thread, which would throw for every other thread touching a shared shard) backs every
/// shard's memory. close() closes it once, freeing every shard's memory together.
public class ShardedOffHeapGroupTable implements AutoCloseable {
  private static final double MEDIUM_TOP1_SHARE_THRESHOLD = 0.01;
  private static final double SMALL_TOP1_SHARE_THRESHOLD = 0.05;
  // Scaling the gate by shard.size() (distinct-key count) was tried and measured wrong, not just
  // mistuned: a Zipfian tail keeps adding distinct keys as cardinality grows even after the HEAD (the
  // only thing top1Share actually depends on) has long since stabilized, so the required-sample count
  // kept climbing with cardinality while the real per-shard sample budget (driven by query throughput,
  // not key cardinality) does not. Measured effect: at skew=1.0 the gate stopped triggering for most
  // shards at all (1.5-9% memory reduction instead of the 96-98% Direction A achieves with the same
  // signal/thresholds) and, backwards from what adaptive capacity needs, got WORSE as cardinality grew
  // from 320K to 20M. Reverted to a flat sample count, same category of empirically-chosen heuristic as
  // the top1Share thresholds above -- 5000 is comfortably above the ~500-1000 range where the realistic
  // (non-unit-valued) uniform workload still showed false-positive shrinkage, and well under a shard's
  // typical total sample volume so the skewed case still has room to trigger.
  private static final long MIN_SAMPLES_BEFORE_ADAPTATION = 5000;

  private final int _numShards;
  private final boolean _adaptiveCapacity;
  private final int _fullCapacity;
  private final int _mediumCapacity;
  private final int _smallCapacity;
  private final Arena _arena;
  private final OffHeapGroupTable[] _shards;
  private final ReentrantReadWriteLock[] _locks;

  // Adaptive-capacity bookkeeping, one slot per shard. Plain arrays, not atomics: every write happens
  // under that shard's write lock already (see upsert()), so there is no separate race to guard against.
  private final double[] _runningMax;
  private final double[] _runningTotal;
  private final long[] _sampleCount; // COUNT of upserts, separate from runningTotal (a SUM of values) --
  // gating on runningTotal directly was a bug: it only coincidentally equals the sample count when every
  // upserted value happens to be 1.0. With real (non-unit) values the gate could pass after a handful of
  // upserts, when runningMax is still just "the single largest raw value seen so far" -- pure small-sample
  // noise, not a real signal. Caught by testing against a realistic random-value workload, not just the
  // all-1.0 workload every earlier verification used.
  private final int[] _currentTier; // 0 = full, 1 = medium, 2 = small; monotonic, only ever increases

  public ShardedOffHeapGroupTable(int numShards, int perShardInitialCapacity) {
    this(numShards, perShardInitialCapacity, perShardInitialCapacity, false);
  }

  /// @param fullCapacity     Per-shard capacity used when adaptiveCapacity is false, or as the FULL tier
  ///                         when it's true. MEDIUM = fullCapacity/10, SMALL = fullCapacity/100 (floor 1),
  ///                         mirroring AdaptiveConcurrentIndexedTable's tiers exactly.
  /// @param adaptiveCapacity If true, each shard narrows its own finishAllShards() target capacity based
  ///                         on its local top1Share signal (see updateSignal below) -- same signal,
  ///                         thresholds, and monotonic-shrink-only behavior as Direction A's
  ///                         AdaptiveConcurrentIndexedTable, just applied to an off-heap shard instead of
  ///                         a ConcurrentIndexedTable one.
  public ShardedOffHeapGroupTable(int numShards, int perShardInitialCapacity, int fullCapacity,
      boolean adaptiveCapacity) {
    _numShards = numShards;
    _adaptiveCapacity = adaptiveCapacity;
    _fullCapacity = fullCapacity;
    _mediumCapacity = Math.max(1, fullCapacity / 10);
    _smallCapacity = Math.max(1, fullCapacity / 100);
    _arena = Arena.ofShared();
    _shards = new OffHeapGroupTable[numShards];
    _locks = new ReentrantReadWriteLock[numShards];
    _runningMax = new double[numShards];
    _runningTotal = new double[numShards];
    _sampleCount = new long[numShards];
    _currentTier = new int[numShards];
    for (int i = 0; i < numShards; i++) {
      _shards[i] = new OffHeapGroupTable(perShardInitialCapacity, _arena);
      _locks[i] = new ReentrantReadWriteLock();
    }
  }

  private int shardFor(int key) {
    return Math.floorMod(Integer.hashCode(key), _numShards);
  }

  /// Thread-safe: takes the target shard's write lock for the full duration of the upsert. Safe to call
  /// concurrently from multiple threads on the same table (that's the whole point) -- different keys
  /// hashing to different shards proceed fully in parallel; keys hashing to the same shard serialize.
  public void upsert(int key, double value) {
    int shard = shardFor(key);
    ReentrantReadWriteLock lock = _locks[shard];
    lock.writeLock().lock();
    try {
      double updatedValue = _shards[shard].upsert(key, value);
      if (_adaptiveCapacity) {
        updateSignal(shard, value, updatedValue);
      }
    } finally {
      lock.writeLock().unlock();
    }
  }

  /// Same top1Share signal as AdaptiveConcurrentIndexedTable: runningTotal accumulates each upsert's raw
  /// incoming value (valid for SUM-like additive aggregates only -- same documented scope as everywhere
  /// else in this investigation); runningMax must reflect the key's current AGGREGATED value (a hot
  /// key's dominance only shows up in its accumulated total, never in one call's raw contribution).
  private void updateSignal(int shard, double rawValue, double updatedValue) {
    if (_currentTier[shard] == 2) {
      return; // already at the smallest tier, nothing left to shrink to
    }
    _runningTotal[shard] += rawValue;
    _sampleCount[shard]++;
    if (updatedValue > _runningMax[shard]) {
      _runningMax[shard] = updatedValue;
    }
    if (_sampleCount[shard] < MIN_SAMPLES_BEFORE_ADAPTATION) {
      // Too few samples yet -- top1Share is noise-dominated (a lucky early high-value draw, or a key
      // that happened to repeat a couple of times first) rather than a real concentration signal.
      return;
    }
    double top1Share = _runningMax[shard] / _runningTotal[shard];
    int targetTier = top1Share < MEDIUM_TOP1_SHARE_THRESHOLD ? 0
        : top1Share < SMALL_TOP1_SHARE_THRESHOLD ? 1 : 2;
    if (targetTier > _currentTier[shard]) {
      _currentTier[shard] = targetTier; // monotonic: only ever shrinks, never grows back
    }
  }

  private int capacityFor(int shard) {
    if (!_adaptiveCapacity) {
      return _fullCapacity;
    }
    return switch (_currentTier[shard]) {
      case 1 -> _mediumCapacity;
      case 2 -> _smallCapacity;
      default -> _fullCapacity;
    };
  }

  /// Trims every shard down to its capacity (fixed at fullCapacity, or per-shard-adaptive if this table
  /// was built with adaptiveCapacity=true), matching finish() semantics (called once, after all upserts
  /// are done, not continuously). Sequential across shards for this first prototype -- could be
  /// parallelized (matching ShardedIndexedTable.finishShardsInParallel) but that is a performance
  /// refinement, not needed to validate correctness first.
  public void finishAllShards() {
    for (int i = 0; i < _numShards; i++) {
      ReentrantReadWriteLock lock = _locks[i];
      lock.writeLock().lock();
      try {
        _shards[i].trimTo(capacityFor(i));
      } finally {
        lock.writeLock().unlock();
      }
    }
  }

  public int totalSize() {
    int total = 0;
    for (OffHeapGroupTable shard : _shards) {
      total += shard.size();
    }
    return total;
  }

  /// Visits every surviving (key, value) pair across all shards, post-finishAllShards(). No lock taken --
  /// caller must ensure no concurrent upserts are in flight (matches finish()-then-read usage pattern).
  public void forEachEntry(EntryVisitor visitor) {
    for (OffHeapGroupTable shard : _shards) {
      int size = shard.size();
      for (int i = 0; i < size; i++) {
        visitor.visit(shard.keyAt(i), shard.valueAt(i));
      }
    }
  }

  public interface EntryVisitor {
    void visit(int key, double value);
  }

  @Override
  public void close() {
    _arena.close();
  }
}
