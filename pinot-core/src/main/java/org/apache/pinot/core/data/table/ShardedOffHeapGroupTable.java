/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.pinot.core.data.table;

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
/// A read-lock-based fast path (fixed capacity takes a shared read lock plus an atomic CAS for an
/// existing key, escalating to the write lock only for a genuinely new key) was tried and MEASURED TO
/// REGRESS performance -- see DESIGN.md Sec 6.2/6.4 for the full story and profiling evidence. Reverted.
/// The write-lock-for-every-upsert design here is not an unexamined default; it was compared directly
/// against a more fine-grained alternative and won.
///
/// Sub-segmenting (fixed capacity only): each outer shard can optionally be split into
/// `numSubSegments` independently-locked OffHeapGroupTable instances instead of one. This keeps the
/// exact same exclusive-lock-per-critical-section model that beat the read-lock fast path above --
/// resize is still always safely inside a normal write lock, no new correctness hazard -- just at
/// finer granularity, so keys hashing to the same OUTER shard but different sub-segments no longer
/// contend. Deliberately NOT extended to adaptive capacity yet: the top1Share signal is tracked per
/// OUTER shard specifically to avoid multiplying the memory-ceiling problem (Sec 4.2) that more outer
/// shards would cause, and splitting the signal-tracking fields across multiple independently-locked
/// sub-segments needs its own concurrency-safety treatment (DoubleAdder/LongAdder/CAS-based max,
/// mirroring AdaptiveConcurrentIndexedTable) that hasn't been done -- passing numSubSegments>1 with
/// adaptiveCapacity=true throws rather than silently doing the wrong thing. finishAllShards() merges
/// all of a shard's sub-segments into sub-segment 0 (single-threaded at that point, reusing
/// OffHeapGroupTable's own upsert()) before trimming, so totalSize()/forEachEntry() only ever need to
/// look at sub-segment 0 per shard -- consistent whether numSubSegments is 1 or more.
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
  private final int _numSubSegments;
  private final boolean _adaptiveCapacity;
  private final int _fullCapacity;
  private final int _mediumCapacity;
  private final int _smallCapacity;
  private final Arena _arena;
  private final OffHeapGroupTable[][] _shards; // [outer shard][sub-segment]
  private final ReentrantReadWriteLock[][] _locks; // [outer shard][sub-segment]

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
    this(numShards, perShardInitialCapacity, perShardInitialCapacity, false, 1);
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
    this(numShards, perShardInitialCapacity, fullCapacity, adaptiveCapacity, 1);
  }

  /// @param numSubSegments Splits each outer shard into this many independently-locked sub-segments
  ///                        (see class Javadoc). 1 = existing behavior exactly. Only valid with
  ///                        adaptiveCapacity=false.
  public ShardedOffHeapGroupTable(int numShards, int perShardInitialCapacity, int fullCapacity,
      boolean adaptiveCapacity, int numSubSegments) {
    if (adaptiveCapacity && numSubSegments > 1) {
      throw new IllegalArgumentException("Sub-segmenting is not yet supported for adaptive capacity");
    }
    _numShards = numShards;
    _numSubSegments = numSubSegments;
    _adaptiveCapacity = adaptiveCapacity;
    _fullCapacity = fullCapacity;
    _mediumCapacity = Math.max(1, fullCapacity / 10);
    _smallCapacity = Math.max(1, fullCapacity / 100);
    _arena = Arena.ofShared();
    _shards = new OffHeapGroupTable[numShards][numSubSegments];
    _locks = new ReentrantReadWriteLock[numShards][numSubSegments];
    _runningMax = new double[numShards];
    _runningTotal = new double[numShards];
    _sampleCount = new long[numShards];
    _currentTier = new int[numShards];
    int perSubSegmentInitialCapacity = Math.max(1, perShardInitialCapacity / numSubSegments);
    for (int i = 0; i < numShards; i++) {
      for (int j = 0; j < numSubSegments; j++) {
        _shards[i][j] = new OffHeapGroupTable(perSubSegmentInitialCapacity, _arena);
        _locks[i][j] = new ReentrantReadWriteLock();
      }
    }
  }

  private int shardFor(int key) {
    return Math.floorMod(Integer.hashCode(key), _numShards);
  }

  /// A second, independent hash for sub-segment selection -- Integer.hashCode(int) is just the int
  /// itself, so shardFor() is really `key mod numShards`. Reusing that same raw value (mod
  /// numSubSegments) for sub-segment selection would correlate the two choices (e.g. every key that's
  /// a multiple of numShards would also land in the same sub-segment). The multiplicative mixing
  /// constant here is the same one OffHeapGroupTable's own index hash() already uses -- a proven
  /// avalanche mix, not a new unverified choice.
  private int subSegmentFor(int key) {
    if (_numSubSegments == 1) {
      return 0;
    }
    int h = key * 0x9E3779B1;
    h ^= h >>> 16;
    return Math.floorMod(h, _numSubSegments);
  }

  /// Thread-safe: takes the target shard's (and sub-segment's) write lock for the full duration of the
  /// upsert. Safe to call concurrently from multiple threads on the same table (that's the whole point)
  /// -- different keys hashing to different shards, or to different sub-segments of the same shard,
  /// proceed fully in parallel; keys hashing to the same shard AND sub-segment serialize.
  public void upsert(int key, double value) {
    int shard = shardFor(key);
    int subSegment = subSegmentFor(key);
    ReentrantReadWriteLock lock = _locks[shard][subSegment];
    lock.writeLock().lock();
    try {
      double updatedValue = _shards[shard][subSegment].upsert(key, value);
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
  /// are done, not continuously). If numSubSegments > 1, every sub-segment beyond the first is merged
  /// into sub-segment 0 first (single-threaded at this point -- all locks for the shard are held for
  /// the duration -- so OffHeapGroupTable's own upsert() is safe to reuse directly rather than writing
  /// new merge logic), then that merged sub-segment 0 is trimmed exactly as the single-sub-segment case
  /// always was. Sequential across shards for this first prototype -- could be parallelized (matching
  /// ShardedIndexedTable.finishShardsInParallel) but that is a performance refinement, not needed to
  /// validate correctness first.
  public void finishAllShards() {
    for (int i = 0; i < _numShards; i++) {
      for (int j = 0; j < _numSubSegments; j++) {
        _locks[i][j].writeLock().lock();
      }
      try {
        OffHeapGroupTable primary = _shards[i][0];
        for (int j = 1; j < _numSubSegments; j++) {
          OffHeapGroupTable other = _shards[i][j];
          int size = other.size();
          for (int k = 0; k < size; k++) {
            primary.upsert(other.keyAt(k), other.valueAt(k));
          }
        }
        primary.trimTo(capacityFor(i));
      } finally {
        for (int j = 0; j < _numSubSegments; j++) {
          _locks[i][j].writeLock().unlock();
        }
      }
    }
  }

  /// Valid post-finishAllShards() only -- see forEachEntry().
  public int totalSize() {
    int total = 0;
    for (OffHeapGroupTable[] shard : _shards) {
      total += shard[0].size(); // finishAllShards() has already merged every sub-segment into [0]
    }
    return total;
  }

  /// Visits every surviving (key, value) pair across all shards, post-finishAllShards() -- by that
  /// point every shard's sub-segments have been merged into sub-segment 0 (see finishAllShards()), so
  /// only [0] needs visiting; before finishAllShards() has run, a shard's true contents are split
  /// across all its sub-segments and this would silently under-report. No lock taken -- caller must
  /// ensure no concurrent upserts are in flight (matches finish()-then-read usage pattern).
  public void forEachEntry(EntryVisitor visitor) {
    for (OffHeapGroupTable[] shard : _shards) {
      OffHeapGroupTable merged = shard[0];
      int size = merged.size();
      for (int i = 0; i < size; i++) {
        visitor.visit(merged.keyAt(i), merged.valueAt(i));
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
