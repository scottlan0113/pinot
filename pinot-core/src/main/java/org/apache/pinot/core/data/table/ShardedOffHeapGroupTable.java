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
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAccumulator;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;
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
/// shard (or per sub-segment, see below) and takes the WRITE lock for every upsert, not just for
/// trim/resize. This is a real, deliberate simplification for a first correctness-focused prototype --
/// it means more serialization per shard than Direction A's ConcurrentIndexedTable shards have, and
/// needs to be measured, not assumed, before claiming a performance verdict against either direction
/// alone. Adaptive-capacity signal tracking (below) uses the same DoubleAdder/DoubleAccumulator/
/// AtomicInteger-with-CAS pattern as AdaptiveConcurrentIndexedTable, not plain fields -- with
/// sub-segmenting, multiple independently-locked sub-segments of the same OUTER shard can update that
/// shard's signal concurrently, so a single exclusive lock is no longer guaranteed to serialize every
/// write to it the way it did in the numSubSegments=1-only original version of this class.
///
/// A read-lock-based fast path (fixed capacity takes a shared read lock plus an atomic CAS for an
/// existing key, escalating to the write lock only for a genuinely new key) was tried and MEASURED TO
/// REGRESS performance -- see DESIGN.md Sec 6.2/6.4 for the full story and profiling evidence. Reverted.
/// The write-lock-for-every-upsert design here is not an unexamined default; it was compared directly
/// against a more fine-grained alternative and won.
///
/// Sub-segmenting: each outer shard can optionally be split into `numSubSegments`
/// independently-locked OffHeapGroupTable instances instead of one. This keeps the exact same
/// exclusive-lock-per-critical-section model that beat the read-lock fast path above -- resize is
/// still always safely inside a normal write lock, no new correctness hazard -- just at finer
/// granularity, so keys hashing to the same OUTER shard but different sub-segments no longer contend.
/// The top1Share signal is tracked per OUTER shard regardless of numSubSegments, specifically to avoid
/// multiplying the memory-ceiling problem (Sec 4.2) that more outer shards would cause -- so signal
/// tracking (below) has to be safe under concurrent access from multiple independently-locked
/// sub-segments of the same shard, unlike a plain field would be. finishAllShards() merges all of a
/// shard's sub-segments into sub-segment 0 (single-threaded at that point, reusing OffHeapGroupTable's
/// own upsert()) before trimming, so totalSize()/forEachEntry() only ever need to look at sub-segment
/// 0 per shard -- consistent whether numSubSegments is 1 or more.
///
/// numSubSegments does NOT monotonically help -- a JMH sweep (DESIGN.md Sec 6.2) found 4 to be the
/// clear sweet spot (a real 6.9-8.1% win over numSubSegments=1), with 8 back near the numSubSegments=1
/// baseline and 16/32 measurably WORSE than never sub-segmenting at all. Callers should default to
/// numSubSegments=4 rather than assuming higher is better. The initial hypothesis -- smaller
/// per-sub-segment initial capacity (from dividing perShardInitialCapacity by numSubSegments) causing
/// more frequent resize -- was tested directly via divideInitialCapacityAcrossSubSegments=false below
/// and REFUTED: giving every sub-segment the full undivided capacity made K=16/32 dramatically worse
/// (e.g. K=32: 48,266 -> 85,800 us/op), not better, because that also scales TOTAL initial allocation
/// with numSubSegments rather than isolating resize frequency at a constant total. Why K=16/32 regress
/// at a properly constant total capacity remains an open question. The numSubSegments=4 sweet spot was
/// found on fixed capacity, but transfers to adaptive capacity too (6.3% win, DESIGN.md Sec 6.5) --
/// only 4 has been checked for adaptive, not the full 8/16/32 sweep, since the same locking model
/// (still exclusive, just finer-grained) makes it a plausible extrapolation rather than a fresh
/// question, but it hasn't been independently re-swept.
///
/// Multi-column keys: numKeyColumns > 1 (DESIGN.md Sec 4.6's "multi-column GROUP BY... not yet tested"
/// scope item) switches every shard's underlying OffHeapGroupTable into its multi-column mode and
/// requires the upsert(int[], double) overload instead of upsert(int, double) -- see OffHeapGroupTable's
/// own class Javadoc for why the two modes are mutually exclusive per table instance. Shard and
/// sub-segment routing both hash the full composite key (not just the first column), and adaptive
/// capacity's top1Share signal is unaffected either way -- it only ever looks at VALUES.
///
/// Non-SUM aggregation: OffHeapGroupTable.AggregationType (MIN/MAX, DESIGN.md Sec 6.7) is supported here
/// too, but NOT combined with adaptiveCapacity=true -- the constructor throws in that combination, since
/// top1Share's "sum of raw contributions approximates the aggregated total" assumption is specific to SUM.
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
  private final int _numKeyColumns;
  private final boolean _adaptiveCapacity;
  private final int _fullCapacity;
  private final int _mediumCapacity;
  private final int _smallCapacity;
  private final Arena _arena;
  private final OffHeapGroupTable[][] _shards; // [outer shard][sub-segment]
  private final ReentrantReadWriteLock[][] _locks; // [outer shard][sub-segment]
  // Set during finishAllShards() (sequential across shards, see its own Javadoc, so a plain boolean is
  // safe -- no concurrent writers). Lets a caller ask "did any shard actually discard records," the
  // signal IndexedTable.isTrimmed() reports for the on-heap tables (ShardedOffHeapIndexedTable needs
  // this to implement that same contract).
  private boolean _anyShardTrimmed;

  // Adaptive-capacity bookkeeping, one slot per OUTER shard -- shared across that shard's
  // sub-segments, which each hold their own independent lock, so these must be genuinely
  // concurrency-safe rather than plain fields (see class Javadoc). Same primitive types
  // AdaptiveConcurrentIndexedTable uses, for the same reason (that class's ConcurrentHashMap shards
  // permit real concurrent access to one shard the way sub-segments here now do too) -- but NOT the
  // same gate logic: AdaptiveConcurrentIndexedTable's own gate compares _runningTotal.sum() (a SUM of
  // values) against a sample-count-shaped threshold, apparently the exact bug class fixed here (see
  // DESIGN.md Sec 4.6) -- not yet fixed there. _sampleCount below is a real, separate LongAdder count,
  // not reused from the value sum.
  private final DoubleAccumulator[] _runningMax;
  private final DoubleAdder[] _runningTotal;
  private final LongAdder[] _sampleCount;
  private final AtomicInteger[] _currentTier; // 0 = full, 1 = medium, 2 = small; monotonic via CAS below

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
  ///                        (see class Javadoc). 1 = existing behavior exactly. Valid with either
  ///                        adaptiveCapacity value -- the top1Share signal below is concurrency-safe
  ///                        regardless of numSubSegments.
  public ShardedOffHeapGroupTable(int numShards, int perShardInitialCapacity, int fullCapacity,
      boolean adaptiveCapacity, int numSubSegments) {
    this(numShards, perShardInitialCapacity, fullCapacity, adaptiveCapacity, numSubSegments, true);
  }

  /// @param divideInitialCapacityAcrossSubSegments If true (matches the 5-arg constructor above and
  ///                        all prior behavior), perShardInitialCapacity is divided by numSubSegments
  ///                        for each sub-segment's own initial off-heap allocation -- more
  ///                        numSubSegments means less initial capacity per piece. If false, EVERY
  ///                        sub-segment starts at perShardInitialCapacity directly (numSubSegments
  ///                        times more total initial memory for the shard). Was added to test whether
  ///                        smaller per-sub-segment capacity (more frequent resize) explained the
  ///                        higher-numSubSegments regression (DESIGN.md Sec 6.2) -- it does not:
  ///                        `false` measured DRAMATICALLY slower (K=32: 48,266 -> 85,800 us/op), not
  ///                        faster, because it also scales TOTAL initial allocation with
  ///                        numSubSegments instead of holding it constant. Kept as a real, working
  ///                        knob (over-allocating capacity is a genuine, separate cost worth being
  ///                        able to reproduce/avoid), not as a fix for the original regression, which
  ///                        this ruled out rather than explained.
  public ShardedOffHeapGroupTable(int numShards, int perShardInitialCapacity, int fullCapacity,
      boolean adaptiveCapacity, int numSubSegments, boolean divideInitialCapacityAcrossSubSegments) {
    this(numShards, perShardInitialCapacity, fullCapacity, adaptiveCapacity, numSubSegments,
        divideInitialCapacityAcrossSubSegments, 1);
  }

  /// @param numKeyColumns Number of int GROUP BY key columns (added while testing DESIGN.md's
  ///                       "multi-column GROUP BY... not yet tested" scope item, Sec 4.6). 1 = existing
  ///                       behavior exactly (upsert(int, double), the scalar overload). > 1 switches
  ///                       every shard's underlying OffHeapGroupTable into its multi-column mode
  ///                       (upsert(int[], double) required instead -- see OffHeapGroupTable's class
  ///                       Javadoc for why the two modes cannot be mixed on one table). Orthogonal to
  ///                       adaptiveCapacity and numSubSegments -- the top1Share signal only depends on
  ///                       VALUES, and sub-segment/shard routing only depend on a hash of the key,
  ///                       neither of which cares how many int columns make up that key.
  public ShardedOffHeapGroupTable(int numShards, int perShardInitialCapacity, int fullCapacity,
      boolean adaptiveCapacity, int numSubSegments, boolean divideInitialCapacityAcrossSubSegments,
      int numKeyColumns) {
    this(numShards, perShardInitialCapacity, fullCapacity, adaptiveCapacity, numSubSegments,
        divideInitialCapacityAcrossSubSegments, numKeyColumns, OffHeapGroupTable.AggregationType.SUM);
  }

  /// @param aggregationType SUM (default everywhere else), MIN, or MAX -- added while testing DESIGN.md's
  ///                       "single numeric SUM-like (additive) aggregate" scope limitation (Sec 4.6).
  ///                       NOT orthogonal to adaptiveCapacity, unlike numKeyColumns: the top1Share signal
  ///                       (below) assumes "sum of every upsert's raw contribution equals the sum of all
  ///                       keys' final aggregated values", which holds for SUM but not MIN/MAX -- a
  ///                       MIN/MAX table with a genuinely low/high value early on would look artificially
  ///                       "concentrated" to a signal built for SUM, an incorrect basis for shrinking.
  ///                       Combining adaptiveCapacity=true with a non-SUM aggregationType throws rather
  ///                       than silently producing a meaningless signal.
  public ShardedOffHeapGroupTable(int numShards, int perShardInitialCapacity, int fullCapacity,
      boolean adaptiveCapacity, int numSubSegments, boolean divideInitialCapacityAcrossSubSegments,
      int numKeyColumns, OffHeapGroupTable.AggregationType aggregationType) {
    this(numShards, perShardInitialCapacity, fullCapacity, adaptiveCapacity, numSubSegments,
        divideInitialCapacityAcrossSubSegments, numKeyColumns, aggregationType, false);
  }

  /// @param segmentZeroFullCapacity Root-caused 2026-08-28 (DESIGN.md Sec 6.2 "Root cause... found"):
  ///                       sub-segment 0 is the single-threaded MERGE TARGET in finishAllShards() --
  ///                       every other sub-segment's records get re-upserted into it -- so at high
  ///                       numSubSegments, starting it at the same small divided capacity as every
  ///                       other sub-segment (the default) forces it through repeated growData()/
  ///                       growIndex() calls while absorbing everyone else's data, a real cost
  ///                       (measured ~13% of total op time at numSubSegments=32). If true, sub-segment
  ///                       0 alone gets perShardInitialCapacity (undivided), while sub-segments 1..N-1
  ///                       still respect divideInitialCapacityAcrossSubSegments exactly as before --
  ///                       unlike that flag (which changes ALL sub-segments and was measured to regress
  ///                       the concurrent phase badly via 32x total over-allocation at
  ///                       numSubSegments=32), this only grows ONE sub-segment per shard, a much
  ///                       smaller and more targeted change. Default false (existing behavior,
  ///                       unchanged) -- not yet confirmed to be a net win end-to-end (only
  ///                       finishAllShards() in isolation has been measured so far), so not the
  ///                       default until it is.
  public ShardedOffHeapGroupTable(int numShards, int perShardInitialCapacity, int fullCapacity,
      boolean adaptiveCapacity, int numSubSegments, boolean divideInitialCapacityAcrossSubSegments,
      int numKeyColumns, OffHeapGroupTable.AggregationType aggregationType, boolean segmentZeroFullCapacity) {
    if (adaptiveCapacity && aggregationType != OffHeapGroupTable.AggregationType.SUM) {
      throw new IllegalArgumentException(
          "adaptiveCapacity requires aggregationType == SUM (top1Share is not a meaningful signal for "
              + aggregationType + ") -- got adaptiveCapacity=true with aggregationType=" + aggregationType);
    }
    _numShards = numShards;
    _numSubSegments = numSubSegments;
    _numKeyColumns = numKeyColumns;
    _adaptiveCapacity = adaptiveCapacity;
    _fullCapacity = fullCapacity;
    _mediumCapacity = Math.max(1, fullCapacity / 10);
    _smallCapacity = Math.max(1, fullCapacity / 100);
    _arena = Arena.ofShared();
    _shards = new OffHeapGroupTable[numShards][numSubSegments];
    _locks = new ReentrantReadWriteLock[numShards][numSubSegments];
    _runningMax = new DoubleAccumulator[numShards];
    _runningTotal = new DoubleAdder[numShards];
    _sampleCount = new LongAdder[numShards];
    _currentTier = new AtomicInteger[numShards];
    int perSubSegmentInitialCapacity = divideInitialCapacityAcrossSubSegments
        ? Math.max(1, perShardInitialCapacity / numSubSegments) : perShardInitialCapacity;
    for (int i = 0; i < numShards; i++) {
      _runningMax[i] = new DoubleAccumulator(Double::max, 0.0);
      _runningTotal[i] = new DoubleAdder();
      _sampleCount[i] = new LongAdder();
      _currentTier[i] = new AtomicInteger(0);
      for (int j = 0; j < numSubSegments; j++) {
        int thisCapacity = (j == 0 && segmentZeroFullCapacity) ? perShardInitialCapacity : perSubSegmentInitialCapacity;
        _shards[i][j] = new OffHeapGroupTable(thisCapacity, numKeyColumns, _arena, aggregationType);
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

  /// Multi-column counterparts to shardFor/subSegmentFor above, same decorrelation reasoning: shard
  /// selection uses a simple/raw hash (Arrays.hashCode, the multi-column analogue of
  /// Integer.hashCode(key) being just `key` itself for the single-column case) while sub-segment
  /// selection uses the avalanche-mixed hash, so the two choices don't correlate.
  private int shardForMulti(int[] keys) {
    return Math.floorMod(Arrays.hashCode(keys), _numShards);
  }

  private int subSegmentForMulti(int[] keys) {
    if (_numSubSegments == 1) {
      return 0;
    }
    return Math.floorMod(avalancheHash(keys), _numSubSegments);
  }

  private static int avalancheHash(int[] keys) {
    int h = 1;
    for (int k : keys) {
      h = h * 31 + k;
    }
    h *= 0x9E3779B1;
    return h ^ (h >>> 16);
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

  /// Multi-column counterpart to upsert(int, double) above -- identical locking/signal-tracking
  /// behavior, keyed by keys.length int columns instead of one. Only valid on a table constructed with
  /// numKeyColumns > 1 matching keys.length (enforced by the underlying OffHeapGroupTable).
  public void upsert(int[] keys, double value) {
    int shard = shardForMulti(keys);
    int subSegment = subSegmentForMulti(keys);
    ReentrantReadWriteLock lock = _locks[shard][subSegment];
    lock.writeLock().lock();
    try {
      double updatedValue = _shards[shard][subSegment].upsert(keys, value);
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
  /// Called with `shard` = the OUTER shard index, never the sub-segment -- multiple sub-segments of the
  /// same outer shard can call this concurrently (each holding only its OWN sub-segment's lock), which
  /// is exactly why every field this method touches is a concurrency-safe accumulator/CAS type rather
  /// than a plain field, unlike this class's original numSubSegments=1-only version.
  private void updateSignal(int shard, double rawValue, double updatedValue) {
    int currentTier = _currentTier[shard].get();
    if (currentTier == 2) {
      return; // already at the smallest tier, nothing left to shrink to
    }
    _runningTotal[shard].add(rawValue);
    _sampleCount[shard].increment(); // a real COUNT, not reused from runningTotal's sum -- see field doc
    _runningMax[shard].accumulate(updatedValue);
    if (_sampleCount[shard].sum() < MIN_SAMPLES_BEFORE_ADAPTATION) {
      // Too few samples yet -- top1Share is noise-dominated (a lucky early high-value draw, or a key
      // that happened to repeat a couple of times first) rather than a real concentration signal.
      return;
    }
    double top1Share = _runningMax[shard].get() / _runningTotal[shard].sum();
    int targetTier = top1Share < MEDIUM_TOP1_SHARE_THRESHOLD ? 0
        : top1Share < SMALL_TOP1_SHARE_THRESHOLD ? 1 : 2;
    // CAS retry loop, mirroring AdaptiveConcurrentIndexedTable exactly: monotonic (only ever shrinks),
    // and safe against another sub-segment's thread racing to advance the same shard's tier at the
    // same time -- whichever CAS wins, the loser just re-reads and re-checks against the new state.
    while (targetTier > currentTier) {
      if (_currentTier[shard].compareAndSet(currentTier, targetTier)) {
        break;
      }
      currentTier = _currentTier[shard].get();
    }
  }

  private int capacityFor(int shard) {
    if (!_adaptiveCapacity) {
      return _fullCapacity;
    }
    return switch (_currentTier[shard].get()) {
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
          if (_numKeyColumns == 1) {
            for (int k = 0; k < size; k++) {
              primary.upsert(other.keyAt(k), other.valueAt(k));
            }
          } else {
            for (int k = 0; k < size; k++) {
              primary.upsert(other.keysAt(k), other.valueAt(k));
            }
          }
        }
        if (primary.trimTo(capacityFor(i))) {
          _anyShardTrimmed = true;
        }
      } finally {
        for (int j = 0; j < _numSubSegments; j++) {
          _locks[i][j].writeLock().unlock();
        }
      }
    }
  }

  /// Valid post-finishAllShards() only. True if trimming actually discarded records on at least one
  /// shard -- false if every shard's final size was already at or under its capacity, in which case
  /// nothing was lost regardless of capacity/adaptive-capacity settings.
  public boolean anyShardTrimmed() {
    return _anyShardTrimmed;
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

  /// Multi-column counterpart to forEachEntry() -- same finishAllShards()-then-read contract.
  public void forEachMultiColumnEntry(MultiColumnEntryVisitor visitor) {
    for (OffHeapGroupTable[] shard : _shards) {
      OffHeapGroupTable merged = shard[0];
      int size = merged.size();
      for (int i = 0; i < size; i++) {
        visitor.visit(merged.keysAt(i), merged.valueAt(i));
      }
    }
  }

  public interface MultiColumnEntryVisitor {
    void visit(int[] keys, double value);
  }

  @Override
  public void close() {
    _arena.close();
  }
}
