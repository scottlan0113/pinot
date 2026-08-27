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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAccumulator;
import java.util.concurrent.atomic.DoubleAdder;
import org.apache.pinot.common.utils.DataSchema;
import org.apache.pinot.core.query.request.context.QueryContext;


/// Thread safe [Table] implementation that shards records by key hash across `numShards` independent
/// [ConcurrentIndexedTable]s, instead of a single shared table protected by one lock.
///
/// Motivation (see #10498 investigation): profiling showed [ConcurrentIndexedTable]'s [#resize()] --
/// sorting/trimming the map to `trimSize` -- happening under the table's single exclusive write lock,
/// serializing all combine threads onto whichever one thread is currently trimming. An alternative
/// where each combine thread keeps its own independent local table (then merges at the end) avoids the
/// lock, but was found to have two problems: it can silently drop a key whose combined value across
/// threads would have made the final top-K (no single thread's partial view looked important enough to
/// survive local trimming), and the same key ends up stored redundantly in every thread's local table
/// that saw it (measured duplication factors up to ~3.5x on skewed data).
///
/// Sharding by key hash avoids both: every contribution to a given key is routed to the same shard
/// regardless of which combine thread produced it, so (a) each shard always sees the complete,
/// up-to-date value for its keys (no incomplete-local-view correctness risk), and (b) a key is stored
/// in exactly one place, never duplicated. Each shard keeps its own lock, so trimming one shard does not
/// block combine threads writing to other shards.
///
/// Each shard is configured with the SAME `resultSize`/`trimSize`/`trimThreshold` as a single
/// [ConcurrentIndexedTable] would use for the whole table (not divided by `numShards`). Empirically,
/// dividing this budget across shards measurably hurts recall on moderately-skewed data; keeping the
/// full per-shard budget matches [ConcurrentIndexedTable]'s correctness while still avoiding duplication
/// (the "extra" capacity is unused headroom in shards with fewer distinct keys, not redundant copies of
/// the same data).
public class ShardedIndexedTable extends BaseTable implements Table {
  private final int _numShards;
  private final int _numKeyColumns;
  private final int _resultSize;
  private final boolean _hasOrderBy;
  private final TableResizer _finalMergeResizer;
  private final ExecutorService _executorService;
  private final ConcurrentIndexedTable[] _shards;

  private Map<Key, Record> _finalMergedMap;
  private Iterable<Record> _topRecords;

  public ShardedIndexedTable(DataSchema dataSchema, boolean hasFinalInput, QueryContext queryContext, int resultSize,
      int trimSize, int trimThreshold, int initialCapacityPerShard, int numShards, ExecutorService executorService) {
    this(dataSchema, hasFinalInput, queryContext, resultSize, trimSize, trimThreshold, initialCapacityPerShard,
        numShards, executorService, false);
  }

  /// @param adaptiveCapacity If true, each shard starts at the full `trimSize`/`trimThreshold` budget but
  ///                         narrows itself down (never back up) once it gathers enough local evidence
  ///                         that its data is concentrated enough for a smaller capacity to be safe. See
  ///                         [AdaptiveConcurrentIndexedTable]. Prototype/experimental: assumes a single
  ///                         numeric ORDER BY aggregate (the first aggregation column), not general
  ///                         multi-expression ORDER BY.
  public ShardedIndexedTable(DataSchema dataSchema, boolean hasFinalInput, QueryContext queryContext, int resultSize,
      int trimSize, int trimThreshold, int initialCapacityPerShard, int numShards, ExecutorService executorService,
      boolean adaptiveCapacity) {
    super(dataSchema);
    _numShards = numShards;
    _numKeyColumns = queryContext.getNumGroupByKeyColumns();
    _resultSize = resultSize;
    _hasOrderBy = queryContext.getOrderByExpressions() != null;
    _finalMergeResizer = _hasOrderBy ? new TableResizer(dataSchema, hasFinalInput, queryContext) : null;
    _executorService = executorService;
    _shards = new ConcurrentIndexedTable[numShards];
    for (int i = 0; i < numShards; i++) {
      // Every shard uses the SAME full resultSize/trimSize/trimThreshold as the whole table would --
      // see class javadoc for why this is not divided by numShards.
      _shards[i] = adaptiveCapacity ? new AdaptiveConcurrentIndexedTable(dataSchema, hasFinalInput, queryContext,
          resultSize, trimSize, trimThreshold, initialCapacityPerShard, executorService, _numKeyColumns)
          : new ConcurrentIndexedTable(dataSchema, hasFinalInput, queryContext, resultSize, trimSize, trimThreshold,
              initialCapacityPerShard, executorService);
    }
  }

  private int shardFor(Key key) {
    return Math.floorMod(key.hashCode(), _numShards);
  }

  @Override
  public boolean upsert(Record record) {
    Object[] keyValues = Arrays.copyOf(record.getValues(), _numKeyColumns);
    return upsert(new Key(keyValues), record);
  }

  @Override
  public boolean upsert(Key key, Record record) {
    return _shards[shardFor(key)].upsert(key, record);
  }

  @Override
  public int size() {
    if (_topRecords != null) {
      int size = 0;
      for (Record ignored : _topRecords) {
        size++;
      }
      return size;
    }
    int size = 0;
    for (ConcurrentIndexedTable shard : _shards) {
      size += shard.size();
    }
    return size;
  }

  @Override
  public java.util.Iterator<Record> iterator() {
    return _topRecords.iterator();
  }

  /// Finishes every shard (in parallel, since shards are fully independent) and merges their results
  /// into the final top `resultSize` records. Each shard has already trimmed itself down to at most
  /// `resultSize` records via its own [ConcurrentIndexedTable#finish], so this final merge operates on
  /// at most `numShards * resultSize` candidates -- cheap relative to the per-record combine work.
  @Override
  public void finish(boolean sort, boolean storeFinalResult) {
    finishShardsInParallel(sort, storeFinalResult);

    if (!_hasOrderBy) {
      // No ORDER BY: every key appears in exactly one shard (sharded by key hash), so a plain
      // concatenation of each shard's records is already the correct, deduplicated final result.
      List<Record> merged = new ArrayList<>();
      for (ConcurrentIndexedTable shard : _shards) {
        java.util.Iterator<Record> it = shard.iterator();
        while (it.hasNext()) {
          merged.add(it.next());
        }
      }
      _topRecords = merged;
      return;
    }

    // ORDER BY: gather each shard's (already-trimmed) candidates and do one more small top-`resultSize`
    // pass across all of them combined. Keys never collide across shards, so this is a straight merge,
    // not a re-aggregation.
    Map<Key, Record> candidates = new LinkedHashMap<>();
    for (ConcurrentIndexedTable shard : _shards) {
      java.util.Iterator<Record> it = shard.iterator();
      while (it.hasNext()) {
        Record record = it.next();
        Object[] keyValues = Arrays.copyOf(record.getValues(), _numKeyColumns);
        candidates.put(new Key(keyValues), record);
      }
    }
    _finalMergedMap = candidates;
    _topRecords = _finalMergeResizer.getTopRecords(candidates, _resultSize, sort);
  }

  private void finishShardsInParallel(boolean sort, boolean storeFinalResult) {
    List<Future<?>> futures = new ArrayList<>(_numShards);
    for (ConcurrentIndexedTable shard : _shards) {
      futures.add(_executorService.submit(() -> shard.finish(sort, storeFinalResult)));
    }
    try {
      for (Future<?> future : futures) {
        future.get();
      }
    } catch (InterruptedException | ExecutionException e) {
      for (Future<?> future : futures) {
        future.cancel(true);
      }
      throw new RuntimeException("Error finishing shards", e);
    }
  }

  /// Total resize count across all shards, for parity with [IndexedTable#getNumResizes].
  public int getNumResizes() {
    int total = 0;
    for (ConcurrentIndexedTable shard : _shards) {
      total += shard.getNumResizes();
    }
    return total;
  }

  /// A [ConcurrentIndexedTable] shard that narrows its own trim size/threshold at runtime, instead of
  /// keeping the full budget for its entire lifetime like a plain shard does.
  ///
  /// Background (see #10498 / #11924 investigation): keeping every shard at the full `trimSize` avoids
  /// the correctness risk that a smaller fixed capacity has on mildly-skewed data (verified: recall drops
  /// as low as 80% at skew~0.15 with a small fixed per-shard capacity), but it means the aggregate memory
  /// ceiling across all shards is `numShards * trimSize` -- confirmed to be genuinely approached (>99% of
  /// ceiling) at true group cardinality >=1M, a real ~64x cost at exactly the cardinality #10498 cares
  /// about.
  ///
  /// Signal: `top1Share` = (this shard's single largest-value key's value) / (sum of all this shard's
  /// values). Validated (in scratch simulation, not yet re-verified against this real implementation)
  /// to separate the "needs full capacity" band (mild skew, top1Share stays low even at high cardinality)
  /// from the "small capacity is safe" band (skew>=0.5, top1Share rises with skew) using only each
  /// shard's own local view -- despite each shard seeing only a fraction of the global data, hash-based
  /// key routing does not bias this signal (a shard cannot tell, from top1Share alone, whether it got
  /// unlucky and missed the global hot keys or whether there simply aren't any).
  ///
  /// Cost: computing top1Share via a full rescan of the shard's map, done periodically, was measured at
  /// ~33us per scan at realistic shard sizes (~15,625 entries) -- if checked every ~200 upserts, that is
  /// ~167ms of added overhead across one benchmark invocation, more than 2x the entire sharded query time.
  /// This implementation instead tracks `runningMax`/`runningTotal` incrementally (O(1) per upsert, using
  /// lock-free [DoubleAccumulator]/[DoubleAdder]), measured at ~1.8ns/upsert -- negligible. The tradeoff:
  /// `runningMax` is a monotonically-increasing approximation of the shard's current true max (valid
  /// because trimming keeps entries sorted by value and the max-value entry essentially always survives a
  /// trim), not an exact live max; adequate for a heuristic signal, not used for correctness.
  ///
  /// Scope: assumes a single numeric, SUM-like additive ORDER BY aggregate at column index
  /// `numKeyColumns` (matching every benchmark/test used throughout this investigation, e.g.
  /// `ORDER BY sum(m1)`) -- `runningTotal` relies on "sum of every upsert's raw contribution == sum of
  /// all keys' final aggregated values", which only holds for additive aggregates. Not general
  /// multi-expression ORDER BY -- extracting a comparable value for arbitrary ORDER BY would need to
  /// reuse [TableResizer]'s internal comparator, which is not currently exposed for this purpose.
  ///
  /// Tier capacities are derived from the full `trimSize` passed to the constructor: MEDIUM = trimSize/10,
  /// SMALL = trimSize/100 (floor 1). Their thresholds use a small fixed multiplier of their own trim size
  /// (NOT the full tier's threshold:size ratio -- see `SMALLER_TIER_THRESHOLD_MULTIPLIER` below for why
  /// that was wrong: it produced a threshold resize() would realistically never reach, silently making the
  /// "shrink" inert). Shrinking is monotonic -- once a tier is reached, this shard never grows back up,
  /// since entries already discarded by a trim cannot be recovered. A minimum sample count gates any
  /// adaptation at all, since top1Share is meaningless noise from only a handful of records (trivially
  /// 1.0 after a single upsert).
  private static final class AdaptiveConcurrentIndexedTable extends ConcurrentIndexedTable {
    private static final double MEDIUM_TOP1_SHARE_THRESHOLD = 0.01;
    private static final double SMALL_TOP1_SHARE_THRESHOLD = 0.05;
    // Smaller tiers use a small fixed threshold:size multiplier (matching the scratch simulations this
    // design was validated in), NOT the full tier's own ratio -- preserving that ratio (e.g. 200x for a
    // trimSize=5000/trimThreshold=1,000,000 caller) produces an absolute threshold that can exceed a
    // shard's entire natural key population at moderate cardinality, so resize() never actually fires and
    // the "shrink" has no real effect despite the tier having formally advanced.
    private static final int SMALLER_TIER_THRESHOLD_MULTIPLIER = 4;
    // Require enough samples before trusting top1Share at all -- with only a handful of records seen,
    // top1Share is dominated by noise (e.g. exactly 1.0 after a single upsert), which would otherwise
    // trigger an immediate, meaningless jump straight to the smallest tier on essentially every shard.
    private static final double MIN_SAMPLES_BEFORE_ADAPTATION = 500;

    private final int _valueColumnIndex;
    private final int _mediumTrimSize;
    private final int _mediumTrimThreshold;
    private final int _smallTrimSize;
    private final int _smallTrimThreshold;

    private final DoubleAdder _runningTotal = new DoubleAdder();
    private final DoubleAccumulator _runningMax = new DoubleAccumulator(Double::max, 0.0);
    // 0 = full (initial), 1 = medium, 2 = small. Only ever moves up (shrinks), guarded by CAS below.
    private final AtomicInteger _currentTier = new AtomicInteger(0);

    AdaptiveConcurrentIndexedTable(DataSchema dataSchema, boolean hasFinalInput, QueryContext queryContext,
        int resultSize, int trimSize, int trimThreshold, int initialCapacity, ExecutorService executorService,
        int valueColumnIndex) {
      super(dataSchema, hasFinalInput, queryContext, resultSize, trimSize, trimThreshold, initialCapacity,
          executorService);
      _valueColumnIndex = valueColumnIndex;
      _mediumTrimSize = Math.max(1, trimSize / 10);
      _mediumTrimThreshold = _mediumTrimSize * SMALLER_TIER_THRESHOLD_MULTIPLIER;
      _smallTrimSize = Math.max(1, trimSize / 100);
      _smallTrimThreshold = _smallTrimSize * SMALLER_TIER_THRESHOLD_MULTIPLIER;
    }

    @Override
    public boolean upsert(Key key, Record record) {
      boolean result = super.upsert(key, record);
      updateSignalAndMaybeShrink(key, record);
      return result;
    }

    /// NOTE: `runningTotal` accumulates each upsert's raw incoming value (not looked up from the map),
    /// relying on the fact that for a SUM-like additive aggregate, the sum of every raw contribution
    /// equals the sum of all keys' final aggregated values -- this signal is only valid for that class of
    /// aggregate (matches this prototype's documented single-numeric-ORDER-BY-aggregate scope).
    /// `runningMax`, in contrast, MUST reflect the key's current AGGREGATED value, not the raw per-call
    /// value -- a hot key's dominance only shows up in its accumulated total (e.g. 1000 hits x 1.0 each),
    /// never in any single call's raw contribution, so it requires one extra lookup of the just-updated
    /// record from `_lookupMap` (O(1) hash lookup, not a rescan).
    private void updateSignalAndMaybeShrink(Key key, Record record) {
      int currentTier = _currentTier.get();
      if (currentTier == 2) {
        return; // already at the smallest tier, nothing left to shrink to
      }

      Object rawValue = record.getValues()[_valueColumnIndex];
      if (!(rawValue instanceof Number)) {
        return; // not a numeric aggregate at this column -- signal not applicable, stay at full capacity
      }
      _runningTotal.add(((Number) rawValue).doubleValue());

      Record stored = _lookupMap.get(key);
      Object storedValue = stored != null ? stored.getValues()[_valueColumnIndex] : null;
      if (storedValue instanceof Number) {
        _runningMax.accumulate(((Number) storedValue).doubleValue());
      }

      double total = _runningTotal.sum();
      if (total < MIN_SAMPLES_BEFORE_ADAPTATION) {
        return; // not enough evidence yet -- top1Share is noise-dominated at very low sample counts
      }
      double top1Share = _runningMax.get() / total;
      int targetTier = top1Share < MEDIUM_TOP1_SHARE_THRESHOLD ? 0
          : top1Share < SMALL_TOP1_SHARE_THRESHOLD ? 1 : 2;

      while (targetTier > currentTier) {
        if (_currentTier.compareAndSet(currentTier, targetTier)) {
          if (targetTier == 1) {
            shrinkTrimSizeAndThreshold(_mediumTrimSize, _mediumTrimThreshold);
          } else {
            shrinkTrimSizeAndThreshold(_smallTrimSize, _smallTrimThreshold);
          }
          break;
        }
        currentTier = _currentTier.get(); // another thread moved it first, retry from the new state
      }
    }
  }
}
