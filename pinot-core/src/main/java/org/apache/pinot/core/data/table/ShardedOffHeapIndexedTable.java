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

import java.lang.ref.Cleaner;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import org.apache.pinot.common.utils.DataSchema;
import org.apache.pinot.common.utils.DataSchema.ColumnDataType;
import org.apache.pinot.core.query.aggregation.function.AggregationFunction;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.segment.spi.AggregationFunctionType;


/// Path 1 prototype for wiring Direction C ([ShardedOffHeapGroupTable]) into the real query engine as an
/// [IndexedTable] -- see the "Direction C - Query Engine Wiring Proposal" design doc linked from
/// apache/pinot#19388. STANDALONE PROTOTYPE: not wired into [org.apache.pinot.core.util.GroupByUtils] or
/// `GroupByCombineOperator` yet, and not yet reviewed by Jackie -- verified so far only via standalone
/// construction with real `QueryContext`/`DataSchema`/[Key]/[Record] objects, the same way
/// `ShardedIndexedTable` (Direction A) was verified before any decision was made to wire it in either.
///
/// Extends [IndexedTable] itself, not just [Table] -- a correction to the design doc's original "widen
/// `GroupByCombineOperator._indexedTable` to `Table`" sketch. Reading the real call sites found that
/// `GroupByResultsBlock`'s own constructor also requires the concrete `IndexedTable` type, so widening the
/// combine operator's field alone would not have been sufficient.
///
/// [IndexedTable]'s inherited upsert/finish/resize machinery is built entirely around
/// `Map<Key, Record>` + [TableResizer] -- this class does not use any of it (a genuinely different
/// implementation sharing the same base class for type-compatibility reasons, not a generalization of it).
/// Every substantive inherited method is overridden; the superclass constructor is satisfied with an
/// always-empty, never-touched `Map`.
///
/// **Eligibility is narrower than "int keys, SUM/MIN/MAX aggregate" alone suggests** -- see
/// [#isEligible(DataSchema, QueryContext)]. [ShardedOffHeapGroupTable] stores exactly one `double` value
/// per key, so a query with more than one aggregation function (e.g. `SELECT SUM(a), MIN(b) ... GROUP BY
/// c`) has no way to fit this model at all, a constraint the design doc's Path 1 sketch did not call out
/// explicitly -- found by reading [org.apache.pinot.core.operator.combine.GroupByCombineOperator]'s real
/// record layout (key columns then one slot per aggregation function), not assumed.
///
/// **Three gaps flagged 2026-08-29, addressed same day:**
/// - `finish(sort=true, ...)` used to sort descending by the single aggregate value only. Now reuses the
///   inherited [#_tableResizer] (built correctly by the `IndexedTable` superclass constructor from the
///   real `QueryContext`, exactly the same object `ConcurrentIndexedTable` etc. use) via
///   [TableResizer#getTopRecords(Map, int, boolean)] -- real ORDER BY support (arbitrary expressions, ASC
///   order, ordering by key columns, ties broken however Pinot's own comparator breaks them), not a
///   hand-rolled approximation. The one real cost: building a `Map<Key, Record>` from the off-heap
///   results, a one-time, post-off-heap-processing allocation -- doesn't reintroduce boxing into the
///   actual upsert hot path, which is what the off-heap design was for in the first place.
/// - `getNumResizes()`/`getResizeTimeMs()` were previously coarse placeholders (0-or-1, untracked 0ms).
///   Now `getNumResizes()` reports the real per-shard trim count
///   ([ShardedOffHeapGroupTable#numShardsTrimmed()]), and `getResizeTimeMs()` times the actual
///   `finishAllShards()` call -- still coarser than [IndexedTable]'s own breakdown (which separates
///   periodic upsert-time resizes from the final one; `ShardedOffHeapGroupTable` only ever resizes once,
///   at `finishAllShards()`, so there is only one number to report), but a real measurement, not a
///   hardcoded stand-in.
/// - Arena/native-memory lifecycle: a [Cleaner]-registered safety net now closes `_shardedTable`'s Arena
///   if this object becomes unreachable without `finish()` ever having run (query error/timeout before
///   `mergeResults()`). This is a backstop, not a fix -- `Cleaner` actions only fire once the JVM notices
///   the object is unreachable, which is not immediate or guaranteed-timely, so a leak is now BOUNDED
///   (freed eventually) rather than UNBOUNDED (never freed until process restart), not eliminated. The
///   real fix -- explicit cleanup tied to the query's own error/timeout handling -- needs real
///   `GroupByCombineOperator` integration, out of scope for a still-standalone prototype.
public class ShardedOffHeapIndexedTable extends IndexedTable {
  private static final Map<Key, Record> UNUSED_LOOKUP_MAP = Map.of();
  private static final int DEFAULT_NUM_SUB_SEGMENTS = 4; // DESIGN.md Sec 6.2's validated sweet spot
  // One shared background thread for the whole JVM, the standard Cleaner usage pattern -- not one thread
  // per table instance.
  private static final Cleaner CLEANER = Cleaner.create();

  private final ShardedOffHeapGroupTable _shardedTable;
  private final int _aggregationColumnIndex;
  private final Cleaner.Cleanable _cleanable;
  private boolean _finished;
  private long _resizeTimeMs;

  /// @param numShards Not derived automatically -- see class Javadoc's "not yet wired into GroupByUtils".
  ///                   Caller picks this directly, matching how every benchmark/test in this
  ///                   investigation has (64, throughout DESIGN.md).
  /// @throws IllegalArgumentException if the query is not eligible -- see
  ///                   [#isEligible(DataSchema, QueryContext)]. Callers are expected to check eligibility
  ///                   BEFORE choosing to construct this class, the same way `GroupByUtils`'s existing
  ///                   selection logic branches on thread count / ORDER BY presence / grouping sets before
  ///                   picking a concrete `IndexedTable` implementation -- this constructor re-validates
  ///                   rather than trusting the caller, since an ineligible query routed here would
  ///                   silently produce wrong results (e.g. a second aggregation function's values would
  ///                   simply never be read), not a safe fallback.
  public ShardedOffHeapIndexedTable(DataSchema dataSchema, boolean hasFinalInput, QueryContext queryContext,
      int resultSize, int trimSize, int trimThreshold, int initialCapacity, int numShards,
      ExecutorService executorService) {
    super(dataSchema, hasFinalInput, queryContext, resultSize, trimSize, trimThreshold, UNUSED_LOOKUP_MAP,
        executorService);
    String ineligibilityReason = ineligibilityReason(dataSchema, queryContext);
    if (ineligibilityReason != null) {
      throw new IllegalArgumentException("Not eligible for ShardedOffHeapIndexedTable: " + ineligibilityReason);
    }
    _aggregationColumnIndex = _numKeyColumns;
    OffHeapGroupTable.AggregationType aggregationType =
        toAggregationType(queryContext.getAggregationFunctions()[0].getType());
    int perShardInitialCapacity = Math.max(1, initialCapacity / numShards);
    int fullCapacity = trimSize == Integer.MAX_VALUE ? resultSize : trimSize;
    _shardedTable = new ShardedOffHeapGroupTable(numShards, perShardInitialCapacity, fullCapacity,
        /* adaptiveCapacity= */ false, DEFAULT_NUM_SUB_SEGMENTS, /* divideInitialCapacityAcrossSubSegments= */ true,
        _numKeyColumns, aggregationType);
    // Safety net, not a fix -- see class Javadoc. The registered action (a method reference bound to
    // _shardedTable, the receiver) must NOT capture `this`, or the ShardedOffHeapIndexedTable itself
    // could never become unreachable and this Cleaner would never fire.
    _cleanable = CLEANER.register(this, _shardedTable::close);
  }

  /// @return null if eligible, otherwise a human-readable reason -- returning the reason (not just a
  ///         boolean) is deliberate: whoever eventually wires this into `GroupByUtils`'s selection logic
  ///         will want it for logging/metrics on how much real traffic falls outside Path 1's scope, one
  ///         of the design doc's own open questions.
  public static String ineligibilityReason(DataSchema dataSchema, QueryContext queryContext) {
    AggregationFunction[] aggregationFunctions = queryContext.getAggregationFunctions();
    if (aggregationFunctions == null || aggregationFunctions.length != 1) {
      return "requires exactly one aggregation function, got "
          + (aggregationFunctions == null ? 0 : aggregationFunctions.length)
          + " -- ShardedOffHeapGroupTable stores exactly one double value per key";
    }
    AggregationFunctionType type = aggregationFunctions[0].getType();
    if (type != AggregationFunctionType.SUM && type != AggregationFunctionType.MIN
        && type != AggregationFunctionType.MAX) {
      return "aggregation function " + type + " is not SUM/MIN/MAX";
    }
    int numKeyColumns = queryContext.getNumGroupByKeyColumns();
    ColumnDataType[] columnDataTypes = dataSchema.getColumnDataTypes();
    for (int i = 0; i < numKeyColumns; i++) {
      if (columnDataTypes[i] != ColumnDataType.INT) {
        return "GROUP BY key column " + i + " is " + columnDataTypes[i] + ", not INT";
      }
    }
    return null;
  }

  public static boolean isEligible(DataSchema dataSchema, QueryContext queryContext) {
    return ineligibilityReason(dataSchema, queryContext) == null;
  }

  private static OffHeapGroupTable.AggregationType toAggregationType(AggregationFunctionType type) {
    return switch (type) {
      case SUM -> OffHeapGroupTable.AggregationType.SUM;
      case MIN -> OffHeapGroupTable.AggregationType.MIN;
      case MAX -> OffHeapGroupTable.AggregationType.MAX;
      default -> throw new IllegalStateException(
          "Unreachable -- ineligibilityReason() should have rejected " + type + " before construction");
    };
  }

  @Override
  public boolean upsert(Key key, Record record) {
    Object[] keyValues = key.getValues();
    double value = ((Number) record.getValues()[_aggregationColumnIndex]).doubleValue();
    if (_numKeyColumns == 1) {
      _shardedTable.upsert((Integer) keyValues[0], value);
    } else {
      int[] keys = new int[_numKeyColumns];
      for (int i = 0; i < _numKeyColumns; i++) {
        keys[i] = (Integer) keyValues[i];
      }
      _shardedTable.upsert(keys, value);
    }
    return true;
  }

  @Override
  public void finish(boolean sort, boolean storeFinalResult) {
    if (_finished) {
      return; // Table interface doesn't document finish() as idempotent, but the on-heap implementations
              // are only ever called once per query (see GroupByCombineOperator); guard against double-clean
              // of the Cleanable if this is ever called twice, rather than assuming it won't be.
    }
    _finished = true;

    long startNanos = System.nanoTime();
    _shardedTable.finishAllShards();
    _resizeTimeMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);

    // Build a real Map<Key, Record> so the inherited _tableResizer (built correctly by the IndexedTable
    // superclass constructor from the real QueryContext) can be reused for ORDER BY -- a one-time,
    // post-off-heap-processing cost, not a per-upsert one, so this does not reintroduce boxing into the
    // hot path the off-heap design exists to avoid.
    Map<Key, Record> lookupMap = new HashMap<>(_shardedTable.totalSize());
    if (_numKeyColumns == 1) {
      _shardedTable.forEachEntry((key, value) -> {
        Record record = toRecord(key, value);
        lookupMap.put(new Key(new Object[]{key}), record);
      });
    } else {
      _shardedTable.forEachMultiColumnEntry((keys, value) -> {
        Record record = toRecord(keys, value);
        lookupMap.put(new Key(boxKeys(keys)), record);
      });
    }

    if (_hasOrderBy) {
      _topRecords = _tableResizer.getTopRecords(lookupMap, _resultSize, sort);
    } else {
      _topRecords = lookupMap.values();
    }

    _cleanable.clean(); // runs _shardedTable.close() exactly once, safe even if already run
  }

  private Record toRecord(int key, double value) {
    Object[] values = new Object[_numKeyColumns + 1];
    values[0] = key;
    values[_aggregationColumnIndex] = value;
    return new Record(values);
  }

  private Record toRecord(int[] keys, double value) {
    Object[] values = new Object[_numKeyColumns + 1];
    for (int i = 0; i < _numKeyColumns; i++) {
      values[i] = keys[i]; // int -> Object autoboxes fine one element at a time
    }
    values[_aggregationColumnIndex] = value;
    return new Record(values);
  }

  // System.arraycopy cannot bulk-box int[] into Object[] -- copy element-by-element instead (this is the
  // exact bug the tests caught in toRecord(int[], double) before this method existed to share the fix).
  private Object[] boxKeys(int[] keys) {
    Object[] boxed = new Object[keys.length];
    for (int i = 0; i < keys.length; i++) {
      boxed[i] = keys[i];
    }
    return boxed;
  }

  @Override
  public boolean isTrimmed() {
    return _shardedTable.anyShardTrimmed();
  }

  @Override
  public int getNumResizes() {
    return _shardedTable.numShardsTrimmed();
  }

  @Override
  public long getResizeTimeMs() {
    return _resizeTimeMs;
  }
}
