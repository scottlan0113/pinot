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
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
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
/// **Known gaps, deliberately not solved here** (this is a prototype demonstrating the core data-flow is
/// sound, not a production-ready implementation):
/// - `finish(sort=true, ...)` sorts descending by the single aggregate value only -- does not reimplement
///   [TableResizer]'s general ORDER BY comparator (arbitrary expressions, ASC order, ordering by key
///   columns, multi-expression ORDER BY are all unsupported here even though `Table`'s contract doesn't
///   forbid them for an eligible query -- callers of this class must not present it with such a query).
/// - `getNumResizes()`/`getResizeTimeMs()` are coarse: [ShardedOffHeapGroupTable] currently only tracks
///   *whether* any shard was trimmed ([ShardedOffHeapGroupTable#anyShardTrimmed()]), not a resize count or
///   timing breakdown per shard the way [IndexedTable] tracks for the on-heap implementations.
/// - Arena/native-memory lifecycle: `close()` is called at the end of `finish()`, once results have been
///   extracted into `_topRecords`. If `finish()` is never reached (query error/timeout before
///   `mergeResults()`), the off-heap Arena leaks -- this is the exact "Arena lifecycle in a long-running
///   server process" risk the design doc already lists as an open question, not newly discovered or
///   solved here.
public class ShardedOffHeapIndexedTable extends IndexedTable {
  private static final Map<Key, Record> UNUSED_LOOKUP_MAP = Map.of();
  private static final int DEFAULT_NUM_SUB_SEGMENTS = 4; // DESIGN.md Sec 6.2's validated sweet spot

  private final ShardedOffHeapGroupTable _shardedTable;
  private final int _aggregationColumnIndex;
  private boolean _finished;

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
              // are only ever called once per query (see GroupByCombineOperator); guard against double-close
              // of the Arena if this is ever called twice, rather than assuming it won't be.
    }
    _finished = true;
    _shardedTable.finishAllShards();
    List<Record> records = new ArrayList<>(_shardedTable.totalSize());
    if (_numKeyColumns == 1) {
      _shardedTable.forEachEntry((key, value) -> records.add(toRecord(key, value)));
    } else {
      _shardedTable.forEachMultiColumnEntry((keys, value) -> records.add(toRecord(keys, value)));
    }
    if (sort) {
      // Known gap -- see class Javadoc: descending by the single aggregate value only, not a general
      // ORDER BY comparator.
      records.sort((a, b) -> Double.compare(
          ((Number) b.getValues()[_aggregationColumnIndex]).doubleValue(),
          ((Number) a.getValues()[_aggregationColumnIndex]).doubleValue()));
    }
    _topRecords = records;
    _shardedTable.close();
  }

  private Record toRecord(int key, double value) {
    Object[] values = new Object[_numKeyColumns + 1];
    values[0] = key;
    values[_aggregationColumnIndex] = value;
    return new Record(values);
  }

  private Record toRecord(int[] keys, double value) {
    // System.arraycopy cannot bulk-box int[] into Object[] -- copy element-by-element instead.
    Object[] values = new Object[_numKeyColumns + 1];
    for (int i = 0; i < _numKeyColumns; i++) {
      values[i] = keys[i];
    }
    values[_aggregationColumnIndex] = value;
    return new Record(values);
  }

  @Override
  public boolean isTrimmed() {
    return _shardedTable.anyShardTrimmed();
  }

  @Override
  public int getNumResizes() {
    // Coarse -- see class Javadoc. ShardedOffHeapGroupTable currently reports only a boolean, not a count.
    return _shardedTable.anyShardTrimmed() ? 1 : 0;
  }

  @Override
  public long getResizeTimeMs() {
    // Not tracked -- see class Javadoc.
    return 0;
  }
}
