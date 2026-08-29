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

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.pinot.common.utils.DataSchema;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.core.query.request.context.utils.QueryContextConverterUtils;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;


/// First real (`mvn test`-run) regression coverage for Direction A ([ShardedIndexedTable] and its nested
/// [AdaptiveConcurrentIndexedTable]) -- previously verified only by ad-hoc scratchpad scripts and JMH
/// benchmarks (see DESIGN.md's own timeline). Not a full characterization suite (that lives in DESIGN.md
/// Sec 4.5's JMH/scratch results) -- just enough to pin down the specific bug below with a real red/green
/// test instead of code-inspection reasoning alone.
public class ShardedIndexedTableTest {
  private static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(4);

  @AfterClass
  public void tearDown() {
    EXECUTOR.shutdownNow();
  }

  @Test
  public void testBasicUpsertAndMerge() {
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1) FROM testTable GROUP BY d1 ORDER BY sum(m1) DESC LIMIT 500");
    DataSchema dataSchema = new DataSchema(new String[]{"d1", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE});
    ShardedIndexedTable table =
        new ShardedIndexedTable(dataSchema, false, queryContext, 500, 500, Integer.MAX_VALUE, 16, 16, EXECUTOR);

    table.upsert(new Key(new Object[]{1}), new Record(new Object[]{1, 10d}));
    table.upsert(new Key(new Object[]{2}), new Record(new Object[]{2, 20d}));
    table.upsert(new Key(new Object[]{1}), new Record(new Object[]{1, 5d})); // repeat key -- must aggregate
    table.upsert(new Key(new Object[]{3}), new Record(new Object[]{3, 30d}));

    table.finish(true, false);
    Assert.assertEquals(table.size(), 3);
  }

  /// Regression test for the sample-count-vs-value-sum gate bug (DESIGN.md Sec 4.6, found 2026-08-27 by
  /// reading AdaptiveConcurrentIndexedTable against the fix already applied to its Direction C
  /// counterpart, fixed here 2026-08-29): `MIN_SAMPLES_BEFORE_ADAPTATION` is meant to gate on a real
  /// upsert COUNT ("with only a handful of records seen, top1Share is dominated by noise" -- the class's
  /// own Javadoc), but the pre-fix code compared it against `_runningTotal.sum()`, a SUM OF VALUES --
  /// coincidentally correct only when every value is 1.0, which every prior Direction A verification
  /// (DESIGN.md Sec 4.5) happened to use. A single upsert whose value alone exceeds the threshold was
  /// enough to open the gate and, since a lone sample's top1Share is trivially 1.0, jump straight to the
  /// smallest capacity tier -- exactly the "immediate, meaningless jump" scenario the minimum-sample gate
  /// exists to prevent.
  @Test
  public void testAdaptiveCapacityDoesNotShrinkOnLowSampleCountEvenWithLargeValues() {
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1) FROM testTable GROUP BY d1 ORDER BY sum(m1) DESC LIMIT 1000");
    DataSchema dataSchema = new DataSchema(new String[]{"d1", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE});

    // numShards=1: forces every key onto the one shard under test, so routing can't add noise. trimSize
    // 1000 (SMALL tier therefore 10) vs. only 50 true samples keeps this deterministic either way.
    ShardedIndexedTable fixedTable = new ShardedIndexedTable(dataSchema, false, queryContext, 1000, 1000,
        1_000_000, 1_000_000, 1, EXECUTOR);
    ShardedIndexedTable adaptiveTable = new ShardedIndexedTable(dataSchema, false, queryContext, 1000, 1000,
        1_000_000, 1_000_000, 1, EXECUTOR, true);

    // One dominant-value key (value 1000 alone already exceeds MIN_SAMPLES_BEFORE_ADAPTATION=500 -- the
    // exact bug trigger), then 49 more small-value keys. True sample count stays 50 throughout, far below
    // 500, so a correctly-gated table must not have made any tier decision yet.
    upsert(fixedTable, 0, 1000d);
    upsert(adaptiveTable, 0, 1000d);
    for (int key = 1; key < 50; key++) {
      upsert(fixedTable, key, 1d);
      upsert(adaptiveTable, key, 1d);
    }

    fixedTable.finish(true, false);
    adaptiveTable.finish(true, false);

    Assert.assertEquals(fixedTable.size(), 50, "fixed capacity should retain every distinct key");
    Assert.assertEquals(adaptiveTable.size(), fixedTable.size(),
        "with only 50 true samples (< MIN_SAMPLES_BEFORE_ADAPTATION), adaptive capacity must not have "
            + "shrunk yet, regardless of how large one upsert's value was");
  }

  private static void upsert(ShardedIndexedTable table, int key, double value) {
    table.upsert(new Key(new Object[]{key}), new Record(new Object[]{key, value}));
  }
}
