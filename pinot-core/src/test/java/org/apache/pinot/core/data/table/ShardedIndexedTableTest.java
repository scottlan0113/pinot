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
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
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

  /// The property every correctness claim in this design rests on: a key's contributions all land on the
  /// same shard, so no key is ever split across two shards and counted twice. Asserted indirectly and
  /// without reaching into the shard array -- if routing were not key-stable, the same key would appear
  /// in more than one shard and the final merge would emit it more than once, or with a partial sum.
  @Test
  public void testSameKeyAlwaysRoutesToSameShard() {
    ShardedIndexedTable table = newTable(sumDescQuery(500), 500, 500, Integer.MAX_VALUE, 16);

    // Every key contributed several times, interleaved, so a key-unstable router would scatter them.
    for (int round = 0; round < 5; round++) {
      for (int key = 0; key < 200; key++) {
        upsert(table, key, 1d);
      }
    }
    table.finish(true, false);

    Map<Integer, Double> byKey = collect(table);
    Assert.assertEquals(byKey.size(), 200, "each distinct key must appear exactly once after the merge");
    for (Map.Entry<Integer, Double> entry : byKey.entrySet()) {
      Assert.assertEquals(entry.getValue(), 5d, "key " + entry.getKey() + " lost or duplicated contributions");
    }
  }

  /// Concurrent upserts from several threads over an overlapping key space. The existing basic test is
  /// single-threaded, so nothing until now exercised the per-shard locking under real contention.
  @Test
  public void testConcurrentUpsertsAggregateCorrectly()
      throws Exception {
    ShardedIndexedTable table = newTable(sumDescQuery(1000), 1000, 1000, Integer.MAX_VALUE, 8);

    int numThreads = 4;
    int keysPerThread = 250;
    int roundsPerThread = 20;
    CountDownLatch start = new CountDownLatch(1);
    CountDownLatch done = new CountDownLatch(numThreads);
    for (int t = 0; t < numThreads; t++) {
      EXECUTOR.submit(() -> {
        try {
          start.await();
          for (int round = 0; round < roundsPerThread; round++) {
            for (int key = 0; key < keysPerThread; key++) {
              upsert(table, key, 1d);
            }
          }
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
        } finally {
          done.countDown();
        }
      });
    }
    start.countDown();
    Assert.assertTrue(done.await(60, TimeUnit.SECONDS), "upsert threads did not finish in time");

    table.finish(true, false);

    Map<Integer, Double> byKey = collect(table);
    Assert.assertEquals(byKey.size(), keysPerThread);
    double expected = numThreads * roundsPerThread;
    for (Map.Entry<Integer, Double> entry : byKey.entrySet()) {
      Assert.assertEquals(entry.getValue(), expected,
          "key " + entry.getKey() + " did not aggregate every concurrent contribution");
    }
  }

  /// A key whose individual contributions are each too small to survive a trim, but whose total belongs
  /// in the top-K, must still be there at the end. This is the failure mode that disqualifies per-thread
  /// tables (DESIGN.md Sec 5): key-hash routing puts all three contributions in one shard, so the shard
  /// trims with the full total in hand rather than three separate small values.
  @Test
  public void testKeySurvivesTrimBecauseContributionsShareAShard()
      throws Exception {
    // trimThreshold well below the key count, so shards really do resize mid-upsert rather than only at
    // finish -- without that this test would pass for the wrong reason.
    ShardedIndexedTable table = newTable(sumDescQuery(10), 10, 10, 20, 4);

    for (int key = 1; key <= 200; key++) {
      upsert(table, key, 1d);
    }
    // Key 0 contributed from three different threads, 4 each: below the filler keys' rank individually,
    // clearly first once summed.
    CountDownLatch done = new CountDownLatch(3);
    for (int t = 0; t < 3; t++) {
      EXECUTOR.submit(() -> {
        try {
          upsert(table, 0, 4d);
        } finally {
          done.countDown();
        }
      });
    }
    Assert.assertTrue(done.await(30, TimeUnit.SECONDS), "contributor threads did not finish in time");

    Assert.assertTrue(table.getNumResizes() > 0,
        "shards never resized, so this test would pass without exercising a trim at all");

    table.finish(true, false);

    Map<Integer, Double> byKey = collect(table);
    Assert.assertTrue(byKey.containsKey(0), "the key with the largest true total was trimmed away");
    Assert.assertEquals(byKey.get(0), 12d, "surviving key kept only part of its contributions");
  }

  /// ORDER BY is routed through the inherited TableResizer, so shapes other than `SUM DESC` have to work
  /// too. Ascending order and ordering by a key column are both exercised here.
  @Test
  public void testOrderByAscendingAndByKeyColumn() {
    ShardedIndexedTable ascending = newTable(
        QueryContextConverterUtils.getQueryContext(
            "SELECT sum(m1) FROM testTable GROUP BY d1 ORDER BY sum(m1) LIMIT 3"), 3, 3, Integer.MAX_VALUE, 8);
    for (int key = 0; key < 20; key++) {
      upsert(ascending, key, key + 1d);
    }
    ascending.finish(true, false);
    List<Record> ascendingRecords = records(ascending);
    Assert.assertEquals(ascendingRecords.size(), 3);
    Assert.assertEquals((double) ascendingRecords.get(0).getValues()[1], 1d, "ascending order not honoured");
    Assert.assertEquals((double) ascendingRecords.get(2).getValues()[1], 3d, "ascending order not honoured");

    ShardedIndexedTable byKeyColumn = newTable(
        QueryContextConverterUtils.getQueryContext(
            "SELECT sum(m1) FROM testTable GROUP BY d1 ORDER BY d1 DESC LIMIT 3"), 3, 3, Integer.MAX_VALUE, 8);
    for (int key = 0; key < 20; key++) {
      upsert(byKeyColumn, key, 1d);
    }
    byKeyColumn.finish(true, false);
    List<Record> byKeyRecords = records(byKeyColumn);
    Assert.assertEquals(byKeyRecords.size(), 3);
    Assert.assertEquals((int) byKeyRecords.get(0).getValues()[0], 19, "ordering by a key column not honoured");
    Assert.assertEquals((int) byKeyRecords.get(2).getValues()[0], 17, "ordering by a key column not honoured");
  }

  /// Multi-column group-by keys: routing hashes the whole composite key, and the final merge rebuilds
  /// keys from the first `numKeyColumns` record values, so an off-by-one there would collapse distinct
  /// keys into one.
  @Test
  public void testMultiColumnKeys() {
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1) FROM testTable GROUP BY d1, d2 ORDER BY sum(m1) DESC LIMIT 500");
    DataSchema dataSchema = new DataSchema(new String[]{"d1", "d2", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.INT,
            DataSchema.ColumnDataType.DOUBLE});
    ShardedIndexedTable table =
        new ShardedIndexedTable(dataSchema, false, queryContext, 500, 500, Integer.MAX_VALUE, 16, 8, EXECUTOR);

    for (int d1 = 0; d1 < 10; d1++) {
      for (int d2 = 0; d2 < 10; d2++) {
        table.upsert(new Key(new Object[]{d1, d2}), new Record(new Object[]{d1, d2, 1d}));
        table.upsert(new Key(new Object[]{d1, d2}), new Record(new Object[]{d1, d2, 2d}));
      }
    }
    table.finish(true, false);

    Assert.assertEquals(table.size(), 100, "distinct (d1, d2) pairs must not collapse");
    Iterator<Record> iterator = table.iterator();
    while (iterator.hasNext()) {
      Assert.assertEquals((double) iterator.next().getValues()[2], 3d);
    }
  }

  /// finish() takes a different branch without ORDER BY -- a plain concatenation of the shards rather
  /// than a merge through the TableResizer. That branch is only correct because keys never straddle
  /// shards, so it deserves its own coverage.
  @Test
  public void testNoOrderByConcatenatesShardsWithoutDuplicating() {
    QueryContext queryContext =
        QueryContextConverterUtils.getQueryContext("SELECT sum(m1) FROM testTable GROUP BY d1 LIMIT 500");
    DataSchema dataSchema = new DataSchema(new String[]{"d1", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE});
    // Without ORDER BY, IndexedTable asserts that trimming is disabled entirely -- both trimSize and
    // trimThreshold must be Integer.MAX_VALUE, not just the threshold.
    ShardedIndexedTable table = new ShardedIndexedTable(dataSchema, false, queryContext, 500, Integer.MAX_VALUE,
        Integer.MAX_VALUE, 16, 8, EXECUTOR);

    for (int round = 0; round < 3; round++) {
      for (int key = 0; key < 120; key++) {
        upsert(table, key, 1d);
      }
    }
    table.finish(false, false);

    Map<Integer, Double> byKey = collect(table);
    Assert.assertEquals(byKey.size(), 120, "no-ORDER-BY concatenation duplicated or dropped keys");
    for (Double value : byKey.values()) {
      Assert.assertEquals(value, 3d);
    }
  }

  private static QueryContext sumDescQuery(int limit) {
    return QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1) FROM testTable GROUP BY d1 ORDER BY sum(m1) DESC LIMIT " + limit);
  }

  private static ShardedIndexedTable newTable(QueryContext queryContext, int resultSize, int trimSize,
      int trimThreshold, int numShards) {
    DataSchema dataSchema = new DataSchema(new String[]{"d1", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE});
    return new ShardedIndexedTable(dataSchema, false, queryContext, resultSize, trimSize, trimThreshold, 16,
        numShards, EXECUTOR);
  }

  private static List<Record> records(ShardedIndexedTable table) {
    List<Record> records = new ArrayList<>();
    Iterator<Record> iterator = table.iterator();
    while (iterator.hasNext()) {
      records.add(iterator.next());
    }
    return records;
  }

  private static Map<Integer, Double> collect(ShardedIndexedTable table) {
    Map<Integer, Double> byKey = new HashMap<>();
    for (Record record : records(table)) {
      Object[] values = record.getValues();
      Assert.assertNull(byKey.put((Integer) values[0], (Double) values[1]),
          "key " + values[0] + " appeared more than once in the final result");
    }
    return byKey;
  }

  private static void upsert(ShardedIndexedTable table, int key, double value) {
    table.upsert(new Key(new Object[]{key}), new Record(new Object[]{key, value}));
  }
}
