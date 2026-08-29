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
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.apache.pinot.common.utils.DataSchema;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.core.query.request.context.utils.QueryContextConverterUtils;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;


/// Correctness coverage for ShardedOffHeapIndexedTable, the Path 1 query-engine-wiring prototype (see
/// apache/pinot#19388's design doc). Standalone construction with real QueryContext/DataSchema/Key/Record
/// objects -- not a real end-to-end query through GroupByCombineOperator, which this class is not yet
/// wired into. Mirrors ShardedOffHeapGroupTableTest's own discipline (real concurrent threads against an
/// independent ground truth, not simulation) but exercises it through the Key/Record boundary this class
/// adds, not ShardedOffHeapGroupTable's own int/double API directly.
public class ShardedOffHeapIndexedTableTest {
  private static final int NUM_SHARDS = 16;
  private static final int NUM_THREADS = 10;
  private static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(NUM_THREADS);

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
    ShardedOffHeapIndexedTable table = new ShardedOffHeapIndexedTable(dataSchema, false, queryContext, 500, 500,
        Integer.MAX_VALUE, 16, NUM_SHARDS, EXECUTOR);

    table.upsert(new Key(new Object[]{1}), new Record(new Object[]{1, 10d}));
    table.upsert(new Key(new Object[]{2}), new Record(new Object[]{2, 20d}));
    table.upsert(new Key(new Object[]{1}), new Record(new Object[]{1, 5d})); // repeat key -- must aggregate
    table.upsert(new Key(new Object[]{3}), new Record(new Object[]{3, 30d}));

    table.finish(true, false);
    Assert.assertEquals(table.size(), 3);

    Map<Integer, Double> result = collectSingleIntKey(table);
    Assert.assertEquals(result.get(1), 15d);
    Assert.assertEquals(result.get(2), 20d);
    Assert.assertEquals(result.get(3), 30d);
  }

  @Test
  public void testFinishSortsDescendingByAggregateValue() {
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1) FROM testTable GROUP BY d1 ORDER BY sum(m1) DESC LIMIT 500");
    DataSchema dataSchema = new DataSchema(new String[]{"d1", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE});
    ShardedOffHeapIndexedTable table = new ShardedOffHeapIndexedTable(dataSchema, false, queryContext, 500, 500,
        Integer.MAX_VALUE, 16, NUM_SHARDS, EXECUTOR);

    table.upsert(new Key(new Object[]{1}), new Record(new Object[]{1, 10d}));
    table.upsert(new Key(new Object[]{2}), new Record(new Object[]{2, 50d}));
    table.upsert(new Key(new Object[]{3}), new Record(new Object[]{3, 30d}));
    table.finish(true, false);

    List<Double> valuesInOrder = new ArrayList<>();
    Iterator<Record> iterator = table.iterator();
    while (iterator.hasNext()) {
      valuesInOrder.add((Double) iterator.next().getValues()[1]);
    }
    Assert.assertEquals(valuesInOrder, List.of(50d, 30d, 10d));
  }

  @Test
  public void testMinAndMaxAggregationEligibleAndCorrect() {
    for (String function : List.of("min", "max")) {
      QueryContext queryContext = QueryContextConverterUtils.getQueryContext(
          "SELECT " + function + "(m1) FROM testTable GROUP BY d1 ORDER BY " + function + "(m1) DESC LIMIT 500");
      DataSchema dataSchema = new DataSchema(new String[]{"d1", function + "(m1)"},
          new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE});
      Assert.assertTrue(ShardedOffHeapIndexedTable.isEligible(dataSchema, queryContext),
          function + " should be eligible");

      ShardedOffHeapIndexedTable table = new ShardedOffHeapIndexedTable(dataSchema, false, queryContext, 500, 500,
          Integer.MAX_VALUE, 16, NUM_SHARDS, EXECUTOR);
      table.upsert(new Key(new Object[]{1}), new Record(new Object[]{1, 10d}));
      table.upsert(new Key(new Object[]{1}), new Record(new Object[]{1, 30d}));
      table.upsert(new Key(new Object[]{1}), new Record(new Object[]{1, 20d}));
      table.finish(true, false);

      Map<Integer, Double> result = collectSingleIntKey(table);
      double expected = function.equals("min") ? 10d : 30d;
      Assert.assertEquals(result.get(1), expected, function + "(m1) for key 1");
    }
  }

  @Test
  public void testMultiColumnKeyBasicUpsertAndMerge() {
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1) FROM testTable GROUP BY d1, d2 ORDER BY sum(m1) DESC LIMIT 500");
    DataSchema dataSchema = new DataSchema(new String[]{"d1", "d2", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.INT,
            DataSchema.ColumnDataType.DOUBLE});
    ShardedOffHeapIndexedTable table = new ShardedOffHeapIndexedTable(dataSchema, false, queryContext, 500, 500,
        Integer.MAX_VALUE, 16, NUM_SHARDS, EXECUTOR);

    table.upsert(new Key(new Object[]{1, 100}), new Record(new Object[]{1, 100, 10d}));
    table.upsert(new Key(new Object[]{1, 200}), new Record(new Object[]{1, 200, 20d})); // shares column 0
    table.upsert(new Key(new Object[]{1, 100}), new Record(new Object[]{1, 100, 5d})); // exact repeat -- merge
    table.finish(true, false);

    Assert.assertEquals(table.size(), 2);
    Iterator<Record> iterator = table.iterator();
    Map<List<Integer>, Double> result = new java.util.HashMap<>();
    while (iterator.hasNext()) {
      Object[] values = iterator.next().getValues();
      result.put(List.of((Integer) values[0], (Integer) values[1]), (Double) values[2]);
    }
    Assert.assertEquals(result.get(List.of(1, 100)), 15d);
    Assert.assertEquals(result.get(List.of(1, 200)), 20d);
  }

  @Test
  public void testIneligibleQueriesRejected() {
    // More than one aggregation function -- ShardedOffHeapGroupTable stores exactly one double per key.
    QueryContext twoAggs = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1), min(m2) FROM testTable GROUP BY d1 ORDER BY sum(m1) DESC LIMIT 500");
    DataSchema twoAggsSchema = new DataSchema(new String[]{"d1", "sum(m1)", "min(m2)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE,
            DataSchema.ColumnDataType.DOUBLE});
    Assert.assertFalse(ShardedOffHeapIndexedTable.isEligible(twoAggsSchema, twoAggs));
    Assert.assertNotNull(ShardedOffHeapIndexedTable.ineligibilityReason(twoAggsSchema, twoAggs));
    Assert.assertThrows(IllegalArgumentException.class,
        () -> new ShardedOffHeapIndexedTable(twoAggsSchema, false, twoAggs, 500, 500, Integer.MAX_VALUE, 16,
            NUM_SHARDS, EXECUTOR));

    // A non-SUM/MIN/MAX aggregation function.
    QueryContext countStar = QueryContextConverterUtils.getQueryContext(
        "SELECT count(*) FROM testTable GROUP BY d1 ORDER BY count(*) DESC LIMIT 500");
    DataSchema countStarSchema = new DataSchema(new String[]{"d1", "count(*)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.LONG});
    Assert.assertFalse(ShardedOffHeapIndexedTable.isEligible(countStarSchema, countStar));

    // A STRING-typed GROUP BY key column -- Direction C only supports INT key columns.
    QueryContext stringKey = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1) FROM testTable GROUP BY d1 ORDER BY sum(m1) DESC LIMIT 500");
    DataSchema stringKeySchema = new DataSchema(new String[]{"d1", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.STRING, DataSchema.ColumnDataType.DOUBLE});
    Assert.assertFalse(ShardedOffHeapIndexedTable.isEligible(stringKeySchema, stringKey));
  }

  @Test
  public void testSameKeyAlwaysRoutesToSameShardAcrossThreads()
      throws Exception {
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1) FROM testTable GROUP BY d1 ORDER BY sum(m1) DESC LIMIT 500000");
    DataSchema dataSchema = new DataSchema(new String[]{"d1", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE});
    int cardinality = 5000;
    ShardedOffHeapIndexedTable table = new ShardedOffHeapIndexedTable(dataSchema, false, queryContext, 500_000,
        500_000, Integer.MAX_VALUE, 256, NUM_SHARDS, EXECUTOR);
    Map<Integer, Double> groundTruth = new ConcurrentHashMap<>();

    List<Future<?>> futures = new ArrayList<>();
    for (int t = 0; t < NUM_THREADS; t++) {
      long seed = 1000L + t;
      futures.add(EXECUTOR.submit(() -> {
        Random random = new Random(seed);
        for (int r = 0; r < 20_000; r++) {
          int key = random.nextInt(cardinality);
          double value = random.nextDouble() * 100;
          table.upsert(new Key(new Object[]{key}), new Record(new Object[]{key, value}));
          groundTruth.merge(key, value, Double::sum);
        }
      }));
    }
    for (Future<?> f : futures) {
      f.get(30, TimeUnit.SECONDS);
    }

    table.finish(true, false);
    Assert.assertEquals(table.size(), groundTruth.size());
    Map<Integer, Double> result = collectSingleIntKey(table);
    for (Map.Entry<Integer, Double> entry : groundTruth.entrySet()) {
      Assert.assertEquals(result.get(entry.getKey()), entry.getValue(), 1e-6, "Mismatch for key " + entry.getKey());
    }
  }

  private static Map<Integer, Double> collectSingleIntKey(ShardedOffHeapIndexedTable table) {
    Map<Integer, Double> result = new java.util.HashMap<>();
    Iterator<Record> iterator = table.iterator();
    while (iterator.hasNext()) {
      Object[] values = iterator.next().getValues();
      result.put((Integer) values[0], (Double) values[1]);
    }
    return result;
  }
}
