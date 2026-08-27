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
package org.apache.pinot.perf;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.apache.commons.lang3.RandomStringUtils;
import org.apache.pinot.common.utils.DataSchema;
import org.apache.pinot.core.data.table.ConcurrentIndexedTable;
import org.apache.pinot.core.data.table.IndexedTable;
import org.apache.pinot.core.data.table.Record;
import org.apache.pinot.core.data.table.ShardedIndexedTable;
import org.apache.pinot.core.data.table.SimpleIndexedTable;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.core.query.request.context.utils.QueryContextConverterUtils;
import org.apache.pinot.core.util.trace.TraceRunnable;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.ChainedOptionsBuilder;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;


/// Same setup/parameters as the validated 8.5x BenchmarkIndexedTable run (cardinality 100x100,
/// TRIM_SIZE=5000, ThreadLocalRandom), plus a shardedIndexedTable() variant using the REAL
/// ShardedIndexedTable class, so all three go through the SAME rigorous JMH methodology (proper forks,
/// warmup, multiple measurement iterations) -- not the single cold ad-hoc run used earlier.
@State(Scope.Benchmark)
public class BenchmarkShardedIndexedTable {
  private static final int TRIM_SIZE = 5000;
  private static final int TRIM_THRESHOLD = 1_000_000;
  private static final int NUM_RECORDS = 100_000;
  private static final int NUM_SHARDS = 64;

  private QueryContext _queryContext;
  private DataSchema _dataSchema;

  private List<String> _d1;
  private List<Integer> _d2;

  private ExecutorService _executorService;

  @Setup
  public void setup() {
    int cardinalityD1 = 100;
    Set<String> d1 = new HashSet<>(cardinalityD1);
    while (d1.size() < cardinalityD1) {
      d1.add(RandomStringUtils.secure().nextAlphabetic(3));
    }
    _d1 = new ArrayList<>(cardinalityD1);
    _d1.addAll(d1);

    int cardinalityD2 = 100;
    _d2 = new ArrayList<>(cardinalityD2);
    for (int i = 0; i < cardinalityD2; i++) {
      _d2.add(i);
    }

    _queryContext = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1), max(m2) FROM testTable GROUP BY d1, d2 ORDER BY sum(m1) LIMIT 500");
    _dataSchema = new DataSchema(new String[]{"d1", "d2", "sum(m1)", "max(m2)"}, new DataSchema.ColumnDataType[]{
        DataSchema.ColumnDataType.STRING, DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE,
        DataSchema.ColumnDataType.DOUBLE
    });

    _executorService = Executors.newFixedThreadPool(10);
  }

  @TearDown
  public void destroy() {
    _executorService.shutdown();
  }

  private Record getNewRecord() {
    ThreadLocalRandom random = ThreadLocalRandom.current();
    Object[] columns = new Object[]{
        _d1.get(random.nextInt(_d1.size())), _d2.get(random.nextInt(_d2.size())), (double) random.nextInt(1000),
        (double) random.nextInt(1000)
    };
    return new Record(columns);
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void concurrentIndexedTable()
      throws InterruptedException {
    int numSegments = 10;
    IndexedTable concurrentIndexedTable =
        new ConcurrentIndexedTable(_dataSchema, false, _queryContext, TRIM_SIZE, TRIM_SIZE, TRIM_THRESHOLD,
            TRIM_THRESHOLD, _executorService);

    CountDownLatch operatorLatch = new CountDownLatch(numSegments);
    Future[] futures = new Future[numSegments];
    for (int i = 0; i < numSegments; i++) {
      futures[i] = _executorService.submit(new TraceRunnable() {
        @Override
        public void runJob() {
          for (int r = 0; r < NUM_RECORDS; r++) {
            concurrentIndexedTable.upsert(getNewRecord());
          }
          operatorLatch.countDown();
        }
      });
    }
    try {
      operatorLatch.await(30, TimeUnit.SECONDS);
      concurrentIndexedTable.finish(false);
    } finally {
      for (Future future : futures) {
        if (!future.isDone()) {
          future.cancel(true);
        }
      }
    }
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void simpleIndexedTable()
      throws InterruptedException, TimeoutException, ExecutionException {
    int numSegments = 10;
    List<IndexedTable> simpleIndexedTables = new ArrayList<>(numSegments);
    List<Callable<Void>> innerSegmentCallables = new ArrayList<>(numSegments);

    for (int i = 0; i < numSegments; i++) {
      IndexedTable simpleIndexedTable =
          new SimpleIndexedTable(_dataSchema, false, _queryContext, TRIM_SIZE, TRIM_SIZE, TRIM_THRESHOLD,
              TRIM_THRESHOLD, _executorService);
      simpleIndexedTables.add(simpleIndexedTable);

      Callable<Void> callable = () -> {
        for (int r = 0; r < NUM_RECORDS; r++) {
          simpleIndexedTable.upsert(getNewRecord());
        }
        simpleIndexedTable.finish(false);
        return null;
      };
      innerSegmentCallables.add(callable);
    }

    List<Future<Void>> futures = _executorService.invokeAll(innerSegmentCallables);
    for (Future<Void> future : futures) {
      future.get(10, TimeUnit.SECONDS);
    }

    IndexedTable mergedTable = null;
    for (IndexedTable indexedTable : simpleIndexedTables) {
      mergedTable = mergedTable == null ? indexedTable : mergeInto(mergedTable, indexedTable);
    }
    mergedTable.finish(false);
  }

  private static IndexedTable mergeInto(IndexedTable target, IndexedTable source) {
    target.merge(source);
    return target;
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedIndexedTable()
      throws InterruptedException {
    int numSegments = 10;
    ShardedIndexedTable shardedIndexedTable =
        new ShardedIndexedTable(_dataSchema, false, _queryContext, TRIM_SIZE, TRIM_SIZE, TRIM_THRESHOLD,
            TRIM_THRESHOLD / NUM_SHARDS, NUM_SHARDS, _executorService);

    CountDownLatch operatorLatch = new CountDownLatch(numSegments);
    Future[] futures = new Future[numSegments];
    for (int i = 0; i < numSegments; i++) {
      futures[i] = _executorService.submit(new TraceRunnable() {
        @Override
        public void runJob() {
          for (int r = 0; r < NUM_RECORDS; r++) {
            shardedIndexedTable.upsert(getNewRecord());
          }
          operatorLatch.countDown();
        }
      });
    }
    try {
      operatorLatch.await(30, TimeUnit.SECONDS);
      shardedIndexedTable.finish(false);
    } finally {
      for (Future future : futures) {
        if (!future.isDone()) {
          future.cancel(true);
        }
      }
    }
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void adaptiveShardedIndexedTable()
      throws InterruptedException {
    int numSegments = 10;
    ShardedIndexedTable adaptiveShardedIndexedTable =
        new ShardedIndexedTable(_dataSchema, false, _queryContext, TRIM_SIZE, TRIM_SIZE, TRIM_THRESHOLD,
            TRIM_THRESHOLD / NUM_SHARDS, NUM_SHARDS, _executorService, true);

    CountDownLatch operatorLatch = new CountDownLatch(numSegments);
    Future[] futures = new Future[numSegments];
    for (int i = 0; i < numSegments; i++) {
      futures[i] = _executorService.submit(new TraceRunnable() {
        @Override
        public void runJob() {
          for (int r = 0; r < NUM_RECORDS; r++) {
            adaptiveShardedIndexedTable.upsert(getNewRecord());
          }
          operatorLatch.countDown();
        }
      });
    }
    try {
      operatorLatch.await(30, TimeUnit.SECONDS);
      adaptiveShardedIndexedTable.finish(false);
    } finally {
      for (Future future : futures) {
        if (!future.isDone()) {
          future.cancel(true);
        }
      }
    }
  }

  public static void main(String[] args)
      throws Exception {
    ChainedOptionsBuilder opt =
        new OptionsBuilder().include(BenchmarkShardedIndexedTable.class.getSimpleName())
            .warmupTime(TimeValue.seconds(3)).warmupIterations(2).measurementTime(TimeValue.seconds(3))
            .measurementIterations(5).forks(3);

    new Runner(opt.build()).run();
  }
}
