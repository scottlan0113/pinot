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

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import org.apache.pinot.common.utils.DataSchema;
import org.apache.pinot.core.data.table.ConcurrentIndexedTable;
import org.apache.pinot.core.data.table.Key;
import org.apache.pinot.core.data.table.Record;
import org.apache.pinot.core.data.table.ShardedOffHeapGroupTable;
import org.apache.pinot.core.data.table.ShardedOffHeapIndexedTable;
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


/// Measures the Path 1 adapter (ShardedOffHeapIndexedTable, apache/pinot#19388's design doc, DESIGN.md
/// Sec 6.9) through its REAL upsert(Key, Record) interface for the first time -- every prior number for
/// Direction C in this investigation (BenchmarkShardedOffHeapGroupTable and friends) went through
/// ShardedOffHeapGroupTable's own raw upsert(int, double) directly, never through the Key/Record boundary
/// the adapter adds specifically to satisfy IndexedTable's real contract.
///
/// Three variants, same uniform workload (cardinality 50,000, matching BenchmarkShardedOffHeapGroupTable
/// exactly so the shape is familiar), all in one fresh run rather than stitched together from numbers
/// measured on different days:
/// - `concurrentIndexedTable`: ConcurrentIndexedTable, what GroupByUtils actually picks in production
///   today for this exact query shape (multi-threaded, trim-enabled, ORDER BY present) -- the real
///   baseline Path 1 needs to beat, not a stand-in for it.
/// - `shardedOffHeapIndexedTable`: the new Path 1 adapter, via real Key/Record objects -- what a real
///   query would actually exercise if this were wired in.
/// - `shardedOffHeapGroupTableDirect`: ShardedOffHeapGroupTable's own upsert(int, double), no Key/Record
///   at all -- isolates the adapter's own boundary-crossing cost (Object[] unboxing per upsert, and the
///   one-time Map<Key,Record> build for ORDER BY in finish()) from Direction C's already-known cost,
///   without relying on a cross-run comparison against BenchmarkShardedOffHeapGroupTable's own numbers.
@State(Scope.Benchmark)
public class BenchmarkShardedOffHeapIndexedTable {
  private static final int TRIM_SIZE = 5000;
  private static final int TRIM_THRESHOLD = 1_000_000;
  private static final int NUM_RECORDS = 100_000;
  private static final int NUM_SHARDS = 64;
  private static final int NUM_SEGMENTS = 10;
  private static final int CARDINALITY = 50_000;

  private QueryContext _queryContext;
  private DataSchema _dataSchema;
  private ExecutorService _executorService;

  @Setup
  public void setup() {
    _queryContext = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1) FROM testTable GROUP BY d1 ORDER BY sum(m1) DESC LIMIT 500");
    _dataSchema = new DataSchema(new String[]{"d1", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE});
    _executorService = Executors.newFixedThreadPool(NUM_SEGMENTS);
  }

  @TearDown
  public void destroy() {
    _executorService.shutdown();
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void concurrentIndexedTable()
      throws InterruptedException {
    ConcurrentIndexedTable table = new ConcurrentIndexedTable(_dataSchema, false, _queryContext, TRIM_SIZE,
        TRIM_SIZE, TRIM_THRESHOLD, TRIM_THRESHOLD, _executorService);

    CountDownLatch operatorLatch = new CountDownLatch(NUM_SEGMENTS);
    Future<?>[] futures = new Future<?>[NUM_SEGMENTS];
    for (int i = 0; i < NUM_SEGMENTS; i++) {
      futures[i] = _executorService.submit(new TraceRunnable() {
        @Override
        public void runJob() {
          ThreadLocalRandom random = ThreadLocalRandom.current();
          for (int r = 0; r < NUM_RECORDS; r++) {
            int key = random.nextInt(CARDINALITY);
            double value = random.nextDouble() * 100;
            table.upsert(new Key(new Object[]{key}), new Record(new Object[]{key, value}));
          }
          operatorLatch.countDown();
        }
      });
    }
    try {
      operatorLatch.await(30, TimeUnit.SECONDS);
      table.finish(false);
    } finally {
      for (Future<?> future : futures) {
        if (!future.isDone()) {
          future.cancel(true);
        }
      }
    }
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedOffHeapIndexedTable()
      throws InterruptedException {
    ShardedOffHeapIndexedTable table = new ShardedOffHeapIndexedTable(_dataSchema, false, _queryContext, TRIM_SIZE,
        TRIM_SIZE, TRIM_THRESHOLD, TRIM_THRESHOLD, NUM_SHARDS, _executorService);

    CountDownLatch operatorLatch = new CountDownLatch(NUM_SEGMENTS);
    Future<?>[] futures = new Future<?>[NUM_SEGMENTS];
    for (int i = 0; i < NUM_SEGMENTS; i++) {
      futures[i] = _executorService.submit(new TraceRunnable() {
        @Override
        public void runJob() {
          ThreadLocalRandom random = ThreadLocalRandom.current();
          for (int r = 0; r < NUM_RECORDS; r++) {
            int key = random.nextInt(CARDINALITY);
            double value = random.nextDouble() * 100;
            table.upsert(new Key(new Object[]{key}), new Record(new Object[]{key, value}));
          }
          operatorLatch.countDown();
        }
      });
    }
    try {
      operatorLatch.await(30, TimeUnit.SECONDS);
      table.finish(false);
    } finally {
      for (Future<?> future : futures) {
        if (!future.isDone()) {
          future.cancel(true);
        }
      }
    }
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedOffHeapGroupTableDirect()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, false, 4)) {
      CountDownLatch operatorLatch = new CountDownLatch(NUM_SEGMENTS);
      Future<?>[] futures = new Future<?>[NUM_SEGMENTS];
      for (int i = 0; i < NUM_SEGMENTS; i++) {
        futures[i] = _executorService.submit(new TraceRunnable() {
          @Override
          public void runJob() {
            ThreadLocalRandom random = ThreadLocalRandom.current();
            for (int r = 0; r < NUM_RECORDS; r++) {
              int key = random.nextInt(CARDINALITY);
              double value = random.nextDouble() * 100;
              table.upsert(key, value);
            }
            operatorLatch.countDown();
          }
        });
      }
      try {
        operatorLatch.await(30, TimeUnit.SECONDS);
        table.finishAllShards();
      } finally {
        for (Future<?> future : futures) {
          if (!future.isDone()) {
            future.cancel(true);
          }
        }
      }
    }
  }

  public static void main(String[] args)
      throws Exception {
    ChainedOptionsBuilder opt =
        new OptionsBuilder().include(BenchmarkShardedOffHeapIndexedTable.class.getSimpleName())
            .jvmArgsAppend("--enable-preview")
            .warmupTime(TimeValue.seconds(3)).warmupIterations(2).measurementTime(TimeValue.seconds(3))
            .measurementIterations(5).forks(3);

    new Runner(opt.build()).run();
  }
}
