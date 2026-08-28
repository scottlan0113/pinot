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
import org.apache.pinot.core.data.table.Record;
import org.apache.pinot.core.data.table.ShardedIndexedTable;
import org.apache.pinot.core.data.table.ShardedOffHeapGroupTable;
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


/// Puts Direction C (`ShardedOffHeapGroupTable`) through the same rigorous JMH methodology (proper
/// forks, warmup, multiple measurement iterations) already established for Direction A in
/// [BenchmarkShardedIndexedTable], instead of the ad-hoc `System.currentTimeMillis()` timing used in
/// `CompareShardedOffHeapVsDirectionA` during initial development. Same workload shape as that ad-hoc
/// comparison (uniform random int keys, cardinality 50,000, 10 concurrent writers) so the numbers are
/// directly comparable to what was already validated -- this benchmark exists to confirm that finding
/// under rigor, not to redesign the comparison.
///
/// `shardedIndexedTable()` here is Direction A under the SAME single-INT-key/single-DOUBLE-SUM workload
/// as the off-heap variants below (not [BenchmarkShardedIndexedTable]'s own STRING+INT/two-aggregate
/// workload) -- apples-to-apples against Direction C is the point of this class.
///
/// `ShardedOffHeapGroupTable` owns off-heap (native) memory, unlike `ShardedIndexedTable` which is
/// plain heap-allocated and garbage collected automatically -- each benchmark invocation below closes
/// its table explicitly, otherwise repeated warmup/measurement iterations within a single fork would
/// leak native memory outside the JVM heap.
@State(Scope.Benchmark)
public class BenchmarkShardedOffHeapGroupTable {
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
  public void shardedIndexedTable()
      throws InterruptedException {
    ShardedIndexedTable table =
        new ShardedIndexedTable(_dataSchema, false, _queryContext, TRIM_SIZE, TRIM_SIZE, TRIM_THRESHOLD,
            TRIM_THRESHOLD / NUM_SHARDS, NUM_SHARDS, _executorService, false);

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
            table.upsert(new Record(new Object[]{key, value}));
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
  public void shardedOffHeapGroupTableFixed()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, false)) {
      runDirectionC(table);
    }
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedOffHeapGroupTableAdaptive()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, true)) {
      runDirectionC(table);
    }
  }

  /// Sub-segmenting (DESIGN.md Sec 6.2) was only measured on the skewed benchmark before this --
  /// numSubSegments=4 was a real win there for both fixed and adaptive capacity, but that says nothing
  /// about a workload where contention is spread over CARDINALITY=50,000 uniformly distinct keys rather
  /// than concentrated by skew. These two variants close that gap.
  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedOffHeapGroupTableFixedSubSegmented4()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, false, 4)) {
      runDirectionC(table);
    }
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedOffHeapGroupTableAdaptiveSubSegmented4()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, true, 4)) {
      runDirectionC(table);
    }
  }

  private void runDirectionC(ShardedOffHeapGroupTable table)
      throws InterruptedException {
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

  public static void main(String[] args)
      throws Exception {
    ChainedOptionsBuilder opt =
        new OptionsBuilder().include(BenchmarkShardedOffHeapGroupTable.class.getSimpleName())
            .jvmArgsAppend("--enable-preview")
            .warmupTime(TimeValue.seconds(3)).warmupIterations(2).measurementTime(TimeValue.seconds(3))
            .measurementIterations(5).forks(3);

    new Runner(opt.build()).run();
  }
}
