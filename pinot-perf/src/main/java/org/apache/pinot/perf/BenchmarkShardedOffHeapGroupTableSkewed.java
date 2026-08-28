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
import org.apache.pinot.core.data.table.OffHeapGroupTable;
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


/// Companion to [BenchmarkShardedOffHeapGroupTable]: that benchmark's uniform-key workload only ever
/// measures adaptive capacity's bookkeeping COST, because uniform data correctly never triggers
/// shrinkage -- there is no shrink BENEFIT for the off-setting wall-clock savings (smaller off-heap
/// structure, less work per upsert once shrunk) to show up against. This benchmark uses a skewed
/// (Zipfian, skew=1.0) key distribution instead, at the exact cardinality/thread/shard parameters
/// `VerifyAdaptiveDirectionC.java` already validated produce dramatic real shrinkage at this
/// cardinality (217,168 -> 3,200 held entries, ~98.5% reduction, recall@10=100%) -- so adaptive
/// capacity here pays the bookkeeping cost AND collects the memory-reduction benefit, closing the one
/// gap `BenchmarkShardedOffHeapGroupTable` left open (DESIGN.md Sec 6.4/6.6).
///
/// Deliberately uses value=1.0 for every upsert (unlike the uniform-workload benchmark's
/// `random.nextDouble() * 100`) specifically to match `VerifyAdaptiveDirectionC.java`'s own workload
/// exactly, so the shrinkage this benchmark exercises is a known, already-validated quantity rather
/// than an untested combination of skewed keys with random values.
///
/// `shardedOffHeapGroupTableFixedSubSegmentedN` sweeps the sub-segmenting idea (DESIGN.md Sec 6.2)
/// across N=4/8/16/32: each outer shard split into N independently-locked pieces, keeping the exact
/// exclusive-lock-per-critical-section model that beat the read-lock fast path, just at finer
/// granularity -- against `shardedOffHeapGroupTableFixed` (N=1, unchanged) as the direct baseline.
/// N=4 alone already measured a real 6.9-8.1% win (§6.2); this sweep exists to find where returns
/// flatten out or reverse, not just confirm the first data point again.
@State(Scope.Benchmark)
public class BenchmarkShardedOffHeapGroupTableSkewed {
  private static final int TRIM_SIZE = 5000;
  private static final int TRIM_THRESHOLD = 1_000_000;
  private static final int NUM_RECORDS = 100_000;
  private static final int NUM_SHARDS = 64;
  private static final int NUM_SEGMENTS = 10;
  private static final int CARDINALITY = 1_000_000;
  private static final double SKEW = 1.0;

  private QueryContext _queryContext;
  private DataSchema _dataSchema;
  private ExecutorService _executorService;
  private double[] _zipfCdf;

  @Setup
  public void setup() {
    _queryContext = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1) FROM testTable GROUP BY d1 ORDER BY sum(m1) DESC LIMIT 500");
    _dataSchema = new DataSchema(new String[]{"d1", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE});
    _executorService = Executors.newFixedThreadPool(NUM_SEGMENTS);
    _zipfCdf = buildZipfCdf(CARDINALITY, SKEW);
  }

  @TearDown
  public void destroy() {
    _executorService.shutdown();
  }

  private static double[] buildZipfCdf(int numGroups, double skew) {
    double[] cdf = new double[numGroups];
    double sum = 0;
    for (int i = 1; i <= numGroups; i++) {
      sum += 1.0 / Math.pow(i, skew);
      cdf[i - 1] = sum;
    }
    return cdf;
  }

  private int sampleZipfKey(ThreadLocalRandom random) {
    double target = random.nextDouble() * _zipfCdf[_zipfCdf.length - 1];
    int lo = 0;
    int hi = _zipfCdf.length - 1;
    while (lo < hi) {
      int mid = (lo + hi) >>> 1;
      if (_zipfCdf[mid] < target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
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
            int key = sampleZipfKey(random);
            table.upsert(new Record(new Object[]{key, 1.0}));
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
  public void shardedOffHeapGroupTableFixedSubSegmented8()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, false, 8)) {
      runDirectionC(table);
    }
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedOffHeapGroupTableFixedSubSegmented16()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, false, 16)) {
      runDirectionC(table);
    }
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedOffHeapGroupTableFixedSubSegmented32()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, false, 32)) {
      runDirectionC(table);
    }
  }

  /// Tests the resize-frequency hypothesis for why K=16/32 regressed (DESIGN.md Sec 6.2): here every
  /// sub-segment gets the FULL perShardInitialCapacity instead of perShardInitialCapacity/numSubSegments
  /// (divideInitialCapacityAcrossSubSegments=false), so if the regression was caused by smaller initial
  /// capacity triggering more frequent growData()/growIndex(), these variants should recover toward
  /// shardedOffHeapGroupTableFixedSubSegmented4's number instead of staying slow.
  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedOffHeapGroupTableFixedSubSegmented16FullCapacity()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, false, 16, false)) {
      runDirectionC(table);
    }
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedOffHeapGroupTableFixedSubSegmented32FullCapacity()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, false, 32, false)) {
      runDirectionC(table);
    }
  }

  /// Root-caused 2026-08-28 (DESIGN.md Sec 6.2 "Root cause of the K=16/32 regression found"): a
  /// dedicated diagnostic isolating finishAllShards() from the concurrent phase found sub-segment 0 (the
  /// single-threaded merge target) growing repeatedly while absorbing every other sub-segment's data,
  /// because it starts at the same small divided capacity as everyone else. These variants give ONLY
  /// sub-segment 0 the full perShardInitialCapacity (segmentZeroFullCapacity=true) while 1..N-1 stay
  /// divided/small -- unlike the *FullCapacity variants above (which grow EVERY sub-segment and were
  /// measured to regress the concurrent phase via 32x total over-allocation), this grows only one
  /// sub-segment per shard. The diagnostic only measured finishAllShards() in isolation; these variants
  /// are the first end-to-end (concurrent + finish) test of whether that translates into a real net win.
  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedOffHeapGroupTableFixedSubSegmented16SegmentZeroFullCapacity()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, false, 16, true, 1, OffHeapGroupTable.AggregationType.SUM, true)) {
      runDirectionC(table);
    }
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void shardedOffHeapGroupTableFixedSubSegmented32SegmentZeroFullCapacity()
      throws InterruptedException {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TRIM_THRESHOLD / NUM_SHARDS,
        TRIM_SIZE, false, 32, true, 1, OffHeapGroupTable.AggregationType.SUM, true)) {
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
            int key = sampleZipfKey(random);
            table.upsert(key, 1.0);
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
        new OptionsBuilder().include(BenchmarkShardedOffHeapGroupTableSkewed.class.getSimpleName())
            .warmupTime(TimeValue.seconds(3)).warmupIterations(2).measurementTime(TimeValue.seconds(3))
            .measurementIterations(5).forks(3);

    new Runner(opt.build()).run();
  }
}
