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

import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.pinot.core.data.table.ShardedOffHeapGroupTable;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.ChainedOptionsBuilder;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;


/// Isolates adaptive capacity's per-upsert overhead from Direction C's shared lock-contention cost.
/// `BenchmarkShardedOffHeapGroupTableSkewed`'s stack profile (see DESIGN.md Sec 6.4) showed adaptive
/// and fixed capacity with statistically IDENTICAL WAITING/RUNNABLE breakdowns under 10 concurrent
/// threads -- meaning lock contention (real, ~52% of thread-samples parked) is a cost shared by both,
/// not what explains the ~11.5% adaptive-vs-fixed gap. This benchmark removes the ExecutorService and
/// per-shard locking from the picture entirely: a single thread upserts directly, so there is no
/// contention to measure and whatever gap remains must come from updateSignal's own per-upsert work.
///
/// Keys are pre-generated once in @Setup (not sampled inline) specifically so `sampleZipfKey`'s own
/// cost -- a real, non-trivial fraction of the concurrent benchmark's profile -- doesn't pollute this
/// measurement; both benchmark methods read from the exact same pre-generated key sequence, so the
/// only thing that can differ between them is the table implementation itself.
@State(Scope.Benchmark)
public class BenchmarkShardedOffHeapGroupTableSingleThread {
  private static final int TRIM_SIZE = 5000;
  private static final int NUM_SHARDS = 64;
  private static final int TOTAL_RECORDS = 1_000_000;
  private static final int CARDINALITY = 1_000_000;
  private static final double SKEW = 1.0;

  private int[] _keys;

  @Setup
  public void setup() {
    double[] cdf = buildZipfCdf(CARDINALITY, SKEW);
    Random random = new Random(42);
    _keys = new int[TOTAL_RECORDS];
    for (int i = 0; i < TOTAL_RECORDS; i++) {
      _keys[i] = sampleZipfKey(cdf, random);
    }
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

  private static int sampleZipfKey(double[] cdf, Random random) {
    double target = random.nextDouble() * cdf[cdf.length - 1];
    int lo = 0;
    int hi = cdf.length - 1;
    while (lo < hi) {
      int mid = (lo + hi) >>> 1;
      if (cdf[mid] < target) {
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
  public void fixedSingleThread() {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TOTAL_RECORDS / NUM_SHARDS,
        TRIM_SIZE, false)) {
      for (int key : _keys) {
        table.upsert(key, 1.0);
      }
      table.finishAllShards();
    }
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public void adaptiveSingleThread() {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, TOTAL_RECORDS / NUM_SHARDS,
        TRIM_SIZE, true)) {
      for (int key : _keys) {
        table.upsert(key, 1.0);
      }
      table.finishAllShards();
    }
  }

  public static void main(String[] args)
      throws Exception {
    ChainedOptionsBuilder opt =
        new OptionsBuilder().include(BenchmarkShardedOffHeapGroupTableSingleThread.class.getSimpleName())
            .warmupTime(TimeValue.seconds(3)).warmupIterations(2).measurementTime(TimeValue.seconds(3))
            .measurementIterations(5).forks(3);

    new Runner(opt.build()).run();
  }
}
