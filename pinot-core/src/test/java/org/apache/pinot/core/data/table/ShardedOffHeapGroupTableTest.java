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
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.Test;


/// Regression coverage for Direction C ([ShardedOffHeapGroupTable]): shard-by-key-hash routing (the
/// correctness guarantee ported from `ShardedIndexedTable`) combined with off-heap storage per shard,
/// including the `top1Share` adaptive-capacity signal ported from `AdaptiveConcurrentIndexedTable`.
/// Replaces the ad-hoc scratchpad scripts (`ShardedOffHeapCorrectnessTest`,
/// `VerifyAdaptiveDirectionC`, the correctness-relevant half of `CompareShardedOffHeapVsDirectionA`)
/// used during initial development with a permanent suite that runs under `mvn test`.
///
/// Two historical bugs motivate the adaptive-capacity tests specifically, not just their happy path:
/// (1) the minimum-sample gate once compared a SUM of upserted values against a count-shaped threshold,
/// which only coincidentally worked when every value was `1.0` -- caught by a non-unit-valued uniform
/// workload collapsing to the smallest capacity tier despite having no real concentration. (2) a fix that
/// scaled the gate by a shard's distinct-key count fixed that but suppressed genuine shrinkage on skewed
/// data, getting worse as cardinality grew. Both `testAdaptiveCapacityNoFalsePositiveOnUniformData` and
/// `testAdaptiveCapacityShrinksOnSkewedData` must keep passing together -- a fix for one regime that
/// breaks the other is exactly the failure mode both historical bugs took.
public class ShardedOffHeapGroupTableTest {
  private static final int NUM_SHARDS = 64;
  private static final int NUM_THREADS = 10;

  @Test
  public void testBasicUpsertAndMerge() {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 16)) {
      table.upsert(1, 10d);
      table.upsert(2, 20d);
      table.upsert(1, 5d); // repeat key, same shard -- must aggregate, not duplicate
      table.upsert(3, 30d);
      table.upsert(2, 1d);

      table.finishAllShards();
      Assert.assertEquals(table.totalSize(), 3);

      Map<Integer, Double> result = new java.util.HashMap<>();
      table.forEachEntry(result::put);
      Assert.assertEquals(result.get(1), 15d);
      Assert.assertEquals(result.get(2), 21d);
      Assert.assertEquals(result.get(3), 30d);
    }
  }

  @Test
  public void testSameKeyAlwaysRoutesToSameShardAcrossThreads()
      throws Exception {
    // A key's true aggregated value can only be correct if every upsert of that key -- regardless of
    // which thread produced it -- lands in the same shard. This is Direction A's correctness guarantee;
    // Direction C must preserve it exactly since off-heap storage is orthogonal to shard routing.
    int cardinality = 5000;
    ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    Map<Integer, Double> groundTruth = new ConcurrentHashMap<>();
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 256)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < NUM_THREADS; t++) {
        long seed = 1000L + t;
        futures.add(executor.submit(() -> {
          Random random = new Random(seed);
          for (int r = 0; r < 20_000; r++) {
            int key = random.nextInt(cardinality);
            double value = random.nextDouble() * 100;
            table.upsert(key, value);
            groundTruth.merge(key, value, Double::sum);
          }
        }));
      }
      for (Future<?> f : futures) {
        f.get(30, TimeUnit.SECONDS);
      }
      executor.shutdown();

      table.finishAllShards();
      Assert.assertEquals(table.totalSize(), groundTruth.size());
      Map<Integer, Double> result = new java.util.HashMap<>();
      table.forEachEntry(result::put);
      for (Map.Entry<Integer, Double> entry : groundTruth.entrySet()) {
        Assert.assertEquals(result.get(entry.getKey()), entry.getValue(), 1e-6,
            "Mismatch for key " + entry.getKey());
      }
    } finally {
      executor.shutdownNow();
    }
  }

  @Test
  public void testFixedCapacityUnderHeavyContention()
      throws Exception {
    // Originally written for a read-lock fast path (ShardedOffHeapGroupTable.upsert /
    // OffHeapGroupTable.tryFastUpdate) that was tried and reverted after being measured to regress
    // performance -- see DESIGN.md Sec 6.2/6.4. Kept as general heavy-contention coverage for the
    // write-lock-per-upsert design that's actually in place: a small cardinality with many threads
    // maximizes contention on a handful of shards, including many threads simultaneously missing an
    // as-yet-nonexistent key and racing to insert it (must produce exactly one entry per key, not
    // duplicates, and lose no upserts either way).
    int cardinality = 20;
    int numThreads = 20;
    int upsertsPerThread = 200_000;
    ExecutorService executor = Executors.newFixedThreadPool(numThreads);
    Map<Integer, Double> groundTruth = new ConcurrentHashMap<>();
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 16, Integer.MAX_VALUE,
        false)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < numThreads; t++) {
        long seed = 2000L + t;
        futures.add(executor.submit(() -> {
          Random random = new Random(seed);
          for (int r = 0; r < upsertsPerThread; r++) {
            int key = random.nextInt(cardinality);
            double value = random.nextDouble() * 100;
            table.upsert(key, value);
            groundTruth.merge(key, value, Double::sum);
          }
        }));
      }
      for (Future<?> f : futures) {
        f.get(60, TimeUnit.SECONDS);
      }
      executor.shutdown();

      table.finishAllShards();
      Assert.assertEquals(table.totalSize(), cardinality, "Expected every key to survive (no trim at "
          + "this capacity) with no duplicates from the fast-path/slow-path insert race");
      Assert.assertEquals(table.totalSize(), groundTruth.size());
      Map<Integer, Double> result = new java.util.HashMap<>();
      table.forEachEntry(result::put);
      for (Map.Entry<Integer, Double> entry : groundTruth.entrySet()) {
        Assert.assertEquals(result.get(entry.getKey()), entry.getValue(), 1e-6,
            "Mismatch for key " + entry.getKey() + " -- lost or double-counted update under contention");
      }
    } finally {
      executor.shutdownNow();
    }
  }

  @Test
  public void testSubSegmentsUnderRealConcurrentThreads()
      throws Exception {
    // Splitting each outer shard into numSubSegments independently-locked sub-segments (§6.2's
    // sub-segment attempt) must still be exactly correct: every key's aggregated value must match
    // ground truth, and finishAllShards()'s merge-then-trim step must not lose or duplicate anything
    // when consolidating sub-segments 1..N-1 into sub-segment 0.
    int cardinality = 5000;
    int numSubSegments = 4;
    ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    Map<Integer, Double> groundTruth = new ConcurrentHashMap<>();
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 256, Integer.MAX_VALUE,
        false, numSubSegments)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < NUM_THREADS; t++) {
        long seed = 3000L + t;
        futures.add(executor.submit(() -> {
          Random random = new Random(seed);
          for (int r = 0; r < 50_000; r++) {
            int key = random.nextInt(cardinality);
            double value = random.nextDouble() * 100;
            table.upsert(key, value);
            groundTruth.merge(key, value, Double::sum);
          }
        }));
      }
      for (Future<?> f : futures) {
        f.get(60, TimeUnit.SECONDS);
      }
      executor.shutdown();

      table.finishAllShards();
      Assert.assertEquals(table.totalSize(), groundTruth.size(), "Expected every key to survive (no "
          + "trim at this capacity); a mismatch here would mean the sub-segment merge lost or "
          + "duplicated a key");
      Map<Integer, Double> result = new java.util.HashMap<>();
      table.forEachEntry(result::put);
      for (Map.Entry<Integer, Double> entry : groundTruth.entrySet()) {
        Assert.assertEquals(result.get(entry.getKey()), entry.getValue(), 1e-6,
            "Mismatch for key " + entry.getKey() + " -- lost or double-counted update across sub-segments");
      }
    } finally {
      executor.shutdownNow();
    }
  }

  @Test
  public void testMultiColumnKeyBasicUpsertAndMerge() {
    // Multi-column counterpart to testBasicUpsertAndMerge (DESIGN.md Sec 4.6's "multi-column GROUP
    // BY... not yet tested" scope item). Two composite keys sharing one column but differing in the
    // other must stay separate; the exact same composite key upserted twice must aggregate.
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 16, 16, false, 1, true, 2)) {
      table.upsert(new int[]{1, 100}, 10d);
      table.upsert(new int[]{1, 200}, 20d); // same column 0, different column 1 -- must stay separate
      table.upsert(new int[]{1, 100}, 5d);  // exact same composite key -- must aggregate
      table.upsert(new int[]{2, 100}, 30d); // different column 0, same column 1 as another row

      table.finishAllShards();
      Assert.assertEquals(table.totalSize(), 3);

      Map<List<Integer>, Double> result = new java.util.HashMap<>();
      table.forEachMultiColumnEntry((keys, value) -> result.put(toList(keys), value));
      Assert.assertEquals(result.get(List.of(1, 100)), 15d);
      Assert.assertEquals(result.get(List.of(1, 200)), 20d);
      Assert.assertEquals(result.get(List.of(2, 100)), 30d);
    }
  }

  @Test
  public void testMultiColumnKeySameCompositeKeyAlwaysRoutesToSameShardAcrossThreads()
      throws Exception {
    // Multi-column counterpart to testSameKeyAlwaysRoutesToSameShardAcrossThreads -- shard/sub-segment
    // routing must hash the FULL composite key consistently regardless of which thread produced a given
    // upsert, the same guarantee the single-column path already has.
    int cardinalityPerColumn = 70; // 70*70 = 4900 distinct composite keys, comparable to the single-column test
    ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    Map<List<Integer>, Double> groundTruth = new ConcurrentHashMap<>();
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 256, 256, false, 1, true, 2)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < NUM_THREADS; t++) {
        long seed = 4000L + t;
        futures.add(executor.submit(() -> {
          Random random = new Random(seed);
          for (int r = 0; r < 20_000; r++) {
            int[] keys = new int[]{random.nextInt(cardinalityPerColumn), random.nextInt(cardinalityPerColumn)};
            double value = random.nextDouble() * 100;
            table.upsert(keys, value);
            groundTruth.merge(toList(keys), value, Double::sum);
          }
        }));
      }
      for (Future<?> f : futures) {
        f.get(30, TimeUnit.SECONDS);
      }
      executor.shutdown();

      table.finishAllShards();
      Assert.assertEquals(table.totalSize(), groundTruth.size());
      Map<List<Integer>, Double> result = new java.util.HashMap<>();
      table.forEachMultiColumnEntry((keys, value) -> result.put(toList(keys), value));
      for (Map.Entry<List<Integer>, Double> entry : groundTruth.entrySet()) {
        Assert.assertEquals(result.get(entry.getKey()), entry.getValue(), 1e-6,
            "Mismatch for composite key " + entry.getKey());
      }
    } finally {
      executor.shutdownNow();
    }
  }

  @Test
  public void testMultiColumnKeyWithAdaptiveCapacityAndSubSegments()
      throws Exception {
    // Smoke test for the full stack together: multi-column keys + adaptive capacity + sub-segments. The
    // skew/recall behavior itself is already covered by the single-column adaptive-capacity tests above
    // -- this only needs to confirm there is no interaction bug when all three are combined (e.g.
    // hashing the wrong thing for shard vs. sub-segment routing on a composite key, or the
    // merge-on-finish path picking the wrong upsert overload for a multi-column table).
    int cardinalityPerColumn = 40;
    ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    Map<List<Integer>, Double> groundTruth = new ConcurrentHashMap<>();
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 64, 64, true, 4, true, 2)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < NUM_THREADS; t++) {
        long seed = 5000L + t;
        futures.add(executor.submit(() -> {
          Random random = new Random(seed);
          for (int r = 0; r < 20_000; r++) {
            int[] keys = new int[]{random.nextInt(cardinalityPerColumn), random.nextInt(cardinalityPerColumn)};
            double value = random.nextDouble() * 100;
            table.upsert(keys, value);
            groundTruth.merge(toList(keys), value, Double::sum);
          }
        }));
      }
      for (Future<?> f : futures) {
        f.get(30, TimeUnit.SECONDS);
      }
      executor.shutdown();

      table.finishAllShards();
      // Capacity here is small enough that trimming may drop some keys -- unlike the no-trim tests
      // above, we don't assert full ground-truth coverage, only that every SURVIVING entry is correct.
      Map<List<Integer>, Double> result = new java.util.HashMap<>();
      table.forEachMultiColumnEntry((keys, value) -> result.put(toList(keys), value));
      for (Map.Entry<List<Integer>, Double> entry : result.entrySet()) {
        Double expected = groundTruth.get(entry.getKey());
        Assert.assertNotNull(expected, "Result contains a composite key not in ground truth: " + entry.getKey());
        Assert.assertEquals(entry.getValue(), expected, 1e-6, "Mismatch for composite key " + entry.getKey());
      }
    } finally {
      executor.shutdownNow();
    }
  }

  private static List<Integer> toList(int[] keys) {
    List<Integer> list = new ArrayList<>(keys.length);
    for (int k : keys) {
      list.add(k);
    }
    return list;
  }

  @Test
  public void testMinAggregation() {
    // Non-SUM aggregation (DESIGN.md Sec 6.7's other new capability, testing the "single numeric
    // SUM-like (additive) aggregate" scope limitation from Sec 4.6).
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 16, 16, false, 1, true, 1,
        OffHeapGroupTable.AggregationType.MIN)) {
      table.upsert(1, 10d);
      table.upsert(1, 3d);
      table.upsert(1, 7d);
      table.upsert(2, 5d);

      table.finishAllShards();
      Assert.assertEquals(table.totalSize(), 2);
      Map<Integer, Double> result = new java.util.HashMap<>();
      table.forEachEntry(result::put);
      Assert.assertEquals(result.get(1), 3d);
      Assert.assertEquals(result.get(2), 5d);
    }
  }

  @Test
  public void testMaxAggregation() {
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 16, 16, false, 1, true, 1,
        OffHeapGroupTable.AggregationType.MAX)) {
      table.upsert(1, 10d);
      table.upsert(1, 3d);
      table.upsert(1, 7d);
      table.upsert(2, 5d);

      table.finishAllShards();
      Assert.assertEquals(table.totalSize(), 2);
      Map<Integer, Double> result = new java.util.HashMap<>();
      table.forEachEntry(result::put);
      Assert.assertEquals(result.get(1), 10d);
      Assert.assertEquals(result.get(2), 5d);
    }
  }

  @Test
  public void testAdaptiveCapacityWithNonSumAggregationThrows() {
    // top1Share's "sum of raw contributions approximates the aggregated total" assumption is specific to
    // SUM (DESIGN.md Sec 4.3/6.7) -- combining it with MIN/MAX must fail loudly, not silently produce a
    // meaningless signal.
    Assert.assertThrows(IllegalArgumentException.class,
        () -> new ShardedOffHeapGroupTable(NUM_SHARDS, 16, 16, true, 1, true, 1,
            OffHeapGroupTable.AggregationType.MIN));
    Assert.assertThrows(IllegalArgumentException.class,
        () -> new ShardedOffHeapGroupTable(NUM_SHARDS, 16, 16, true, 1, true, 1,
            OffHeapGroupTable.AggregationType.MAX));
  }

  @Test
  public void testMinMaxAggregationUnderRealConcurrentThreads()
      throws Exception {
    // Same routing/merge correctness guarantee as the SUM concurrency tests, exercised against MIN/MAX
    // instead. The exclusive per-(shard, sub-segment) write lock makes concurrency safety independent of
    // which merge function runs inside it in principle, but AggregationType is a genuinely new code path
    // worth its own end-to-end check rather than assuming that reasoning holds without verifying it.
    int cardinality = 5000;
    ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    Map<Integer, Double> groundTruthMin = new ConcurrentHashMap<>();
    Map<Integer, Double> groundTruthMax = new ConcurrentHashMap<>();
    try (ShardedOffHeapGroupTable minTable = new ShardedOffHeapGroupTable(NUM_SHARDS, 256, 256, false, 1, true, 1,
        OffHeapGroupTable.AggregationType.MIN);
        ShardedOffHeapGroupTable maxTable = new ShardedOffHeapGroupTable(NUM_SHARDS, 256, 256, false, 1, true, 1,
            OffHeapGroupTable.AggregationType.MAX)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < NUM_THREADS; t++) {
        long seed = 7000L + t;
        futures.add(executor.submit(() -> {
          Random random = new Random(seed);
          for (int r = 0; r < 20_000; r++) {
            int key = random.nextInt(cardinality);
            double value = random.nextDouble() * 100;
            minTable.upsert(key, value);
            maxTable.upsert(key, value);
            groundTruthMin.merge(key, value, Math::min);
            groundTruthMax.merge(key, value, Math::max);
          }
        }));
      }
      for (Future<?> f : futures) {
        f.get(30, TimeUnit.SECONDS);
      }
      executor.shutdown();

      minTable.finishAllShards();
      maxTable.finishAllShards();
      Map<Integer, Double> minResult = new java.util.HashMap<>();
      minTable.forEachEntry(minResult::put);
      Map<Integer, Double> maxResult = new java.util.HashMap<>();
      maxTable.forEachEntry(maxResult::put);

      Assert.assertEquals(minResult.size(), groundTruthMin.size());
      Assert.assertEquals(maxResult.size(), groundTruthMax.size());
      for (Map.Entry<Integer, Double> entry : groundTruthMin.entrySet()) {
        Assert.assertEquals(minResult.get(entry.getKey()), entry.getValue(), 1e-9,
            "MIN mismatch for key " + entry.getKey());
      }
      for (Map.Entry<Integer, Double> entry : groundTruthMax.entrySet()) {
        Assert.assertEquals(maxResult.get(entry.getKey()), entry.getValue(), 1e-9,
            "MAX mismatch for key " + entry.getKey());
      }
    } finally {
      executor.shutdownNow();
    }
  }

  @Test
  public void testMultiColumnKeyWithMinAggregationAndSubSegments()
      throws Exception {
    // Smoke test combining both new capabilities from this stretch of work (DESIGN.md Sec 6.7): a
    // multi-column key with a non-SUM aggregation, plus sub-segments -- confirms there is no interaction
    // bug, e.g. the wrong merge function applied during finishAllShards()'s sub-segment consolidation.
    int cardinalityPerColumn = 40;
    ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    Map<List<Integer>, Double> groundTruth = new ConcurrentHashMap<>();
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 256, 256, false, 4, true, 2,
        OffHeapGroupTable.AggregationType.MIN)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < NUM_THREADS; t++) {
        long seed = 8000L + t;
        futures.add(executor.submit(() -> {
          Random random = new Random(seed);
          for (int r = 0; r < 20_000; r++) {
            int[] keys = new int[]{random.nextInt(cardinalityPerColumn), random.nextInt(cardinalityPerColumn)};
            double value = random.nextDouble() * 100;
            table.upsert(keys, value);
            groundTruth.merge(toList(keys), value, Math::min);
          }
        }));
      }
      for (Future<?> f : futures) {
        f.get(30, TimeUnit.SECONDS);
      }
      executor.shutdown();

      table.finishAllShards();
      Assert.assertEquals(table.totalSize(), groundTruth.size());
      Map<List<Integer>, Double> result = new java.util.HashMap<>();
      table.forEachMultiColumnEntry((keys, value) -> result.put(toList(keys), value));
      for (Map.Entry<List<Integer>, Double> entry : groundTruth.entrySet()) {
        Assert.assertEquals(result.get(entry.getKey()), entry.getValue(), 1e-9,
            "MIN mismatch for composite key " + entry.getKey());
      }
    } finally {
      executor.shutdownNow();
    }
  }

  @Test
  public void testAdaptiveCapacityWithSubSegmentsShrinksOnSkewedData()
      throws Exception {
    // Sub-segmenting was extended to adaptive capacity (§6.2) by moving the top1Share signal
    // (runningTotal/sampleCount/runningMax/currentTier) to DoubleAdder/LongAdder/DoubleAccumulator/
    // AtomicInteger-with-CAS, since multiple independently-locked sub-segments of the same OUTER shard
    // can now update that shard's signal concurrently. This must produce the SAME adaptive behavior as
    // numSubSegments=1 on the same skewed workload testAdaptiveCapacityShrinksOnSkewedData already
    // covers: real memory reduction, recall@10 never worse than fixed.
    SkewResult r = runZipfWorkload(1.0, 1_000_000, 5000, 4);
    Assert.assertTrue(r._adaptiveSize < r._fixedSize,
        "Expected real memory reduction at skew=1.0 with sub-segments, got fixed=" + r._fixedSize
            + " adaptive=" + r._adaptiveSize);
    Assert.assertTrue(r._adaptiveSize < r._fixedSize / 2,
        "Expected a large reduction at skew=1.0 with sub-segments, only got fixed=" + r._fixedSize
            + " adaptive=" + r._adaptiveSize);
    Assert.assertEquals(r._recallFixed, 1.0);
    Assert.assertEquals(r._recallAdaptive, 1.0,
        "Adaptive capacity with sub-segments must never cost recall on the true top-10");
  }

  @Test
  public void testAdaptiveCapacityNoFalsePositiveOnUniformData()
      throws Exception {
    // Regression case for both historical gate bugs: with keys AND values both effectively uniform
    // (no real concentration anywhere), adaptive capacity must hold the exact same set of keys as fixed
    // capacity. Any shrinkage here is a false positive.
    int cardinality = 50_000;
    int fullCapacity = 5000;
    long fixedSize = runUniformWorkload(cardinality, fullCapacity, false, 1);
    long adaptiveSize = runUniformWorkload(cardinality, fullCapacity, true, 1);
    Assert.assertEquals(adaptiveSize, fixedSize,
        "Adaptive capacity shrank on uniform, non-concentrated data -- the minimum-sample gate is "
            + "triggering on noise, not a real top1Share signal");
    Assert.assertEquals(adaptiveSize, cardinality);
  }

  @Test
  public void testAdaptiveCapacityWithSubSegmentsNoFalsePositiveOnUniformData()
      throws Exception {
    // Same false-positive regression case as testAdaptiveCapacityNoFalsePositiveOnUniformData, but with
    // numSubSegments=4 -- multiple sub-segments' threads racing to update the same shard's signal via
    // DoubleAdder/LongAdder must not, itself, create spurious concentration (e.g. double-counting a
    // sample, or a torn read of runningTotal/sampleCount) that looks like a real top1Share signal.
    int cardinality = 50_000;
    int fullCapacity = 5000;
    long fixedSize = runUniformWorkload(cardinality, fullCapacity, false, 4);
    long adaptiveSize = runUniformWorkload(cardinality, fullCapacity, true, 4);
    Assert.assertEquals(adaptiveSize, fixedSize,
        "Adaptive capacity with sub-segments shrank on uniform, non-concentrated data");
    Assert.assertEquals(adaptiveSize, cardinality);
  }

  private long runUniformWorkload(int cardinality, int fullCapacity, boolean adaptiveCapacity,
      int numSubSegments)
      throws Exception {
    ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 1_000_000 / NUM_SHARDS,
        fullCapacity, adaptiveCapacity, numSubSegments)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < NUM_THREADS; t++) {
        long seed = 6000L + t;
        futures.add(executor.submit(() -> {
          Random random = new Random(seed);
          for (int r = 0; r < 100_000; r++) {
            int key = random.nextInt(cardinality);
            double value = random.nextDouble() * 100;
            table.upsert(key, value);
          }
        }));
      }
      for (Future<?> f : futures) {
        f.get(60, TimeUnit.SECONDS);
      }
      executor.shutdown();
      table.finishAllShards();
      return table.totalSize();
    } finally {
      executor.shutdownNow();
    }
  }

  @Test
  public void testAdaptiveCapacityMildSkewStaysAtFullCapacity()
      throws Exception {
    // Mild skew (0.15) is not concentrated enough to justify shrinking -- adaptive must match fixed
    // exactly, same as the uniform case, just via the Zipfian sampling path instead.
    SkewResult r = runZipfWorkload(0.15, 320_000, 5000);
    Assert.assertEquals(r._adaptiveSize, r._fixedSize);
    Assert.assertEquals(r._recallAdaptive, 1.0);
  }

  @Test
  public void testAdaptiveCapacityShrinksOnSkewedData()
      throws Exception {
    // The regime #10498 actually cares about: real concentration must produce real memory reduction,
    // without ever costing recall on the true top-10 keys by value. This is what both historical gate
    // bugs eventually broke (the sum-vs-count bug made this over-shrink even on uniform data; the
    // distinct-key-count scaling fix suppressed shrinkage here specifically).
    SkewResult r = runZipfWorkload(1.0, 1_000_000, 5000);
    Assert.assertTrue(r._adaptiveSize < r._fixedSize,
        "Expected real memory reduction at skew=1.0, got fixed=" + r._fixedSize + " adaptive=" + r._adaptiveSize);
    Assert.assertTrue(r._adaptiveSize < r._fixedSize / 2,
        "Expected a large reduction at skew=1.0 (Direction A achieves 96-98%), only got fixed="
            + r._fixedSize + " adaptive=" + r._adaptiveSize);
    Assert.assertEquals(r._recallFixed, 1.0);
    Assert.assertEquals(r._recallAdaptive, 1.0, "Adaptive capacity must never cost recall on the true top-10");
  }

  private static class SkewResult {
    long _fixedSize;
    long _adaptiveSize;
    double _recallFixed;
    double _recallAdaptive;
  }

  private SkewResult runZipfWorkload(double skew, int cardinality, int fullCapacity)
      throws Exception {
    return runZipfWorkload(skew, cardinality, fullCapacity, 1);
  }

  private SkewResult runZipfWorkload(double skew, int cardinality, int fullCapacity, int numSubSegments)
      throws Exception {
    double[] cdf = buildZipfCdf(cardinality, skew);
    Map<Integer, Double> groundTruth = new ConcurrentHashMap<>();
    Map<Integer, Double> fixedResult = runZipfTable(cdf, fullCapacity, false, numSubSegments, groundTruth);
    Map<Integer, Double> adaptiveResult = runZipfTable(cdf, fullCapacity, true, numSubSegments, null);

    List<Integer> trueTop10 = topKKeys(groundTruth, 10);
    SkewResult r = new SkewResult();
    r._fixedSize = fixedResult.size();
    r._adaptiveSize = adaptiveResult.size();
    r._recallFixed = recallAt10(fixedResult, trueTop10);
    r._recallAdaptive = recallAt10(adaptiveResult, trueTop10);
    return r;
  }

  private Map<Integer, Double> runZipfTable(double[] cdf, int fullCapacity, boolean adaptiveCapacity,
      int numSubSegments, Map<Integer, Double> groundTruthOut)
      throws Exception {
    ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
    boolean recordGroundTruth = groundTruthOut != null;
    try (ShardedOffHeapGroupTable table = new ShardedOffHeapGroupTable(NUM_SHARDS, 1024, fullCapacity,
        adaptiveCapacity, numSubSegments)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < NUM_THREADS; t++) {
        long seed = 9000L + t;
        futures.add(executor.submit(() -> {
          Random random = new Random(seed);
          for (int r = 0; r < 100_000; r++) {
            int key = sampleZipf(cdf, random);
            double value = 1.0;
            table.upsert(key, value);
            if (recordGroundTruth) {
              groundTruthOut.merge(key, value, Double::sum);
            }
          }
        }));
      }
      for (Future<?> f : futures) {
        f.get(60, TimeUnit.SECONDS);
      }
      executor.shutdown();
      table.finishAllShards();

      Map<Integer, Double> result = new ConcurrentHashMap<>();
      table.forEachEntry(result::put);
      return result;
    } finally {
      executor.shutdownNow();
    }
  }

  private static List<Integer> topKKeys(Map<Integer, Double> totals, int k) {
    List<Map.Entry<Integer, Double>> entries = new ArrayList<>(totals.entrySet());
    entries.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
    List<Integer> result = new ArrayList<>();
    for (int i = 0; i < k && i < entries.size(); i++) {
      result.add(entries.get(i).getKey());
    }
    return result;
  }

  private static double recallAt10(Map<Integer, Double> result, List<Integer> trueTop10) {
    List<Integer> resultTop10 = topKKeys(result, 10);
    int hits = 0;
    for (int key : resultTop10) {
      if (trueTop10.contains(key)) {
        hits++;
      }
    }
    return hits / 10.0;
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

  private static int sampleZipf(double[] cdf, Random random) {
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
}
