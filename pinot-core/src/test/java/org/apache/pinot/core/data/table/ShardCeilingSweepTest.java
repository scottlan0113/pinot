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
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.pinot.common.utils.DataSchema;
import org.apache.pinot.core.query.request.context.QueryContext;
import org.apache.pinot.core.query.request.context.utils.QueryContextConverterUtils;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;


/// Characterises what sharding does to the number of entries held before the final merge, and what a
/// per-shard `trimThreshold` would cost in recall. Backs DESIGN.md Sec 4.2, which these measurements
/// corrected: that section previously claimed the aggregate ceiling was `numShards * trimSize`,
/// approached to >99% at cardinality >= 1M, while Sec 4.5's own table reported held entries above that
/// figure. Neither the formula nor the ">99%" holds.
///
/// These are characterisation tests: several assertions below lock in behaviour that is a *problem*
/// (trimming that never fires), so that a fix has to update them deliberately rather than silently.
/// Method names say which. They also replace the ad-hoc scratch scripts Sec 4.6 flags as missing
/// regression coverage.
public class ShardCeilingSweepTest {
  private static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(4);
  private static final int TRIM_SIZE = 5_000;
  private static final int RESULT_SIZE = 5_000;

  private static final int RECALL_CARDINALITY = 200_000;
  private static final int RECALL_UPSERTS = 1_000_000;
  private static final int RECALL_BASE_THRESHOLD = 50_000;
  private static final long SEED = 42L;

  /// Rank -> key id, so a key's Zipfian rank is not correlated with its hash bucket.
  private static final int[] RANK_TO_KEY = buildPermutation(RECALL_CARDINALITY, SEED);

  private record Held(int entries, int resizes) {
  }

  private record Recall(int shardThreshold, int distinctKeys, int held, int resizes, double recallAt10) {
  }

  @AfterClass
  public void tearDown() {
    EXECUTOR.shutdownNow();
  }

  /// Sharding divides the key population by `numShards`, so a `trimThreshold` that fires on the whole
  /// table can fire on no shard at all. Asserts the current (unfixed) behaviour: the un-sharded table
  /// trims to `trimSize`, every sharded configuration from 4 shards up holds every key and never trims.
  @Test
  public void shardingMakesTrimmingInertAtProductionThreshold() {
    Map<Integer, Held> atOneMillion = new HashMap<>();
    for (int trimThreshold : new int[]{1_000_000, 50_000}) {
      System.out.printf("%n=== trimSize=%,d trimThreshold=%,d, uniform keys ===%n", TRIM_SIZE, trimThreshold);
      System.out.printf("%12s %12s %14s %10s %14s%n", "cardinality", "numShards", "heldEntries", "resizes",
          "numShards*trim");
      for (int cardinality : new int[]{200_000, 1_000_000}) {
        for (int numShards : new int[]{1, 4, 8, 16, 32, 64}) {
          Held held = runHeld(cardinality, numShards, trimThreshold);
          if (trimThreshold == 1_000_000 && cardinality == 1_000_000) {
            atOneMillion.put(numShards, held);
          }
        }
      }
    }

    Held unsharded = atOneMillion.get(1);
    Assert.assertEquals(unsharded.entries(), TRIM_SIZE, "un-sharded table should trim down to trimSize");
    Assert.assertTrue(unsharded.resizes() >= 1, "un-sharded table should have trimmed at least once");

    for (int numShards : new int[]{4, 8, 16, 32, 64}) {
      Held sharded = atOneMillion.get(numShards);
      Assert.assertEquals(sharded.resizes(), 0,
          "PROBLEM being characterised: no shard reaches trimThreshold at numShards=" + numShards);
      Assert.assertEquals(sharded.entries(), 1_000_000,
          "PROBLEM being characterised: every key is held, un-trimmed, at numShards=" + numShards);
      Assert.assertTrue(sharded.entries() > (long) numShards * TRIM_SIZE,
          "held entries exceed the numShards*trimSize figure DESIGN.md Sec 4.2 used to claim as a ceiling");
    }
  }

  /// Does making each shard's `trimThreshold` small enough to actually fire cost recall? DESIGN.md
  /// Sec 4.2 previously said dividing "the budget" hurts recall, but the evidence behind that (Sec 4.5,
  /// Sec 5.7) is about a small `trimSize`; `trimThreshold` is a separate constructor parameter and
  /// decides only whether a shard trims at all.
  ///
  /// Single-threaded on purpose: key-hash routing puts a key in exactly one shard regardless of which
  /// thread wrote it, so trim decisions -- the thing recall depends on -- do not change with
  /// concurrency, and a deterministic harness makes the policies comparable. Values are non-constant
  /// (1..10) so a key's SUM is not merely its frequency; DESIGN.md Sec 4.6 flagged constant-valued data
  /// as the reason an earlier gate bug hid behind clean-looking numbers.
  @Test
  public void dividingTrimThresholdBoundsMemoryWithoutCostingRecall() {
    Map<String, Recall> results = new HashMap<>();
    for (double skew : new double[]{0.15, 0.5, 1.0}) {
      double[] cdf = buildZipfianCdf(RECALL_CARDINALITY, skew);
      System.out.printf("%n=== skew=%.2f cardinality=%,d upserts=%,d trimSize=%,d baseThreshold=%,d ===%n", skew,
          RECALL_CARDINALITY, RECALL_UPSERTS, TRIM_SIZE, RECALL_BASE_THRESHOLD);
      System.out.printf("%10s %-16s %12s %14s %14s %10s %10s%n", "numShards", "policy", "shardThresh", "distinctKeys",
          "heldEntries", "resizes", "recall@10");
      for (int numShards : new int[]{1, 4, 16, 64}) {
        results.put(key(skew, numShards, "full"), runRecall(cdf, numShards, "full", RECALL_BASE_THRESHOLD));
        results.put(key(skew, numShards, "divided"),
            runRecall(cdf, numShards, "divided", Math.max(1, RECALL_BASE_THRESHOLD / numShards)));
        results.put(key(skew, numShards, "divided-floor4x"),
            runRecall(cdf, numShards, "divided-floor4x", Math.max(RECALL_BASE_THRESHOLD / numShards, 4 * TRIM_SIZE)));
      }
    }

    // Recall only moves in the mild-skew band; everywhere else every policy is exact.
    for (double skew : new double[]{0.5, 1.0}) {
      for (int numShards : new int[]{1, 4, 16, 64}) {
        for (String policy : new String[]{"full", "divided", "divided-floor4x"}) {
          Assert.assertEquals(results.get(key(skew, numShards, policy)).recallAt10(), 1d,
              "recall should be exact at skew " + skew + ", " + numShards + " shards, policy " + policy);
        }
      }
    }

    // The current code's perfect recall at mild skew is what "never trimmed" looks like, not quality.
    Recall currentCode = results.get(key(0.15, 64, "full"));
    Assert.assertEquals(currentCode.resizes(), 0, "PROBLEM being characterised: undivided threshold never fires");
    Assert.assertEquals(currentCode.held(), currentCode.distinctKeys(), "every distinct key is held");
    Assert.assertEquals(currentCode.recallAt10(), 1d, "and so nothing is lost -- at the cost of no bound at all");

    // Dividing the threshold restores a bound. At 16 shards the per-shard population exceeds trimSize,
    // so each shard is pinned there and the aggregate lands exactly on numShards * trimSize.
    Recall divided16 = results.get(key(0.15, 16, "divided"));
    Assert.assertEquals(divided16.held(), 16 * TRIM_SIZE, "held should pin to numShards * trimSize");
    Assert.assertEquals(divided16.recallAt10(), 1d, "and cost no recall");

    // Against the honest baseline -- the un-sharded table -- dividing the threshold wins on both axes.
    Recall baseline = results.get(key(0.15, 1, "full"));
    Recall divided4 = results.get(key(0.15, 4, "divided"));
    Assert.assertTrue(divided4.held() < baseline.held(),
        "divided threshold should hold fewer entries than the un-sharded baseline");
    Assert.assertTrue(divided4.recallAt10() >= baseline.recallAt10(),
        "divided threshold should not lose recall against the un-sharded baseline");
  }

  /// Is recall@10 even well posed at mild skew? If rank 10 and rank 1000 have nearly the same SUM, a
  /// recall difference in that band is reordering among indistinguishable keys, not a lost hot key.
  @Test
  public void mildSkewTopTenIsANearTie() {
    System.out.printf("%n=== ground-truth SUM by rank (cardinality=%,d, upserts=%,d, values 1..10) ===%n",
        RECALL_CARDINALITY, RECALL_UPSERTS);
    System.out.printf("%8s %12s %12s %12s %12s %12s%n", "skew", "rank1", "rank10", "rank100", "rank1000", "rank5000");
    Map<Double, List<Double>> bySkew = new HashMap<>();
    for (double skew : new double[]{0.15, 0.5, 1.0}) {
      double[] cdf = buildZipfianCdf(RECALL_CARDINALITY, skew);
      Map<Integer, Double> truth = new HashMap<>();
      Random keyRandom = new Random(SEED);
      Random valueRandom = new Random(SEED + 1);
      for (int i = 0; i < RECALL_UPSERTS; i++) {
        truth.merge(RANK_TO_KEY[drawRank(cdf, keyRandom.nextDouble())], 1d + valueRandom.nextInt(10), Double::sum);
      }
      List<Double> sums = new ArrayList<>(truth.values());
      sums.sort((a, b) -> Double.compare(b, a));
      bySkew.put(skew, sums);
      System.out.printf("%8.2f %12.0f %12.0f %12.0f %12.0f %12.0f%n", skew, at(sums, 1), at(sums, 10), at(sums, 100),
          at(sums, 1000), at(sums, 5000));
    }

    Assert.assertTrue(at(bySkew.get(0.15), 10) / at(bySkew.get(0.15), 1000) < 2d,
        "at mild skew the top-10 should be a near-tie with rank 1000, making recall@10 ill-posed there");
    Assert.assertTrue(at(bySkew.get(1.0), 10) / at(bySkew.get(1.0), 1000) > 50d,
        "at skew 1.0 the top-10 should be cleanly separated, making recall@10 meaningful");
  }

  private static Held runHeld(int cardinality, int numShards, int trimThreshold) {
    ShardedIndexedTable table = newTable(numShards, trimThreshold);
    for (int key = 0; key < cardinality; key++) {
      table.upsert(new Key(new Object[]{key}), new Record(new Object[]{key, 1d}));
    }
    Held held = new Held(table.size(), table.getNumResizes());
    System.out.printf("%12d %12d %,14d %10d %,14d%n", cardinality, numShards, held.entries(), held.resizes(),
        (long) numShards * TRIM_SIZE);
    return held;
  }

  private static Recall runRecall(double[] cdf, int numShards, String policy, int shardThreshold) {
    ShardedIndexedTable table = newTable(numShards, shardThreshold);
    Map<Integer, Double> truth = new HashMap<>();
    Random keyRandom = new Random(SEED);
    Random valueRandom = new Random(SEED + 1);

    for (int i = 0; i < RECALL_UPSERTS; i++) {
      int key = RANK_TO_KEY[drawRank(cdf, keyRandom.nextDouble())];
      double value = 1 + valueRandom.nextInt(10);
      truth.merge(key, value, Double::sum);
      table.upsert(new Key(new Object[]{key}), new Record(new Object[]{key, value}));
    }

    // Snapshot before finish(): size() switches to the merged top-records view afterwards. Kept as
    // explicit statements rather than constructor arguments, so the ordering does not rely on Java's
    // left-to-right argument evaluation.
    int held = table.size();
    int resizes = table.getNumResizes();
    double recall = recallAt10(table, truth);

    Recall result = new Recall(shardThreshold, truth.size(), held, resizes, recall);
    System.out.printf("%10d %-16s %,12d %,14d %,14d %10d %10.2f%n", numShards, policy, result.shardThreshold(),
        result.distinctKeys(), result.held(), result.resizes(), result.recallAt10());
    return result;
  }

  /// Reads the table's final top records, so it must run after the held-entry snapshot is taken.
  private static double recallAt10(ShardedIndexedTable table, Map<Integer, Double> truth) {
    table.finish(true, false);

    List<Map.Entry<Integer, Double>> trueEntries = new ArrayList<>(truth.entrySet());
    trueEntries.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

    List<Object[]> rows = new ArrayList<>();
    Iterator<Record> iterator = table.iterator();
    while (iterator.hasNext()) {
      rows.add(iterator.next().getValues());
    }
    rows.sort((a, b) -> Double.compare(((Number) b[1]).doubleValue(), ((Number) a[1]).doubleValue()));

    Set<Integer> returnedTop10 = new HashSet<>();
    for (int i = 0; i < Math.min(10, rows.size()); i++) {
      returnedTop10.add(((Number) rows.get(i)[0]).intValue());
    }
    int hits = 0;
    for (int i = 0; i < Math.min(10, trueEntries.size()); i++) {
      if (returnedTop10.contains(trueEntries.get(i).getKey())) {
        hits++;
      }
    }
    return hits / 10d;
  }

  private static ShardedIndexedTable newTable(int numShards, int trimThreshold) {
    QueryContext queryContext = QueryContextConverterUtils.getQueryContext(
        "SELECT sum(m1) FROM testTable GROUP BY d1 ORDER BY sum(m1) DESC LIMIT " + RESULT_SIZE);
    DataSchema dataSchema = new DataSchema(new String[]{"d1", "sum(m1)"},
        new DataSchema.ColumnDataType[]{DataSchema.ColumnDataType.INT, DataSchema.ColumnDataType.DOUBLE});
    return new ShardedIndexedTable(dataSchema, false, queryContext, RESULT_SIZE, TRIM_SIZE, trimThreshold, 16,
        numShards, EXECUTOR);
  }

  private static String key(double skew, int numShards, String policy) {
    return skew + "/" + numShards + "/" + policy;
  }

  private static double at(List<Double> sorted, int rank) {
    return rank <= sorted.size() ? sorted.get(rank - 1) : -1;
  }

  private static int[] buildPermutation(int size, long seed) {
    int[] permutation = new int[size];
    for (int i = 0; i < size; i++) {
      permutation[i] = i;
    }
    Random random = new Random(seed);
    for (int i = size - 1; i > 0; i--) {
      int j = random.nextInt(i + 1);
      int tmp = permutation[i];
      permutation[i] = permutation[j];
      permutation[j] = tmp;
    }
    return permutation;
  }

  private static double[] buildZipfianCdf(int cardinality, double skew) {
    double[] cdf = new double[cardinality];
    double sum = 0;
    for (int i = 0; i < cardinality; i++) {
      sum += 1d / Math.pow(i + 1, skew);
      cdf[i] = sum;
    }
    for (int i = 0; i < cardinality; i++) {
      cdf[i] /= sum;
    }
    return cdf;
  }

  private static int drawRank(double[] cdf, double u) {
    int index = Arrays.binarySearch(cdf, u);
    if (index < 0) {
      index = -index - 1;
    }
    return Math.min(index, cdf.length - 1);
  }
}
