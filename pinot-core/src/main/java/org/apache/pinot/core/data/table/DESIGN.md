<!--

    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

-->
# GROUP BY Combine-Level Memory & Performance — Design Document

Status: **investigation / candidate designs — not finalized.** Three candidate directions are
described below (§4, §5, §6 — the third combines the first two); no decision has been made
between them, and none is proposed for merge as-is. This document exists to record findings
before they are lost, per Jackie's request, ahead of converging on one design.

Tracked issues: [apache/pinot#10498](https://github.com/apache/pinot/issues/10498),
[apache/pinot#11924](https://github.com/apache/pinot/issues/11924) ("GROUP BY Optimizations").

Related PR (merged, benchmark methodology fix found during this investigation):
[apache/pinot#19368](https://github.com/apache/pinot/pull/19368).

Candidate branch (Direction A prototype, not a PR):
`scottlan0113/pinot` branch `explore/adaptive-shard-capacity`.

## 1. Problem statement

`GroupByCombineOperator` merges the per-segment GROUP BY results from all segments on a
server into a single `IndexedTable`, using multiple worker threads concurrently calling
`upsert()` on that one table (see `GroupByCombineOperator.processSegments()`). The production
implementation, `ConcurrentIndexedTable`, protects its backing map with a single
`ReentrantReadWriteLock`.

Two related concerns motivate this investigation:

1. **Performance**: at high thread counts, does the single shared lock become a bottleneck,
   and if so, under what conditions (cardinality, skew)?
2. **Memory**: independent of the performance question, GROUP BY at high cardinality can hold
   a large number of live groups in memory before trimming, with associated JVM heap / GC
   cost.

## 2. Investigation timeline

- **2026-08-25**: Read the GROUP BY pipeline end to end. Revisited an earlier discussion
  between Jackie and **walterddr** on #10498, in which walterddr split the hypothesized
  bottleneck into two cases: low cardinality (threads contend for the same hot keys, lock
  contention) vs. high cardinality (rising per-group overhead). An initial benchmark run
  using the existing `BenchmarkIndexedTable` concluded "no evidence lock is the bottleneck"
  in either case — **this conclusion was later found to be wrong**, caused by an unrigorous
  benchmark (see §8, and the fix in #19368: a shared `java.util.Random` field created false
  cross-thread contention, and `NUM_RECORDS` was too close to the key cardinality to exercise
  realistic repeat-hit behavior). Jackie's response proposed two directions: (a) profile to
  find the real bottleneck, and (b) consider an off-heap storage solution (see §5).
- **2026-08-26**: JFR profiling of the corrected benchmark found `TableResizer`'s sort/trim
  work (invoked from `IndexedTable.resize()`) accounting for ~43.5% of CPU time in the
  combine-operator hot path, running entirely under `ConcurrentIndexedTable`'s exclusive
  write lock — confirming lock contention as a real, measurable bottleneck once the benchmark
  itself was fixed. This motivated Direction A (§4): shard the table by key hash so trimming
  one shard does not block writers to other shards.
- **2026-08-27**: Direction A implemented against the real `ShardedIndexedTable` /
  `IndexedTable` classes (not only simulation) and verified end to end (§4.5). Jackie
  countered with Direction B (§5): revisit a per-thread-local design, but backed by off-heap
  storage rather than JVM-heap objects, specifically to address GC pressure.

## 3. Root cause: lock contention under `ConcurrentIndexedTable`

`ConcurrentIndexedTable.upsertWithOrderBy()` takes the table's read lock for every `upsert()`
call (to allow concurrent writers), but escalates to the write lock — blocking every other
writer — whenever `_lookupMap.size() >= _trimThreshold`, in order to run
`IndexedTable.resize()`. `resize()` calls `TableResizer.resizeRecordsMap()`, which performs a
heap-based top-K selection over the entire map (see `TableResizer.getTopRecordsHeap()`) — an
O(map size) operation. JFR profiling confirmed this accounts for ~43.5% of combine-operator
CPU time, and it runs while every other combine thread is blocked on the write lock.

## 4. Direction A: sharding + adaptive capacity

### 4.1 Sharding architecture

`ShardedIndexedTable` (new class, `pinot-core/.../data/table/ShardedIndexedTable.java`, not
yet wired into `GroupByUtils`'s table-selection factories — see §4.6) replaces one
`ConcurrentIndexedTable` guarded by one lock with `numShards` (64 in all benchmarks below)
independent `ConcurrentIndexedTable`s, each with its own lock. Every contribution to a given
key is routed to `_shards[Math.floorMod(key.hashCode(), numShards)]` regardless of which
combine thread produced it (`Key.hashCode()` is content-based — `Arrays.hashCode` — not
identity-based), so each shard always holds the complete, up-to-date value for its keys, and
a key is stored in exactly one place, never duplicated. This differs from a naive
per-thread-local-table design (`SimpleIndexedTable`, already present in the codebase, used
when the combine operator runs single-threaded): a per-thread design must trim each thread's
partial view independently before merging, which can silently drop a key whose combined
value across threads would have made the final top-K, and duplicates any key seen by more
than one thread's table.

### 4.2 The resulting memory ceiling problem

Each shard is configured with the **same** full `resultSize`/`trimSize`/`trimThreshold` a
single `ConcurrentIndexedTable` would use for the whole table (not divided by `numShards`) —
dividing the budget per shard was found to measurably hurt recall on moderately-skewed data.
This means the aggregate memory ceiling across all shards is `numShards * trimSize` — e.g.
64 * 5000 = 320,000, vs. the un-sharded design's fixed 5,000. Direct measurement confirmed
this ceiling is genuinely approached (>99%) at true group cardinality >= 1M — a real ~64x
memory cost at exactly the cardinality #10498 is concerned with.

### 4.3 Adaptive capacity design

To recover most of the ceiling cost without the correctness risk of a fixed, uniformly small
per-shard capacity, each shard can optionally narrow its own trim capacity at runtime based
on a local signal, instead of keeping the full budget for its entire lifetime.

**Signal.** `top1Share` = (a shard's single largest-value key's current aggregated value) /
(sum of all values that shard has processed). Validated to separate the "needs full capacity"
regime (mild skew — top1Share stays low regardless of cardinality) from the "small capacity
is safe" regime (skew >= 0.5 — top1Share rises with skew), using only each shard's own local
view. A simpler candidate signal — `distinctKeys / recordsProcessed` — was tried first and
rejected: it stayed flat across the skew range that matters (0.0–0.5), because hash-based
key routing scatters hot keys across shards independent of rank, so a shard's local
distinct-key coverage does not track global skew.

**Tiers.** Three capacity tiers, derived from the full `trimSize` passed to the constructor:
FULL (as configured) when `top1Share < 0.01`; MEDIUM (`trimSize / 10`) when
`0.01 <= top1Share < 0.05`; SMALL (`trimSize / 100`) otherwise. Every shard starts at FULL.

**Safety properties.** (1) Monotonic: a shard only ever narrows, never widens, since entries
already discarded by a trim cannot be recovered. (2) Minimum-sample gate: no tier transition
is considered before a shard has processed at least 500 units of value, since `top1Share` is
noise-dominated at very low sample counts (trivially 1.0 after a single upsert).

**Cost.** Computing `top1Share` incrementally (`DoubleAdder`/`DoubleAccumulator`, updated
O(1) per upsert, plus one O(1) hash lookup into the shard's own map to read back the current
aggregated value for the key just written) measured at ~1.8 ns/upsert — negligible. An
earlier, rejected approach recomputed the signal via a full periodic rescan of the shard's
map; at realistic shard sizes (~15,625 entries) that cost ~33 us/scan, which projected to
~167 ms of added overhead across one benchmark invocation — more than 2x the entire sharded
query time, and would have erased the sharding speedup. This is the same class of "did you
actually measure the added cost" scrutiny that applies to sharding itself (§4.5).

### 4.4 Implementation bugs found and fixed

Three defects were found while verifying the real implementation against skewed data
(not just a smoke test) — all in code written during this investigation, not in existing
`ConcurrentIndexedTable`/`IndexedTable`/`TableResizer` behavior, which was independently
re-verified to be unaffected:

1. `runningMax` was accumulating each upsert's raw incoming value instead of the key's
   current aggregated value read back from the map. Since `runningTotal` correctly grows
   with volume but a single call's raw value never reflects a hot key's accumulated
   dominance, `top1Share` silently trended toward zero regardless of true skew, and no
   shard ever narrowed. Fixed by reading the post-merge value via `_lookupMap.get(key)`.
2. Smaller tiers' `trimThreshold` was initially computed by preserving the full tier's own
   threshold:size ratio (200x for a 5000/1,000,000 configuration), producing an absolute
   threshold a shard's natural key population could never reach at moderate cardinality —
   so `resize()` never actually fired and the "narrowed" tier had no real effect. Fixed with
   a small fixed multiplier (4x) instead.
3. Early real-class correctness tests used `ORDER BY sum(m1)` without `DESC`; standard SQL
   defaults to ascending order, so trimming was retaining the smallest values instead of the
   largest. This affected only the correctness-verification test harness, not the
   `AdaptiveConcurrentIndexedTable` production code, and not the JMH performance numbers or
   memory/size measurements (neither depends on which end of the ordering is kept).

### 4.5 Verification results

All results below are against the real `ShardedIndexedTable` / `AdaptiveConcurrentIndexedTable`
classes, not a standalone simulation, after the fixes in §4.4.

**Performance** (JMH, 3 forks x 5 iterations, 10 threads x 100k records, cardinality
100x100, us/op):

| Implementation | Result |
|---|---|
| `ConcurrentIndexedTable` | 407413 ± 11895 |
| `SimpleIndexedTable` | 93076 ± 2361 |
| `ShardedIndexedTable` (fixed capacity) | 62525 ± 7326 |
| `ShardedIndexedTable` (adaptive capacity) | 58825 ± 1690 |

Sharded is ~6.9x faster than the unsharded concurrent table and ~1.6x faster than the
per-thread simple design; enabling adaptive capacity on top of sharding costs no measurable
additional time (bookkeeping overhead is within noise) — sharding accounts for essentially
all of the speedup, and adaptive capacity's contribution is to memory, not latency.

**Memory** (entries held before final merge, real classes):

| Skew | Cardinality | Fixed capacity | Adaptive capacity | Reduction |
|---|---|---|---|---|
| 0.15 (mild) | 320K | 304,605 | 304,467 | ~0% (by design — see below) |
| 0.50 | 320K | 286,588 | 114,598 | 60% |
| 0.50 | 1M / 5M / 20M | 557,000 / 837,385 / 942,449 | 352,464 / 790,259 / 929,560 | 37% / 6% / 1% |
| 1.00 | 320K | 149,540 | 7,762 | 94.8% |
| 1.00 | 1M / 5M / 20M | 217,142 / 301,016 / 361,025 | 8,386 / 7,591 / 8,274 | 96.1% / 97.5% / 97.7% |

At mild skew, adaptive capacity produces byte-identical behavior to fixed capacity — no
memory saved, but no correctness risk introduced either, since `top1Share` correctly stays
in the FULL tier when data genuinely is not concentrated. At skew >= 1 (arguably the more
realistic shape for a genuine production hot key), the reduction is large and stable across
the full tested cardinality range up to 20M — the exact regime #10498 is most concerned
about. At skew = 0.5, savings are real but shrink as cardinality grows, because a Zipfian
distribution with skew < 1 genuinely flattens at extreme cardinality (the rank-probability
normalizing sum diverges as cardinality grows); this band was already known to be safe for
even a small fixed capacity, so the shrinking savings carry no correctness risk.

**Correctness** (recall@10 against an independently computed ground truth, real classes):
adaptive capacity's recall never dropped below the fixed-full-capacity baseline in any
tested (skew, cardinality) combination — always 100% where the fixed-full baseline was
100%. A fixed *small* capacity, by contrast, dropped as low as 0% recall at mild skew and
high cardinality, reproducing the original `ShardSizeCalibration` finding that motivated
keeping full capacity per shard in the first place (§4.2).

### 4.6 Known scope limitations

- Assumes a single numeric, SUM-like (additive) ORDER BY aggregate. `runningTotal`'s
  correctness relies on "sum of every upsert's raw contribution equals the sum of all keys'
  final aggregated values," which holds for SUM/COUNT but not for MIN/MAX/AVG/COUNT DISTINCT
  or other non-additive aggregates. Extracting a signal for arbitrary ORDER BY would need to
  reuse `TableResizer`'s internal comparator, not currently exposed for this purpose.
- Multi-column GROUP BY is architecturally expected to work (the value-column index is
  computed relative to `numKeyColumns`, not hardcoded to a single key column) but has not
  been tested.
- `ShardedIndexedTable` is not wired into `GroupByUtils.getTrimEnabledIndexedTable()` /
  `GroupByCombineOperator._indexedTable` (typed as `IndexedTable`, a sibling class, not a
  supertype of `ShardedIndexedTable`) — everything above is verified via standalone
  construction, not a real end-to-end query. Wiring it in would require widening that field's
  (and the factory's) type to `Table`, plus adding `isTrimmed()`/`getResizeTimeMs()` to
  `ShardedIndexedTable`.
- No regression test suite yet — verification so far is ad-hoc scratch scripts, not
  committed tests. This should exist before proposing a real PR.
- **Found 2026-08-27, NOT yet fixed, NOT yet confirmed against a realistic-value workload**:
  `AdaptiveConcurrentIndexedTable`'s minimum-sample gate (`updateSignalAndMaybeShrink`) checks
  `_runningTotal.sum() < MIN_SAMPLES_BEFORE_ADAPTATION` — comparing a **sum** of upserted
  values against a **sample-count**-shaped threshold. This is the identical bug class Direction
  C's own port of this signal had (§6.5), fixed there with an explicit sample counter separate
  from the value sum. Every verification in §4.5 above used a workload where this coincidentally
  doesn't matter (found by code inspection while researching precedent for Direction C's
  adaptive-capacity sub-segment port, not by re-running Direction A's own tests with a
  non-unit-valued workload — that confirmation step hasn't been done). If confirmed, this would
  mean the memory-reduction and recall numbers in §4.5 need re-verification under a realistic
  (non-1.0-valued) workload before being relied on as-is, the same way Direction C's original
  numbers needed re-verification after this bug class was found there. Not fixed here yet —
  flagged for a decision (fix now vs. defer) rather than silently patched, since §4.5's numbers
  have already been reported externally and a fix would need those claims revisited, not just
  the code.

## 5. Direction B: per-thread tables + off-heap (initial prototype)

### 5.1 Proposal (Jackie, 2026-08-27)

Rather than sharding a single logical table across independently-locked shards, give each
combine thread its own separate table (as `SimpleIndexedTable` already does), and implement
that per-thread table using off-heap storage (native memory, outside the JVM-managed heap —
e.g. via `sun.misc.Unsafe` or the `java.lang.foreign` API) instead of JVM objects. Because
each such table is only ever touched by one thread, no concurrent access needs to be
supported by the off-heap structure itself — a substantially simpler implementation problem
than making a *shared* off-heap structure thread-safe would be.

### 5.2 Relationship to the existing per-thread ("simple") design

This is architecturally the same shape as `SimpleIndexedTable` — independent per-thread
tables, merged at the end — which this investigation already evaluated and found to have two
distinct costs (see the "simple" findings referenced throughout §2–§4): a correctness risk
(a key's true importance can only be known after merging all threads' partial views; trimming
one thread's table locally can drop a key that would have mattered) and memory duplication
(the same key stored redundantly in every thread's table that observed it — measured up to
~3.45x on skewed data in earlier testing). Off-heap storage does not, by itself, change either
of those properties:

- **Duplication cost**: off-heap makes each duplicated entry cheaper to store (no Java object
  header/alignment overhead) and removes GC pressure from holding many such entries, but does
  not reduce the duplication *factor* — the same key is still stored once per thread that saw
  it. Off-heap plausibly makes a given duplication factor more tolerable in absolute
  memory/GC terms; it has not been measured whether it makes the duplication factor itself
  irrelevant.
- **Correctness risk**: unaffected by storage location. Whether a locally-trimmed thread table
  is backed by a JVM heap object or a native memory buffer, a key with a low value in one
  thread's partial view can still be discarded before that thread ever learns the key's true
  combined importance across threads. This risk is structural to "trim locally before
  merging," independent of on-heap vs. off-heap.

### 5.3 What off-heap solves vs. does not solve

Off-heap and Direction A's adaptive capacity address different, complementary axes of the
memory problem, not the same one:

| | Off-heap | Adaptive capacity (Direction A) |
|---|---|---|
| Reduces | Cost **per entry held** (object overhead, GC pressure) | **Count** of entries held (the ceiling) |
| Does not reduce | Entry count | Per-entry storage cost |

The two are, in principle, stackable rather than competing: a design could shard (or use
per-thread tables) *and* store entries off-heap *and* apply adaptive capacity to bound the
count — each addresses a different cost.

### 5.4 Initial prototype

`OffHeapGroupTable`
(`pinot-core/src/main/java/org/apache/pinot/core/data/table/OffHeapGroupTable.java`) is a
first, deliberately narrow prototype: single INT group-by key, single DOUBLE SUM-like
aggregate (same scope as Direction A, for comparability), built on
`java.lang.foreign.Arena`/`MemorySegment` (the modern JDK 22+ FFM API — chosen over
`sun.misc.Unsafe`, which is unsupported and has a long-signaled removal path). Each record is
a fixed-width 16-byte off-heap slot (4-byte key + 8-byte double, naturally aligned); an
on-heap open-addressing `int[]`-based index (key → slot) provides O(1) average lookup without
reintroducing per-entry object overhead. `Arena.ofConfined()` is used deliberately: it makes
the single-thread-only assumption a JVM-enforced invariant (another thread touching this
table's memory throws), not just a convention. Caller must call `close()` — off-heap memory
is not garbage collected.

**Correctness** (scratch verification, not yet a committed test): cross-checked against a
plain `HashMap<Integer, Double>` ground truth over 50,000 random upserts (including negative
values and heavy key repetition, cardinality 2,000) — every key's final aggregated value
matched exactly. Table growth (off-heap segment reallocation + index rehashing) and
trim-to-top-K (sort + compact into a smaller segment) were also verified correct in isolation.

**GC / memory** (scratch verification; the specific claim in Jackie's proposal): compared
against a plain on-heap `HashMap<Integer, Double>` baseline — not the real `SimpleIndexedTable`,
to isolate exactly the "many boxed entries in JVM heap" cost this design targets, without
Pinot-specific overhead (Key/Record wrappers, TableResizer) diluting the comparison; a real
`SimpleIndexedTable` would carry more per-entry overhead than plain `HashMap`, so this
comparison is if anything conservative toward the on-heap side. Workload: 10 simulated
per-thread tables, 200,000 upserts each, cardinality 100,000 (864,732 total entries held,
identical between both approaches), measured via `GarbageCollectorMXBean`, `-Xmx512m`,
post-JIT-warmup, repeated twice for stability:

| | wall-clock | GC time | GC count |
|---|---|---|---|
| on-heap (`HashMap`) | 119 ms | 37–38 ms | 9 |
| off-heap (`OffHeapGroupTable`) | 72–74 ms | 1–2 ms | 1–9 (unstable, but always cheap) |

GC time dropped ~95% and was consistent across both runs. GC *count* was not a reliable
signal (varied 1 vs. 9 between runs), but GC *time* was always far lower for off-heap even
when counts matched — i.e. the benefit is not fewer collections but much cheaper ones (far
less live object graph for the collector to trace). Wall-clock time was also lower for
off-heap in this test (~38–40% faster) — avoiding millions of small allocations (boxed
`Integer`/`Double`, `HashMap.Node`) has a direct allocation-cost benefit, not only a
downstream GC-cost benefit. This directly supports Jackie's stated rationale ("这样可以解决
java GC的问题").

**What this prototype does NOT yet address** — the open questions from §5.2 are unchanged by
these results, since they test a different axis:

- The correctness risk from trimming a thread's partial view before merging (a key's true
  importance may only be known after merging) is unaddressed by design — this prototype has
  no merge or trim-before-merge logic exercised yet beyond the isolated `trimTo()` unit check.

### 5.5 Duplication factor, measured on this prototype

`DuplicationFactorOffHeap.java`: `NUM_THREADS=10` threads each independently draw
`RECORDS_PER_THREAD=50,000` samples from the same Zipfian distribution (cardinality 20,000,
per-thread local `trimTo(500)` before merge, matching `finish()` semantics — trim once at
finalization, not continuously). Duplication factor = (sum of each thread's post-trim size) /
(size of the union of keys that survive in at least one thread's post-trim table):

| Skew | Surviving-key union | Total held | Duplication factor |
|---|---|---|---|
| 0.00 | 4,458 | 5,000 | 1.12x |
| 0.15 | 3,932 | 5,000 | 1.27x |
| 0.30 | 2,465 | 5,000 | 2.03x |
| 0.50 | 1,344 | 5,000 | 3.72x |
| 0.80 | 867 | 5,000 | 5.77x |
| 1.00 | 824 | 5,000 | 6.07x |

Monotonically increasing with skew, same direction as the earlier on-heap "simple" finding —
**confirms duplication factor is a property of the per-thread-local-trim architecture, not of
storage backend**, as hypothesized in §5.2: off-heap does not change which keys end up
duplicated, only what each duplicate costs. The absolute magnitude here (up to 6.07x) is
higher than the previously-cited 1.63x–3.45x range; that range used different parameters
(trim capacity, thread/record counts) from a different session and should not be assumed to
transfer — this is a parameter-dependent number that needs re-measuring under whatever the
real production trim/thread configuration would be, not a fixed constant.

(A first version of this measurement had a bug: it computed the denominator as every key ever
*drawn* before trimming, which is close to full cardinality regardless of skew and produced a
nonsensical duplication factor below 1.0. Fixed by only counting keys that *survive* each
thread's local trim.)

### 5.6 Does off-heap's GC benefit survive duplication?

`CompareWithDuplication.java`: reran the §5.4 GC/wall-clock comparison, but now with the
realistic per-thread-local-trim workload (matching §5.5's setup, not §5.4's untrimmed one),
at the same scale as §5.4 (10 threads x 200,000 upserts, cardinality 100,000, trim to 5,000),
at low-duplication (skew 0.0) and high-duplication (skew 1.0) extremes:

| Skew | Impl | Total held | Wall-clock | GC time |
|---|---|---|---|---|
| 0.0 | on-heap | 50,000 | 283 ms | 29 ms |
| 0.0 | off-heap | 50,000 | 286 ms | 2 ms |
| 1.0 | on-heap | 50,000 | 173 ms | 16 ms |
| 1.0 | off-heap | 50,000 | 165 ms | 3 ms |

**GC-time advantage is robust**: off-heap stays far lower (81–93% less GC time) at both the
low- and high-duplication extreme — duplication does not erase this benefit, since both
implementations pay the identical duplication factor and off-heap still wins. This is the
core claim from §5.4 holding up under a more realistic workload.

**Wall-clock advantage was NOT robust once trimming was included, in the first version of
this prototype** — off-heap was a wash-to-worse at skew 0.0 (286 vs 283 ms) and only
modestly better at skew 1.0 (165 vs 173 ms), unlike §5.4's untrimmed workload where off-heap
was ~38–40% faster outright. Root cause: `trimTo()` originally sorted by boxing into
`Double[]`/`Integer[]` (`java.util.Arrays.sort` only accepts a `Comparator` for `Object[]`,
not `double[]`/`int[]`), reintroducing exactly the per-entry allocation cost this design is
meant to avoid, just during the trim step instead of the upsert step.

**Fixed**: replaced the boxed sort with a hand-rolled, zero-boxing quicksort operating on two
parallel primitive arrays (`double[] values`, `int[] indices`), sorted together. First fix
attempt used a standard 2-way (Lomuto-style) partition and was measured to be
**catastrophically slower** — 2117 ms vs the original 286 ms at skew 0.0 — because this
workload's values are `upsert(key, 1.0)`-accumulated small-integer-ish counts with heavy
duplication (many keys landing on the same count), which degrades a naive 2-way partition
toward O(n^2): when many elements equal the pivot, only one element gets excluded per
partitioning pass. Fixed properly with 3-way (Dutch national flag) partitioning, which groups
all pivot-equal elements into a single pass with no further recursion needed on them — the
standard remedy for duplicate-heavy inputs. Re-measured after the real fix, reproduced across
2 runs:

| Skew | Impl | Wall-clock | GC time |
|---|---|---|---|
| 0.0 | on-heap | 270–279 ms | 11–21 ms |
| 0.0 | off-heap | 217–220 ms | 2–3 ms |
| 1.0 | on-heap | 167 ms | 9–11 ms |
| 1.0 | off-heap | 134–135 ms | 1 ms |

Both wall-clock (~19–22% faster) and GC time (~82–91% less) now favor off-heap at both
duplication extremes, consistently across both runs. Duplication factor itself (§5.5) was
independently re-confirmed unchanged by this fix (1.11x–6.05x vs. the earlier
1.12x–6.07x — same numbers within run-to-run noise), as expected: fixing the trim
implementation's speed doesn't change which keys the trim selects.

### 5.7 Correctness risk, measured on this prototype

`CorrectnessRiskOffHeap.java`: same multi-thread-Zipfian setup as §5.5/§5.6 (10 threads,
50,000 upserts each, cardinality 20,000), but now measuring recall@10 instead of duplication
or GC. Ground truth is each key's TRUE total value, summed across every thread's contribution
regardless of trim; the "reported" result merges each thread's `OffHeapGroupTable` only
*after* it locally trims to `TRIM_CAPACITY` (matching `finish()` semantics), summing values
for any key that happens to survive in more than one thread. Swept across skew and three trim
capacities, 5 trials each (avg recall@10, %):

| Skew | capacity=500 | capacity=100 | capacity=20 |
|---|---|---|---|
| 0.15 | 92 | 80 | 62 |
| 0.30 | 100 | 96 | 96 |
| 0.50 | 100 | 100 | 100 |
| 0.80 | 100 | 100 | 100 |
| 1.00 | 100 | 100 | 100 |

**The risk is real and reproduces through the real `OffHeapGroupTable`/`trimTo()` path, not
just as an abstract architectural concern.** The pattern matches Direction A's
`ShardSizeCalibration` finding exactly: risk concentrates in the mild-skew band (0.15,
mildly at 0.30) and gets monotonically worse as trim capacity shrinks (92% → 80% → 62% at
skew 0.15); skew >= 0.5 is robust at 100% recall even at the most aggressive capacity tested
(20) — this is the same architectural property surfacing in a different implementation, as
expected, since off-heap storage does not touch trim *logic*, only trim *cost* (§5.2). The
absolute recall numbers here are milder than the most severe cases previously found for the
on-heap "simple" design (which used different parameters, from a different session) — as
with the duplication-factor magnitude (§5.5), this is parameter-dependent and should not be
assumed to transfer; what's robust is the *shape* (mild-skew risk band, monotonic in
capacity), independently reproduced here.

**Practical implication for Direction B**: whatever trim capacity a real deployment would use
per thread matters a great deal for how much correctness risk is being accepted, exactly as
it did for Direction A before adaptive capacity was introduced to manage it (§4.2-§4.3). This
prototype has no analogous mitigation — Direction A's `top1Share` adaptive-capacity mechanism
could in principle be ported to a per-thread off-heap table the same way it was added to a
sharded one, but that has not been attempted.

### 5.8 Open questions

- No mitigation for the correctness risk exists in this prototype (see above) — porting
  something like Direction A's adaptive capacity signal to a per-thread table is a plausible
  next step, unexplored.
- Prototype is not integrated with real `QueryContext`/`DataSchema`/multi-column keys, and is
  not wired into the query engine.
- No regression test suite — current verification is scratch scripts only, same caveat as
  Direction A (§4.6).

## 6. Direction C: combining A and B (sharded off-heap)

Prompted by the user asking, after §4/§5 were both characterized, whether the two directions
could simply be combined rather than chosen between.

### 6.1 Why "just combine them" is not trivial

Direction A's correctness guarantee depends on routing by **key hash** (§4.1): a key always
lands on the same shard regardless of which thread produced it, so no shard ever has an
incomplete view. Direction B's off-heap tractability depends on routing by **thread** (§5.1):
each table is touched by exactly one thread, so `Arena.ofConfined()` can enforce
single-thread access at the JVM level with no concurrency control needed at all. These two
routing schemes are in direct tension — switching from thread-routing to key-hash-routing
means a single shard can now be written by *multiple* threads concurrently, which is exactly
the "shared off-heap structure" problem `Arena.ofConfined()` was used to avoid needing to
solve.

### 6.2 Design: shard by key hash, off-heap storage per shard, lock per shard

`ShardedOffHeapGroupTable`
(`pinot-core/src/main/java/org/apache/pinot/core/data/table/OffHeapGroupTable.java`'s new
`(capacity, Arena)` constructor + `ShardedOffHeapGroupTable.java`): `numShards` shards, each
an `OffHeapGroupTable`, all sharing one `Arena.ofShared()` (not `ofConfined()` — a shared
arena permits access from any thread, since JVM-level single-thread enforcement is no longer
available once multiple threads must reach the same shard) with one lifecycle (`close()`
frees every shard's memory together). Routing mirrors Direction A exactly:
`Math.floorMod(Integer.hashCode(key), numShards)`. Each shard is additionally guarded by its
own `ReentrantReadWriteLock` — this is *not* optional the way `ConcurrentIndexedTable`'s read
lock is for the common case: `OffHeapGroupTable` has no internal concurrency control
analogous to `ConcurrentHashMap`, so every `upsert()` takes the shard's **write** lock for its
full duration, not just resize/trim. This is a real, deliberate simplification for a first
prototype — more serialization per shard than Direction A's `ConcurrentHashMap`-backed shards
have — flagged as something to measure, not assume away.

`OffHeapGroupTable` itself gained a second constructor,
`OffHeapGroupTable(int initialCapacity, Arena arena)`, taking an externally-owned arena
instead of creating its own confined one; an instance built this way does not close the arena
in its own `close()` (the owner — here, `ShardedOffHeapGroupTable` — closes it once, after
every shard is done). The original single-arg constructor (own confined arena, owns its own
lifecycle) is unchanged, so every existing Direction B usage/test still works as before.

**Tried a finer-grained fast path, measured a real regression, reverted.** Stack profiling
(§6.4) found ~52% of thread-samples parked on this exact write lock, identically for fixed and
adaptive capacity — real evidence the "measure, don't assume" flag above deserved a real
attempt, not just a note. Built a fast path for fixed capacity only: `upsert()` takes a shared
*read* lock first, tries `OffHeapGroupTable.tryFastUpdate(key, value)` (a new method — probes
the index read-only, then atomically updates the value slot for an existing key), and only
escalates to the write lock when the key genuinely doesn't exist yet (which mutates the index
and is never safe under a shared lock). Adaptive capacity was deliberately left untouched in
this attempt — its signal tracking uses plain fields safe only under exclusive access, and
making that safe under a shared lock (`DoubleAdder`/`LongAdder`/CAS-based max, mirroring
`AdaptiveConcurrentIndexedTable`) is a separate piece of work.

Caught a real bug before it went anywhere near correctness testing: `VarHandle.getAndAdd`
throws `UnsupportedOperationException` for a `MemorySegment` double `VarHandle` in this JDK —
found by actually running the new concurrency stress test (see below), not assumed to work
from the API surface. Fixed with a compare-and-swap retry loop
(`VarHandle.compareAndSet`) instead, the more universally-supported access mode.

Verified correctness thoroughly given the added concurrency risk: a new stress test
(`testFixedCapacityUnderHeavyContention`, kept — see below) — 20 threads, cardinality 20,
maximizing both CAS contention on existing keys and the race of many threads simultaneously
missing the fast path for a not-yet-inserted key. 28 total clean runs (20 ad-hoc + 8 through
the real `mvn test` pipeline) with zero failures before performance was measured at all.

Then measured performance, expecting an improvement: instead, a confirmed, reproducible
**regression**. Two independent JMH runs of the skewed benchmark: 48,463 then 51,087 us/op,
both well above the write-lock-only baseline of 43,723 — getting worse, not better, across
runs. Profiled the regression directly: the new run's stack sample showed
`ReentrantReadWriteLock$Sync.tryAcquireShared`/`tryReleaseShared` and
`ThreadLocal$ThreadLocalMap.cleanSomeSlots` as real, distinct costs that never appeared in the
write-lock-only profile — `ReentrantReadWriteLock`'s read side carries `ThreadLocal`-based
hold-count bookkeeping (needed to correctly support reentrant read locks and detect the "last
reader releasing" case) that the write-only path never pays. For a critical section this short
(one index probe plus one CAS on a double), that fixed per-acquisition overhead outweighed
whatever parallelism the shared lock bought back. This matches well-documented Java concurrency
guidance: `ReentrantReadWriteLock` earns its keep on long/expensive critical sections, not
uniformly — it was the wrong tool for a critical section this cheap, not a broken idea in
general.

Reverted completely: `OffHeapGroupTable.tryFastUpdate` and its `VarHandle` field removed,
`ShardedOffHeapGroupTable.upsert()` restored to always take the write lock. The stress test
(renamed from `testFixedCapacityFastPathUnderHeavyContention` to
`testFixedCapacityUnderHeavyContention`) was kept — heavy-contention correctness coverage for
the write-lock design that's actually in place is still valuable, independent of why it was
originally written. The `getAndAdd`-unsupported-for-double finding is real and worth keeping
even though the attempt it was found inside was reverted: anyone reaching for `VarHandle`
atomics on off-heap `double` values in a future attempt should use `compareAndSet`, not
`getAndAdd`, from the start.

**Third attempt, sub-segmenting: kept the exact locking model that beat the fast path, changed
the granularity instead — this one worked.** Asked what an experienced concurrent-systems
engineer would try next (given the read-lock fast path's failure mode was specifically
`ReentrantReadWriteLock`'s own overhead, not the idea of finer granularity in general): split
each OUTER shard into `numSubSegments` independently-locked `OffHeapGroupTable` instances
instead of one, keeping the *exact same* exclusive-write-lock-for-every-upsert model that was
just measured to beat the alternative — no new correctness hazard, since resize is still always
safely inside a normal write lock, just scoped to a smaller sub-segment. `numSubSegments=1`
(the default, via two new overloaded constructors) is byte-identical in behavior to before.
Deliberately scoped to fixed capacity only, same reasoning as the fast-path attempt: the
`top1Share` signal is tracked per OUTER shard specifically to avoid multiplying the memory-
ceiling problem (§4.2) that more outer shards would cause, and splitting that signal tracking
across independently-locked sub-segments needs its own concurrency-safety treatment not yet
done — `numSubSegments>1` with `adaptiveCapacity=true` throws rather than silently doing the
wrong thing.

`finishAllShards()` merges every shard's sub-segments 1..N-1 into sub-segment 0 (single-
threaded at that point, holding every sub-segment's lock — reuses `OffHeapGroupTable.upsert()`
directly rather than new merge logic) before trimming; `totalSize()`/`forEachEntry()`
correspondingly only need to look at sub-segment 0 per shard once `finishAllShards()` has run.

Verified correctness first, same discipline as the fast-path attempt: two new tests
(`testSubSegmentsUnderRealConcurrentThreads` — real concurrent threads, `numSubSegments=4`,
cross-checked against ground truth, specifically exercises the merge-at-finish path;
`testSubSegmentsRejectedForAdaptiveCapacity` — the guard throws as documented), plus the
existing suite re-verified at the `numSubSegments=1` default. 18 total clean runs (15 ad-hoc +
3 through real `mvn test`) before performance was measured.

Then measured performance on the skewed benchmark (`numSubSegments=4`,
`shardedOffHeapGroupTableFixedSubSegmented` vs. the existing `shardedOffHeapGroupTableFixed`).
First run had a single-iteration outlier in the *baseline* (`shardedOffHeapGroupTableFixed`
Fork 3: one iteration at 52,528 us/op against a tight 44,959-45,404 band everywhere else) —
not trusted, rerun. Second run clean on every fork of every benchmark:

| Benchmark | Score (avgt, us/op) | vs. Direction A |
|---|---|---|
| Direction A | 95,857.789 ± 2,630.367 | baseline |
| Fixed (`numSubSegments=1`) | 45,594.705 ± 250.722 | 2.10x faster |
| **Fixed, sub-segmented (`numSubSegments=4`)** | **42,181.160 ± 458.286** | **2.27x faster** |

Sub-segmenting is **8.1% faster than plain fixed capacity** in this run, **6.9% faster** in the
first (outlier-excluded) run — consistent direction and magnitude across both. This is the
first of three optimization attempts in this section that actually helped. Kept: the new
5-argument constructor is purely additive (existing 2- and 4-argument constructors delegate to
`numSubSegments=1`, unchanged behavior), so nothing about this needed reverting the way the
first two attempts did.

**Swept `numSubSegments` past 4 to see whether more is better — it is not.** Added
`shardedOffHeapGroupTableFixedSubSegmented8/16/32` alongside the existing 4, all against the
same `shardedOffHeapGroupTableFixed` (`numSubSegments=1`) baseline. Result, clean on every fork
(one run needed a targeted rerun of just K=32 — its first pass had two elevated iterations in
one fork; the rerun's aggregate agreed with the original within ~0.1%, confirming the number
despite fork-to-fork spread higher than K=4/K=8 showed):

| `numSubSegments` | Score (us/op) | vs. Direction A | vs. K=4 |
|---|---|---|---|
| 1 (baseline) | 45,326 | 2.06x | +6.7% slower |
| **4** | **42,497** | **2.19x** | **best** |
| 8 | 45,182 | 2.06x | +6.3% slower |
| 16 | 46,586 | 2.00x | +9.6% slower |
| 32 | 46,532 | 2.00x | +9.5% slower |

Non-monotonic, not just diminishing returns: performance improves from K=1 to K=4, then gets
*worse* than the K=1 baseline from K=8 onward, worst at K=16-32. **K=4 is the clear sweet spot
among the values tested, and remains the recommended default** — `NUM_SUB_SEGMENTS` in the
benchmark and any future caller should use 4 unless a different workload's own sweep says
otherwise.

**The resize-frequency hypothesis above was tested directly and REFUTED — with a methodology
flaw worth recording, not just the negative result.** Added a `divideInitialCapacityAcrossSubSegments`
constructor flag (default `true` = existing behavior) and benchmarked
`FixedSubSegmented16FullCapacity`/`32FullCapacity` (`false`: every sub-segment gets the full,
undivided `perShardInitialCapacity` instead of a fraction of it) against the original divided
variants. If smaller per-sub-segment capacity (more frequent resize) were the cause, the
undivided variants should have recovered toward K=4's ~42,800 us/op. Instead, both got
*dramatically* slower — K16: 44,865 -> 61,659 us/op; K32: 48,266 -> 85,800 us/op (noisier too,
though every iteration checked individually, no single outlier — genuinely more variable, not a
measurement artifact). Clean and reproducible in both directions, so the hypothesis is
confirmed wrong, not just unconfirmed.

The flaw: "undivided" was not a clean isolation of "resize frequency" — since capacity per
sub-segment stays at the FULL `perShardInitialCapacity` regardless of `numSubSegments`, total
initial off-heap allocation for a shard scales *with* `numSubSegments` (32x more total memory
allocated upfront at K=32, not the same total split differently). This experiment therefore
measured "does massively over-allocating initial capacity hurt performance" (yes, apparently —
consistent with the raw cost of allocating and OS-mapping that much more off-heap memory before
any upsert work even begins), not "does resize frequency matter" — those got conflated by this
specific test design. **The original question — does per-sub-segment resize frequency, at a
constant TOTAL capacity, explain the K=16/32 regression — remains open.** A properly isolated
follow-up would need to hold total per-shard capacity constant while varying how many pieces
it's split into (e.g. distributing unevenly, or pre-sizing each sub-segment from actual expected
load rather than an even division) — not attempted here, this investigation stopped after
learning the over-allocation cost is real and large enough to dominate whatever it was mixed
with.

**Sub-segmenting on the uniform workload, closing the last gap in this story.** Every
sub-segmenting result above was measured on the skewed benchmark; the uniform-key benchmark
(`BenchmarkShardedOffHeapGroupTable`, cardinality 50,000, no skew) had never been checked, so it
was unknown whether `numSubSegments=4` might help under skew specifically and do nothing (or
hurt) when contention is spread evenly instead of concentrated. Added
`shardedOffHeapGroupTableFixedSubSegmented4`/`AdaptiveSubSegmented4` there. Clean on every fork
of all 4 relevant benchmarks (no outliers, unusually tight even by this document's standards):

| Benchmark | `numSubSegments=1` | `numSubSegments=4` | Improvement |
|---|---|---|---|
| Fixed | 24,804.061 ± 285.126 | 14,911.970 ± 330.486 | **66.3% faster** |
| Adaptive | 54,765.603 ± 334.866 | 37,855.017 ± 202.076 | **44.7% faster** |

Far larger than the skewed workload's 6.3-8.1% — not a smaller or absent effect, a much bigger
one. Plausible reason: uniform data spreads ~781 keys evenly across every one of the 64 shards,
so all 64 benefit from finer-grained locking roughly equally; skewed data concentrates traffic
onto a handful of genuinely hot shards, while many others were already low-contention even at
`numSubSegments=1`, diluting the aggregate win. Not confirmed by profiling — a reasoned
explanation consistent with the numbers, not a measured mechanism. Net: sub-segmenting has now
been checked on both workload shapes this document uses throughout, and helps substantially in
both — `numSubSegments=4` is not a skew-specific trick.

### 6.3 Correctness: real concurrent threads, not simulation

Every Direction B measurement (§5) used *simulated* per-thread tables — one thread at a time,
sequentially, standing in for "a thread" — because a single `OffHeapGroupTable` was never
meant to be touched by more than one thread, so there was nothing concurrent to test. This
design's entire point is genuine concurrent access to shared shards, so it needed a different
kind of test: `ShardedOffHeapCorrectnessTest.java` runs `NUM_THREADS=10` real threads (via
`ExecutorService`) concurrently upserting 100,000 records each (cardinality 50,000) into one
shared `ShardedOffHeapGroupTable`, cross-checked against a `ConcurrentHashMap`-based ground
truth fed the identical stream. Passed cleanly across 4 runs (1 + 3 repeats, since
concurrency bugs are often non-deterministic and a single clean pass does not rule one out):
every one of 50,000 distinct keys' aggregated value matched ground truth exactly, with no
size mismatch. This is Direction A's zero-correctness-risk property (§4.1), now confirmed to
carry over to the off-heap-backed version — expected, since the same key-hash routing
argument applies regardless of where a shard's bytes live, but confirmed rather than assumed.

### 6.4 Performance vs. Direction A

`CompareShardedOffHeapVsDirectionA.java`: same workload against both the real
`ShardedIndexedTable` (Direction A, on-heap, `ConcurrentHashMap` per shard) and
`ShardedOffHeapGroupTable` (this section) — 10 real threads, 100,000 upserts each,
cardinality 50,000, 64 shards, trim size 5,000. Measured via `GarbageCollectorMXBean`,
`-Xmx512m`, post-warmup, 3 trials per run, reproduced across 2 full runs:

| | Wall-clock (avg of 3) | GC time (avg of 3) |
|---|---|---|
| Direction A (on-heap sharded) | 79 ms | 5.3 ms |
| Direction C (sharded off-heap) | 38–46 ms | ~0.0 ms |

Direction C was faster (~1.7–2.1x) and had essentially zero GC time in both full runs,
despite the write-lock-per-upsert simplification (§6.2) that was flagged as a possible
regression risk — it was not one in this test. **Caveat**: this comparison uses
`ShardedOffHeapGroupTable`'s simplified API (raw `int`/`double`, no `Record`/`Key` boxing at
the call site) against Direction A's real `Record`-based API, which requires that boxing —
some of the gap may be attributable to API/scope simplification (§5.1's single-INT-key,
single-DOUBLE-aggregate scope, same caveat as everywhere else in Direction B) rather than
purely to storage location.

**JMH-rigor follow-up (`BenchmarkShardedOffHeapGroupTable.java`, same workload — uniform
random `int` keys, cardinality 50,000, 10 threads, 64 shards, trim size 5,000 — now under
3 forks x 5 iterations x 3s, matching `BenchmarkShardedIndexedTable`'s own methodology
exactly)**:

| Benchmark | Score (avgt, us/op) | Error (99.9% CI) | vs. Direction A |
|---|---|---|---|
| Direction A (`shardedIndexedTable`) | 44,517.158 | ± 1,051.373 | baseline |
| Direction C fixed (`shardedOffHeapGroupTableFixed`) | 24,548.270 | ± 179.708 | **1.81x faster** |
| Direction C adaptive (`shardedOffHeapGroupTableAdaptive`) | 41,043.798 | ± 231.983 | **1.09x faster** |

No outlier forks (all 15 iterations per benchmark within a tight band; Direction A's own
run-to-run variance is proportionally larger than either off-heap variant's, plausibly from
on-heap GC pauses the off-heap variants don't have, not from an unstable measurement). This
confirms the ad-hoc fixed-capacity finding under real rigor — 1.81x is in the same range as
the 1.7-2.1x estimated above.

**But it surfaces a real, previously-invisible cost of adaptive capacity that the ad-hoc
numbers understated**: adaptive capacity's margin over Direction A shrinks to just ~9%, far
below fixed capacity's 1.81x, and well below the ~1.49x the earlier ad-hoc timing suggested
for adaptive specifically (`/tmp/compare-3way-flatfix.log`, not reproduced here). This
workload uses uniform keys — by design, per §6.5's own verification, adaptive capacity
correctly never shrinks anything here (no false positive). That means every upsert still pays
`updateSignal`'s bookkeeping cost (running sum, sample count, running max, tier comparison)
for a signal that never pays for itself with a smaller structure. Fixed and adaptive share the
same off-heap upsert path, so the entire 24,548 -> 41,044 us/op gap (adaptive taking ~67%
longer than fixed) is attributable to that bookkeeping alone. This is exactly the kind of gap
JMH rigor exists to catch (§8) — the ad-hoc measurement's noise floor was large enough to
mask most of it.

This is not a reason to abandon adaptive capacity — its entire value proposition is memory
reduction under skew (§6.5's ~98% at skew=1.0), and this benchmark deliberately uses the one
workload shape (uniform) where that value can never materialize, making it a worst-case
overhead measurement, not a representative one. The skewed-workload follow-up below answers
whether the memory savings pay back the overhead when adaptive capacity actually gets to use
them.

**Skewed-workload follow-up (`BenchmarkShardedOffHeapGroupTableSkewed.java`)**: same JMH rigor,
but Zipfian keys (skew=1.0, cardinality 1,000,000) instead of uniform — the exact parameters
`VerifyAdaptiveDirectionC.java` already validated collapse adaptive capacity from 217,168 to
3,200 held entries (~98.5% reduction, recall@10=100%) — so this benchmark's adaptive variant
pays the bookkeeping cost *and* collects the shrink benefit, unlike the uniform benchmark
above. First run showed an outlier fork on `shardedOffHeapGroupTableFixed` (Fork 1:
48,000-56,000 us/op vs. Forks 2-3's tight ~43,500-45,000 band, Score 46,293 ± 4,869) — not
trusted, rerun per this document's own rigor standard (§8). Second run was clean, all 9 forks
across all 3 benchmarks tight, no outliers:

| Benchmark | Score (avgt, us/op) | Error (99.9% CI) | vs. Direction A |
|---|---|---|---|
| Direction A | 95,229.021 | ± 1,515.194 | baseline |
| Direction C fixed | 43,722.839 | ± 167.442 | **2.18x faster** |
| Direction C adaptive | 48,762.513 | ± 325.504 | **1.95x faster** |

This corrects a premature read of the first (outlier-contaminated) run, which made fixed and
adaptive look statistically tied — they are not. Adaptive is still a consistent ~11.5% slower
than fixed (48,763 vs. 43,723 us/op) even with the shrink benefit included. But that overhead
shrank dramatically from the uniform benchmark's ~67% gap, and adaptive's margin over Direction
A nearly doubled (1.09x -> 1.95x) between the uniform and skewed workloads — the memory-reduction
benefit clearly buys back most, not all, of the bookkeeping cost.

A second, unplanned finding: Direction A's own absolute time roughly doubled under skew
(44,517 -> 95,229 us/op) while both off-heap variants grew far less (fixed: 24,548 -> 43,723,
~1.78x; adaptive: 41,044 -> 48,763, ~1.19x) — skew appears to cost Direction A's architecture
specifically, not just make everything uniformly slower. Plausible cause: concurrent updates to
the same hot key inside `ConcurrentHashMap` still serialize at the bucket level, and a
Zipfian-skewed workload concentrates far more traffic onto that single bucket than a uniform
one ever would. **Not confirmed** — this is an observation, not a profiled finding (the JMH
output's own standard caveat applies: correlation in aggregate numbers isn't a mechanism), and
is noted here rather than asserted as settled.

**Net**: under the skewed regime #10498 actually cares about, Direction C's advantage over
Direction A is larger, not smaller, than the uniform benchmark suggested, for both fixed and
adaptive capacity. Adaptive capacity's bookkeeping cost is real but is more than justified by
its ~98.5% memory reduction, and it still beats Direction A by a wide margin even carrying that
cost.

**Attempted optimization, measured to not help, reverted**: the natural next question is
whether that remaining bookkeeping cost can be cut. `updateSignal()` recomputes `top1Share` (a
division) and its two threshold comparisons on every single post-gate upsert; the obvious fix
is to amortize that check to once every N upserts instead, with a final unconditional check in
`finishAllShards()` so the sampling can only delay detecting a tier crossing, never change the
final capacity decision. Implemented (`TOP1_SHARE_CHECK_INTERVAL = 100`), verified correct
(regression suite still 5/5), then measured on both JMH benchmarks: no measurable improvement.
Skewed-adaptive moved from 48,762.513 ± 325.504 to 48,383.349 ± 293.989 us/op — directionally
lower, but the two 99.9% CIs overlap ([48,437-49,088] vs. [48,089-48,677]), so this is not
distinguishable from ordinary run-to-run noise. Uniform-adaptive was statistically unchanged
(41,043.798 -> 41,107.419 us/op). Reverted rather than kept as unproven complexity. Likely
reason it didn't help: the division+comparisons amortized away were probably never the
dominant cost in `updateSignal` — `_runningTotal[shard] += rawValue`, `_sampleCount[shard]++`,
and the `_runningMax` compare-and-maybe-write still run on *every* upsert regardless (removing
them isn't safe, the signal needs them), and are the more likely target for a future attempt.
Not confirmed by profiling — noted as a hypothesis for whoever picks this up next, not a
finding.

**Follow-up: actually profiled, then isolated the cost cleanly.** JMH's built-in stack sampler
(`-prof stack`) on the skewed 10-thread benchmark showed adaptive and fixed capacity with
*statistically identical* thread-state breakdowns — both ~52.1% WAITING (parked on the per-shard
write lock), both showing `AbstractQueuedLongSynchronizer.acquire` in the RUNNABLE portion at
matching percentages. This rules out lock contention as the explanation for the adaptive-vs-fixed
gap (it's a cost shared equally by both, not specific to adaptive) — real evidence for the
separate write-lock-per-upsert open item (§6.2/§6.6), just not for this one. Neither variant
showed `updateSignal` as a distinct hot frame, most likely JIT-inlined into the enclosing loop and
too fine-grained (a few nanoseconds) for a millisecond-scale sampling profiler to resolve.

To get a clean number anyway, built `BenchmarkShardedOffHeapGroupTableSingleThread.java`: same
Zipfian keys (pre-generated once in `@Setup`, identical sequence fed to both variants), but a
single thread upserting directly — no `ExecutorService`, no per-shard lock contention to dilute
the measurement. Result, no outlier forks: `fixedSingleThread` 18,104.056 ± 109.955 us/op,
`adaptiveSingleThread` 21,810.335 ± 299.060 us/op — adaptive **~20.5% slower**, working out to
**~3.7 nanoseconds of extra cost per upsert**. This is larger, not smaller, than the concurrent
benchmark's ~11.5% gap, which at first looks backwards until the arithmetic is written out: both
variants pay close to the same *absolute* lock-wait time under contention, and adding the same
constant to two numbers that differ moves their *ratio* toward 1 — the concurrent number
understates the true per-upsert cost because a shared, unrelated cost dilutes the percentage.
The single-threaded number is the honest one.

At ~3.7ns/upsert (roughly a dozen CPU cycles for two array increments, a compare-and-maybe-write,
and two branch checks), this looks close to irreducible without removing part of what the signal
needs to work — consistent with why amortizing the top1Share check alone didn't move the needle:
that check was never more than a small slice of an already-small cost. Current read: this is a
small, well-understood, tightly-bounded overhead, not an unsolved performance problem — further
optimization here has a low ceiling and is not recommended as a priority.

### 6.5 Adaptive capacity ported from Direction A

`ShardedOffHeapGroupTable` gained an optional `adaptiveCapacity` flag, porting Direction A's
`top1Share` signal onto these off-heap shards. One simplification was available here that
`AdaptiveConcurrentIndexedTable` didn't have: since every upsert already holds the shard's
exclusive write lock (§6.2), the signal can use plain `double`/`long` fields instead of
lock-free `DoubleAdder`/`DoubleAccumulator` — there is no separate concurrency problem left
to solve for the signal itself. `OffHeapGroupTable.upsert()` was changed to return the key's
post-upsert aggregated value, so the signal doesn't need a second lookup.

**First verification** (`VerifyAdaptiveDirectionC.java`, real concurrent threads, same
skew/cardinality sweep as Direction A's original validation): results closely mirror
Direction A's — mild skew unchanged (byte-identical to fixed capacity), skew>=0.5 real
reduction, skew=1.0 stable ~98-99% reduction from 1M to 20M cardinality, recall@10 never
below 100% in any tested cell. No new bugs this time — both of Direction A's original bugs
(§4.4) don't even apply to this design: `runningMax` was read from `upsert()`'s return value
from the start, and `trimTo()` is only ever called once at `finishAllShards()` (matching
`finish()` semantics), so there's no "threshold never reached during continuous upsert-time
resizing" failure mode to hit in the first place.

**A real bug was found anyway**, by testing against a more realistic workload than the
all-`1.0`-valued one above: `CompareShardedOffHeapVsDirectionA.java`'s workload uses
`random.nextDouble() * 100` as the upserted value (closer to a real `sum(m1)`-style
aggregate). Under adaptive capacity, this **uniform, non-concentrated** workload collapsed
every shard straight to the SMALL floor (`totalHeld=3200`, down from the true 50,000
distinct keys) — a sign something was very wrong, since uniform data shouldn't trigger any
shrink at all. Root cause: `MIN_SAMPLES_BEFORE_ADAPTATION` was compared against
`_runningTotal` (a **sum of upserted values**), not an actual **count of upserts** — this
only coincidentally behaves correctly when every value is exactly `1.0` (true of every
earlier verification, including the clean result above), which is exactly why it went
unnoticed until a non-unit-valued workload was tried. With mean value ~50, the gate passed
after roughly 10 upserts — far too early, when `runningMax` was still just "the single
largest individual draw seen so far," not a real signal. Fixed by adding an explicit
`_sampleCount` counter, incremented once per upsert regardless of value, and gating on that.

**The fix helped substantially but left a smaller residual issue**, also only visible with
non-unit values: post-fix, the same uniform workload now holds ~36,500 of the true 50,000
keys (down from the 3200 floor, but still ~27% lower than it should be, since uniform data
should trigger no shrinkage). Working hypothesis at the time: `MIN_SAMPLES_BEFORE_ADAPTATION=500`
(now correctly counting upserts) is still small relative to a shard's own local key
cardinality (~781 keys in this workload) — after 500 draws spread across ~781 possible keys,
most keys have been seen once or twice at most, so "the max so far" can still be dominated by
which key got lucky with a couple of high-value draws, not genuine repeated dominance.

**First attempt at that hypothesis made things worse, not better.** The gate was changed to
scale with the shard's own observed distinct-key count —
`requiredSamples = max(100, 10 * shard.size())` — which did fully resolve the uniform-workload
false positive (`totalHeld=50000` exactly, matching true cardinality in all trials). But
rerunning `VerifyAdaptiveDirectionC.java` (the skew/cardinality sweep that had never been
re-checked after this change) showed the fix had broken the actually-important case: at
skew=1.0 the aggregate memory reduction dropped to 1.5-9% (vs. Direction A's 96-98% with the
same signal/thresholds), and — backwards from what's needed — it got *worse* as cardinality
grew from 320K to 20M. Root cause: `NUM_THREADS x RECORDS_PER_THREAD / NUM_SHARDS` fixes the
*available* sample budget per shard at ~15,625 regardless of cardinality, but a Zipfian
distribution's long tail keeps adding *distinct* keys as cardinality grows even though its
*head* (the only thing `top1Share` depends on) stabilizes almost immediately — so
`shard.size()` was the wrong quantity to scale against: it conflates "tail is long" with
"signal is unreliable," and required-samples kept climbing past the available budget for most
shards, especially at higher cardinality. This was caught by the same discipline as every
other fix in this document — rerun the previously-good case after changing anything, not just
the case that motivated the change — and is recorded here rather than silently discarded,
since it's a real, structural reason to avoid this class of fix in the future, not just a bad
parameter choice.

**Final fix: reverted to a flat sample count, retuned.** `MIN_SAMPLES_BEFORE_ADAPTATION` went
from `500` to `5000` — same shape as the original (pre-bug) design, just a bigger constant,
chosen empirically the same way the `top1Share` tier thresholds already were. Verified against
both regimes together, in the same session, before considering this closed:
- Uniform/realistic-value workload (`CompareShardedOffHeapVsDirectionA.java`): `totalHeld=50000`
  exactly, matching true cardinality in all 3 trials — the false positive is fully gone, not
  just reduced.
- Skew/cardinality sweep (`VerifyAdaptiveDirectionC.java`, real concurrent threads): skew=0.15
  unchanged (304,482 held either way — correctly no shrinkage), skew=0.5 shows real reduction
  (286,603 -> 134,940, ~53%), skew=1.0 collapses to the SMALL floor on every shard
  (149,358 -> 3,200, ~98%) and **stays exactly at 3,200 from cardinality 320K through 20M** —
  matching (slightly exceeding) Direction A's own 96-98% number. `recall@10` stayed 100% in
  every cell of both sweeps.

This is now considered resolved, with one honest caveat carried into §6.6: `5000` is tuned
against this test's specific sample volume (~15,625/shard), not derived from first principles
— a workload with a much smaller per-shard sample budget (e.g. far fewer records per query, or
many more shards) has not been tried and could plausibly need a smaller constant.

**Sub-segmenting extended to adaptive capacity.** Before implementing, checked for existing
precedent rather than designing from scratch (per-request) — re-read
`AdaptiveConcurrentIndexedTable` (Direction A's own concurrent top1Share tracking) closely,
and found it likely has the identical sample-count-vs-sum bug fixed here (§4.6, flagged as a
separate, undecided item — not fixed as part of this work). Its
`DoubleAdder`/`DoubleAccumulator`/`AtomicInteger`-with-CAS pattern for the accumulator/max/tier
fields is otherwise directly reusable and was ported as-is; the gate itself uses this class's
own already-corrected, explicit `LongAdder` sample count, not a copy of Direction A's
`_runningTotal.sum()`-based check.

Design: `_runningMax`/`_runningTotal`/`_sampleCount`/`_currentTier` moved from plain
`double[]`/`long[]`/`int[]` (safe only because every adaptive upsert held one single exclusive
lock in the original, `numSubSegments=1`-only version of this class) to
`DoubleAccumulator[]`/`DoubleAdder[]`/`LongAdder[]`/`AtomicInteger[]`, one slot per OUTER shard
— now genuinely necessary, since multiple sub-segments of the same outer shard, each with its
own independent lock, can call `updateSignal` concurrently. `_currentTier`'s CAS retry loop
mirrors `AdaptiveConcurrentIndexedTable`'s exactly (monotonic shrink-only, safe against two
sub-segments racing to advance the same shard's tier at once). The
`adaptiveCapacity && numSubSegments > 1` constructor guard that used to throw is gone —
replaced by making the combination genuinely safe rather than rejecting it.

Verified correctness before measuring performance, same discipline as every prior concurrency
change here: replaced the now-obsolete `testSubSegmentsRejectedForAdaptiveCapacity` (which
tested the guard that no longer exists) with two new tests —
`testAdaptiveCapacityWithSubSegmentsShrinksOnSkewedData` and
`testAdaptiveCapacityWithSubSegmentsNoFalsePositiveOnUniformData`, extending the existing
skew/uniform helpers with a `numSubSegments` parameter rather than duplicating them. 18 total
clean runs (15 ad-hoc + 3 through real `mvn test`) before any performance measurement.

Performance, measured on the skewed benchmark (adding
`shardedOffHeapGroupTableAdaptiveSubSegmented4`): the first comparison had a real outlier (one
iteration in `AdaptiveSubSegmented4`'s Fork 3 spiked to 55,961 us/op against a tight
~50,000-50,745 band everywhere else in that fork) — not trusted, rerun. Clean the second time:

| Benchmark | Score (avgt, us/op) |
|---|---|
| Adaptive (`numSubSegments=1`) | 53,453.597 ± 1,197.892 |
| **Adaptive, sub-segmented (`numSubSegments=4`)** | **50,302.185 ± 305.670** |

Sub-segmenting is **6.3% faster** for adaptive capacity too — the same lever that helped fixed
capacity (6.9-8.1%) transfers, roughly matching in magnitude.

**One honest, secondary finding this surfaced, not asked for but real**: `numSubSegments=1`
adaptive capacity (now atomics-based) measured 53,454-54,097 us/op across two separate runs —
consistently higher than the ~48,763-51,096 us/op range this same configuration measured at
*before* this port, when it used plain fields. This is a cross-run comparison (different JMH
invocations at different points in this investigation, not a same-run A/B test), so it isn't
held to the same standard as the sub-segmenting win above — but the gap (~5-10%) is larger than
the run-to-run drift seen elsewhere in this document for genuinely unmodified code (Direction
A's own baseline has shown ~5% spread across clean runs). Plausible explanation: Direction A's
own "~1.8ns/upsert, negligible" measurement for these same primitive types was made under real
concurrent contention, where `DoubleAdder`/`DoubleAccumulator` are specifically designed to be
cheap *relative to* a naive shared counter under contention — but at `numSubSegments=1`, every
access is already serialized by the single exclusive lock, so there is no contention for the
atomic types to be cheap *relative to*; their overhead here is closer to pure, unamortized cost.
Not confirmed by profiling — noted as a real, secondary cost of this change, not swept under
the sub-segmenting win.

### 6.6 Open questions

- The adaptive-capacity gate (§6.5) is a flat, empirically-chosen sample-count constant
  (`MIN_SAMPLES_BEFORE_ADAPTATION=5000`), verified against both a uniform/realistic-value
  workload and a skew/cardinality sweep together — not yet verified against a workload with a
  much smaller per-shard sample volume than the ~15,625 both existing tests happen to share
  (see §6.5's closing caveat).
- Performance is now JMH-rigor under both a uniform and a skewed workload (§6.4). Resolved:
  fixed capacity is 1.81-2.18x faster than Direction A depending on skew; adaptive capacity is
  1.09-1.95x faster than Direction A and 11.5-67% slower than fixed capacity depending on skew
  (worse on uniform data, where it has no shrink benefit to offset the bookkeeping cost; much
  better under the skew #10498 actually targets). Not yet explained: why Direction A's own
  absolute time roughly doubles under skew while both off-heap variants grow far less (§6.4's
  skewed follow-up) — a real, reproduced observation, not yet a profiled mechanism.
- Adaptive capacity's bookkeeping cost is now resolved, not open: stack profiling ruled out lock
  contention (identical between fixed and adaptive), and a single-threaded isolation benchmark
  measured it directly at ~3.7ns/upsert (~20.5% of a single-threaded upsert's own cost, diluted
  to ~11.5% under real contention because both variants pay nearly the same absolute lock-wait
  time — §6.4's follow-up). Read as a small, tightly-bounded cost with a low ceiling for further
  optimization, not an unsolved problem.
- The write-lock-for-every-upsert design is no longer untested, and is not an open question: a
  read-write-split fast path was built, verified correct (28 clean concurrency runs), and
  measured to REGRESS performance (43,723 -> 48,463 -> 51,087 us/op) — `ReentrantReadWriteLock`'s
  read-side `ThreadLocal` bookkeeping costs more than the write lock saves for a critical section
  this short (§6.2). Reverted. Sub-segmenting each outer shard (still exclusive locks throughout,
  just finer-grained) was tried next and DID help — fixed capacity with `numSubSegments=4` is
  6.9-8.1% faster than `numSubSegments=1` across two clean runs (§6.2) — kept, not reverted. A
  sweep found this does NOT keep improving with larger `numSubSegments`: 8 is back near the
  baseline, 16/32 are measurably worse than never sub-segmenting at all — non-monotonic, not
  diminishing returns. The "smaller per-sub-segment capacity causes more frequent resize"
  hypothesis for that regression was tested directly and REFUTED (giving every sub-segment full
  undivided capacity made K=16/32 dramatically *worse*, not better, since it also scales total
  allocation with `numSubSegments`) — why K=16/32 regress at a properly constant total capacity
  is still genuinely unknown, not just unconfirmed. `numSubSegments=4` remains the recommended
  default regardless. A fully lock-free scheme (no `ReentrantReadWriteLock` at all) remains a
  different, unexplored, higher-risk idea that a design-phase review (not yet an implementation
  attempt) found a real correctness hazard in around concurrent resize — see the session notes
  for the
  reasoning; not pursued given sub-segmenting already delivered a real win at much lower risk.
  Sub-segmenting was subsequently extended to adaptive capacity too (§6.5) — DoubleAdder/
  DoubleAccumulator/AtomicInteger-with-CAS per outer shard, the same primitives
  AdaptiveConcurrentIndexedTable already uses, with the corrected count-based gate rather than a
  copy of that class's likely-buggy one (§4.6). `numSubSegments=4` is a real 6.3% win for
  adaptive too, closely matching fixed capacity's own margin. Real, secondary, NOT yet resolved
  finding from the same work: `numSubSegments=1` adaptive capacity, now atomics-based, measures
  ~5-10% slower across two runs than the same configuration measured before this port when it
  used plain fields — a cross-run comparison, not confirmed by profiling, but larger than this
  document's own baseline for ordinary run-to-run drift.
- Not wired into the query engine (same as both parent directions). A regression test suite now
  exists (`ShardedOffHeapGroupTableTest.java`, `pinot-core/src/test/java/org/apache/pinot/core/
  data/table/`) covering basic correctness, same-key-same-shard routing under real concurrent
  threads, and both adaptive-capacity regimes (no false positive on uniform data, real
  shrinkage + recall@10=100% on skewed data) — runs under plain `mvn test`, no special setup.
- Duplication factor does not apply here (key-hash routing means no key is ever duplicated
  across shards, same as Direction A) — worth stating explicitly since it's easy to
  incorrectly assume Direction B's duplication numbers (§5.5) carry over.

## 7. Open decision

No decision has been made among Direction A (sharding + adaptive capacity), Direction B
(per-thread + off-heap), and Direction C (sharding + off-heap, §6) — all three are now
characterized on real implementations across the same three axes:

| | A: sharding + adaptive capacity | B: per-thread + off-heap | C: sharding + off-heap |
|---|---|---|---|
| Performance vs. baseline | ~6.9x faster than `ConcurrentIndexedTable` (§4.5) | ~19-22% faster + 82-91% less GC than on-heap per-thread (§5.6) | JMH-confirmed under both uniform and skewed workloads: fixed capacity 1.81-3.00x faster (3.00x with sub-segmenting on uniform data, its biggest measured margin, §6.2), adaptive capacity 1.09-1.95x faster (wider margin under the skew #10498 targets) + ~0 GC vs. Direction A itself (§6.4) |
| Memory | Tunable via `top1Share`: 0% to 96-98% reduction depending on skew (§4.5) | Duplication factor 1.1x-6.1x depending on skew, unmitigated (§5.5) | No duplication (key-hash routing); `top1Share` ported (§6.5) — 0% to ~98% reduction depending on skew, stable from 320K to 20M cardinality, matching Direction A; two gate-tuning bugs found and fixed, verified against both a uniform and a skewed workload together |
| Correctness risk | None found (§4.5) | Real: recall@10 drops to 62-92% at mild skew (§5.7), unmitigated | None found — same key-hash-routing guarantee as A, confirmed under real concurrent threads (§6.3) |
| Wired into query engine | No (§4.6) | No (§5.8) | No (§6.6) |
| Regression tests | No (§4.6) | No (§5.8) | Yes — `ShardedOffHeapGroupTableTest.java`, runs under `mvn test` (§6.6) |

**Direction C currently looks like the strongest candidate**: it inherits Direction A's
correctness guarantee exactly (confirmed, not just argued by analogy — §6.3), its performance
comparison is now JMH-confirmed under both a uniform and a skewed workload, and it now has
Direction A's adaptive-capacity idea ported for the memory-ceiling problem (§4.2) with memory
numbers matching Direction A's own (§6.5) — two real gate-tuning bugs were found and fixed
along the way, both verified against a uniform and a skewed workload together, not just
whichever one motivated the fix. It also now has a permanent regression suite, unlike either
parent direction. Adaptive capacity's wall-clock margin over Direction A got *less* flattering
under rigor on uniform data (~9%, not the ~1.5x ad-hoc numbers suggested) but recovered to
~1.95x under the skewed workload #10498 actually targets — the memory-reduction benefit does
buy back most (not quite all) of the bookkeeping cost once it has skew to work with (§6.4).
Direction B's standalone results remain useful as the underlying evidence that off-heap storage
genuinely lowers GC pressure and that duplication does not erase that benefit (§5.5-§5.6) — but
as a complete design, Direction B carries a correctness risk that Direction C does not, for what
was (in this comparison) *better* performance, not worse — which weakens the case for choosing
plain Direction B over Direction C specifically. The write-lock-per-upsert design (§6.2) is also
no longer just an unexamined simplification — it was directly compared against a finer-grained
read-write-split alternative and won, then improved further (not just defended) by
sub-segmenting each shard while keeping that same exclusive-lock model, a real 6.9-8.1% win.
The numbers above reflect a design that was tested against its own obvious alternatives, not
merely the first thing that worked. Next step is Jackie's input on which direction(s) to keep
pursuing — most plausibly Direction C, with
query-engine wiring as the concrete remaining work before it's ready to compare against
Direction A on fully equal footing.

## 8. A note on benchmark rigor

Both the original "no lock bottleneck" false negative (§2) and this investigation's own
early sharding-speedup claim were initially measured without JMH's fork/warmup/multi-iteration
discipline, and both times the unrigorous number was misleading (respectively: hid a real
bottleneck; overstated a speedup that later needed re-verification with a proper 3-fork x
5-iteration methodology before it could be trusted). Any new performance claim in this
investigation — including for whichever direction is eventually pursued — should be held to
that same standard before being treated as a finding.

A third instance: Direction C's adaptive-capacity variant looked ~1.5x faster than Direction A
under ad-hoc timing (§6.4's original table); under JMH rigor that margin collapsed to ~1.09x
(§6.4's follow-up) once the measurement was precise enough to isolate the signal-tracking
bookkeeping cost from the actual upsert cost. The direction of the error was different this
time (optimistic, not just noisy) but the lesson is the same one this section already draws.

A fourth, narrower instance: even JMH's own fork/warmup/multi-iteration discipline doesn't make
a single run automatically trustworthy. The skewed-workload benchmark's first run (§6.4) had one
outlier fork on `shardedOffHeapGroupTableFixed` (48,000-56,000 us/op vs. two other forks' tight
~43,500-45,000 band) that would have made fixed and adaptive capacity look statistically tied —
a materially different, more flattering conclusion for adaptive than the true result. Caught
only by checking individual fork data, not just the aggregated Score/Error line, and resolved by
a full rerun (clean on all 9 forks the second time). The standard from §2/§6.4 extends here too:
JMH rigor lowers the *rate* of misleading numbers, it doesn't eliminate the need to look at the
underlying iterations before trusting an aggregate.

A fifth instance, this time working as intended rather than catching a prior mistake: the
read-lock fast-path attempt (§6.2) was motivated by real profiling evidence and implemented
carefully (28 clean concurrency runs before performance was even measured), but still turned out
to make things worse, not better. The first JMH run alone (48,463 us/op vs. 43,723 baseline)
would have been reason enough to suspect a regression, but an independent second run (51,087 —
getting worse, not converging back toward baseline) is what ruled out "unlucky noisy run" as the
explanation before committing to a revert. Worth naming because it cuts the other way from the
first four instances: those all caught an unrigorous number being *wrong*; this one confirmed a
rigorously-measured number was simply *bad news*, and the discipline was to trust it and revert
rather than look for a reason to keep the change.

A sixth instance, on the sub-segmenting attempt (§6.2) that finally did help: the first run's
*baseline* (`shardedOffHeapGroupTableFixed`, unmodified code, `numSubSegments=1`) had a single
iteration spike to 52,528 us/op inside an otherwise tight Fork 3 (44,959-45,404 everywhere else
in that fork). A less careful read would have compared the new variant against this inflated
baseline and reported an even bigger win than real — overclaiming in the *flattering* direction,
which is easier to miss than a suspicious result, precisely because it doesn't trigger the same
"wait, that's odd" reaction. Caught the same way as every other instance: checking individual
fork/iteration data, not the aggregate line, before trusting a comparison in either direction.

## 9. References

- [apache/pinot#10498](https://github.com/apache/pinot/issues/10498)
- [apache/pinot#11924](https://github.com/apache/pinot/issues/11924)
- [apache/pinot#19368](https://github.com/apache/pinot/pull/19368) (merged — benchmark
  methodology fix found during this investigation)
- `pinot-core/src/main/java/org/apache/pinot/core/data/table/ShardedIndexedTable.java`
  (Direction A prototype, includes `AdaptiveConcurrentIndexedTable`)
- `pinot-core/src/main/java/org/apache/pinot/core/data/table/IndexedTable.java`
  (`shrinkTrimSizeAndThreshold` addition supporting Direction A)
- `pinot-core/src/main/java/org/apache/pinot/core/data/table/OffHeapGroupTable.java`
  (Direction B prototype; also the per-shard storage used by Direction C)
- `pinot-core/src/main/java/org/apache/pinot/core/data/table/ShardedOffHeapGroupTable.java`
  (Direction C prototype)
- `pinot-perf/src/main/java/org/apache/pinot/perf/BenchmarkShardedIndexedTable.java`
  (Direction A JMH benchmark)
- `pinot-perf/src/main/java/org/apache/pinot/perf/BenchmarkShardedOffHeapGroupTable.java`
  (Direction A vs. C JMH benchmark, uniform-key workload, §6.4's follow-up)
- `pinot-perf/src/main/java/org/apache/pinot/perf/BenchmarkShardedOffHeapGroupTableSkewed.java`
  (Direction A vs. C JMH benchmark, skewed-key workload, §6.4's skewed follow-up; also the
  sub-segmenting benchmark, §6.2)
- `pinot-perf/src/main/java/org/apache/pinot/perf/BenchmarkShardedOffHeapGroupTableSingleThread.java`
  (isolates adaptive capacity's per-upsert cost from lock contention, §6.4's profiling follow-up)
- `pinot-core/src/test/java/org/apache/pinot/core/data/table/ShardedOffHeapGroupTableTest.java`
  (Direction C regression suite)
