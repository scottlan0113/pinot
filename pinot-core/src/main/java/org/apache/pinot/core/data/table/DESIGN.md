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

Status: **investigation / candidate designs — not finalized.** Two candidate directions are
described below (§4, §5); no decision has been made between them, and neither is proposed
for merge as-is. This document exists to record findings before they are lost, per Jackie's
request, ahead of converging on one design.

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
  benchmark (see §7, and the fix in #19368: a shared `java.util.Random` field created false
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

## 5. Direction B: per-thread tables + off-heap (proposed, unexplored)

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

### 5.4 Open questions

- Has not been prototyped. No implementation, no benchmark, no correctness measurement yet.
- What off-heap mechanism to use (`sun.misc.Unsafe` and its long-signaled removal, vs. the
  modern `java.lang.foreign.MemorySegment`/`Arena` API) is an open choice with real
  tradeoffs (JDK version support, API maturity, safety).
- Whether the duplication-factor cost becomes acceptable once off-heap, or whether it still
  needs to be bounded (e.g. combined with some form of the sharding idea to eliminate
  duplication rather than merely making it cheaper), is unmeasured.
- Whether the correctness risk from local pre-merge trimming is acceptable in practice (the
  earlier "simple" investigation found it negligible at realistic skew >= 0.5, but real and
  measurable — up to significant recall loss — in a mild-skew band) has not been revisited
  in combination with this proposal.

## 6. Open decision

No decision has been made between Direction A (sharding + adaptive capacity, fully
implemented and verified against real classes, not yet wired into the query engine or
regression-tested) and Direction B (per-thread + off-heap, proposed, entirely unexplored).
The two are not mutually exclusive in principle (§5.3). Next step is Jackie's input on which
to pursue, or whether to prototype Direction B far enough to compare it against Direction A's
already-measured numbers before deciding.

## 7. A note on benchmark rigor

Both the original "no lock bottleneck" false negative (§2) and this investigation's own
early sharding-speedup claim were initially measured without JMH's fork/warmup/multi-iteration
discipline, and both times the unrigorous number was misleading (respectively: hid a real
bottleneck; overstated a speedup that later needed re-verification with a proper 3-fork x
5-iteration methodology before it could be trusted). Any new performance claim in this
investigation — including for whichever direction is eventually pursued — should be held to
that same standard before being treated as a finding.

## 8. References

- [apache/pinot#10498](https://github.com/apache/pinot/issues/10498)
- [apache/pinot#11924](https://github.com/apache/pinot/issues/11924)
- [apache/pinot#19368](https://github.com/apache/pinot/pull/19368) (merged — benchmark
  methodology fix found during this investigation)
- `pinot-core/src/main/java/org/apache/pinot/core/data/table/ShardedIndexedTable.java`
  (Direction A prototype, includes `AdaptiveConcurrentIndexedTable`)
- `pinot-core/src/main/java/org/apache/pinot/core/data/table/IndexedTable.java`
  (`shrinkTrimSizeAndThreshold` addition supporting Direction A)
- `pinot-perf/src/main/java/org/apache/pinot/perf/BenchmarkShardedIndexedTable.java`
