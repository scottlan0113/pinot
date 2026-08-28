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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;


/// Prototype for Jackie's "Direction B" proposal (2026-08-27): a per-thread-scoped GROUP BY table backed
/// by off-heap memory instead of JVM-heap Java objects, to cut per-entry object overhead and GC pressure.
/// Deliberately single-threaded (Arena.ofConfined() enforces this at the JVM level -- any other thread
/// touching this table's memory throws, not just a documented convention) -- this is what makes an
/// off-heap implementation tractable without also needing to solve concurrent off-heap access.
///
/// Scope matches Direction A for apples-to-apples comparison: single INT group-by key, single DOUBLE
/// aggregate. SUM is still the only aggregation every benchmark in this investigation actually exercises
/// (GROUP BY d1 / sum(m1)) and remains the default everywhere -- MIN/MAX (see AggregationType below) were
/// added to test DESIGN.md's "single numeric SUM-like (additive) aggregate" scope limitation (Sec 4.6),
/// correctness-verified but not JMH-measured.
///
/// Layout: each off-heap record is `recordSize` bytes -- `4 * numKeyColumns` bytes of consecutive int key
/// columns (offset 0), then the 8-byte double value at the next 8-byte-aligned offset (padding in between
/// if `4 * numKeyColumns` is not already a multiple of 8). At numKeyColumns=1 this is exactly 16 bytes --
/// 4-byte int key (offset 0) + 8-byte double value (offset 8) -- the original, unchanged layout. An
/// on-heap open-addressing index (key -> off-heap slot index) provides O(1) average lookup; this index
/// itself is small primitive arrays, not boxed objects, so it does not reintroduce the per-entry object
/// overhead this design is trying to avoid.
///
/// Growth: MemorySegment is fixed-size once allocated. Growing reallocates a bigger segment and copies old
/// data in; the abandoned old segment is not individually freed (Arena.ofConfined() frees everything at
/// once on close()) -- this holds some extra native memory DURING a growth event within one table's
/// lifetime, not a permanent leak, since the whole table is short-lived (one query's combine phase).
///
/// Multi-column keys (added while testing DESIGN.md's "multi-column GROUP BY... not yet tested" scope
/// item): numKeyColumns > 1 packs that many consecutive int columns into the key region instead of one.
/// This is a deliberately SEPARATE code path from the original single-column one, not a generalization of
/// it in place -- upsert(int, double)/keyAt(int) are UNCHANGED (same direct-int-comparison index) so every
/// performance number measured against them all session stays valid; upsert(int[], double)/keysAt(int)
/// use a different index instead, keyed by a hash of the composite key rather than the key itself (an
/// arbitrary number of int columns cannot fit in one on-heap int the way a single column can) -- an
/// on-heap hash match is then verified against the actual off-heap key bytes before being trusted, to
/// handle hash collisions correctly. A table is exclusively single-column or multi-column for its whole
/// lifetime, fixed by numKeyColumns at construction; calling the wrong overload for a table's mode throws
/// rather than silently corrupting the index (the two modes store incompatible things -- a raw key vs. a
/// hash -- in the same physical on-heap arrays, so mixing them on one instance is not just unsupported,
/// it is actively unsafe).
///
/// NOTE: caller MUST call close() when done with this table, or the native memory leaks for real --
/// unlike JVM heap objects, nothing here is garbage collected.
public class OffHeapGroupTable implements AutoCloseable {
  /// SUM is this class's original, still-default, only-ever-benchmarked aggregation -- MIN/MAX (added
  /// while testing DESIGN.md's "single numeric SUM-like (additive) aggregate" scope limitation, Sec 4.6)
  /// are real but comparatively lightly-exercised additions, correctness-tested, not JMH-measured. Note
  /// for whoever adds a MIN/MAX benchmark later: mixing aggregation types across table instances
  /// constructed in the same JVM/fork as the existing SUM-only benchmarks could in principle affect the
  /// JIT's inline cache for upsert()'s merge call and disturb previously-measured SUM numbers -- keep
  /// non-SUM benchmarking in its own fork if that ever matters.
  public enum AggregationType {
    SUM(Double::sum),
    MIN(Math::min),
    MAX(Math::max);

    private final DoubleBinaryOperator _merge;

    AggregationType(DoubleBinaryOperator merge) {
      _merge = merge;
    }

    double merge(double existing, double incoming) {
      return _merge.applyAsDouble(existing, incoming);
    }
  }

  private static final long KEY_OFFSET = 0;
  private static final int EMPTY_SLOT = -1; // occupancy sentinel for _indexSlots, both modes (see below)
  private static final double MAX_LOAD_FACTOR = 0.7;

  private final Arena _arena;
  private final boolean _ownsArena;
  private final int _numKeyColumns;
  private final AggregationType _aggregationType;
  private final long _valueOffset;
  private final long _recordSize;
  private MemorySegment _segment;
  private int _dataCapacity;
  private int _size;

  // Occupancy for BOTH modes is tracked by _indexSlots[i] == EMPTY_SLOT, never by a reserved value in
  // _indexKeys -- _indexKeys[i] is meaningful only once _indexSlots[i] says the slot is occupied.
  // Single-column mode: _indexKeys[i] holds the raw key. Multi-column mode: _indexKeys[i] holds
  // hashKeys(...) of the composite key. (An earlier version of the single-column mode used
  // Integer.MIN_VALUE as a reserved "empty" key value stored directly in _indexKeys -- a real key of
  // exactly that value was then indistinguishable from an empty slot, silently breaking lookup/merge/
  // rehashing for it. Unifying both modes onto _indexSlots' own sentinel, which was already correct and
  // already always maintained, removes that failure mode entirely rather than special-casing around it.)
  private int[] _indexKeys;
  private int[] _indexSlots;
  private int _indexCapacity;
  private int _indexSize;

  /// Single-thread-owned, single-column table: creates its own confined Arena (JVM-enforced single-thread
  /// access, matching the original Direction B proposal) and frees it in close(). Existing behavior, unchanged.
  public OffHeapGroupTable(int initialCapacity) {
    this(initialCapacity, 1, Arena.ofConfined(), true, AggregationType.SUM);
  }

  /// Shard-owned, single-column table: uses an externally-provided Arena instead of creating its own, so
  /// multiple shards (and the threads that access them, under an external lock -- see
  /// ShardedOffHeapGroupTable) can share one Arena with one lifecycle. Must be Arena.ofShared() (or
  /// otherwise safe for the actual access pattern), NOT ofConfined() -- a confined arena only permits
  /// access from the thread that created it, which would throw for every other thread touching a shared
  /// shard. close() on an instance built this way does NOT close the arena -- the owner (whoever passed
  /// it in) is responsible for closing it exactly once, after all sharers are done.
  public OffHeapGroupTable(int initialCapacity, Arena arena) {
    this(initialCapacity, 1, arena, false, AggregationType.SUM);
  }

  /// Shard-owned, multi-column table -- same Arena-sharing contract as the two-arg constructor above,
  /// with numKeyColumns > 1 consecutive int key columns instead of one. See class Javadoc for why this
  /// is a strictly separate mode rather than a drop-in generalization of the single-column one.
  public OffHeapGroupTable(int initialCapacity, int numKeyColumns, Arena arena) {
    this(initialCapacity, numKeyColumns, arena, false, AggregationType.SUM);
  }

  /// Shard-owned table with an explicit AggregationType (default SUM everywhere else) -- MIN/MAX added
  /// while testing DESIGN.md's "single numeric SUM-like (additive) aggregate" scope limitation (Sec 4.6).
  /// Applies to both single- and multi-column tables; the aggregation type is orthogonal to how many key
  /// columns there are.
  public OffHeapGroupTable(int initialCapacity, int numKeyColumns, Arena arena, AggregationType aggregationType) {
    this(initialCapacity, numKeyColumns, arena, false, aggregationType);
  }

  private OffHeapGroupTable(int initialCapacity, int numKeyColumns, Arena arena, boolean ownsArena,
      AggregationType aggregationType) {
    if (numKeyColumns < 1) {
      throw new IllegalArgumentException("numKeyColumns must be >= 1, got " + numKeyColumns);
    }
    _arena = arena;
    _ownsArena = ownsArena;
    _numKeyColumns = numKeyColumns;
    _aggregationType = aggregationType;
    long keyBytes = 4L * numKeyColumns;
    _valueOffset = ((keyBytes + 7) / 8) * 8; // round up to 8-byte alignment for the double
    _recordSize = _valueOffset + 8;
    _dataCapacity = Math.max(16, initialCapacity);
    _segment = _arena.allocate(_recordSize * _dataCapacity, 8);
    _size = 0;

    _indexCapacity = nextPowerOfTwo(Math.max(16, (int) (_dataCapacity / MAX_LOAD_FACTOR)));
    _indexKeys = new int[_indexCapacity];
    _indexSlots = new int[_indexCapacity];
    Arrays.fill(_indexSlots, EMPTY_SLOT); // _indexKeys needs no fill -- unread until _indexSlots says occupied
    _indexSize = 0;
  }

  /// Single-column upsert: combines `value` into the running aggregate for `key` using this table's
  /// AggregationType (SUM by default), inserting a new record if `key` has not been seen before. Returns
  /// the key's new aggregated value post-upsert -- callers that want to track a concentration signal
  /// (e.g. adaptive capacity, see ShardedOffHeapGroupTable -- SUM-only, see its own Javadoc) need this
  /// and would otherwise have to do a second, redundant lookup for it. Only valid on a table constructed
  /// with numKeyColumns == 1 -- throws otherwise, since calling this on a multi-column table would
  /// silently ignore every key column past the first.
  public double upsert(int key, double value) {
    requireSingleColumn();
    int probe = indexOf(key);
    if (probe >= 0) {
      int slot = _indexSlots[probe];
      double existing = _segment.get(ValueLayout.JAVA_DOUBLE, slot * _recordSize + _valueOffset);
      double updated = _aggregationType.merge(existing, value);
      _segment.set(ValueLayout.JAVA_DOUBLE, slot * _recordSize + _valueOffset, updated);
      return updated;
    }

    if (_size >= _dataCapacity) {
      growData();
    }
    int slot = _size++;
    _segment.set(ValueLayout.JAVA_INT, slot * _recordSize + KEY_OFFSET, key);
    _segment.set(ValueLayout.JAVA_DOUBLE, slot * _recordSize + _valueOffset, value);

    if (_indexSize >= _indexCapacity * MAX_LOAD_FACTOR) {
      growIndex();
    }
    insertIntoIndex(key, slot);
    return value;
  }

  /// Multi-column upsert: same aggregation semantics and new-composite-key-insertion behavior as the
  /// single-column upsert above, but keyed by `keys.length` consecutive int columns instead of one. Only
  /// valid on a table constructed with numKeyColumns > 1 and matching keys.length -- see class Javadoc for
  /// why single- and multi-column usage cannot be mixed on the same table instance.
  public double upsert(int[] keys, double value) {
    requireMultiColumn(keys);
    int probe = indexOfMulti(keys);
    if (probe >= 0) {
      int slot = _indexSlots[probe];
      double existing = _segment.get(ValueLayout.JAVA_DOUBLE, slot * _recordSize + _valueOffset);
      double updated = _aggregationType.merge(existing, value);
      _segment.set(ValueLayout.JAVA_DOUBLE, slot * _recordSize + _valueOffset, updated);
      return updated;
    }

    if (_size >= _dataCapacity) {
      growData();
    }
    int slot = _size++;
    for (int c = 0; c < _numKeyColumns; c++) {
      _segment.set(ValueLayout.JAVA_INT, slot * _recordSize + KEY_OFFSET + 4L * c, keys[c]);
    }
    _segment.set(ValueLayout.JAVA_DOUBLE, slot * _recordSize + _valueOffset, value);

    if (_indexSize >= _indexCapacity * MAX_LOAD_FACTOR) {
      growIndex();
    }
    insertHashIntoIndex(hashKeys(keys), slot);
    return value;
  }

  /// Trims down to the top `topK` records by value (descending), discarding the rest. Mirrors
  /// TableResizer's role for the on-heap tables -- a full sort here since per-thread tables are expected
  /// to be small enough (already trimmed segment-level results) that this is not the hot path.
  public void trimTo(int topK) {
    if (_size <= topK) {
      return;
    }
    // Sort descending by value with zero boxing: java.util.Arrays.sort only accepts a Comparator for
    // Object[], not double[]/int[], so a naive fix would box into Double[]/Integer[] -- which just moves
    // the per-entry allocation cost this design is trying to avoid from upsert() into trimTo(). Instead,
    // extract into two parallel primitive arrays and sort both together with a hand-rolled quicksort.
    double[] values = new double[_size];
    int[] indices = new int[_size];
    for (int i = 0; i < _size; i++) {
      values[i] = _segment.get(ValueLayout.JAVA_DOUBLE, (long) i * _recordSize + _valueOffset);
      indices[i] = i;
    }
    quickSortDescending(values, indices, 0, _size - 1);

    MemorySegment trimmedSegment = _arena.allocate(_recordSize * topK, 8);
    for (int newSlot = 0; newSlot < topK; newSlot++) {
      int oldSlot = indices[newSlot];
      MemorySegment.copy(_segment, (long) oldSlot * _recordSize, trimmedSegment, (long) newSlot * _recordSize,
          _recordSize);
    }
    _segment = trimmedSegment;
    _dataCapacity = topK;
    _size = topK;

    // Rebuild the index to match the new (compacted) slot numbering.
    _indexSize = 0;
    Arrays.fill(_indexSlots, EMPTY_SLOT);
    if (_numKeyColumns == 1) {
      for (int slot = 0; slot < _size; slot++) {
        int key = _segment.get(ValueLayout.JAVA_INT, slot * _recordSize + KEY_OFFSET);
        insertIntoIndex(key, slot);
      }
    } else {
      for (int slot = 0; slot < _size; slot++) {
        insertHashIntoIndex(hashKeysAt(slot), slot);
      }
    }
  }

  public int size() {
    return _size;
  }

  public int numKeyColumns() {
    return _numKeyColumns;
  }

  /// Single-column accessor -- only valid on a numKeyColumns == 1 table, see upsert(int, double).
  public int keyAt(int slot) {
    requireSingleColumn();
    return _segment.get(ValueLayout.JAVA_INT, slot * _recordSize + KEY_OFFSET);
  }

  /// Multi-column accessor -- returns a freshly-allocated array of this table's numKeyColumns values.
  /// Not on the upsert hot path (called from finishAllShards()'s sub-segment merge and trimTo()'s own
  /// index rebuild, both already-necessarily-allocating/single-threaded contexts), so the per-call
  /// allocation is acceptable here in a way it deliberately is not in upsert() itself. Works for any
  /// numKeyColumns, including 1 -- unlike upsert(), there is no index-corruption hazard in a pure read.
  public int[] keysAt(int slot) {
    int[] keys = new int[_numKeyColumns];
    for (int c = 0; c < _numKeyColumns; c++) {
      keys[c] = _segment.get(ValueLayout.JAVA_INT, slot * _recordSize + KEY_OFFSET + 4L * c);
    }
    return keys;
  }

  public double valueAt(int slot) {
    return _segment.get(ValueLayout.JAVA_DOUBLE, slot * _recordSize + _valueOffset);
  }

  @Override
  public void close() {
    if (_ownsArena) {
      _arena.close();
    }
    // else: this table doesn't own the arena (constructed via the (capacity, arena) constructor) -- the
    // owner is responsible for closing it once, after every table sharing it is done.
  }

  // ---------- single-column on-heap index (raw key -> off-heap slot). Occupancy is checked via
  // _indexSlots[i] != EMPTY_SLOT (see the field comment above _indexKeys/_indexSlots for why), not via a
  // reserved value in _indexKeys -- every other aspect (hash function, probing sequence, what's stored)
  // is unchanged from the original single-purpose version of this class, so every performance number
  // measured against it all session was re-verified to still hold after this fix (DESIGN.md Sec 6.7). ----

  private int indexOf(int key) {
    int mask = _indexCapacity - 1;
    int i = hash(key) & mask;
    while (_indexSlots[i] != EMPTY_SLOT) {
      if (_indexKeys[i] == key) {
        return i;
      }
      i = (i + 1) & mask;
    }
    return -1;
  }

  private void insertIntoIndex(int key, int slot) {
    int mask = _indexCapacity - 1;
    int i = hash(key) & mask;
    while (_indexSlots[i] != EMPTY_SLOT) {
      i = (i + 1) & mask;
    }
    _indexKeys[i] = key;
    _indexSlots[i] = slot;
    _indexSize++;
  }

  // ---------- multi-column on-heap index (hash of composite key -> off-heap slot, verified against the
  // actual off-heap key bytes on hash match to handle collisions correctly) ----------

  private int indexOfMulti(int[] keys) {
    int mask = _indexCapacity - 1;
    int keyHash = hashKeys(keys);
    int i = keyHash & mask;
    while (_indexSlots[i] != EMPTY_SLOT) {
      if (_indexKeys[i] == keyHash && keysEqualAt(_indexSlots[i], keys)) {
        return i;
      }
      i = (i + 1) & mask;
    }
    return -1;
  }

  private void insertHashIntoIndex(int keyHash, int slot) {
    int mask = _indexCapacity - 1;
    int i = keyHash & mask;
    while (_indexSlots[i] != EMPTY_SLOT) {
      i = (i + 1) & mask;
    }
    _indexKeys[i] = keyHash;
    _indexSlots[i] = slot;
    _indexSize++;
  }

  private boolean keysEqualAt(int slot, int[] keys) {
    long base = slot * _recordSize + KEY_OFFSET;
    for (int c = 0; c < _numKeyColumns; c++) {
      if (_segment.get(ValueLayout.JAVA_INT, base + 4L * c) != keys[c]) {
        return false;
      }
    }
    return true;
  }

  private int hashKeys(int[] keys) {
    int h = 1;
    for (int k : keys) {
      h = h * 31 + k;
    }
    return mix(h);
  }

  private int hashKeysAt(int slot) {
    long base = slot * _recordSize + KEY_OFFSET;
    int h = 1;
    for (int c = 0; c < _numKeyColumns; c++) {
      h = h * 31 + _segment.get(ValueLayout.JAVA_INT, base + 4L * c);
    }
    return mix(h);
  }

  private void requireSingleColumn() {
    if (_numKeyColumns != 1) {
      throw new IllegalStateException(
          "upsert(int, double) / keyAt(int) require a table constructed with numKeyColumns == 1, this table has "
              + _numKeyColumns);
    }
  }

  private void requireMultiColumn(int[] keys) {
    if (_numKeyColumns == 1) {
      throw new IllegalStateException("this table has numKeyColumns == 1 -- use upsert(int, double) instead");
    }
    if (keys.length != _numKeyColumns) {
      throw new IllegalArgumentException("expected " + _numKeyColumns + " key columns, got " + keys.length);
    }
  }

  // ---------- shared growth / hashing helpers ----------

  private void growIndex() {
    int[] oldKeys = _indexKeys;
    int[] oldSlots = _indexSlots;
    _indexCapacity *= 2;
    _indexKeys = new int[_indexCapacity];
    _indexSlots = new int[_indexCapacity];
    Arrays.fill(_indexSlots, EMPTY_SLOT); // _indexKeys needs no fill -- unread until _indexSlots says occupied
    _indexSize = 0;
    // Occupancy check (oldSlots[i] != EMPTY_SLOT) is identical for both modes now -- only which insert
    // function rebuilds the entry differs: single-column re-derives nothing (oldKeys[i] IS the raw key,
    // re-probed by insertIntoIndex against the NEW capacity/mask), multi-column's oldKeys[i] already
    // holds the composite key's hash (insertHashIntoIndex takes a hash directly, not the keys themselves).
    for (int i = 0; i < oldSlots.length; i++) {
      if (oldSlots[i] != EMPTY_SLOT) {
        if (_numKeyColumns == 1) {
          insertIntoIndex(oldKeys[i], oldSlots[i]);
        } else {
          insertHashIntoIndex(oldKeys[i], oldSlots[i]);
        }
      }
    }
  }

  private void growData() {
    int newCapacity = _dataCapacity * 2;
    MemorySegment newSegment = _arena.allocate(_recordSize * newCapacity, 8);
    MemorySegment.copy(_segment, 0, newSegment, 0, _recordSize * _size);
    _segment = newSegment; // old segment is abandoned, freed only when the whole Arena closes -- see class javadoc
    _dataCapacity = newCapacity;
  }

  private static int hash(int key) {
    return mix(key);
  }

  private static int mix(int h) {
    h = h * 0x9E3779B1;
    return h ^ (h >>> 16);
  }

  private static int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) {
      p <<= 1;
    }
    return p;
  }

  // ---------- zero-boxing dual-array quicksort (descending by `values`, `indices` carried along) ----------

  private static final int INSERTION_SORT_CUTOFF = 16;

  /// 3-way (Dutch national flag) partitioning quicksort. A naive 2-way partition degrades to O(n^2) when
  /// many elements are equal -- exactly the shape of this data (upsert(key, 1.0)-accumulated integer-ish
  /// counts collide constantly, e.g. many keys landing on count=2, count=3, ...). A first version of this
  /// method used 2-way partitioning and was measured to be catastrophically slower than the boxed
  /// Arrays.sort() it replaced (2117ms vs 286ms in one comparison) specifically because of this. 3-way
  /// partitioning groups all values equal to the pivot into a single pass with no further recursion needed
  /// on them, which is the standard fix for duplicate-heavy inputs.
  private static void quickSortDescending(double[] values, int[] indices, int lo, int hi) {
    while (lo < hi) {
      if (hi - lo < INSERTION_SORT_CUTOFF) {
        insertionSortDescending(values, indices, lo, hi);
        return;
      }
      int mid = lo + (hi - lo) / 2;
      if (values[mid] > values[lo]) {
        swap(values, indices, lo, mid);
      }
      if (values[hi] > values[lo]) {
        swap(values, indices, lo, hi);
      }
      if (values[mid] > values[hi]) {
        swap(values, indices, mid, hi);
      }
      double pivot = values[hi]; // median-of-three now sits at hi

      int lt = lo; // [lo, lt-1]: values > pivot
      int i = lo;  // [lt, i-1]: values == pivot (scanned so far)
      int gt = hi; // [gt+1, hi]: values < pivot
      while (i <= gt) {
        if (values[i] > pivot) { // descending: larger values partition to the left
          swap(values, indices, lt++, i++);
        } else if (values[i] < pivot) {
          swap(values, indices, i, gt--);
        } else {
          i++;
        }
      }
      // [lt, gt] all equal pivot -- correctly placed already, no need to recurse into it.

      if (lt - lo < hi - gt) {
        quickSortDescending(values, indices, lo, lt - 1);
        lo = gt + 1;
      } else {
        quickSortDescending(values, indices, gt + 1, hi);
        hi = lt - 1;
      }
    }
  }

  private static void insertionSortDescending(double[] values, int[] indices, int lo, int hi) {
    for (int i = lo + 1; i <= hi; i++) {
      double v = values[i];
      int idx = indices[i];
      int j = i - 1;
      while (j >= lo && values[j] < v) {
        values[j + 1] = values[j];
        indices[j + 1] = indices[j];
        j--;
      }
      values[j + 1] = v;
      indices[j + 1] = idx;
    }
  }

  private static void swap(double[] values, int[] indices, int i, int j) {
    double tv = values[i];
    values[i] = values[j];
    values[j] = tv;
    int ti = indices[i];
    indices[i] = indices[j];
    indices[j] = ti;
  }
}
