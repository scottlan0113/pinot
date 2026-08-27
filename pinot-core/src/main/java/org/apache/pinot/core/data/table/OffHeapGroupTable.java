package org.apache.pinot.perf;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;


/// Prototype for Jackie's "Direction B" proposal (2026-08-27): a per-thread-scoped GROUP BY table backed
/// by off-heap memory instead of JVM-heap Java objects, to cut per-entry object overhead and GC pressure.
/// Deliberately single-threaded (Arena.ofConfined() enforces this at the JVM level -- any other thread
/// touching this table's memory throws, not just a documented convention) -- this is what makes an
/// off-heap implementation tractable without also needing to solve concurrent off-heap access.
///
/// Scope matches Direction A for apples-to-apples comparison: single INT group-by key, single DOUBLE
/// SUM-like aggregate (mirrors every benchmark used throughout this investigation, GROUP BY d1 / sum(m1)).
///
/// Layout: each off-heap record is 16 bytes -- 4-byte int key (offset 0) + 8-byte double value (offset 8,
/// naturally aligned; bytes 4-7 are padding). An on-heap open-addressing int->int index (key -> off-heap
/// slot index) provides O(1) average lookup; this index itself is small primitive arrays, not boxed
/// objects, so it does not reintroduce the per-entry object overhead this design is trying to avoid.
///
/// Growth: MemorySegment is fixed-size once allocated. Growing reallocates a bigger segment and copies old
/// data in; the abandoned old segment is not individually freed (Arena.ofConfined() frees everything at
/// once on close()) -- this holds some extra native memory DURING a growth event within one table's
/// lifetime, not a permanent leak, since the whole table is short-lived (one query's combine phase).
///
/// NOTE: caller MUST call close() when done with this table, or the native memory leaks for real --
/// unlike JVM heap objects, nothing here is garbage collected.
public class OffHeapGroupTable implements AutoCloseable {
  private static final long RECORD_SIZE = 16;
  private static final long KEY_OFFSET = 0;
  private static final long VALUE_OFFSET = 8;
  private static final int EMPTY_KEY = Integer.MIN_VALUE;
  private static final double MAX_LOAD_FACTOR = 0.7;

  private final Arena _arena;
  private final boolean _ownsArena;
  private MemorySegment _segment;
  private int _dataCapacity;
  private int _size;

  private int[] _indexKeys;
  private int[] _indexSlots;
  private int _indexCapacity;
  private int _indexSize;

  /// Single-thread-owned table: creates its own confined Arena (JVM-enforced single-thread access,
  /// matching the original Direction B proposal) and frees it in close(). Existing behavior, unchanged.
  public OffHeapGroupTable(int initialCapacity) {
    this(initialCapacity, Arena.ofConfined(), true);
  }

  /// Shard-owned table: uses an externally-provided Arena instead of creating its own, so multiple
  /// shards (and the threads that access them, under an external lock -- see ShardedOffHeapGroupTable)
  /// can share one Arena with one lifecycle. Must be Arena.ofShared() (or otherwise safe for the actual
  /// access pattern), NOT ofConfined() -- a confined arena only permits access from the thread that
  /// created it, which would throw for every other thread touching a shared shard. close() on an
  /// instance built this way does NOT close the arena -- the owner (whoever passed it in) is responsible
  /// for closing it exactly once, after all sharers are done.
  public OffHeapGroupTable(int initialCapacity, Arena arena) {
    this(initialCapacity, arena, false);
  }

  private OffHeapGroupTable(int initialCapacity, Arena arena, boolean ownsArena) {
    _arena = arena;
    _ownsArena = ownsArena;
    _dataCapacity = Math.max(16, initialCapacity);
    _segment = _arena.allocate(RECORD_SIZE * _dataCapacity, 8);
    _size = 0;

    _indexCapacity = nextPowerOfTwo(Math.max(16, (int) (_dataCapacity / MAX_LOAD_FACTOR)));
    _indexKeys = new int[_indexCapacity];
    Arrays.fill(_indexKeys, EMPTY_KEY);
    _indexSlots = new int[_indexCapacity];
    _indexSize = 0;
  }

  /// Upsert: adds `value` to the running aggregate for `key` (SUM semantics), inserting a new record if
  /// `key` has not been seen before.
  public void upsert(int key, double value) {
    int probe = indexOf(key);
    if (probe >= 0) {
      int slot = _indexSlots[probe];
      double existing = _segment.get(ValueLayout.JAVA_DOUBLE, slot * RECORD_SIZE + VALUE_OFFSET);
      _segment.set(ValueLayout.JAVA_DOUBLE, slot * RECORD_SIZE + VALUE_OFFSET, existing + value);
      return;
    }

    if (_size >= _dataCapacity) {
      growData();
    }
    int slot = _size++;
    _segment.set(ValueLayout.JAVA_INT, slot * RECORD_SIZE + KEY_OFFSET, key);
    _segment.set(ValueLayout.JAVA_DOUBLE, slot * RECORD_SIZE + VALUE_OFFSET, value);

    if (_indexSize >= _indexCapacity * MAX_LOAD_FACTOR) {
      growIndex();
    }
    insertIntoIndex(key, slot);
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
      values[i] = _segment.get(ValueLayout.JAVA_DOUBLE, (long) i * RECORD_SIZE + VALUE_OFFSET);
      indices[i] = i;
    }
    quickSortDescending(values, indices, 0, _size - 1);

    MemorySegment trimmedSegment = _arena.allocate(RECORD_SIZE * topK, 8);
    for (int newSlot = 0; newSlot < topK; newSlot++) {
      int oldSlot = indices[newSlot];
      MemorySegment.copy(_segment, (long) oldSlot * RECORD_SIZE, trimmedSegment, (long) newSlot * RECORD_SIZE,
          RECORD_SIZE);
    }
    _segment = trimmedSegment;
    _dataCapacity = topK;
    _size = topK;

    // Rebuild the index to match the new (compacted) slot numbering.
    Arrays.fill(_indexKeys, EMPTY_KEY);
    _indexSize = 0;
    for (int slot = 0; slot < _size; slot++) {
      int key = _segment.get(ValueLayout.JAVA_INT, slot * RECORD_SIZE + KEY_OFFSET);
      insertIntoIndex(key, slot);
    }
  }

  public int size() {
    return _size;
  }

  public int keyAt(int slot) {
    return _segment.get(ValueLayout.JAVA_INT, slot * RECORD_SIZE + KEY_OFFSET);
  }

  public double valueAt(int slot) {
    return _segment.get(ValueLayout.JAVA_DOUBLE, slot * RECORD_SIZE + VALUE_OFFSET);
  }

  @Override
  public void close() {
    if (_ownsArena) {
      _arena.close();
    }
    // else: this table doesn't own the arena (constructed via the (capacity, arena) constructor) -- the
    // owner is responsible for closing it once, after every table sharing it is done.
  }

  // ---------- on-heap open-addressing index (key -> off-heap slot), primitive int[] only ----------

  private int indexOf(int key) {
    int mask = _indexCapacity - 1;
    int i = hash(key) & mask;
    while (_indexKeys[i] != EMPTY_KEY) {
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
    while (_indexKeys[i] != EMPTY_KEY) {
      i = (i + 1) & mask;
    }
    _indexKeys[i] = key;
    _indexSlots[i] = slot;
    _indexSize++;
  }

  private void growIndex() {
    int[] oldKeys = _indexKeys;
    int[] oldSlots = _indexSlots;
    _indexCapacity *= 2;
    _indexKeys = new int[_indexCapacity];
    Arrays.fill(_indexKeys, EMPTY_KEY);
    _indexSlots = new int[_indexCapacity];
    _indexSize = 0;
    for (int i = 0; i < oldKeys.length; i++) {
      if (oldKeys[i] != EMPTY_KEY) {
        insertIntoIndex(oldKeys[i], oldSlots[i]);
      }
    }
  }

  private void growData() {
    int newCapacity = _dataCapacity * 2;
    MemorySegment newSegment = _arena.allocate(RECORD_SIZE * newCapacity, 8);
    MemorySegment.copy(_segment, 0, newSegment, 0, RECORD_SIZE * _size);
    _segment = newSegment; // old segment is abandoned, freed only when the whole Arena closes -- see class javadoc
    _dataCapacity = newCapacity;
  }

  private static int hash(int key) {
    int h = key * 0x9E3779B1;
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
