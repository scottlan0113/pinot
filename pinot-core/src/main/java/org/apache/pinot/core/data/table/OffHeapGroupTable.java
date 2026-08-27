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
  private MemorySegment _segment;
  private int _dataCapacity;
  private int _size;

  private int[] _indexKeys;
  private int[] _indexSlots;
  private int _indexCapacity;
  private int _indexSize;

  public OffHeapGroupTable(int initialCapacity) {
    _arena = Arena.ofConfined();
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
    int[] order = new int[_size];
    for (int i = 0; i < _size; i++) {
      order[i] = i;
    }
    Double[] boxedForSort = new Double[_size]; // small, short-lived boxing just for the comparator; not the
    // steady-state per-entry storage this design is trying to avoid
    for (int i = 0; i < _size; i++) {
      boxedForSort[i] = _segment.get(ValueLayout.JAVA_DOUBLE, i * RECORD_SIZE + VALUE_OFFSET);
    }
    Integer[] boxedOrder = new Integer[_size];
    for (int i = 0; i < _size; i++) {
      boxedOrder[i] = i;
    }
    Arrays.sort(boxedOrder, (a, b) -> Double.compare(boxedForSort[b], boxedForSort[a]));

    MemorySegment trimmedSegment = _arena.allocate(RECORD_SIZE * topK, 8);
    for (int newSlot = 0; newSlot < topK; newSlot++) {
      int oldSlot = boxedOrder[newSlot];
      MemorySegment.copy(_segment, oldSlot * RECORD_SIZE, trimmedSegment, newSlot * RECORD_SIZE, RECORD_SIZE);
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
    _arena.close();
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
}
