package org.apache.pinot.perf;

import java.lang.foreign.Arena;
import java.util.concurrent.locks.ReentrantReadWriteLock;


/// Combines Direction A's correctness architecture (shard by key hash, so a key always routes to the
/// same shard regardless of which thread produced it -- no "local trim before merge" risk) with
/// Direction B's off-heap storage per shard (GC-pressure benefit, measured in OffHeapGroupTable's own
/// verification). Proposed after the user asked whether the two directions could be combined instead of
/// choosing one; this is a first prototype, not yet compared rigorously against either direction alone.
///
/// Concurrency: OffHeapGroupTable has NO internal thread-safety (unlike ConcurrentIndexedTable's
/// ConcurrentHashMap, which safely handles concurrent upserts to different keys on its own). Every
/// upsert to a shard therefore needs full exclusive locking against every other access to that same
/// shard -- there is no fine-grained "read lock for the common case" available the way
/// ConcurrentIndexedTable gets from ConcurrentHashMap. This version uses one ReentrantReadWriteLock per
/// shard and takes the WRITE lock for every upsert, not just for trim/resize. This is a real, deliberate
/// simplification for a first correctness-focused prototype -- it means more serialization per shard
/// than Direction A's ConcurrentIndexedTable shards have, and needs to be measured, not assumed, before
/// claiming a performance verdict against either direction alone.
///
/// Arena lifecycle: one Arena.ofShared() (NOT ofConfined() -- confined arenas restrict access to their
/// creating thread, which would throw for every other thread touching a shared shard) backs every
/// shard's memory. close() closes it once, freeing every shard's memory together.
public class ShardedOffHeapGroupTable implements AutoCloseable {
  private final int _numShards;
  private final Arena _arena;
  private final OffHeapGroupTable[] _shards;
  private final ReentrantReadWriteLock[] _locks;

  public ShardedOffHeapGroupTable(int numShards, int perShardInitialCapacity) {
    _numShards = numShards;
    _arena = Arena.ofShared();
    _shards = new OffHeapGroupTable[numShards];
    _locks = new ReentrantReadWriteLock[numShards];
    for (int i = 0; i < numShards; i++) {
      _shards[i] = new OffHeapGroupTable(perShardInitialCapacity, _arena);
      _locks[i] = new ReentrantReadWriteLock();
    }
  }

  private int shardFor(int key) {
    return Math.floorMod(Integer.hashCode(key), _numShards);
  }

  /// Thread-safe: takes the target shard's write lock for the full duration of the upsert. Safe to call
  /// concurrently from multiple threads on the same table (that's the whole point) -- different keys
  /// hashing to different shards proceed fully in parallel; keys hashing to the same shard serialize.
  public void upsert(int key, double value) {
    int shard = shardFor(key);
    ReentrantReadWriteLock lock = _locks[shard];
    lock.writeLock().lock();
    try {
      _shards[shard].upsert(key, value);
    } finally {
      lock.writeLock().unlock();
    }
  }

  /// Trims every shard down to perShardTopK, matching finish() semantics (called once, after all
  /// upserts are done, not continuously). Sequential across shards for this first prototype -- could be
  /// parallelized (matching ShardedIndexedTable.finishShardsInParallel) but that is a performance
  /// refinement, not needed to validate correctness first.
  public void finishAllShards(int perShardTopK) {
    for (int i = 0; i < _numShards; i++) {
      ReentrantReadWriteLock lock = _locks[i];
      lock.writeLock().lock();
      try {
        _shards[i].trimTo(perShardTopK);
      } finally {
        lock.writeLock().unlock();
      }
    }
  }

  public int totalSize() {
    int total = 0;
    for (OffHeapGroupTable shard : _shards) {
      total += shard.size();
    }
    return total;
  }

  /// Visits every surviving (key, value) pair across all shards, post-finishAllShards(). No lock taken --
  /// caller must ensure no concurrent upserts are in flight (matches finish()-then-read usage pattern).
  public void forEachEntry(EntryVisitor visitor) {
    for (OffHeapGroupTable shard : _shards) {
      int size = shard.size();
      for (int i = 0; i < size; i++) {
        visitor.visit(shard.keyAt(i), shard.valueAt(i));
      }
    }
  }

  public interface EntryVisitor {
    void visit(int key, double value);
  }

  @Override
  public void close() {
    _arena.close();
  }
}
