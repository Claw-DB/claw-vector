// embeddings/cache.rs — LRU cache for embedding vectors keyed by text.
use std::num::NonZeroUsize;

use lru::LruCache;
use sha2::{Digest, Sha256};

/// Snapshot of embedding cache statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmbeddingCacheStats {
    /// Number of cache hits.
    pub hit_count: u64,
    /// Number of cache misses.
    pub miss_count: u64,
    /// Number of cached entries.
    pub len: usize,
}

/// LRU cache for embedding vectors keyed by a SHA-256 hash of the input text.
pub struct EmbeddingCache {
    /// Inner LRU cache mapping text digests to embeddings.
    pub inner: LruCache<String, Vec<f32>>,
    /// Number of cache hits.
    pub hit_count: u64,
    /// Number of cache misses.
    pub miss_count: u64,
}

impl EmbeddingCache {
    /// Create a new cache with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        let capacity = NonZeroUsize::new(capacity.max(1)).unwrap();
        EmbeddingCache {
            inner: LruCache::new(capacity),
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Compute the cache key for an input text.
    pub fn key(text: &str) -> String {
        let digest = Sha256::digest(text.as_bytes());
        hex::encode(digest)
    }

    /// Look up a cached vector for `text`, returning `None` on a cache miss.
    pub fn get(&mut self, text: &str) -> Option<Vec<f32>> {
        let key = Self::key(text);
        if let Some(vector) = self.inner.get(&key) {
            self.hit_count += 1;
            Some(vector.clone())
        } else {
            self.miss_count += 1;
            None
        }
    }

    /// Insert or update a cached vector for `text`.
    pub fn insert(&mut self, text: &str, vector: Vec<f32>) {
        let key = Self::key(text);
        self.inner.put(key, vector);
    }

    /// Return a snapshot of cache hit/miss counts and entry count.
    pub fn stats(&self) -> EmbeddingCacheStats {
        EmbeddingCacheStats {
            hit_count: self.hit_count,
            miss_count: self.miss_count,
            len: self.inner.len(),
        }
    }
}
