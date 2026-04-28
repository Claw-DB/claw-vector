// embeddings/cache.rs — LRU cache for embedding vectors keyed by text.
use std::collections::HashMap;

/// A simple LRU cache that stores embedding vectors keyed by the original text.
pub struct EmbeddingCache {
    map: HashMap<String, Vec<f32>>,
    order: std::collections::VecDeque<String>,
    capacity: usize,
}

impl EmbeddingCache {
    /// Create a new cache with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        EmbeddingCache {
            map: HashMap::with_capacity(capacity),
            order: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Look up a cached vector for `text`, returning `None` on a cache miss.
    pub fn get(&mut self, text: &str) -> Option<Vec<f32>> {
        if let Some(v) = self.map.get(text) {
            // Move to the back of the LRU order.
            self.order.retain(|k| k != text);
            self.order.push_back(text.to_string());
            Some(v.clone())
        } else {
            None
        }
    }

    /// Insert or update a cached vector for `text`, evicting the LRU entry if necessary.
    pub fn insert(&mut self, text: String, vector: Vec<f32>) {
        if self.map.contains_key(&text) {
            self.order.retain(|k| k != &text);
        } else if self.map.len() >= self.capacity {
            if let Some(evict) = self.order.pop_front() {
                self.map.remove(&evict);
            }
        }
        self.order.push_back(text.clone());
        self.map.insert(text, vector);
    }

    /// Return the current number of cached entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Return `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
