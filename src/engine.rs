// engine.rs — VectorEngine struct: top-level lifecycle coordinator that ties together
// the index selector, SQLite store, and embedding client.
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use chrono::Utc;
use tracing::instrument;

use crate::{
    config::VectorConfig,
    error::{VectorError, VectorResult},
    index::selector::IndexSelector,
    store::sqlite::SqliteStore,
    types::{Collection, DistanceMetric, IndexType, SearchQuery, SearchResult, VectorRecord},
};

/// Shared state for a single collection.
struct CollectionState {
    index: IndexSelector,
    record_map: HashMap<usize, SearchResult>,
    next_id: usize,
}

/// High-level engine that manages collections, indices, and persistence.
pub struct VectorEngine {
    config: VectorConfig,
    store: Arc<SqliteStore>,
    collections: RwLock<HashMap<String, RwLock<CollectionState>>>,
}

impl VectorEngine {
    /// Create a new engine, opening (or creating) the SQLite database.
    pub async fn new(config: VectorConfig) -> VectorResult<Self> {
        let store = SqliteStore::open(&config.db_path).await?;
        Ok(VectorEngine {
            config,
            store: Arc::new(store),
            collections: RwLock::new(HashMap::new()),
        })
    }

    /// Return a reference to the underlying store (for administrative operations).
    pub fn store(&self) -> &SqliteStore {
        &self.store
    }

    /// Create a new collection with the given parameters.
    #[instrument(skip_all)]
    pub async fn create_collection(
        &self,
        name: impl Into<String>,
        dimensions: usize,
        distance: DistanceMetric,
    ) -> VectorResult<Collection> {
        let name = name.into();
        {
            let cols = self.collections.read().map_err(|e| VectorError::Index(e.to_string()))?;
            if cols.contains_key(&name) {
                return Err(VectorError::Collection {
                    name: name.clone(),
                    reason: "already exists".into(),
                });
            }
        }
        let col = Collection {
            name: name.clone(),
            dimensions,
            distance,
            index_type: IndexType::Flat,
            created_at: Utc::now(),
            vector_count: 0,
            metadata: serde_json::Value::Null,
            ef_construction: self.config.ef_construction,
            m_connections: self.config.m_connections,
        };
        self.store.save_collection(&col).await?;
        let state = CollectionState {
            index: IndexSelector::new(dimensions, distance, &self.config),
            record_map: HashMap::new(),
            next_id: 0,
        };
        self.collections
            .write()
            .map_err(|e| VectorError::Index(e.to_string()))?
            .insert(name, RwLock::new(state));
        Ok(col)
    }

    /// Insert a vector record into the named collection.
    #[instrument(skip_all)]
    pub async fn insert(&self, record: VectorRecord) -> VectorResult<uuid::Uuid> {
        let id = record.id;

        // Acquire the write lock, compute internal_id, and update in-memory state.
        // The lock guards must be dropped before any `.await` point.
        let internal_id = {
            let cols = self.collections.read().map_err(|e| VectorError::Index(e.to_string()))?;
            let state_lock = cols.get(&record.collection).ok_or_else(|| VectorError::Collection {
                name: record.collection.clone(),
                reason: "not found".into(),
            })?;
            let mut state = state_lock.write().map_err(|e| VectorError::Index(e.to_string()))?;
            let internal_id = state.next_id;
            state.next_id += 1;
            let search_result = SearchResult {
                id: record.id,
                score: 0.0,
                vector: Some(record.vector.clone()),
                metadata: record.metadata.clone(),
                text: record.text.clone(),
            };
            state.record_map.insert(internal_id, search_result);
            state.index.insert(internal_id, record.vector.clone(), &self.config)?;
            internal_id
        };

        self.store.save_record(&record, internal_id).await?;
        Ok(id)
    }

    /// Search the named collection for the nearest neighbours of `query`.
    #[instrument(skip_all)]
    pub async fn search(&self, query: SearchQuery) -> VectorResult<Vec<SearchResult>> {
        let ef = query.ef_search.unwrap_or(self.config.ef_search);
        let cols = self.collections.read().map_err(|e| VectorError::Index(e.to_string()))?;
        let state_lock = cols.get(&query.collection).ok_or_else(|| VectorError::Collection {
            name: query.collection.clone(),
            reason: "not found".into(),
        })?;
        let state = state_lock.read().map_err(|e| VectorError::Index(e.to_string()))?;
        crate::search::ann::AnnSearch::search(&state.index, &query, ef, &state.record_map)
    }
}
