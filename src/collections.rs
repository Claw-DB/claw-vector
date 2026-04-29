// collections.rs — collection management and persistence orchestration.
use std::{collections::HashMap, path::PathBuf, sync::Arc};

use chrono::Utc;
use tokio::sync::RwLock;
use tracing::{instrument, warn};
use uuid::Uuid;

use crate::{
    config::VectorConfig,
    error::{VectorError, VectorResult},
    index::selector::IndexSelector,
    store::{mmap::MmapVectorFile, sqlite::VectorStore},
    types::{Collection, DistanceMetric, IndexType, VectorRecord},
};

/// Coordinates collection metadata, vector files, and in-memory indexes.
pub struct CollectionManager {
    /// Runtime configuration shared by all collections.
    pub config: VectorConfig,
    /// Persistent metadata store.
    pub store: Arc<VectorStore>,
    /// In-memory index implementations keyed by collection name.
    pub indexes: Arc<RwLock<HashMap<String, IndexSelector>>>,
    /// Memory-mapped vector files keyed by collection name.
    pub mmap_files: Arc<RwLock<HashMap<String, MmapVectorFile>>>,
}

impl CollectionManager {
    /// Build a manager and restore all persisted collections.
    #[instrument(skip(store))]
    pub async fn new(config: VectorConfig, store: Arc<VectorStore>) -> VectorResult<Self> {
        std::fs::create_dir_all(&config.index_dir)?;

        let manager = CollectionManager {
            config,
            store,
            indexes: Arc::new(RwLock::new(HashMap::new())),
            mmap_files: Arc::new(RwLock::new(HashMap::new())),
        };

        let collections = manager.store.list_collections().await?;
        for collection in collections {
            manager.restore_collection(&collection).await?;
        }

        Ok(manager)
    }

    /// Create a new collection and initialize its index and vector file.
    #[instrument(skip(self))]
    pub async fn create_collection(
        &self,
        name: &str,
        dimensions: usize,
        distance: DistanceMetric,
    ) -> VectorResult<Collection> {
        if name.trim().is_empty() {
            return Err(VectorError::Collection {
                name: name.to_string(),
                reason: "name must not be empty".into(),
            });
        }

        let collection = Collection {
            name: name.to_string(),
            dimensions,
            distance,
            index_type: IndexType::Flat,
            created_at: Utc::now(),
            vector_count: 0,
            metadata: serde_json::json!({}),
            ef_construction: self.config.ef_construction,
            m_connections: self.config.m_connections,
        };

        let dir = self.collection_dir(name);
        std::fs::create_dir_all(&dir)?;
        let mmap = MmapVectorFile::create(
            &self.vector_file_path(name),
            dimensions,
            self.config.max_elements.max(1),
        )?;
        let index = IndexSelector::new(dimensions, distance, &self.config);

        self.store.create_collection(&collection).await?;

        self.indexes.write().await.insert(name.to_string(), index);
        self.mmap_files.write().await.insert(name.to_string(), mmap);

        Ok(collection)
    }

    /// Fetch a collection definition by name.
    #[instrument(skip(self))]
    pub async fn get_collection(&self, name: &str) -> VectorResult<Collection> {
        self.store.get_collection(name).await
    }

    /// Delete a collection and all of its persisted state.
    #[instrument(skip(self))]
    pub async fn delete_collection(&self, name: &str) -> VectorResult<()> {
        let removed_index = self.indexes.write().await.remove(name);
        let removed_mmap = self.mmap_files.write().await.remove(name);

        if removed_index.is_none() || removed_mmap.is_none() {
            let exists = self.store.get_collection(name).await.is_ok();
            if !exists {
                return Err(VectorError::NotFound {
                    entity: "collection".into(),
                    id: name.to_string(),
                });
            }
        }

        if let Err(err) = self.store.delete_collection(name).await {
            if let Some(index) = removed_index {
                self.indexes.write().await.insert(name.to_string(), index);
            }
            if let Some(mmap) = removed_mmap {
                self.mmap_files.write().await.insert(name.to_string(), mmap);
            }
            return Err(err);
        }

        let collection_dir = self.collection_dir(name);
        if collection_dir.exists() {
            std::fs::remove_dir_all(collection_dir)?;
        }

        Ok(())
    }

    /// List all persisted collections.
    #[instrument(skip(self))]
    pub async fn list_collections(&self) -> VectorResult<Vec<Collection>> {
        self.store.list_collections().await
    }

    /// Insert a single vector record into its collection.
    #[instrument(skip(self, record))]
    pub async fn insert_vector(&self, record: VectorRecord) -> VectorResult<Uuid> {
        let collection = self.store.get_collection(&record.collection).await?;
        if record.vector.len() != collection.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: collection.dimensions,
                got: record.vector.len(),
            });
        }

        let internal_id = self.store.next_internal_id(&record.collection).await?;
        let record_id = record.id;
        self.apply_in_memory_insert(&record, internal_id).await?;

        if let Err(err) = self.store.insert_record(&record, internal_id).await {
            self.rollback_in_memory_insert(&record.collection, internal_id)
                .await;
            return Err(err);
        }

        if let Err(err) = self
            .store
            .increment_vector_count(&record.collection, 1)
            .await
        {
            let _ = self.store.delete_record(record.id).await;
            self.rollback_in_memory_insert(&record.collection, internal_id)
                .await;
            return Err(err);
        }

        self.sync_collection_index_type(&record.collection).await?;

        Ok(record_id)
    }

    /// Insert multiple vector records atomically.
    #[instrument(skip(self, records))]
    pub async fn insert_batch(&self, records: Vec<VectorRecord>) -> VectorResult<Vec<Uuid>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let mut next_ids = HashMap::<String, usize>::new();
        let mut deltas = HashMap::<String, i64>::new();
        let mut staged = Vec::with_capacity(records.len());
        let mut ids = Vec::with_capacity(records.len());

        for record in records {
            let collection = self.store.get_collection(&record.collection).await?;
            if record.vector.len() != collection.dimensions {
                return Err(VectorError::DimensionMismatch {
                    expected: collection.dimensions,
                    got: record.vector.len(),
                });
            }

            let next_id = if let Some(next_id) = next_ids.get_mut(&record.collection) {
                let current = *next_id;
                *next_id += 1;
                current
            } else {
                let current = self.store.next_internal_id(&record.collection).await?;
                next_ids.insert(record.collection.clone(), current + 1);
                current
            };

            *deltas.entry(record.collection.clone()).or_insert(0) += 1;
            ids.push(record.id);
            staged.push((record, next_id));
        }

        for (record, internal_id) in &staged {
            if let Err(err) = self.apply_in_memory_insert(record, *internal_id).await {
                self.rollback_batch_in_memory(&staged).await;
                return Err(err);
            }
        }

        if let Err(err) = self.store.batch_insert_records(&staged).await {
            self.rollback_batch_in_memory(&staged).await;
            return Err(err);
        }

        for (collection, delta) in deltas {
            if let Err(err) = self.store.increment_vector_count(&collection, delta).await {
                for (record, _) in &staged {
                    let _ = self.store.delete_record(record.id).await;
                }
                self.rollback_batch_in_memory(&staged).await;
                return Err(err);
            }
            self.sync_collection_index_type(&collection).await?;
        }

        Ok(ids)
    }

    /// Delete a vector from a collection by UUID.
    #[instrument(skip(self))]
    pub async fn delete_vector(&self, collection: &str, id: Uuid) -> VectorResult<bool> {
        let (record, internal_id) = match self.store.get_record(id).await {
            Ok(value) => value,
            Err(VectorError::NotFound { .. }) => return Ok(false),
            Err(err) => return Err(err),
        };

        if record.collection != collection {
            return Ok(false);
        }

        {
            let mut indexes = self.indexes.write().await;
            let index = indexes
                .get_mut(collection)
                .ok_or_else(|| VectorError::NotFound {
                    entity: "collection".into(),
                    id: collection.to_string(),
                })?;
            index.delete(internal_id)?;
        }

        {
            let mut mmap_files = self.mmap_files.write().await;
            let mmap = mmap_files
                .get_mut(collection)
                .ok_or_else(|| VectorError::NotFound {
                    entity: "collection".into(),
                    id: collection.to_string(),
                })?;
            mmap.delete_vector(internal_id)?;
            mmap.flush()?;
        }

        self.store.delete_record(id).await?;
        self.store.increment_vector_count(collection, -1).await?;
        Ok(true)
    }

    /// Load a full vector record, including its raw vector from the mmap file.
    #[instrument(skip(self))]
    pub async fn get_vector(&self, collection: &str, id: Uuid) -> VectorResult<VectorRecord> {
        let (mut record, internal_id) = self.store.get_record(id).await?;
        if record.collection != collection {
            return Err(VectorError::NotFound {
                entity: "record".into(),
                id: id.to_string(),
            });
        }

        let mmap_files = self.mmap_files.read().await;
        let mmap = mmap_files
            .get(collection)
            .ok_or_else(|| VectorError::NotFound {
                entity: "collection".into(),
                id: collection.to_string(),
            })?;
        record.vector = mmap.read_vector(internal_id)?;
        Ok(record)
    }

    /// Persist all loaded indexes to disk.
    #[instrument(skip(self))]
    pub async fn persist_indexes(&self) -> VectorResult<()> {
        let indexes = self.indexes.read().await;
        for (name, index) in indexes.iter() {
            index.save(&self.config.index_dir, name)?;
        }
        Ok(())
    }

    /// Read a raw vector by collection and internal id.
    pub async fn read_vector_by_internal_id(
        &self,
        collection: &str,
        internal_id: usize,
    ) -> VectorResult<Vec<f32>> {
        let mmap_files = self.mmap_files.read().await;
        let mmap = mmap_files
            .get(collection)
            .ok_or_else(|| VectorError::NotFound {
                entity: "collection".into(),
                id: collection.to_string(),
            })?;
        mmap.read_vector(internal_id)
    }

    /// Return the number of loaded indexes.
    pub async fn loaded_index_count(&self) -> usize {
        self.indexes.read().await.len()
    }

    /// Return the number of loaded mmap vector files.
    pub async fn loaded_mmap_count(&self) -> usize {
        self.mmap_files.read().await.len()
    }

    async fn restore_collection(&self, collection: &Collection) -> VectorResult<()> {
        let dir = self.collection_dir(&collection.name);
        std::fs::create_dir_all(&dir)?;

        let mmap_path = self.vector_file_path(&collection.name);
        let mmap = if mmap_path.exists() {
            MmapVectorFile::open(&mmap_path)?
        } else {
            MmapVectorFile::create(
                &mmap_path,
                collection.dimensions,
                self.config
                    .max_elements
                    .max(collection.vector_count as usize + 1),
            )?
        };

        let index = match IndexSelector::load(
            &self.config.index_dir,
            &collection.name,
            &self.config,
            collection.distance,
            collection.dimensions,
        ) {
            Ok(index) => index,
            Err(err) => {
                warn!(
                    collection = %collection.name,
                    error = %err,
                    "failed to load persisted index, rebuilding from mmap"
                );
                self.rebuild_index(collection, &mmap).await?
            }
        };

        self.indexes
            .write()
            .await
            .insert(collection.name.clone(), index);
        self.mmap_files
            .write()
            .await
            .insert(collection.name.clone(), mmap);
        self.sync_collection_index_type(&collection.name).await?;
        Ok(())
    }

    async fn rebuild_index(
        &self,
        collection: &Collection,
        mmap: &MmapVectorFile,
    ) -> VectorResult<IndexSelector> {
        let mut index =
            IndexSelector::new(collection.dimensions, collection.distance, &self.config);
        let records = self
            .store
            .list_records_for_collection(&collection.name)
            .await?;
        let mut items = Vec::with_capacity(records.len());
        for (_, internal_id) in records {
            let vector = mmap.read_vector(internal_id)?;
            items.push((internal_id, vector));
        }
        index.insert_batch(items, &self.config)?;
        Ok(index)
    }

    async fn apply_in_memory_insert(
        &self,
        record: &VectorRecord,
        internal_id: usize,
    ) -> VectorResult<()> {
        {
            let mut mmap_files = self.mmap_files.write().await;
            let mmap =
                mmap_files
                    .get_mut(&record.collection)
                    .ok_or_else(|| VectorError::NotFound {
                        entity: "collection".into(),
                        id: record.collection.clone(),
                    })?;
            mmap.write_vector(internal_id, &record.vector)?;
            mmap.flush()?;
        }

        let mut indexes = self.indexes.write().await;
        let index = indexes
            .get_mut(&record.collection)
            .ok_or_else(|| VectorError::NotFound {
                entity: "collection".into(),
                id: record.collection.clone(),
            })?;
        index.insert(internal_id, record.vector.clone(), &self.config)?;
        Ok(())
    }

    async fn rollback_in_memory_insert(&self, collection: &str, internal_id: usize) {
        if let Some(index) = self.indexes.write().await.get_mut(collection) {
            let _ = index.delete(internal_id);
        }
        if let Some(mmap) = self.mmap_files.write().await.get_mut(collection) {
            let _ = mmap.delete_vector(internal_id);
            let _ = mmap.flush();
        }
    }

    async fn rollback_batch_in_memory(&self, staged: &[(VectorRecord, usize)]) {
        for (record, internal_id) in staged.iter().rev() {
            self.rollback_in_memory_insert(&record.collection, *internal_id)
                .await;
        }
    }

    fn collection_dir(&self, name: &str) -> PathBuf {
        self.config.index_dir.join(name)
    }

    fn vector_file_path(&self, name: &str) -> PathBuf {
        self.collection_dir(name).join("vectors.bin")
    }

    async fn sync_collection_index_type(&self, collection: &str) -> VectorResult<()> {
        let current_type = {
            let indexes = self.indexes.read().await;
            let index = indexes
                .get(collection)
                .ok_or_else(|| VectorError::NotFound {
                    entity: "collection".into(),
                    id: collection.to_string(),
                })?;
            if index.is_hnsw() {
                IndexType::HNSW
            } else {
                IndexType::Flat
            }
        };
        self.store
            .update_collection_index_type(collection, current_type)
            .await
    }
}
