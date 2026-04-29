// index/selector.rs — auto-selecting index that migrates from FlatIndex to HnswIndex
// when the collection surpasses HNSW_THRESHOLD (1 000 vectors).
use std::path::Path;

use tracing::instrument;

use crate::{
    config::VectorConfig,
    error::{VectorError, VectorResult},
    index::{flat::FlatIndex, hnsw::HnswIndex},
    types::DistanceMetric,
};

/// Collection size above which the selector automatically migrates to HNSW.
pub const HNSW_THRESHOLD: usize = 1_000;

/// Transparently routes between a [`FlatIndex`] and a [`HnswIndex`].
pub enum IndexSelector {
    /// Brute-force index for small collections.
    Flat(FlatIndex),
    /// Approximate NN index for larger collections.
    Hnsw(Box<HnswIndex>),
}

impl IndexSelector {
    /// Create a new selector (always starts as Flat).
    pub fn new(dimensions: usize, distance: DistanceMetric, _config: &VectorConfig) -> Self {
        IndexSelector::Flat(FlatIndex::new(dimensions, distance))
    }

    /// Insert a single vector, migrating to HNSW if the threshold is crossed.
    #[instrument(skip(self, vector, config))]
    pub fn insert(
        &mut self,
        id: usize,
        vector: Vec<f32>,
        config: &VectorConfig,
    ) -> VectorResult<()> {
        match self {
            IndexSelector::Flat(flat) => {
                flat.insert(id, vector)?;
                if flat.len() > HNSW_THRESHOLD {
                    self.migrate_to_hnsw(config)?;
                }
            }
            IndexSelector::Hnsw(hnsw) => hnsw.insert(id, &vector)?,
        }
        Ok(())
    }

    /// Insert a batch of vectors, migrating to HNSW if the threshold is crossed.
    #[instrument(skip(self, items, config))]
    pub fn insert_batch(
        &mut self,
        items: Vec<(usize, Vec<f32>)>,
        config: &VectorConfig,
    ) -> VectorResult<()> {
        match self {
            IndexSelector::Flat(flat) => {
                flat.insert_batch(items)?;
                if flat.len() > HNSW_THRESHOLD {
                    self.migrate_to_hnsw(config)?;
                }
            }
            IndexSelector::Hnsw(hnsw) => hnsw.insert_batch(&items)?,
        }
        Ok(())
    }

    /// Search for `top_k` nearest neighbours of `query`.
    #[instrument(skip(self, query))]
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> VectorResult<Vec<(usize, f32)>> {
        match self {
            IndexSelector::Flat(flat) => flat.search(query, top_k),
            IndexSelector::Hnsw(hnsw) => hnsw.search(query, top_k, ef_search),
        }
    }

    /// Delete a vector by id. Returns `true` if the id was present.
    #[instrument(skip(self))]
    pub fn delete(&mut self, id: usize) -> VectorResult<bool> {
        match self {
            IndexSelector::Flat(flat) => flat.delete(id),
            IndexSelector::Hnsw(hnsw) => {
                hnsw.delete(id)?;
                Ok(true)
            }
        }
    }

    /// Return the number of live elements.
    pub fn len(&self) -> usize {
        match self {
            IndexSelector::Flat(f) => f.len(),
            IndexSelector::Hnsw(h) => h.len(),
        }
    }

    /// Return `true` if the selector contains no live elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return `true` if the selector is backed by HNSW.
    pub fn is_hnsw(&self) -> bool {
        matches!(self, IndexSelector::Hnsw(_))
    }

    /// Migrate from FlatIndex to HnswIndex, replacing `self`.
    #[instrument(skip(self, config))]
    pub fn migrate_to_hnsw(&mut self, config: &VectorConfig) -> VectorResult<()> {
        let hnsw = match self {
            IndexSelector::Flat(flat) => {
                tracing::info!(elements = flat.len(), "migrating flat index to HNSW");
                flat.to_hnsw(config)?
            }
            IndexSelector::Hnsw(_) => return Ok(()),
        };
        *self = IndexSelector::Hnsw(Box::new(hnsw));
        Ok(())
    }

    /// Persist the index under `<dir>/<collection>/`.
    #[instrument(skip(self))]
    pub fn save(&self, dir: &Path, collection: &str) -> VectorResult<()> {
        let col_dir = dir.join(collection);
        std::fs::create_dir_all(&col_dir)?;
        let kind = if self.is_hnsw() { "hnsw" } else { "flat" };
        std::fs::write(
            col_dir.join("index.meta.json"),
            serde_json::to_string(&serde_json::json!({ "index_type": kind }))?,
        )?;
        match self {
            IndexSelector::Flat(flat) => {
                std::fs::write(
                    col_dir.join("flat.json"),
                    serde_json::to_string(&flat.all_vectors()?)?,
                )?;
            }
            IndexSelector::Hnsw(hnsw) => hnsw.save(&col_dir)?,
        }
        Ok(())
    }

    /// Reload a previously saved index from `<dir>/<collection>/`.
    #[instrument(skip(config))]
    pub fn load(
        dir: &Path,
        collection: &str,
        config: &VectorConfig,
        distance: DistanceMetric,
        dimensions: usize,
    ) -> VectorResult<Self> {
        let col_dir = dir.join(collection);
        let meta: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(col_dir.join("index.meta.json"))?)?;
        match meta["index_type"]
            .as_str()
            .ok_or_else(|| VectorError::Index("missing index_type".into()))?
        {
            "flat" => {
                let vecs: Vec<(usize, Vec<f32>)> =
                    serde_json::from_str(&std::fs::read_to_string(col_dir.join("flat.json"))?)?;
                let flat = FlatIndex::new(dimensions, distance);
                flat.insert_batch(vecs)?;
                Ok(IndexSelector::Flat(flat))
            }
            "hnsw" => Ok(IndexSelector::Hnsw(Box::new(HnswIndex::load(
                &col_dir, config, distance,
            )?))),
            other => Err(VectorError::Index(format!("unknown index_type '{other}'"))),
        }
    }
}
