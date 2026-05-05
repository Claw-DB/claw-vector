// index/selector.rs — auto-selecting index that migrates from FlatIndex to HnswIndex
// when the collection surpasses HNSW_THRESHOLD (1 000 vectors).
use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{
    config::VectorConfig,
    error::{VectorError, VectorResult},
    index::{flat::FlatIndex, hnsw::HnswIndex},
    types::DistanceMetric,
};

/// Collection size above which the selector automatically migrates to HNSW.
pub const HNSW_THRESHOLD: usize = 1_000;

#[derive(Debug, Serialize, Deserialize)]
struct PersistedIndex {
    index_type: String,
    points: Vec<(usize, Vec<f32>)>,
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexManifest {
    blake3: String,
    vector_count: u64,
    dimensions: u32,
    saved_at_ms: u64,
}

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
    pub fn save(&self, dir: &Path, workspace_id: &str, collection: &str) -> VectorResult<()> {
        let col_dir = dir.join(workspace_id).join(collection);
        std::fs::create_dir_all(&col_dir)?;

        let persisted = match self {
            IndexSelector::Flat(flat) => PersistedIndex {
                index_type: "flat".to_string(),
                points: flat.all_vectors()?,
            },
            IndexSelector::Hnsw(hnsw) => PersistedIndex {
                index_type: "hnsw".to_string(),
                points: hnsw.snapshot_points()?,
            },
        };

        let payload = serde_json::to_vec(&persisted)?;
        let final_path = idx_file(&col_dir, collection);
        let tmp_path = idx_tmp_file(&col_dir, collection);
        std::fs::write(&tmp_path, &payload)?;
        std::fs::rename(&tmp_path, &final_path)?;

        let saved_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0);
        let dimensions = match self {
            IndexSelector::Flat(flat) => flat.dimensions,
            IndexSelector::Hnsw(_) => {
                if persisted.points.is_empty() {
                    0
                } else {
                    persisted.points[0].1.len()
                }
            }
        };
        let manifest = IndexManifest {
            blake3: blake3::hash(&payload).to_hex().to_string(),
            vector_count: persisted.points.len() as u64,
            dimensions: dimensions as u32,
            saved_at_ms,
        };
        std::fs::write(
            idx_manifest_file(&col_dir, collection),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        Ok(())
    }

    /// Reload a previously saved index from `<dir>/<collection>/`.
    #[instrument(skip(config))]
    pub fn load(
        dir: &Path,
        workspace_id: &str,
        collection: &str,
        config: &VectorConfig,
        distance: DistanceMetric,
        dimensions: usize,
    ) -> VectorResult<Self> {
        let col_dir = dir.join(workspace_id).join(collection);
        let final_path = idx_file(&col_dir, collection);
        let tmp_path = idx_tmp_file(&col_dir, collection);
        let manifest_path = idx_manifest_file(&col_dir, collection);

        if tmp_path.exists() && final_path.exists() {
            let _ = std::fs::remove_file(&tmp_path);
        }

        if !manifest_path.exists() {
            return Err(VectorError::Index("missing index manifest".into()));
        }

        let manifest: IndexManifest = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        let payload = std::fs::read(&final_path)?;
        let digest = blake3::hash(&payload);
        let expected = hex::decode(manifest.blake3)
            .map_err(|err| VectorError::Index(format!("invalid manifest checksum: {err}")))?;
        if !constant_time_eq(digest.as_bytes(), &expected) {
            return Err(VectorError::Index("index checksum mismatch".into()));
        }

        let persisted: PersistedIndex = serde_json::from_slice(&payload)?;
        match persisted.index_type.as_str() {
            "flat" => {
                let flat = FlatIndex::new(dimensions, distance);
                flat.insert_batch(persisted.points)?;
                Ok(IndexSelector::Flat(flat))
            }
            "hnsw" => {
                let hnsw = HnswIndex::new_with_dimensions(config, distance, dimensions)?;
                hnsw.insert_batch(&persisted.points)?;
                Ok(IndexSelector::Hnsw(Box::new(hnsw)))
            }
            other => Err(VectorError::Index(format!("unknown index_type '{other}'"))),
        }
    }
}

fn idx_file(path: &Path, collection: &str) -> PathBuf {
    path.join(format!("{collection}.idx"))
}

fn idx_tmp_file(path: &Path, collection: &str) -> PathBuf {
    path.join(format!("{collection}.idx.tmp"))
}

fn idx_manifest_file(path: &Path, collection: &str) -> PathBuf {
    path.join(format!("{collection}.idx.manifest"))
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut diff = 0u8;
    for (a, b) in left.iter().zip(right.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}
