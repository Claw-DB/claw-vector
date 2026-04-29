// index/hnsw.rs — HNSW index wrapper around the hnsw_rs crate.
//
// Uses a "shadow map" of (id → vector) for persistence and migration, avoiding
// hnsw_rs's own file-format which carries awkward lifetime constraints.
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        RwLock,
    },
};

use hnsw_rs::prelude::*;
use tracing::instrument;

use crate::{
    config::VectorConfig,
    error::{VectorError, VectorResult},
    types::DistanceMetric,
};

// ─── HnswStats ───────────────────────────────────────────────────────────────

/// Runtime statistics snapshot for a [`HnswIndex`].
#[derive(Debug, Clone)]
pub struct HnswStats {
    /// Number of live (non-deleted) elements.
    pub element_count: usize,
    /// Maximum capacity the index was configured for.
    pub max_elements: usize,
    /// HNSW `ef_construction` build parameter.
    pub ef_construction: usize,
    /// HNSW `M` connections parameter.
    pub m_connections: usize,
    /// Number of layers observed in the current graph.
    pub layers: usize,
}

// ─── HnswInner (type-erased distance variant) ─────────────────────────────────

enum HnswInner {
    L2(Hnsw<'static, f32, DistL2>),
    Cosine(Hnsw<'static, f32, DistCosine>),
    Dot(Hnsw<'static, f32, DistDot>),
}

impl HnswInner {
    fn insert(&self, id: usize, vector: &[f32]) {
        match self {
            HnswInner::L2(h) => h.insert((vector, id)),
            HnswInner::Cosine(h) => h.insert((vector, id)),
            HnswInner::Dot(h) => h.insert((vector, id)),
        }
    }

    fn parallel_insert(&self, refs: &[(&Vec<f32>, usize)]) {
        match self {
            HnswInner::L2(h) => h.parallel_insert(refs),
            HnswInner::Cosine(h) => h.parallel_insert(refs),
            HnswInner::Dot(h) => h.parallel_insert(refs),
        }
    }

    fn search(&self, query: &[f32], top_k: usize, ef_search: usize) -> Vec<Neighbour> {
        match self {
            HnswInner::L2(h) => h.search(query, top_k, ef_search),
            HnswInner::Cosine(h) => h.search(query, top_k, ef_search),
            HnswInner::Dot(h) => h.search(query, top_k, ef_search),
        }
    }

    fn ef_construction(&self) -> usize {
        match self {
            HnswInner::L2(h) => h.get_ef_construction(),
            HnswInner::Cosine(h) => h.get_ef_construction(),
            HnswInner::Dot(h) => h.get_ef_construction(),
        }
    }

    fn max_nb_connection(&self) -> usize {
        match self {
            HnswInner::L2(h) => h.get_max_nb_connection() as usize,
            HnswInner::Cosine(h) => h.get_max_nb_connection() as usize,
            HnswInner::Dot(h) => h.get_max_nb_connection() as usize,
        }
    }

    fn max_level_observed(&self) -> usize {
        match self {
            HnswInner::L2(h) => h.get_max_level_observed() as usize,
            HnswInner::Cosine(h) => h.get_max_level_observed() as usize,
            HnswInner::Dot(h) => h.get_max_level_observed() as usize,
        }
    }
}

// ─── HnswIndex ───────────────────────────────────────────────────────────────

/// Thread-safe HNSW index with support for three distance metrics.
pub struct HnswIndex {
    inner: HnswInner,
    /// Shadow copy of (id → vector) used for serialisation and migration.
    points: RwLock<HashMap<usize, Vec<f32>>>,
    /// Expected vector dimensionality.
    dimensions: usize,
    /// Distance metric in use.
    distance: DistanceMetric,
    /// Count of live (non-deleted) elements.
    element_count: AtomicUsize,
    /// Maximum capacity the index was configured for.
    max_elements: usize,
    /// Logically deleted ids (tombstones).
    deleted: RwLock<HashSet<usize>>,
}

impl HnswIndex {
    /// Build a new empty HNSW index from the given config and distance metric.
    #[instrument(skip(config))]
    pub fn new(config: &VectorConfig, distance: DistanceMetric) -> VectorResult<Self> {
        Self::new_with_dimensions(config, distance, config.default_dimensions)
    }

    /// Build a new empty HNSW index for an explicit `dimensions` count.
    pub fn new_with_dimensions(
        config: &VectorConfig,
        distance: DistanceMetric,
        dimensions: usize,
    ) -> VectorResult<Self> {
        let inner = build_inner(
            config.m_connections,
            config.max_elements,
            16,
            config.ef_construction,
            distance,
        );
        Ok(HnswIndex {
            inner,
            points: RwLock::new(HashMap::new()),
            dimensions,
            distance,
            element_count: AtomicUsize::new(0),
            max_elements: config.max_elements,
            deleted: RwLock::new(HashSet::new()),
        })
    }

    /// Insert a single vector, validating its dimensionality.
    #[instrument(skip(self, vector))]
    pub fn insert(&self, id: usize, vector: &[f32]) -> VectorResult<()> {
        if vector.len() != self.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }
        self.inner.insert(id, vector);
        self.points
            .write()
            .map_err(|e| VectorError::Index(e.to_string()))?
            .insert(id, vector.to_vec());
        self.element_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Insert a batch of vectors in parallel via hnsw_rs `parallel_insert`.
    #[instrument(skip(self, items))]
    pub fn insert_batch(&self, items: &[(usize, Vec<f32>)]) -> VectorResult<()> {
        for (_, v) in items {
            if v.len() != self.dimensions {
                return Err(VectorError::DimensionMismatch {
                    expected: self.dimensions,
                    got: v.len(),
                });
            }
        }
        let refs: Vec<(&Vec<f32>, usize)> = items.iter().map(|(id, v)| (v, *id)).collect();
        self.inner.parallel_insert(&refs);
        let mut pts = self
            .points
            .write()
            .map_err(|e| VectorError::Index(e.to_string()))?;
        for (id, v) in items {
            pts.insert(*id, v.clone());
        }
        self.element_count.fetch_add(items.len(), Ordering::Relaxed);
        Ok(())
    }

    /// Search for the `top_k` nearest neighbours of `query`.
    ///
    /// Returns `(internal_id, distance)` pairs sorted by ascending distance.
    #[instrument(skip(self, query))]
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> VectorResult<Vec<(usize, f32)>> {
        if query.len() != self.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }
        let deleted = self
            .deleted
            .read()
            .map_err(|e| VectorError::Index(e.to_string()))?;
        let neighbours = self.inner.search(query, top_k + deleted.len(), ef_search);
        let mut results: Vec<(usize, f32)> = neighbours
            .into_iter()
            .filter(|n| !deleted.contains(&n.d_id))
            .map(|n| (n.d_id, n.distance))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    /// Mark a vector as deleted (tombstone — hnsw_rs does not support physical removal).
    #[instrument(skip(self))]
    pub fn delete(&self, id: usize) -> VectorResult<()> {
        let mut deleted = self
            .deleted
            .write()
            .map_err(|e| VectorError::Index(e.to_string()))?;
        if deleted.insert(id) {
            self.points
                .write()
                .map_err(|e| VectorError::Index(e.to_string()))?
                .remove(&id);
            self.element_count.fetch_sub(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Return the number of live (non-deleted) elements.
    pub fn len(&self) -> usize {
        self.element_count.load(Ordering::Relaxed)
    }

    /// Return `true` if the index contains no live elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Persist the index under `path`.
    ///
    /// Writes `hnsw.meta.json` and `hnsw.points.bin`.
    #[instrument(skip(self))]
    pub fn save(&self, path: &Path) -> VectorResult<()> {
        std::fs::create_dir_all(path)?;
        let pts = self
            .points
            .read()
            .map_err(|e| VectorError::Index(e.to_string()))?;
        let meta = serde_json::json!({
            "distance": self.distance,
            "dimensions": self.dimensions,
            "element_count": self.element_count.load(Ordering::Relaxed),
            "max_elements": self.max_elements,
        });
        std::fs::write(path.join("hnsw.meta.json"), serde_json::to_string(&meta)?)?;

        // Binary format: [n: u64][(id: u64)(v0..vN: f32) ...]
        let mut buf = Vec::with_capacity(8 + pts.len() * (8 + self.dimensions * 4));
        buf.extend_from_slice(&(pts.len() as u64).to_le_bytes());
        for (&id, vec) in pts.iter() {
            buf.extend_from_slice(&(id as u64).to_le_bytes());
            for &v in vec {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(path.join("hnsw.points.bin"), buf)?;
        Ok(())
    }

    /// Reload a previously saved index by re-inserting all persisted points.
    #[instrument(skip(config))]
    pub fn load(
        path: &Path,
        config: &VectorConfig,
        distance: DistanceMetric,
    ) -> VectorResult<Self> {
        let meta: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(path.join("hnsw.meta.json"))?)?;
        let dimensions = meta["dimensions"]
            .as_u64()
            .ok_or_else(|| VectorError::Index("missing dimensions".into()))?
            as usize;
        let max_elements = meta["max_elements"]
            .as_u64()
            .unwrap_or(config.max_elements as u64) as usize;

        let raw = std::fs::read(path.join("hnsw.points.bin"))?;
        let points = decode_points_bin(&raw, dimensions)?;

        let mut cfg = config.clone();
        cfg.default_dimensions = dimensions;
        cfg.max_elements = max_elements;
        let index = Self::new_with_dimensions(&cfg, distance, dimensions)?;
        index.insert_batch(&points)?;
        Ok(index)
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> HnswStats {
        HnswStats {
            element_count: self.element_count.load(Ordering::Relaxed),
            max_elements: self.max_elements,
            ef_construction: self.inner.ef_construction(),
            m_connections: self.inner.max_nb_connection(),
            layers: self.inner.max_level_observed(),
        }
    }
}

fn build_inner(
    m: usize,
    max_elem: usize,
    max_layer: usize,
    ef_c: usize,
    distance: DistanceMetric,
) -> HnswInner {
    match distance {
        DistanceMetric::Euclidean => {
            HnswInner::L2(Hnsw::new(m, max_elem, max_layer, ef_c, DistL2 {}))
        }
        DistanceMetric::Cosine => {
            HnswInner::Cosine(Hnsw::new(m, max_elem, max_layer, ef_c, DistCosine {}))
        }
        DistanceMetric::DotProduct => {
            HnswInner::Dot(Hnsw::new(m, max_elem, max_layer, ef_c, DistDot {}))
        }
    }
}

fn decode_points_bin(raw: &[u8], dimensions: usize) -> VectorResult<Vec<(usize, Vec<f32>)>> {
    if raw.len() < 8 {
        return Ok(Vec::new());
    }
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let bpr = 8 + dimensions * 4;
    if raw.len() < 8 + n * bpr {
        return Err(VectorError::Index("hnsw.points.bin is truncated".into()));
    }
    let mut points = Vec::with_capacity(n);
    let mut off = 8usize;
    for _ in 0..n {
        let id = u64::from_le_bytes(raw[off..off + 8].try_into().unwrap()) as usize;
        off += 8;
        let floats: Vec<f32> = raw[off..off + dimensions * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        off += dimensions * 4;
        points.push((id, floats));
    }
    Ok(points)
}

// SAFETY: Hnsw<'static, T, D> owns all its data; interior mutation is guarded by parking_lot.
unsafe impl Send for HnswIndex {}
unsafe impl Sync for HnswIndex {}
