// store/mmap.rs — memory-mapped vector file storage for fast bulk reads.
use memmap2::MmapOptions;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};

use crate::error::{VectorError, VectorResult};

/// A flat file of raw f32 vectors accessible via memory-mapping.
pub struct MmapVectorFile {
    /// Expected number of floats per vector.
    pub dimensions: usize,
    path: std::path::PathBuf,
}

impl MmapVectorFile {
    /// Open or create a mmap vector file at `path` with the given `dimensions`.
    pub fn open(path: &Path, dimensions: usize) -> VectorResult<Self> {
        if !path.exists() {
            File::create(path)?;
        }
        Ok(MmapVectorFile { dimensions, path: path.to_path_buf() })
    }

    /// Append a slice of vectors to the file in little-endian binary format.
    pub fn append_vectors(&self, vectors: &[Vec<f32>]) -> VectorResult<()> {
        let mut file = OpenOptions::new().append(true).open(&self.path)?;
        for vec in vectors {
            if vec.len() != self.dimensions {
                return Err(VectorError::DimensionMismatch { expected: self.dimensions, got: vec.len() });
            }
            for &v in vec.iter() {
                file.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Memory-map the file and copy out all stored vectors.
    pub fn read_all(&self) -> VectorResult<Vec<Vec<f32>>> {
        let file = File::open(&self.path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 { return Ok(Vec::new()); }
        // SAFETY: the file is opened read-only; data is copied immediately.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let bpv = self.dimensions * 4;
        if len % bpv != 0 {
            return Err(VectorError::Index(format!("mmap file length {len} not divisible by {bpv}")));
        }
        let n = len / bpv;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * bpv;
            let floats: Vec<f32> = mmap[off..off + bpv]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            result.push(floats);
        }
        Ok(result)
    }

    /// Return the number of stored vectors.
    pub fn vector_count(&self) -> VectorResult<usize> {
        let len = std::fs::metadata(&self.path)?.len() as usize;
        if self.dimensions == 0 { return Ok(0); }
        Ok(len / (self.dimensions * 4))
    }
}
