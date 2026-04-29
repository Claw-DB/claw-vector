// store/mmap.rs — memory-mapped vector file storage for fast random-access reads.
use std::{
    fs::OpenOptions,
    path::{Path, PathBuf},
};

use byteorder::{ByteOrder, LittleEndian};
use memmap2::{MmapMut, MmapOptions};

use crate::error::{VectorError, VectorResult};

const HEADER_SIZE: usize = 64;
const MAGIC: &[u8; 8] = b"CLAWVEC1";
const VERSION: u32 = 1;

/// Header stored at the start of every mmap vector file.
#[derive(Debug, Clone)]
pub struct VecFileHeader {
    /// Format magic bytes.
    pub magic: [u8; 8],
    /// Format version.
    pub version: u32,
    /// Vector dimensionality.
    pub dimensions: u32,
    /// Highest written slot count.
    pub element_count: u64,
    /// Reserved bytes for future metadata.
    pub reserved: [u8; 40],
}

/// A memory-mapped file containing fixed-size raw `f32` vectors.
pub struct MmapVectorFile {
    /// Writable mmap covering the full file.
    pub mmap: MmapMut,
    /// Decoded file header.
    pub header: VecFileHeader,
    /// File path on disk.
    pub path: PathBuf,
}

impl MmapVectorFile {
    /// Create a new vector file with capacity for `max_elements` vectors.
    pub fn create(path: &Path, dimensions: usize, max_elements: usize) -> VectorResult<Self> {
        if dimensions == 0 {
            return Err(VectorError::Config(
                "mmap vector file dimensions must be greater than zero".into(),
            ));
        }
        if max_elements == 0 {
            return Err(VectorError::Config(
                "mmap vector file max_elements must be greater than zero".into(),
            ));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file_size = HEADER_SIZE + max_elements * dimensions * std::mem::size_of::<f32>();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(file_size as u64)?;

        let mmap = map_mut(&file)?;
        let header = VecFileHeader {
            magic: *MAGIC,
            version: VERSION,
            dimensions: dimensions as u32,
            element_count: 0,
            reserved: [0; 40],
        };

        let mut file = MmapVectorFile {
            mmap,
            header,
            path: path.to_path_buf(),
        };
        file.sync_header();
        file.flush()?;
        Ok(file)
    }

    /// Open an existing vector file and validate its header.
    pub fn open(path: &Path) -> VectorResult<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let metadata = file.metadata()?;
        if metadata.len() < HEADER_SIZE as u64 {
            return Err(VectorError::Index(format!(
                "mmap vector file '{}' is too small to contain a header",
                path.display()
            )));
        }

        let mmap = map_mut(&file)?;
        let header = read_header(&mmap)?;
        Ok(MmapVectorFile {
            mmap,
            header,
            path: path.to_path_buf(),
        })
    }

    /// Write a vector to a fixed slot, updating the element count if needed.
    pub fn write_vector(&mut self, internal_id: usize, vector: &[f32]) -> VectorResult<()> {
        let dimensions = self.dimensions();
        if vector.len() != dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: dimensions,
                got: vector.len(),
            });
        }

        let offset = self.vector_offset(internal_id)?;
        let byte_len = std::mem::size_of_val(vector);
        LittleEndian::write_f32_into(vector, &mut self.mmap[offset..offset + byte_len]);

        let next_count = internal_id as u64 + 1;
        if next_count > self.header.element_count {
            self.header.element_count = next_count;
            self.sync_header();
        }

        Ok(())
    }

    /// Read a vector from a fixed slot.
    pub fn read_vector(&self, internal_id: usize) -> VectorResult<Vec<f32>> {
        if internal_id >= self.element_count() {
            return Err(VectorError::NotFound {
                entity: "vector".into(),
                id: internal_id.to_string(),
            });
        }

        let offset = self.vector_offset(internal_id)?;
        let byte_len = self.dimensions() * std::mem::size_of::<f32>();
        let mut vector = vec![0.0f32; self.dimensions()];
        LittleEndian::read_f32_into(&self.mmap[offset..offset + byte_len], &mut vector);
        Ok(vector)
    }

    /// Zero out a vector slot without changing the file capacity.
    pub fn delete_vector(&mut self, internal_id: usize) -> VectorResult<()> {
        let offset = self.vector_offset(internal_id)?;
        let byte_len = self.dimensions() * std::mem::size_of::<f32>();
        self.mmap[offset..offset + byte_len].fill(0);
        Ok(())
    }

    /// Flush pending changes to disk.
    pub fn flush(&self) -> VectorResult<()> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Return the number of written slots tracked in the header.
    pub fn element_count(&self) -> usize {
        self.header.element_count as usize
    }

    /// Return the dimensionality of vectors stored in the file.
    pub fn dimensions(&self) -> usize {
        self.header.dimensions as usize
    }

    /// Return the total file size in bytes.
    pub fn file_size_bytes(&self) -> u64 {
        self.mmap.len() as u64
    }

    fn sync_header(&mut self) {
        self.mmap[..8].copy_from_slice(&self.header.magic);
        LittleEndian::write_u32(&mut self.mmap[8..12], self.header.version);
        LittleEndian::write_u32(&mut self.mmap[12..16], self.header.dimensions);
        LittleEndian::write_u64(&mut self.mmap[16..24], self.header.element_count);
        self.mmap[24..HEADER_SIZE].copy_from_slice(&self.header.reserved);
    }

    fn vector_offset(&self, internal_id: usize) -> VectorResult<usize> {
        let bytes_per_vector = self.dimensions() * std::mem::size_of::<f32>();
        let offset = HEADER_SIZE + internal_id * bytes_per_vector;
        let end = offset + bytes_per_vector;
        if end > self.mmap.len() {
            return Err(VectorError::Index(format!(
                "vector slot {internal_id} exceeds mmap file capacity for '{}'",
                self.path.display()
            )));
        }
        Ok(offset)
    }
}

fn read_header(mmap: &[u8]) -> VectorResult<VecFileHeader> {
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&mmap[..8]);
    if &magic != MAGIC {
        return Err(VectorError::Index("invalid mmap vector file magic".into()));
    }

    let version = LittleEndian::read_u32(&mmap[8..12]);
    if version != VERSION {
        return Err(VectorError::Index(format!(
            "unsupported mmap vector file version {version}"
        )));
    }

    let dimensions = LittleEndian::read_u32(&mmap[12..16]);
    if dimensions == 0 {
        return Err(VectorError::Index(
            "mmap vector file dimensions must be greater than zero".into(),
        ));
    }

    let element_count = LittleEndian::read_u64(&mmap[16..24]);
    let mut reserved = [0u8; 40];
    reserved.copy_from_slice(&mmap[24..HEADER_SIZE]);

    Ok(VecFileHeader {
        magic,
        version,
        dimensions,
        element_count,
        reserved,
    })
}

fn map_mut(file: &std::fs::File) -> VectorResult<MmapMut> {
    // SAFETY: the file handle remains alive for the duration of mapping creation and the
    // returned mmap owns the OS mapping independently of the file descriptor afterwards.
    unsafe { MmapOptions::new().map_mut(file).map_err(Into::into) }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::MmapVectorFile;

    #[test]
    fn create_write_read_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");
        let mut file = MmapVectorFile::create(&path, 3, 8).unwrap();

        file.write_vector(0, &[1.0, 2.0, 3.0]).unwrap();
        file.write_vector(3, &[4.0, 5.0, 6.0]).unwrap();
        file.flush().unwrap();

        assert_eq!(file.read_vector(0).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(file.read_vector(3).unwrap(), vec![4.0, 5.0, 6.0]);
        assert_eq!(file.element_count(), 4);
    }

    #[test]
    fn delete_vector_zeros_slot() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");
        let mut file = MmapVectorFile::create(&path, 2, 4).unwrap();

        file.write_vector(1, &[7.0, 9.0]).unwrap();
        file.delete_vector(1).unwrap();

        assert_eq!(file.read_vector(1).unwrap(), vec![0.0, 0.0]);
    }

    #[test]
    fn open_restores_header_and_data() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");
        {
            let mut file = MmapVectorFile::create(&path, 2, 4).unwrap();
            file.write_vector(2, &[3.5, 8.5]).unwrap();
            file.flush().unwrap();
        }

        let reopened = MmapVectorFile::open(&path).unwrap();
        assert_eq!(reopened.dimensions(), 2);
        assert_eq!(reopened.element_count(), 3);
        assert_eq!(reopened.read_vector(2).unwrap(), vec![3.5, 8.5]);
    }
}
