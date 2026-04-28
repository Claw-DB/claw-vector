// store/mod.rs — public re-exports for the store sub-module.
/// Memory-mapped vector file storage.
pub mod mmap;
/// SQLite persistence for vector records and collection metadata.
pub mod sqlite;

pub use sqlite::SqliteStore;
