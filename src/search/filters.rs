// search/filters.rs — metadata filter DSL extension methods and helpers.
use crate::types::MetadataFilter;

/// Extension trait adding helper constructors to [`MetadataFilter`].
pub trait MetadataFilterExt {
    /// Build an equality filter: `key == value`.
    fn eq(key: impl Into<String>, value: serde_json::Value) -> MetadataFilter;

    /// Build an AND filter from a list of sub-filters.
    fn and(filters: Vec<MetadataFilter>) -> MetadataFilter;

    /// Build an OR filter from a list of sub-filters.
    fn or(filters: Vec<MetadataFilter>) -> MetadataFilter;

    /// Build a NOT filter wrapping another filter.
    fn not(filter: MetadataFilter) -> MetadataFilter;
}

impl MetadataFilterExt for MetadataFilter {
    fn eq(key: impl Into<String>, value: serde_json::Value) -> MetadataFilter {
        MetadataFilter::Eq { key: key.into(), value }
    }

    fn and(filters: Vec<MetadataFilter>) -> MetadataFilter {
        MetadataFilter::And(filters)
    }

    fn or(filters: Vec<MetadataFilter>) -> MetadataFilter {
        MetadataFilter::Or(filters)
    }

    fn not(filter: MetadataFilter) -> MetadataFilter {
        MetadataFilter::Not(Box::new(filter))
    }
}
