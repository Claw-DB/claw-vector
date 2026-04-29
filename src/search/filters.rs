// search/filters.rs — metadata filter DSL extension methods and helpers.
use crate::{
    error::{VectorError, VectorResult},
    types::MetadataFilter,
};

/// Extension trait adding helper constructors to [`MetadataFilter`].
pub trait MetadataFilterExt {
    /// Build an equality filter: `key == value`.
    fn eq(key: impl Into<String>, value: serde_json::Value) -> MetadataFilter;

    /// Build a greater-than filter: `key > value`.
    fn gt(key: impl Into<String>, value: f64) -> MetadataFilter;

    /// Build a less-than filter: `key < value`.
    fn lt(key: impl Into<String>, value: f64) -> MetadataFilter;

    /// Build a case-insensitive string contains filter.
    fn contains(key: impl Into<String>, value: impl Into<String>) -> MetadataFilter;

    /// Build a membership filter.
    fn one_of(key: impl Into<String>, values: Vec<serde_json::Value>) -> MetadataFilter;

    /// Build an existence filter.
    fn exists(key: impl Into<String>) -> MetadataFilter;

    /// Build an AND filter from a list of sub-filters.
    fn and(filters: Vec<MetadataFilter>) -> MetadataFilter;

    /// Build an OR filter from a list of sub-filters.
    fn or(filters: Vec<MetadataFilter>) -> MetadataFilter;

    /// Build a NOT filter wrapping another filter.
    fn not(filter: MetadataFilter) -> MetadataFilter;
}

impl MetadataFilterExt for MetadataFilter {
    fn eq(key: impl Into<String>, value: serde_json::Value) -> MetadataFilter {
        MetadataFilter::Eq {
            key: key.into(),
            value,
        }
    }

    fn gt(key: impl Into<String>, value: f64) -> MetadataFilter {
        MetadataFilter::Gt {
            key: key.into(),
            value,
        }
    }

    fn lt(key: impl Into<String>, value: f64) -> MetadataFilter {
        MetadataFilter::Lt {
            key: key.into(),
            value,
        }
    }

    fn contains(key: impl Into<String>, value: impl Into<String>) -> MetadataFilter {
        MetadataFilter::Contains {
            key: key.into(),
            value: value.into(),
        }
    }

    fn one_of(key: impl Into<String>, values: Vec<serde_json::Value>) -> MetadataFilter {
        MetadataFilter::In {
            key: key.into(),
            values,
        }
    }

    fn exists(key: impl Into<String>) -> MetadataFilter {
        MetadataFilter::Exists { key: key.into() }
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

/// Traverse nested JSON using dot notation with optional array indices.
pub fn parse_json_path<'a>(
    metadata: &'a serde_json::Value,
    path: &str,
) -> Option<&'a serde_json::Value> {
    if path.trim().is_empty() {
        return None;
    }

    let mut current = metadata;
    for segment in path.split('.') {
        if segment.is_empty() {
            return None;
        }

        current = match current {
            serde_json::Value::Object(map) => map.get(segment)?,
            serde_json::Value::Array(items) => {
                let index = segment.parse::<usize>().ok()?;
                items.get(index)?
            }
            _ => return None,
        };
    }

    Some(current)
}

/// Apply a metadata filter to a JSON metadata document.
pub fn apply_filter(filter: &MetadataFilter, metadata: &serde_json::Value) -> bool {
    match filter {
        MetadataFilter::Eq { key, value } => parse_json_path(metadata, key) == Some(value),
        MetadataFilter::Gt { key, value } => parse_json_path(metadata, key)
            .and_then(serde_json::Value::as_f64)
            .is_some_and(|candidate| candidate > *value),
        MetadataFilter::Lt { key, value } => parse_json_path(metadata, key)
            .and_then(serde_json::Value::as_f64)
            .is_some_and(|candidate| candidate < *value),
        MetadataFilter::Contains { key, value } => parse_json_path(metadata, key)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|candidate| {
                candidate
                    .to_ascii_lowercase()
                    .contains(&value.to_ascii_lowercase())
            }),
        MetadataFilter::In { key, values } => parse_json_path(metadata, key)
            .is_some_and(|candidate| values.iter().any(|value| value == candidate)),
        MetadataFilter::Exists { key } => parse_json_path(metadata, key).is_some(),
        MetadataFilter::And(filters) => filters.iter().all(|nested| apply_filter(nested, metadata)),
        MetadataFilter::Or(filters) => filters.iter().any(|nested| apply_filter(nested, metadata)),
        MetadataFilter::Not(filter) => !apply_filter(filter, metadata),
    }
}

/// Validate a metadata filter before search execution.
pub fn validate_filter(filter: &MetadataFilter) -> VectorResult<()> {
    validate_filter_inner(filter, 1)
}

fn validate_filter_inner(filter: &MetadataFilter, depth: usize) -> VectorResult<()> {
    if depth > 10 {
        return Err(VectorError::FilterError(
            "metadata filter nesting exceeds maximum depth of 10".into(),
        ));
    }

    match filter {
        MetadataFilter::Eq { key, .. }
        | MetadataFilter::Gt { key, .. }
        | MetadataFilter::Lt { key, .. }
        | MetadataFilter::Contains { key, .. }
        | MetadataFilter::In { key, .. }
        | MetadataFilter::Exists { key } => {
            if key.trim().is_empty() {
                return Err(VectorError::FilterError(
                    "metadata filter key must not be empty".into(),
                ));
            }
            Ok(())
        }
        MetadataFilter::And(filters) => {
            if filters.is_empty() {
                return Err(VectorError::FilterError(
                    "metadata AND filter must contain at least one clause".into(),
                ));
            }
            for nested in filters {
                validate_filter_inner(nested, depth + 1)?;
            }
            Ok(())
        }
        MetadataFilter::Or(filters) => {
            if filters.is_empty() {
                return Err(VectorError::FilterError(
                    "metadata OR filter must contain at least one clause".into(),
                ));
            }
            for nested in filters {
                validate_filter_inner(nested, depth + 1)?;
            }
            Ok(())
        }
        MetadataFilter::Not(filter) => validate_filter_inner(filter, depth + 1),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{apply_filter, parse_json_path, validate_filter};
    use crate::types::MetadataFilter;

    fn sample_metadata() -> serde_json::Value {
        json!({
            "user": {
                "role": "Admin",
                "age": 34,
                "tags": ["alpha", "beta"],
                "profile": {
                    "score": 91.5,
                    "active": true
                }
            },
            "items": [
                { "name": "First", "price": 9.5 },
                { "name": "Second", "price": 12.0 }
            ],
            "title": "Vector Search Primer",
            "priority": 7,
            "status": "published"
        })
    }

    #[test]
    fn parse_json_path_reads_top_level_value() {
        let metadata = sample_metadata();
        assert_eq!(
            parse_json_path(&metadata, "status"),
            Some(&json!("published"))
        );
    }

    #[test]
    fn parse_json_path_reads_nested_value() {
        let metadata = sample_metadata();
        assert_eq!(
            parse_json_path(&metadata, "user.profile.score"),
            Some(&json!(91.5))
        );
    }

    #[test]
    fn parse_json_path_reads_array_index() {
        let metadata = sample_metadata();
        assert_eq!(
            parse_json_path(&metadata, "items.1.name"),
            Some(&json!("Second"))
        );
    }

    #[test]
    fn parse_json_path_supports_nested_array_scalar() {
        let metadata = sample_metadata();
        assert_eq!(
            parse_json_path(&metadata, "user.tags.0"),
            Some(&json!("alpha"))
        );
    }

    #[test]
    fn parse_json_path_returns_none_for_missing_path() {
        let metadata = sample_metadata();
        assert_eq!(parse_json_path(&metadata, "user.profile.rank"), None);
    }

    #[test]
    fn parse_json_path_returns_none_for_invalid_array_index() {
        let metadata = sample_metadata();
        assert_eq!(parse_json_path(&metadata, "items.two.name"), None);
    }

    #[test]
    fn apply_filter_eq_matches_nested_value() {
        let metadata = sample_metadata();
        let filter = MetadataFilter::Eq {
            key: "user.role".into(),
            value: json!("Admin"),
        };
        assert!(apply_filter(&filter, &metadata));
    }

    #[test]
    fn apply_filter_gt_matches_numeric_threshold() {
        let metadata = sample_metadata();
        let filter = MetadataFilter::Gt {
            key: "user.age".into(),
            value: 30.0,
        };
        assert!(apply_filter(&filter, &metadata));
    }

    #[test]
    fn apply_filter_lt_matches_numeric_threshold() {
        let metadata = sample_metadata();
        let filter = MetadataFilter::Lt {
            key: "items.0.price".into(),
            value: 10.0,
        };
        assert!(apply_filter(&filter, &metadata));
    }

    #[test]
    fn apply_filter_contains_is_case_insensitive() {
        let metadata = sample_metadata();
        let filter = MetadataFilter::Contains {
            key: "title".into(),
            value: "search".into(),
        };
        assert!(apply_filter(&filter, &metadata));
    }

    #[test]
    fn apply_filter_in_matches_candidate_set() {
        let metadata = sample_metadata();
        let filter = MetadataFilter::In {
            key: "status".into(),
            values: vec![json!("draft"), json!("published")],
        };
        assert!(apply_filter(&filter, &metadata));
    }

    #[test]
    fn apply_filter_exists_detects_nested_key() {
        let metadata = sample_metadata();
        let filter = MetadataFilter::Exists {
            key: "user.profile.active".into(),
        };
        assert!(apply_filter(&filter, &metadata));
    }

    #[test]
    fn apply_filter_and_requires_all_clauses() {
        let metadata = sample_metadata();
        let filter = MetadataFilter::And(vec![
            MetadataFilter::Eq {
                key: "status".into(),
                value: json!("published"),
            },
            MetadataFilter::Gt {
                key: "priority".into(),
                value: 5.0,
            },
        ]);
        assert!(apply_filter(&filter, &metadata));
    }

    #[test]
    fn apply_filter_or_matches_any_clause() {
        let metadata = sample_metadata();
        let filter = MetadataFilter::Or(vec![
            MetadataFilter::Eq {
                key: "status".into(),
                value: json!("draft"),
            },
            MetadataFilter::Contains {
                key: "title".into(),
                value: "primer".into(),
            },
        ]);
        assert!(apply_filter(&filter, &metadata));
    }

    #[test]
    fn apply_filter_not_negates_match() {
        let metadata = sample_metadata();
        let filter = MetadataFilter::Not(Box::new(MetadataFilter::Eq {
            key: "status".into(),
            value: json!("draft"),
        }));
        assert!(apply_filter(&filter, &metadata));
    }

    #[test]
    fn apply_filter_handles_complex_boolean_expression() {
        let metadata = sample_metadata();
        let filter = MetadataFilter::And(vec![
            MetadataFilter::Or(vec![
                MetadataFilter::Eq {
                    key: "user.role".into(),
                    value: json!("Admin"),
                },
                MetadataFilter::Eq {
                    key: "user.role".into(),
                    value: json!("Editor"),
                },
            ]),
            MetadataFilter::Not(Box::new(MetadataFilter::Lt {
                key: "user.profile.score".into(),
                value: 90.0,
            })),
        ]);
        assert!(apply_filter(&filter, &metadata));
    }

    #[test]
    fn validate_filter_accepts_valid_nested_filter() {
        let filter = MetadataFilter::And(vec![
            MetadataFilter::Exists {
                key: "user.role".into(),
            },
            MetadataFilter::Not(Box::new(MetadataFilter::Contains {
                key: "title".into(),
                value: "draft".into(),
            })),
        ]);
        assert!(validate_filter(&filter).is_ok());
    }

    #[test]
    fn validate_filter_rejects_empty_and() {
        let err = validate_filter(&MetadataFilter::And(vec![])).unwrap_err();
        assert!(err.to_string().contains("AND filter"));
    }

    #[test]
    fn validate_filter_rejects_empty_or() {
        let err = validate_filter(&MetadataFilter::Or(vec![])).unwrap_err();
        assert!(err.to_string().contains("OR filter"));
    }

    #[test]
    fn validate_filter_rejects_blank_keys() {
        let err = validate_filter(&MetadataFilter::Exists { key: " ".into() }).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn validate_filter_rejects_excessive_depth() {
        let mut filter = MetadataFilter::Exists {
            key: "status".into(),
        };
        for _ in 0..10 {
            filter = MetadataFilter::Not(Box::new(filter));
        }
        let err = validate_filter(&filter).unwrap_err();
        assert!(err.to_string().contains("maximum depth"));
    }
}
