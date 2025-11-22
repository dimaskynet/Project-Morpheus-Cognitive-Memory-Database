//! CRDT (Conflict-free Replicated Data Types) for versioning and conflict resolution

use crate::types::SourceId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Vector clock for tracking causality in distributed systems
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorClock {
    /// Map from node ID to logical timestamp
    pub clock: HashMap<String, u64>,
}

impl VectorClock {
    pub fn new() -> Self {
        Self {
            clock: HashMap::new(),
        }
    }

    /// Increment clock for a given node
    pub fn increment(&mut self, node_id: &str) {
        *self.clock.entry(node_id.to_string()).or_insert(0) += 1;
    }

    /// Get timestamp for a node
    pub fn get(&self, node_id: &str) -> u64 {
        self.clock.get(node_id).copied().unwrap_or(0)
    }

    /// Compare two vector clocks for causality
    pub fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut self_ahead = false;
        let mut other_ahead = false;

        // Check all nodes in both clocks
        let all_nodes: HashSet<_> = self.clock.keys()
            .chain(other.clock.keys())
            .collect();

        for node in all_nodes {
            let self_time = self.get(node);
            let other_time = other.get(node);

            match self_time.cmp(&other_time) {
                Ordering::Greater => self_ahead = true,
                Ordering::Less => other_ahead = true,
                Ordering::Equal => {}
            }
        }

        match (self_ahead, other_ahead) {
            (true, false) => Some(Ordering::Greater),
            (false, true) => Some(Ordering::Less),
            (false, false) => Some(Ordering::Equal),
            (true, true) => None, // Concurrent
        }
    }

    /// Merge two vector clocks, taking the maximum of each component
    pub fn merge(&mut self, other: &Self) {
        for (node, &time) in &other.clock {
            self.clock
                .entry(node.clone())
                .and_modify(|t| *t = (*t).max(time))
                .or_insert(time);
        }
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}

/// A versioned fact with CRDT properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactVersion {
    pub id: Uuid,
    pub content: FactPayload,
    pub vector_clock: VectorClock,
    pub lamport_timestamp: u64,
    pub confidence: f32,
    pub sources: Vec<SourceId>,
    pub supersedes: Option<Uuid>,
    pub metadata: FactMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactPayload {
    pub subject: String,
    pub predicate: String,
    pub object: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactMetadata {
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub is_derived: bool,
    pub tags: HashSet<String>,
}

impl FactVersion {
    /// Create a new fact version
    pub fn new(
        content: FactPayload,
        node_id: &str,
        confidence: f32,
        sources: Vec<SourceId>,
    ) -> Self {
        let mut vector_clock = VectorClock::new();
        vector_clock.increment(node_id);

        Self {
            id: Uuid::new_v4(),
            content,
            vector_clock,
            lamport_timestamp: 1,
            confidence,
            sources,
            supersedes: None,
            metadata: FactMetadata {
                created_at: Utc::now(),
                created_by: node_id.to_string(),
                is_derived: false,
                tags: HashSet::new(),
            },
        }
    }

    /// CRDT merge operation for concurrent facts
    pub fn merge(&self, other: &Self) -> Self {
        match self.vector_clock.partial_cmp(&other.vector_clock) {
            Some(Ordering::Less) => other.clone(),
            Some(Ordering::Greater) => self.clone(),
            Some(Ordering::Equal) | None => {
                // Concurrent or equal - use LWW (Last Writer Wins) with Lamport timestamp
                if self.lamport_timestamp > other.lamport_timestamp {
                    self.clone()
                } else if self.lamport_timestamp < other.lamport_timestamp {
                    other.clone()
                } else {
                    // Same Lamport timestamp - use UUID for deterministic tie-breaking
                    if self.id > other.id {
                        self.clone()
                    } else {
                        other.clone()
                    }
                }
            }
        }
    }

    /// Check if this fact supersedes another
    pub fn supersedes(&self, other: &Self) -> bool {
        if let Some(superseded_id) = self.supersedes {
            if superseded_id == other.id {
                return true;
            }
        }

        // Check vector clock dominance
        matches!(
            self.vector_clock.partial_cmp(&other.vector_clock),
            Some(Ordering::Greater)
        )
    }
}

/// LWW-Element-Set CRDT for managing collections of facts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LWWElementSet<T: Clone + Eq + std::hash::Hash> {
    adds: HashMap<T, DateTime<Utc>>,
    removes: HashMap<T, DateTime<Utc>>,
}

impl<T: Clone + Eq + std::hash::Hash> LWWElementSet<T> {
    pub fn new() -> Self {
        Self {
            adds: HashMap::new(),
            removes: HashMap::new(),
        }
    }

    pub fn add(&mut self, element: T, timestamp: DateTime<Utc>) {
        self.adds.insert(element, timestamp);
    }

    pub fn remove(&mut self, element: T, timestamp: DateTime<Utc>) {
        self.removes.insert(element, timestamp);
    }

    pub fn contains(&self, element: &T) -> bool {
        match (self.adds.get(element), self.removes.get(element)) {
            (Some(add_time), Some(remove_time)) => add_time > remove_time,
            (Some(_), None) => true,
            _ => false,
        }
    }

    pub fn merge(&mut self, other: &Self) {
        // Merge adds
        for (elem, &time) in &other.adds {
            self.adds
                .entry(elem.clone())
                .and_modify(|t| *t = (*t).max(time))
                .or_insert(time);
        }

        // Merge removes
        for (elem, &time) in &other.removes {
            self.removes
                .entry(elem.clone())
                .and_modify(|t| *t = (*t).max(time))
                .or_insert(time);
        }
    }

    pub fn elements(&self) -> HashSet<T> {
        self.adds
            .keys()
            .filter(|elem| self.contains(elem))
            .cloned()
            .collect()
    }
}

impl<T: Clone + Eq + std::hash::Hash> Default for LWWElementSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_comparison() {
        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment("node1");
        clock2.increment("node2");

        // Concurrent clocks
        assert_eq!(clock1.partial_cmp(&clock2), None);

        clock1.increment("node2");
        // clock1 is now ahead
        assert_eq!(clock1.partial_cmp(&clock2), Some(Ordering::Greater));
        assert_eq!(clock2.partial_cmp(&clock1), Some(Ordering::Less));
    }

    #[test]
    fn test_vector_clock_merge() {
        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment("node1");
        clock1.increment("node1");
        clock2.increment("node2");

        clock1.merge(&clock2);

        assert_eq!(clock1.get("node1"), 2);
        assert_eq!(clock1.get("node2"), 1);
    }

    #[test]
    fn test_fact_version_merge() {
        let fact1 = FactVersion::new(
            FactPayload {
                subject: "user".to_string(),
                predicate: "likes".to_string(),
                object: serde_json::json!("coffee"),
            },
            "node1",
            0.9,
            vec![],
        );

        let mut fact2 = fact1.clone();
        fact2.id = Uuid::new_v4();
        fact2.vector_clock.increment("node2");
        fact2.content.object = serde_json::json!("tea");

        let merged = fact1.merge(&fact2);
        assert_eq!(merged.content.object, serde_json::json!("tea"));
    }

    #[test]
    fn test_lww_element_set() {
        let mut set = LWWElementSet::new();
        let now = Utc::now();
        let later = now + chrono::Duration::seconds(1);

        set.add("item1", now);
        assert!(set.contains(&"item1"));

        set.remove("item1", later);
        assert!(!set.contains(&"item1"));

        // Re-add with even later timestamp
        let much_later = later + chrono::Duration::seconds(1);
        set.add("item1", much_later);
        assert!(set.contains(&"item1"));
    }

    #[test]
    fn test_lww_element_set_merge() {
        let mut set1 = LWWElementSet::new();
        let mut set2 = LWWElementSet::new();
        let now = Utc::now();

        set1.add("a", now);
        set2.add("b", now);

        set1.merge(&set2);

        assert!(set1.contains(&"a"));
        assert!(set1.contains(&"b"));
        assert_eq!(set1.elements().len(), 2);
    }
}