//! Deterministic conflict resolution engine

use crate::types::*;
use cmd_core::crdt::FactVersion;
use cmd_core::types::SourceId;
use chrono::Utc;
use std::collections::HashMap;
use tracing::info;

/// Deterministic resolver for conflicts
pub struct DeterministicResolver {
    /// Trust scores for different sources
    source_trust: HashMap<SourceId, f32>,

    /// Domain-specific rules
    domain_rules: HashMap<String, DomainRules>,

    /// Statistics tracking
    stats: ResolutionStats,
}

/// Domain-specific resolution rules
#[derive(Debug, Clone)]
pub struct DomainRules {
    /// Domain name
    pub domain: String,

    /// Priority attributes (which attributes are more important)
    pub priority_attributes: Vec<String>,

    /// Immutable attributes (cannot be changed once set)
    pub immutable_attributes: Vec<String>,

    /// Consensus threshold for this domain
    pub consensus_threshold: f32,
}

impl DeterministicResolver {
    pub fn new() -> Self {
        Self {
            source_trust: HashMap::new(),
            domain_rules: HashMap::new(),
            stats: ResolutionStats::default(),
        }
    }

    /// Set trust score for a source
    pub fn set_source_trust(&mut self, source: SourceId, trust: f32) {
        self.source_trust.insert(source, trust.clamp(0.0, 1.0));
    }

    /// Add domain-specific rules
    pub fn add_domain_rules(&mut self, rules: DomainRules) {
        self.domain_rules.insert(rules.domain.clone(), rules);
    }

    /// Detect the type of conflict between two facts
    pub fn detect_conflict_type(
        &self,
        existing: &FactVersion,
        new_fact: &FactVersion,
    ) -> ConflictType {
        // Check if it's the same fact from different times
        if existing.content.subject == new_fact.content.subject
            && existing.content.predicate == new_fact.content.predicate
        {
            // Check if objects are contradictory
            if self.are_contradictory(&existing.content.object, &new_fact.content.object) {
                ConflictType::ContradictoryFacts
            } else if existing.metadata.created_at < new_fact.metadata.created_at {
                ConflictType::TemporalSupersession
            } else if self.is_partial_overlap(&existing.content.object, &new_fact.content.object) {
                ConflictType::PartialOverlap
            } else {
                ConflictType::AttributeUpdate
            }
        } else if existing.sources != new_fact.sources {
            ConflictType::SourceDisagreement
        } else if self.is_duplicate(existing, new_fact) {
            ConflictType::DuplicateInformation
        } else {
            ConflictType::ReferenceAmbiguity
        }
    }

    /// Main resolution function
    pub fn resolve(
        &mut self,
        conflict: Conflict,
    ) -> Result<ResolutionResult, String> {
        let start_time = std::time::Instant::now();

        info!(
            "Resolving conflict {} of type {:?}",
            conflict.id, conflict.conflict_type
        );

        // Get the appropriate strategy
        let strategy = self.select_strategy(&conflict);

        // Apply the strategy
        let result = match strategy {
            ResolutionStrategy::KeepNewest => self.resolve_keep_newest(&conflict),
            ResolutionStrategy::KeepOldest => self.resolve_keep_oldest(&conflict),
            ResolutionStrategy::MergeViaCRDT => self.resolve_merge_crdt(&conflict),
            ResolutionStrategy::TrustHigherSource => self.resolve_trust_source(&conflict),
            ResolutionStrategy::ApplyLogicRules => self.resolve_apply_logic(&conflict),
            ResolutionStrategy::VersionBranching => self.resolve_version_branch(&conflict),
            ResolutionStrategy::ConsensusVoting => self.resolve_consensus(&conflict),
            ResolutionStrategy::DomainSpecificRules(ref domain) => {
                self.resolve_domain_specific(&conflict, domain)
            }
            ResolutionStrategy::DeferToUser | ResolutionStrategy::LocalLLM => {
                // These are not handled by deterministic resolver
                return Err("Non-deterministic strategy selected".to_string());
            }
        };

        // Update statistics
        let duration = start_time.elapsed();
        self.update_stats(&conflict.conflict_type, &strategy, result.is_ok(), duration);

        result
    }

    /// Select the best strategy for a conflict
    fn select_strategy(&self, conflict: &Conflict) -> ResolutionStrategy {
        // Check if we have precedents that worked well
        if let Some(precedent) = self.find_best_precedent(conflict) {
            if precedent.success && precedent.relevance > 0.8 {
                return precedent.strategy_used.clone();
            }
        }

        // Check for domain-specific rules
        if let Some(domain) = &conflict.context.domain {
            if self.domain_rules.contains_key(domain) {
                return ResolutionStrategy::DomainSpecificRules(domain.clone());
            }
        }

        // Use default strategy for the conflict type
        conflict.conflict_type.default_strategy()
    }

    /// Keep the newest fact
    fn resolve_keep_newest(&self, conflict: &Conflict) -> Result<ResolutionResult, String> {
        let (newer, older) = if conflict.existing_fact.metadata.created_at
            < conflict.new_fact.metadata.created_at
        {
            (conflict.new_fact.clone(), conflict.existing_fact.clone())
        } else {
            (conflict.existing_fact.clone(), conflict.new_fact.clone())
        };

        Ok(ResolutionResult {
            strategy_used: ResolutionStrategy::KeepNewest,
            resolved_facts: vec![newer.clone()],
            discarded_facts: vec![older],
            confidence: 0.95,
            explanation: format!(
                "Kept newer fact created at {}",
                newer.metadata.created_at
            ),
            needs_review: false,
        })
    }

    /// Keep the oldest fact
    fn resolve_keep_oldest(&self, conflict: &Conflict) -> Result<ResolutionResult, String> {
        let (older, newer) = if conflict.existing_fact.metadata.created_at
            < conflict.new_fact.metadata.created_at
        {
            (conflict.existing_fact.clone(), conflict.new_fact.clone())
        } else {
            (conflict.new_fact.clone(), conflict.existing_fact.clone())
        };

        Ok(ResolutionResult {
            strategy_used: ResolutionStrategy::KeepOldest,
            resolved_facts: vec![older.clone()],
            discarded_facts: vec![newer],
            confidence: 0.90,
            explanation: format!(
                "Kept older fact created at {}",
                older.metadata.created_at
            ),
            needs_review: false,
        })
    }

    /// Merge facts using CRDT
    fn resolve_merge_crdt(&self, conflict: &Conflict) -> Result<ResolutionResult, String> {
        let merged = conflict.existing_fact.merge(&conflict.new_fact);

        Ok(ResolutionResult {
            strategy_used: ResolutionStrategy::MergeViaCRDT,
            resolved_facts: vec![merged],
            discarded_facts: vec![],
            confidence: 0.85,
            explanation: "Facts merged using CRDT operations".to_string(),
            needs_review: false,
        })
    }

    /// Trust the source with higher confidence
    fn resolve_trust_source(&self, conflict: &Conflict) -> Result<ResolutionResult, String> {
        let existing_trust = self.calculate_source_trust(&conflict.existing_fact);
        let new_trust = self.calculate_source_trust(&conflict.new_fact);

        let (trusted, discarded, trust_score) = if existing_trust > new_trust {
            (conflict.existing_fact.clone(), conflict.new_fact.clone(), existing_trust)
        } else {
            (conflict.new_fact.clone(), conflict.existing_fact.clone(), new_trust)
        };

        Ok(ResolutionResult {
            strategy_used: ResolutionStrategy::TrustHigherSource,
            resolved_facts: vec![trusted],
            discarded_facts: vec![discarded],
            confidence: trust_score,
            explanation: format!(
                "Trusted source with confidence score {:.2}",
                trust_score
            ),
            needs_review: trust_score < 0.7,
        })
    }

    /// Apply logical rules
    fn resolve_apply_logic(&self, conflict: &Conflict) -> Result<ResolutionResult, String> {
        // Simple logical consistency check
        // In production, this would use a proper Datalog engine

        // Check for transitive consistency
        let is_consistent = self.check_logical_consistency(
            &conflict.existing_fact,
            &conflict.new_fact,
            &conflict.context.related_facts,
        );

        if is_consistent {
            // Both facts can coexist
            Ok(ResolutionResult {
                strategy_used: ResolutionStrategy::ApplyLogicRules,
                resolved_facts: vec![conflict.existing_fact.clone(), conflict.new_fact.clone()],
                discarded_facts: vec![],
                confidence: 0.80,
                explanation: "Both facts are logically consistent".to_string(),
                needs_review: false,
            })
        } else {
            // Choose the one with better logical support
            let existing_support = self.count_logical_support(
                &conflict.existing_fact,
                &conflict.context.related_facts,
            );
            let new_support = self.count_logical_support(
                &conflict.new_fact,
                &conflict.context.related_facts,
            );

            let (kept, discarded) = if existing_support >= new_support {
                (conflict.existing_fact.clone(), conflict.new_fact.clone())
            } else {
                (conflict.new_fact.clone(), conflict.existing_fact.clone())
            };

            Ok(ResolutionResult {
                strategy_used: ResolutionStrategy::ApplyLogicRules,
                resolved_facts: vec![kept],
                discarded_facts: vec![discarded],
                confidence: 0.75,
                explanation: "Kept fact with stronger logical support".to_string(),
                needs_review: true,
            })
        }
    }

    /// Create version branches for both facts
    fn resolve_version_branch(&self, conflict: &Conflict) -> Result<ResolutionResult, String> {
        // Reduce confidence for both versions
        let mut existing = conflict.existing_fact.clone();
        let mut new_fact = conflict.new_fact.clone();
        existing.confidence *= 0.8;
        new_fact.confidence *= 0.8;

        // Mark them as alternatives
        // In production, this would add metadata linking them

        Ok(ResolutionResult {
            strategy_used: ResolutionStrategy::VersionBranching,
            resolved_facts: vec![existing, new_fact],
            discarded_facts: vec![],
            confidence: 0.70,
            explanation: "Created version branches for both facts".to_string(),
            needs_review: true,
        })
    }

    /// Use consensus from multiple sources
    fn resolve_consensus(&self, conflict: &Conflict) -> Result<ResolutionResult, String> {
        // Count votes from sources
        let mut votes: HashMap<String, f32> = HashMap::new();

        // Vote for existing fact
        for source in &conflict.existing_fact.sources {
            let trust = self.source_trust.get(source).unwrap_or(&0.5);
            let key = format!("{:?}", conflict.existing_fact.content);
            *votes.entry(key).or_insert(0.0) += trust;
        }

        // Vote for new fact
        for source in &conflict.new_fact.sources {
            let trust = self.source_trust.get(source).unwrap_or(&0.5);
            let key = format!("{:?}", conflict.new_fact.content);
            *votes.entry(key).or_insert(0.0) += trust;
        }

        // Find winner
        let winner_key = votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone())
            .unwrap();

        let existing_key = format!("{:?}", conflict.existing_fact.content);
        let (kept, discarded) = if winner_key == existing_key {
            (conflict.existing_fact.clone(), conflict.new_fact.clone())
        } else {
            (conflict.new_fact.clone(), conflict.existing_fact.clone())
        };

        Ok(ResolutionResult {
            strategy_used: ResolutionStrategy::ConsensusVoting,
            resolved_facts: vec![kept],
            discarded_facts: vec![discarded],
            confidence: votes[&winner_key] / votes.values().sum::<f32>(),
            explanation: format!("Consensus selected with {:.0}% agreement",
                votes[&winner_key] / votes.values().sum::<f32>() * 100.0),
            needs_review: false,
        })
    }

    /// Apply domain-specific rules
    fn resolve_domain_specific(
        &self,
        conflict: &Conflict,
        domain: &str,
    ) -> Result<ResolutionResult, String> {
        let rules = self.domain_rules.get(domain)
            .ok_or_else(|| format!("No rules for domain: {}", domain))?;

        // Check immutable attributes
        let existing_attrs = self.extract_attributes(&conflict.existing_fact);
        let new_attrs = self.extract_attributes(&conflict.new_fact);

        for attr in &rules.immutable_attributes {
            if existing_attrs.contains_key(attr) && new_attrs.contains_key(attr) {
                if existing_attrs[attr] != new_attrs[attr] {
                    // Immutable attribute changed - keep existing
                    return Ok(ResolutionResult {
                        strategy_used: ResolutionStrategy::DomainSpecificRules(domain.to_string()),
                        resolved_facts: vec![conflict.existing_fact.clone()],
                        discarded_facts: vec![conflict.new_fact.clone()],
                        confidence: 0.95,
                        explanation: format!("Immutable attribute '{}' cannot be changed", attr),
                        needs_review: false,
                    });
                }
            }
        }

        // If no immutable conflicts, merge with priority
        let mut merged = conflict.existing_fact.clone();

        for attr in &rules.priority_attributes {
            if new_attrs.contains_key(attr) {
                // Update with new value for priority attributes
                // In production, this would properly update the fact content
                merged.metadata.created_at = Utc::now();
            }
        }

        Ok(ResolutionResult {
            strategy_used: ResolutionStrategy::DomainSpecificRules(domain.to_string()),
            resolved_facts: vec![merged],
            discarded_facts: vec![],
            confidence: 0.85,
            explanation: format!("Applied domain rules for {}", domain),
            needs_review: false,
        })
    }

    // Helper methods

    fn are_contradictory(&self, obj1: &serde_json::Value, obj2: &serde_json::Value) -> bool {
        // Simple contradiction check
        // In production, this would be more sophisticated
        if let (serde_json::Value::Bool(b1), serde_json::Value::Bool(b2)) = (obj1, obj2) {
            return b1 != b2;
        }

        if let (serde_json::Value::String(s1), serde_json::Value::String(s2)) = (obj1, obj2) {
            // Check for explicit negation
            return (s1.starts_with("not_") && &s1[4..] == s2.as_str())
                || (s2.starts_with("not_") && &s2[4..] == s1.as_str());
        }

        false
    }

    fn is_partial_overlap(&self, obj1: &serde_json::Value, obj2: &serde_json::Value) -> bool {
        // Check if objects partially overlap
        if let (serde_json::Value::Object(m1), serde_json::Value::Object(m2)) = (obj1, obj2) {
            let keys1: std::collections::HashSet<_> = m1.keys().collect();
            let keys2: std::collections::HashSet<_> = m2.keys().collect();
            let intersection = keys1.intersection(&keys2).count();
            let union = keys1.union(&keys2).count();

            return intersection > 0 && intersection < union;
        }

        false
    }

    fn is_duplicate(&self, fact1: &FactVersion, fact2: &FactVersion) -> bool {
        fact1.content.subject == fact2.content.subject
            && fact1.content.predicate == fact2.content.predicate
            && fact1.content.object == fact2.content.object
    }

    fn calculate_source_trust(&self, fact: &FactVersion) -> f32 {
        let mut total_trust = 0.0;
        let mut count = 0;

        for source in &fact.sources {
            let trust = self.source_trust.get(source).unwrap_or(&0.5);
            total_trust += trust;
            count += 1;
        }

        if count > 0 {
            (total_trust / count as f32) * fact.confidence
        } else {
            fact.confidence * 0.5
        }
    }

    fn check_logical_consistency(
        &self,
        fact1: &FactVersion,
        fact2: &FactVersion,
        related: &[FactVersion],
    ) -> bool {
        // Simplified logical consistency check
        // In production, would use proper inference engine

        // Check for direct contradiction
        if self.are_contradictory(&fact1.content.object, &fact2.content.object) {
            return false;
        }

        // Check transitivity violations
        for related_fact in related {
            if related_fact.content.subject == fact1.content.object.as_str().unwrap_or("")
                && related_fact.content.object == fact2.content.object
            {
                // A->B, B->C implies A->C should be consistent
                continue;
            }
        }

        true
    }

    fn count_logical_support(&self, fact: &FactVersion, related: &[FactVersion]) -> usize {
        related
            .iter()
            .filter(|f| {
                // Count facts that support this one
                f.content.object == serde_json::Value::String(fact.content.subject.clone())
                    || f.content.subject == fact.content.object.as_str().unwrap_or("")
            })
            .count()
    }

    fn find_best_precedent<'a>(&self, conflict: &'a Conflict) -> Option<&'a ResolutionPrecedent> {
        conflict
            .context
            .precedents
            .iter()
            .filter(|p| p.conflict_type == conflict.conflict_type && p.success)
            .max_by(|a, b| a.relevance.partial_cmp(&b.relevance).unwrap())
    }

    fn extract_attributes(&self, fact: &FactVersion) -> HashMap<String, String> {
        let mut attrs = HashMap::new();

        if let serde_json::Value::Object(map) = &fact.content.object {
            for (key, value) in map {
                attrs.insert(key.clone(), value.to_string());
            }
        }

        attrs
    }

    fn update_stats(
        &mut self,
        conflict_type: &ConflictType,
        strategy: &ResolutionStrategy,
        success: bool,
        duration: std::time::Duration,
    ) {
        self.stats.total_conflicts += 1;

        if matches!(
            strategy,
            ResolutionStrategy::KeepNewest
                | ResolutionStrategy::KeepOldest
                | ResolutionStrategy::MergeViaCRDT
                | ResolutionStrategy::TrustHigherSource
                | ResolutionStrategy::ApplyLogicRules
                | ResolutionStrategy::VersionBranching
        ) {
            self.stats.deterministic_resolutions += 1;
        } else {
            self.stats.external_resolutions += 1;
        }

        // Update average time
        let new_time = duration.as_micros() as u64;
        let total_time = self.stats.avg_resolution_time_us * (self.stats.total_conflicts - 1);
        self.stats.avg_resolution_time_us = (total_time + new_time) / self.stats.total_conflicts;

        // Update success rate
        if success {
            let successful = (self.stats.success_rate * (self.stats.total_conflicts - 1) as f32) + 1.0;
            self.stats.success_rate = successful / self.stats.total_conflicts as f32;
        } else {
            let successful = self.stats.success_rate * (self.stats.total_conflicts - 1) as f32;
            self.stats.success_rate = successful / self.stats.total_conflicts as f32;
        }

        // Update breakdown
        let conflict_key = format!("{:?}", conflict_type);
        *self.stats.by_type.entry(conflict_key).or_insert(0) += 1;

        let strategy_key = format!("{:?}", strategy);
        *self.stats.by_strategy.entry(strategy_key).or_insert(0) += 1;
    }
}

impl Default for DeterministicResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmd_core::crdt::FactPayload;

    #[test]
    fn test_temporal_supersession() {
        let mut resolver = DeterministicResolver::new();

        let old_fact = FactVersion::new(
            FactPayload {
                subject: "user".to_string(),
                predicate: "prefers".to_string(),
                object: serde_json::json!("light_mode"),
            },
            "node1",
            0.9,
            vec![],
        );

        let mut new_fact = old_fact.clone();
        new_fact.content.object = serde_json::json!("dark_mode");
        new_fact.metadata.created_at = Utc::now();

        let conflict = Conflict {
            id: "test-conflict".to_string(),
            conflict_type: ConflictType::TemporalSupersession,
            existing_fact: old_fact.clone(),
            new_fact: new_fact.clone(),
            confidence: 0.9,
            context: ConflictContext {
                related_facts: vec![],
                source_trust: HashMap::new(),
                domain: None,
                precedents: vec![],
            },
        };

        let result = resolver.resolve(conflict).unwrap();
        assert_eq!(result.strategy_used, ResolutionStrategy::KeepNewest);
        assert_eq!(result.resolved_facts.len(), 1);
        assert_eq!(result.resolved_facts[0].content.object, serde_json::json!("dark_mode"));
    }

    #[test]
    fn test_crdt_merge() {
        let mut resolver = DeterministicResolver::new();

        let fact1 = FactVersion::new(
            FactPayload {
                subject: "user".to_string(),
                predicate: "settings".to_string(),
                object: serde_json::json!({"theme": "dark"}),
            },
            "node1",
            0.8,
            vec![],
        );

        let fact2 = FactVersion::new(
            FactPayload {
                subject: "user".to_string(),
                predicate: "settings".to_string(),
                object: serde_json::json!({"language": "en"}),
            },
            "node2",
            0.8,
            vec![],
        );

        let conflict = Conflict {
            id: "test-conflict-2".to_string(),
            conflict_type: ConflictType::PartialOverlap,
            existing_fact: fact1,
            new_fact: fact2,
            confidence: 0.8,
            context: ConflictContext {
                related_facts: vec![],
                source_trust: HashMap::new(),
                domain: None,
                precedents: vec![],
            },
        };

        let result = resolver.resolve(conflict).unwrap();
        assert_eq!(result.strategy_used, ResolutionStrategy::MergeViaCRDT);
        assert_eq!(result.resolved_facts.len(), 1);
    }
}