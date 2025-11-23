//! Source trust management and consensus strategies

use cmd_core::types::SourceId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Manages trust scores for different sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceTrustManager {
    /// Trust scores for each source (0.0 to 1.0)
    trust_scores: HashMap<SourceId, TrustScore>,

    /// Default trust for new sources
    default_trust: f32,

    /// Trust decay configuration
    decay_config: TrustDecayConfig,

    /// Performance history for adaptive trust
    performance_history: HashMap<SourceId, PerformanceStats>,
}

/// Trust score with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScore {
    /// Current trust value (0.0 to 1.0)
    pub value: f32,

    /// Last time trust was updated
    pub last_updated: SystemTime,

    /// Number of interactions
    pub interaction_count: u64,

    /// Source category (user, system, external)
    pub category: SourceCategory,
}

/// Categories of sources with different base trust levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceCategory {
    /// User-provided information (highest trust)
    User,

    /// System-generated facts (high trust)
    System,

    /// External sources like APIs (medium trust)
    External,

    /// Derived or inferred facts (lower trust)
    Derived,
}

/// Configuration for trust decay over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustDecayConfig {
    /// How quickly trust decays without reinforcement
    pub decay_rate: f32,

    /// Minimum trust threshold
    pub min_trust: f32,

    /// Maximum trust threshold
    pub max_trust: f32,

    /// Time period for decay calculation
    pub decay_period: Duration,
}

/// Performance statistics for a source
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Successful resolutions
    pub successes: u64,

    /// Failed resolutions
    pub failures: u64,

    /// Conflicts with other sources
    pub conflicts: u64,

    /// Average resolution confidence
    pub avg_confidence: f32,
}

impl SourceTrustManager {
    /// Create a new trust manager with default settings
    pub fn new() -> Self {
        Self {
            trust_scores: HashMap::new(),
            default_trust: 0.5,
            decay_config: TrustDecayConfig {
                decay_rate: 0.01,
                min_trust: 0.1,
                max_trust: 0.95,
                decay_period: Duration::from_secs(86400), // 1 day
            },
            performance_history: HashMap::new(),
        }
    }

    /// Get trust score for a source
    pub fn get_trust(&mut self, source: &SourceId) -> f32 {
        if let Some(trust_score) = self.trust_scores.get_mut(source) {
            // Apply decay based on time since last update
            let now = SystemTime::now();
            if let Ok(elapsed) = now.duration_since(trust_score.last_updated) {
                let decay_periods = elapsed.as_secs_f32() / self.decay_config.decay_period.as_secs_f32();
                let decay_factor = (-self.decay_config.decay_rate * decay_periods).exp();

                // Decay towards default trust, not to zero
                trust_score.value = self.default_trust +
                    (trust_score.value - self.default_trust) * decay_factor;

                // Clamp to valid range
                trust_score.value = trust_score.value
                    .max(self.decay_config.min_trust)
                    .min(self.decay_config.max_trust);

                trust_score.last_updated = now;
            }
            trust_score.value
        } else {
            self.default_trust
        }
    }

    /// Update trust based on resolution outcome
    pub fn update_trust(
        &mut self,
        source: &SourceId,
        success: bool,
        confidence: f32,
    ) {
        let trust_score = self.trust_scores
            .entry(source.clone())
            .or_insert(TrustScore {
                value: self.default_trust,
                last_updated: SystemTime::now(),
                interaction_count: 0,
                category: SourceCategory::External,
            });

        // Update performance stats
        let stats = self.performance_history
            .entry(source.clone())
            .or_default();

        if success {
            stats.successes += 1;
        } else {
            stats.failures += 1;
        }

        // Update average confidence
        let total_interactions = stats.successes + stats.failures;
        stats.avg_confidence =
            (stats.avg_confidence * (total_interactions - 1) as f32 + confidence)
            / total_interactions as f32;

        // Calculate trust adjustment
        let performance_ratio = if total_interactions > 0 {
            stats.successes as f32 / total_interactions as f32
        } else {
            0.5
        };

        // Weighted update based on confidence and performance
        let target_trust = performance_ratio * confidence;
        let learning_rate = 0.1; // How quickly trust adjusts

        trust_score.value = trust_score.value * (1.0 - learning_rate)
            + target_trust * learning_rate;

        // Apply category-based adjustments
        let category_multiplier = match trust_score.category {
            SourceCategory::User => 1.2,    // User sources gain/lose trust faster
            SourceCategory::System => 1.1,  // System sources are more stable
            SourceCategory::External => 1.0, // Normal adjustment
            SourceCategory::Derived => 0.9,  // Derived sources adjust slower
        };

        trust_score.value = self.default_trust +
            (trust_score.value - self.default_trust) * category_multiplier;

        // Clamp to valid range
        trust_score.value = trust_score.value
            .max(self.decay_config.min_trust)
            .min(self.decay_config.max_trust);

        trust_score.last_updated = SystemTime::now();
        trust_score.interaction_count += 1;
    }

    /// Set category for a source
    pub fn set_category(&mut self, source: &SourceId, category: SourceCategory) {
        let trust_score = self.trust_scores
            .entry(source.clone())
            .or_insert(TrustScore {
                value: self.default_trust,
                last_updated: SystemTime::now(),
                interaction_count: 0,
                category: category.clone(),
            });

        trust_score.category = category;

        // Adjust default trust based on category
        trust_score.value = match trust_score.category {
            SourceCategory::User => 0.8,
            SourceCategory::System => 0.7,
            SourceCategory::External => 0.5,
            SourceCategory::Derived => 0.4,
        };
    }

    /// Get consensus trust for multiple sources
    pub fn consensus_trust(&mut self, sources: &[SourceId]) -> f32 {
        if sources.is_empty() {
            return 0.0;
        }

        let trust_values: Vec<f32> = sources
            .iter()
            .map(|s| self.get_trust(s))
            .collect();

        // Weighted average with emphasis on higher trust sources
        let sum_weights: f32 = trust_values.iter().map(|t| t * t).sum();
        let weighted_sum: f32 = trust_values.iter().map(|t| t * t * t).sum();

        if sum_weights > 0.0 {
            weighted_sum / sum_weights
        } else {
            self.default_trust
        }
    }

    /// Detect conflicts between sources
    pub fn record_conflict(&mut self, source1: &SourceId, source2: &SourceId) {
        for source in [source1, source2] {
            let stats = self.performance_history
                .entry(source.clone())
                .or_default();
            stats.conflicts += 1;

            // Slightly reduce trust for sources with many conflicts
            if let Some(trust_score) = self.trust_scores.get_mut(source) {
                let conflict_penalty = 0.02;
                trust_score.value = (trust_score.value - conflict_penalty)
                    .max(self.decay_config.min_trust);
            }
        }
    }

    /// Get sources ranked by trust
    pub fn get_ranked_sources(&mut self) -> Vec<(SourceId, f32)> {
        // Collect source IDs first to avoid borrow conflict
        let sources: Vec<SourceId> = self.trust_scores.keys().cloned().collect();

        // Now get trust scores for each source
        let mut ranked: Vec<(SourceId, f32)> = sources
            .into_iter()
            .map(|source| {
                let trust = self.get_trust(&source);
                (source, trust)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Get performance report for a source
    pub fn get_performance(&self, source: &SourceId) -> Option<&PerformanceStats> {
        self.performance_history.get(source)
    }

    /// Reset trust for a source
    pub fn reset_trust(&mut self, source: &SourceId) {
        self.trust_scores.remove(source);
        self.performance_history.remove(source);
    }
}

impl Default for SourceTrustManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceCategory {
    /// Get base trust for this category
    pub fn base_trust(&self) -> f32 {
        match self {
            Self::User => 0.8,
            Self::System => 0.7,
            Self::External => 0.5,
            Self::Derived => 0.4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_initialization() {
        let mut manager = SourceTrustManager::new();
        let source = SourceId::new();

        assert_eq!(manager.get_trust(&source), 0.5);
    }

    #[test]
    fn test_trust_update() {
        let mut manager = SourceTrustManager::new();
        let source = SourceId::new();

        // Successful resolution should increase trust
        manager.update_trust(&source, true, 0.9);
        assert!(manager.get_trust(&source) > 0.5);

        // Failed resolution should decrease trust
        let high_trust = manager.get_trust(&source);
        manager.update_trust(&source, false, 0.3);
        assert!(manager.get_trust(&source) < high_trust);
    }

    #[test]
    fn test_category_trust() {
        let mut manager = SourceTrustManager::new();
        let source = SourceId::new();

        manager.set_category(&source, SourceCategory::User);
        assert_eq!(manager.get_trust(&source), 0.8);

        manager.set_category(&source, SourceCategory::Derived);
        assert_eq!(manager.get_trust(&source), 0.4);
    }

    #[test]
    fn test_consensus_trust() {
        let mut manager = SourceTrustManager::new();
        let source1 = SourceId::new();
        let source2 = SourceId::new();
        let source3 = SourceId::new();

        manager.set_category(&source1, SourceCategory::User);
        manager.set_category(&source2, SourceCategory::System);
        manager.set_category(&source3, SourceCategory::External);

        let consensus = manager.consensus_trust(&[source1, source2, source3]);
        assert!(consensus > 0.5); // Should be weighted towards higher trust sources
    }

    #[test]
    fn test_conflict_recording() {
        let mut manager = SourceTrustManager::new();
        let source1 = SourceId::new();
        let source2 = SourceId::new();

        manager.set_category(&source1, SourceCategory::System);
        let initial_trust = manager.get_trust(&source1);

        manager.record_conflict(&source1, &source2);

        assert!(manager.get_trust(&source1) < initial_trust);
        assert_eq!(manager.get_performance(&source1).unwrap().conflicts, 1);
    }

    #[test]
    fn test_ranked_sources() {
        let mut manager = SourceTrustManager::new();
        let source1 = SourceId::new();
        let source2 = SourceId::new();
        let source3 = SourceId::new();

        manager.set_category(&source1, SourceCategory::User);
        manager.set_category(&source2, SourceCategory::Derived);
        manager.set_category(&source3, SourceCategory::System);

        let ranked = manager.get_ranked_sources();
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].0, source1); // User should be highest
        assert_eq!(ranked[1].0, source3); // System second
        assert_eq!(ranked[2].0, source2); // Derived lowest
    }
}