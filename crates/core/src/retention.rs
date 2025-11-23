//! Mathematical model for memory retention based on Ebbinghaus forgetting curve

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Model for calculating memory retention strength over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionModel {
    /// Base retention strength (S₀) - initial importance/salience
    pub base_retention: f32,

    /// Decay rate (α) - how quickly memory fades
    pub decay_rate: f32,

    /// Stability (τ) - resistance to forgetting
    pub stability: f32,

    /// Last time this memory was recalled
    pub last_recall: DateTime<Utc>,

    /// Number of successful recalls
    pub recall_count: u32,
}

impl RetentionModel {
    /// Create a new retention model with initial confidence
    pub fn new(initial_confidence: f32) -> Self {
        Self {
            base_retention: initial_confidence,
            decay_rate: 0.05, // Default decay rate
            stability: 1.0,   // Initial stability
            last_recall: Utc::now(),
            recall_count: 0,
        }
    }

    /// Calculate retention strength at a given time
    /// S(t) = S₀ × e^(-Δt/τ)
    pub fn retention_strength(&self, now: DateTime<Utc>) -> f32 {
        let delta_seconds = (now - self.last_recall).num_seconds() as f32;

        // Prevent division by zero and ensure positive stability
        let tau = self.stability.max(0.1);

        self.base_retention * (-delta_seconds / (tau * 86400.0)).exp()
    }

    /// Update the model after a recall event
    pub fn update_on_recall(&mut self, success: bool) {
        if success {
            // Successful recall strengthens memory (spaced repetition)
            self.stability *= 1.5;
            self.recall_count += 1;

            // Boost base retention slightly with each successful recall
            self.base_retention = (self.base_retention * 1.05).min(1.0);
        } else {
            // Failed recall indicates memory is weaker than expected
            self.stability *= 0.9;
        }

        self.last_recall = Utc::now();
    }

    /// Predict when memory will fall below threshold
    pub fn predict_forget_time(&self, threshold: f32) -> Option<DateTime<Utc>> {
        if threshold >= self.base_retention {
            return None; // Already below threshold
        }

        // Solve for t: threshold = S₀ × e^(-t/τ)
        // t = -τ × ln(threshold/S₀)
        let ratio = threshold / self.base_retention;
        let seconds = -self.stability * 86400.0 * ratio.ln();

        if seconds > 0.0 {
            Some(self.last_recall + Duration::seconds(seconds as i64))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_retention_decay() {
        let model = RetentionModel::new(1.0);
        let now = model.last_recall;

        // Initial strength should be close to base retention
        assert!((model.retention_strength(now) - 1.0).abs() < 0.001);

        // After one day, strength should decay
        let tomorrow = now + Duration::days(1);
        let strength_tomorrow = model.retention_strength(tomorrow);
        assert!(strength_tomorrow < 1.0);
        assert!(strength_tomorrow > 0.0);

        // Strength should continue to decay
        let week_later = now + Duration::days(7);
        let strength_week = model.retention_strength(week_later);
        assert!(strength_week < strength_tomorrow);
    }

    #[test]
    fn test_successful_recall_boosts_stability() {
        let mut model = RetentionModel::new(0.8);
        let initial_stability = model.stability;

        model.update_on_recall(true);

        assert!(model.stability > initial_stability);
        assert_eq!(model.recall_count, 1);
        assert!(model.base_retention > 0.8);
    }

    #[test]
    fn test_failed_recall_reduces_stability() {
        let mut model = RetentionModel::new(0.8);
        let initial_stability = model.stability;

        model.update_on_recall(false);

        assert!(model.stability < initial_stability);
        assert_eq!(model.recall_count, 0);
    }

    #[test]
    fn test_predict_forget_time() {
        let model = RetentionModel::new(1.0);

        // Should predict when memory falls to 0.5
        let forget_time = model.predict_forget_time(0.5);
        assert!(forget_time.is_some());

        // Memory at that time should be approximately the threshold
        if let Some(time) = forget_time {
            let strength_at_forget = model.retention_strength(time);
            assert!((strength_at_forget - 0.5).abs() < 0.01);
        }

        // Cannot predict if threshold is above base retention
        assert!(model.predict_forget_time(1.1).is_none());
    }
}