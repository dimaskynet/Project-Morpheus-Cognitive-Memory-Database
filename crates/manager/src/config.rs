//! Configuration for Memory Manager

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Memory Manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerConfig {
    /// Enable automatic conflict detection and resolution
    pub enable_conflict_detection: bool,

    /// Threshold for forgetting memories (0.0 to 1.0)
    pub forgetting_threshold: f32,

    /// Interval for running consolidation process
    pub consolidation_interval: Duration,

    /// Interval for running forgetting process
    pub forgetting_interval: Duration,

    /// Maximum number of memories to keep in cache
    pub cache_size: usize,

    /// Enable automatic HDC encoding
    pub auto_hdc_encoding: bool,

    /// Maximum search results to return
    pub max_search_results: usize,

    /// Enable background consolidation
    pub enable_background_consolidation: bool,

    /// Enable background forgetting
    pub enable_background_forgetting: bool,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            enable_conflict_detection: true,
            forgetting_threshold: 0.3,
            consolidation_interval: Duration::from_secs(3600), // 1 hour
            forgetting_interval: Duration::from_secs(1800),    // 30 minutes
            cache_size: 10000,
            auto_hdc_encoding: true,
            max_search_results: 100,
            enable_background_consolidation: false,
            enable_background_forgetting: false,
        }
    }
}

impl ManagerConfig {
    /// Create a new configuration with custom values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set conflict detection enabled
    pub fn with_conflict_detection(mut self, enabled: bool) -> Self {
        self.enable_conflict_detection = enabled;
        self
    }

    /// Set forgetting threshold
    pub fn with_forgetting_threshold(mut self, threshold: f32) -> Self {
        self.forgetting_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set consolidation interval
    pub fn with_consolidation_interval(mut self, interval: Duration) -> Self {
        self.consolidation_interval = interval;
        self
    }

    /// Set forgetting interval
    pub fn with_forgetting_interval(mut self, interval: Duration) -> Self {
        self.forgetting_interval = interval;
        self
    }

    /// Set cache size
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Enable or disable auto HDC encoding
    pub fn with_auto_hdc_encoding(mut self, enabled: bool) -> Self {
        self.auto_hdc_encoding = enabled;
        self
    }

    /// Set maximum search results
    pub fn with_max_search_results(mut self, max: usize) -> Self {
        self.max_search_results = max;
        self
    }

    /// Enable background consolidation
    pub fn with_background_consolidation(mut self, enabled: bool) -> Self {
        self.enable_background_consolidation = enabled;
        self
    }

    /// Enable background forgetting
    pub fn with_background_forgetting(mut self, enabled: bool) -> Self {
        self.enable_background_forgetting = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ManagerConfig::default();
        assert!(config.enable_conflict_detection);
        assert_eq!(config.forgetting_threshold, 0.3);
        assert!(config.auto_hdc_encoding);
    }

    #[test]
    fn test_config_builder() {
        let config = ManagerConfig::new()
            .with_conflict_detection(false)
            .with_forgetting_threshold(0.5)
            .with_cache_size(5000);

        assert!(!config.enable_conflict_detection);
        assert_eq!(config.forgetting_threshold, 0.5);
        assert_eq!(config.cache_size, 5000);
    }

    #[test]
    fn test_threshold_clamping() {
        let config1 = ManagerConfig::new().with_forgetting_threshold(1.5);
        assert_eq!(config1.forgetting_threshold, 1.0);

        let config2 = ManagerConfig::new().with_forgetting_threshold(-0.5);
        assert_eq!(config2.forgetting_threshold, 0.0);
    }
}
