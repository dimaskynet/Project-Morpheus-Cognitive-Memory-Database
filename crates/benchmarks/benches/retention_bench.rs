//! Benchmarks for retention model and memory operations

use cmd_core::retention::RetentionModel;
use cmd_core::memory::{MemoryUnit, Modality};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use chrono::{Utc, Duration};

/// Benchmark retention strength calculations
fn bench_retention_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("RetentionModel");

    group.bench_function("strength_calculation", |b| {
        let model = RetentionModel::new(0.9);
        let now = Utc::now();

        b.iter(|| {
            black_box(model.retention_strength(now))
        });
    });

    group.bench_function("recall_update", |b| {
        let mut model = RetentionModel::new(0.8);

        b.iter(|| {
            model.record_recall(true);
            black_box(&model);
        });
    });

    group.bench_function("forget_time_prediction", |b| {
        let model = RetentionModel::new(0.85);

        b.iter(|| {
            black_box(model.predict_forget_time(0.3))
        });
    });

    // Benchmark retention over time periods
    for days in [1, 7, 30, 365].iter() {
        group.bench_with_input(
            BenchmarkId::new("strength_after_days", days),
            days,
            |b, &days| {
                let model = RetentionModel::new(0.9);
                let future = Utc::now() + Duration::days(days);

                b.iter(|| {
                    black_box(model.retention_strength(future))
                });
            },
        );
    }

    // Benchmark stability adjustments
    group.bench_function("stability_boost", |b| {
        let mut model = RetentionModel::new(0.85);

        b.iter(|| {
            model.record_recall(true);
            let new_stability = model.stability;
            black_box(new_stability);
        });
    });

    group.bench_function("stability_decay", |b| {
        let mut model = RetentionModel::new(0.85);

        b.iter(|| {
            model.record_recall(false);
            let new_stability = model.stability;
            black_box(new_stability);
        });
    });

    group.finish();
}

/// Benchmark memory unit operations
fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("MemoryUnit");

    // Benchmark memory creation
    group.bench_function("create_text_memory", |b| {
        let content = b"This is a test memory content that represents typical user interaction";
        let embeddings = vec![0.1f32; 1536]; // Typical embedding size

        b.iter(|| {
            black_box(MemoryUnit::new(
                Modality::Text,
                content.to_vec(),
                embeddings.clone(),
            ))
        });
    });

    group.bench_function("create_multimodal_memory", |b| {
        let text_content = b"Image description";
        let image_data = vec![0u8; 100_000]; // 100KB image
        let embeddings = vec![0.1f32; 1536];

        b.iter(|| {
            let mut memory = MemoryUnit::new(
                Modality::Image,
                image_data.clone(),
                embeddings.clone(),
            );
            memory.raw_content = text_content.to_vec();
            black_box(memory)
        });
    });

    // Benchmark memory access
    group.bench_function("access_memory", |b| {
        let mut memory = create_test_memory();

        b.iter(|| {
            memory.record_access();
            black_box(&memory);
        });
    });

    // Benchmark retention strength over multiple memories
    for num_memories in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_retention_check", num_memories),
            num_memories,
            |b, &num_memories| {
                let memories: Vec<MemoryUnit> = (0..num_memories)
                    .map(|i| {
                        let mut m = create_test_memory();
                        m.retention = RetentionModel::new(0.5 + (i as f32 / num_memories as f32) * 0.5);
                        m
                    })
                    .collect();

                b.iter(|| {
                    let mut count = 0;
                    for memory in &memories {
                        if memory.retention_strength() > 0.5 {
                            count += 1;
                        }
                    }
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

/// Compare memory efficiency with traditional approaches
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("MemoryEfficiency");

    // Traditional approach: store everything
    group.bench_function("traditional_full_storage", |b| {
        b.iter(|| {
            let mut memories = Vec::new();
            for i in 0..1000 {
                let content = format!("Memory content {} with lots of redundant information", i);
                memories.push(TraditionalMemory {
                    content,
                    embedding: vec![0.1f32; 1536],
                    timestamp: Utc::now(),
                    metadata: std::collections::HashMap::new(),
                });
            }
            black_box(memories)
        });
    });

    // CMD approach: with retention and forgetting
    group.bench_function("cmd_with_retention", |b| {
        b.iter(|| {
            let mut memories = Vec::new();
            for i in 0..1000 {
                let mut memory = create_test_memory();
                memory.retention = RetentionModel::new(0.3 + (i as f32 / 1000.0) * 0.7);

                // Only keep memories above threshold
                if memory.retention_strength() > 0.4 {
                    memories.push(memory);
                }
            }
            black_box(memories)
        });
    });

    // Memory size comparison
    group.bench_function("size_traditional", |b| {
        b.iter(|| {
            let memory = TraditionalMemory {
                content: "x".repeat(1000),
                embedding: vec![0.1f32; 1536],
                timestamp: Utc::now(),
                metadata: std::collections::HashMap::new(),
            };
            let size = std::mem::size_of_val(&memory)
                + memory.content.len()
                + memory.embedding.len() * std::mem::size_of::<f32>();
            black_box(size)
        });
    });

    group.bench_function("size_cmd", |b| {
        b.iter(|| {
            let memory = create_test_memory();
            let size = std::mem::size_of_val(&memory)
                + memory.raw_content.len()
                + memory.embeddings.text.len() * std::mem::size_of::<f32>();
            black_box(size)
        });
    });

    group.finish();
}

/// Benchmark forgetting curve accuracy
fn bench_forgetting_curve(c: &mut Criterion) {
    let mut group = c.benchmark_group("ForgettingCurve");

    // Test different stability values
    for stability in [0.1, 0.5, 1.0, 5.0, 10.0].iter() {
        group.bench_with_input(
            BenchmarkId::new("ebbinghaus_curve", stability),
            stability,
            |b, &stability| {
                let model = RetentionModel {
                    base_retention: 1.0,
                    stability,
                    last_recall: Utc::now(),
                    recall_count: 0,
                };

                let test_times: Vec<_> = (0..100)
                    .map(|i| Utc::now() + Duration::hours(i))
                    .collect();

                b.iter(|| {
                    let strengths: Vec<f32> = test_times
                        .iter()
                        .map(|&t| model.retention_strength(t))
                        .collect();
                    black_box(strengths)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark spaced repetition scheduling
fn bench_spaced_repetition(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpacedRepetition");

    group.bench_function("optimal_review_time", |b| {
        let model = RetentionModel::new(0.85);

        b.iter(|| {
            // Find optimal review time (when strength drops to 0.8)
            let target_strength = 0.8;
            let review_time = model.predict_forget_time(target_strength);
            black_box(review_time)
        });
    });

    group.bench_function("batch_scheduling", |b| {
        let memories: Vec<RetentionModel> = (0..1000)
            .map(|i| RetentionModel::new(0.5 + (i as f32 / 2000.0)))
            .collect();

        b.iter(|| {
            let mut schedule = Vec::new();
            for model in &memories {
                if let Some(time) = model.predict_forget_time(0.7) {
                    schedule.push(time);
                }
            }
            black_box(schedule)
        });
    });

    group.finish();
}

// Helper functions

fn create_test_memory() -> MemoryUnit {
    MemoryUnit::new(
        Modality::Text,
        b"Test memory content".to_vec(),
        vec![0.1f32; 512],
    )
}

#[derive(Clone)]
struct TraditionalMemory {
    content: String,
    embedding: Vec<f32>,
    timestamp: chrono::DateTime<Utc>,
    metadata: std::collections::HashMap<String, String>,
}

criterion_group!(
    benches,
    bench_retention_calculations,
    bench_memory_operations,
    bench_memory_efficiency,
    bench_forgetting_curve,
    bench_spaced_repetition
);
criterion_main!(benches);