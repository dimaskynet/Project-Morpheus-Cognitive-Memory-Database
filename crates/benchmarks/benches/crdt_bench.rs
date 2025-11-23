//! Benchmarks for CRDT operations

use cmd_core::crdt::{FactVersion, VectorClock, LWWElementSet};
use cmd_core::types::{SourceId, MemoryId};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::collections::HashMap;
use uuid::Uuid;

/// Benchmark VectorClock merge operations
fn bench_vector_clock_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("VectorClock");

    for size in [1, 10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("merge", size),
            size,
            |b, &size| {
                // Create two vector clocks with different nodes
                let mut clock1 = VectorClock::new();
                let mut clock2 = VectorClock::new();

                for i in 0..size {
                    let node_id = format!("node_{}", i);
                    clock1.increment(&node_id);
                    if i % 2 == 0 {
                        clock2.increment(&node_id);
                        clock2.increment(&node_id);
                    }
                }

                b.iter(|| {
                    black_box(clock1.merge(&clock2))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark FactVersion merge operations
fn bench_fact_version_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("FactVersion");

    // Test different conflict scenarios
    group.bench_function("merge_no_conflict", |b| {
        let fact1 = create_test_fact(1.0);
        let fact2 = create_test_fact(0.8);

        b.iter(|| {
            black_box(fact1.merge(&fact2))
        });
    });

    group.bench_function("merge_concurrent", |b| {
        let mut fact1 = create_test_fact(1.0);
        let mut fact2 = create_test_fact(0.9);

        // Make them concurrent
        fact1.vector_clock.increment("node1");
        fact2.vector_clock.increment("node2");

        b.iter(|| {
            black_box(fact1.merge(&fact2))
        });
    });

    group.bench_function("merge_complex", |b| {
        let mut fact1 = create_test_fact(0.95);
        let mut fact2 = create_test_fact(0.85);

        // Add multiple sources and complex vector clocks
        for i in 0..10 {
            fact1.sources.push(SourceId::new());
            fact2.sources.push(SourceId::new());
            fact1.vector_clock.increment(&format!("node_{}", i));
            fact2.vector_clock.increment(&format!("node_{}", i * 2));
        }

        b.iter(|| {
            black_box(fact1.merge(&fact2))
        });
    });

    group.finish();
}

/// Benchmark LWWElementSet operations
fn bench_lww_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("LWWElementSet");

    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("add", size),
            size,
            |b, &size| {
                let mut set = LWWElementSet::<String>::new();
                let elements: Vec<String> = (0..size).map(|i| format!("element_{}", i)).collect();

                b.iter(|| {
                    for elem in &elements {
                        set.add(elem.clone());
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("merge", size),
            size,
            |b, &size| {
                let mut set1 = LWWElementSet::<String>::new();
                let mut set2 = LWWElementSet::<String>::new();

                for i in 0..size {
                    let elem = format!("element_{}", i);
                    if i % 2 == 0 {
                        set1.add(elem);
                    } else {
                        set2.add(elem);
                    }
                }

                b.iter(|| {
                    black_box(set1.merge(&set2))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("contains", size),
            size,
            |b, &size| {
                let mut set = LWWElementSet::<String>::new();
                for i in 0..size {
                    set.add(format!("element_{}", i));
                }
                let search_elem = format!("element_{}", size / 2);

                b.iter(|| {
                    black_box(set.contains(&search_elem))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory overhead comparison
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("MemoryOverhead");

    group.bench_function("crdt_vs_naive", |b| {
        b.iter(|| {
            // CRDT version with full tracking
            let mut facts_crdt: Vec<FactVersion> = Vec::new();
            for i in 0..1000 {
                let mut fact = create_test_fact(0.9);
                fact.vector_clock.increment(&format!("node_{}", i % 10));
                facts_crdt.push(fact);
            }

            // Calculate memory usage
            let crdt_size = std::mem::size_of_val(&facts_crdt)
                + facts_crdt.capacity() * std::mem::size_of::<FactVersion>();

            black_box(crdt_size)
        });
    });

    group.finish();
}

/// Helper function to create test facts
fn create_test_fact(confidence: f32) -> FactVersion {
    FactVersion {
        id: Uuid::new_v4(),
        content: cmd_core::crdt::Triple {
            subject: "test_subject".to_string(),
            predicate: "test_predicate".to_string(),
            object: serde_json::json!("test_object"),
        },
        confidence,
        sources: vec![SourceId::new()],
        vector_clock: VectorClock::new(),
        metadata: cmd_core::crdt::FactMetadata {
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            access_count: 0,
            importance: 0.5,
        },
    }
}

criterion_group!(
    benches,
    bench_vector_clock_merge,
    bench_fact_version_merge,
    bench_lww_set,
    bench_memory_overhead
);
criterion_main!(benches);