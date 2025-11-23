//! Benchmarks for conflict resolution

use cmd_resolver::deterministic::DeterministicResolver;
use cmd_resolver::strategies::SourceTrustManager;
use cmd_resolver::types::{Conflict, ConflictType, ResolutionContext, ResolutionPrecedent};
use cmd_core::crdt::{FactVersion, VectorClock};
use cmd_core::types::SourceId;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use uuid::Uuid;

/// Benchmark different resolution strategies
fn bench_resolution_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("ResolutionStrategies");

    let mut resolver = DeterministicResolver::new();

    // Benchmark each strategy type
    group.bench_function("keep_newest", |b| {
        let conflict = create_temporal_conflict();
        b.iter(|| {
            black_box(resolver.resolve(conflict.clone()))
        });
    });

    group.bench_function("merge_crdt", |b| {
        let conflict = create_concurrent_conflict();
        b.iter(|| {
            black_box(resolver.resolve(conflict.clone()))
        });
    });

    group.bench_function("trust_source", |b| {
        let conflict = create_trust_conflict();
        b.iter(|| {
            black_box(resolver.resolve(conflict.clone()))
        });
    });

    group.bench_function("consensus_voting", |b| {
        let conflict = create_consensus_conflict();
        b.iter(|| {
            black_box(resolver.resolve(conflict.clone()))
        });
    });

    group.bench_function("logical_consistency", |b| {
        let conflict = create_logical_conflict();
        b.iter(|| {
            black_box(resolver.resolve(conflict.clone()))
        });
    });

    group.finish();
}

/// Benchmark trust management operations
fn bench_trust_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("TrustManagement");

    for num_sources in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("update_trust", num_sources),
            num_sources,
            |b, &num_sources| {
                let mut manager = SourceTrustManager::new();
                let sources: Vec<SourceId> = (0..num_sources)
                    .map(|_| SourceId::new())
                    .collect();

                b.iter(|| {
                    for source in &sources {
                        manager.update_trust(source, true, 0.85);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("consensus_trust", num_sources),
            num_sources,
            |b, &num_sources| {
                let mut manager = SourceTrustManager::new();
                let sources: Vec<SourceId> = (0..num_sources)
                    .map(|_| SourceId::new())
                    .collect();

                // Initialize with different trust levels
                for (i, source) in sources.iter().enumerate() {
                    let trust = 0.3 + (i as f32 / num_sources as f32) * 0.6;
                    manager.update_trust(source, true, trust);
                }

                b.iter(|| {
                    black_box(manager.consensus_trust(&sources))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ranked_sources", num_sources),
            num_sources,
            |b, &num_sources| {
                let mut manager = SourceTrustManager::new();

                for i in 0..num_sources {
                    let source = SourceId::new();
                    manager.update_trust(&source, i % 2 == 0, 0.5 + (i as f32 / num_sources as f32) * 0.4);
                }

                b.iter(|| {
                    black_box(manager.get_ranked_sources())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark resolution at scale
fn bench_resolution_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("ResolutionScale");

    for num_conflicts in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_resolve", num_conflicts),
            num_conflicts,
            |b, &num_conflicts| {
                let mut resolver = DeterministicResolver::new();
                let conflicts: Vec<Conflict> = (0..num_conflicts)
                    .map(|i| {
                        match i % 5 {
                            0 => create_temporal_conflict(),
                            1 => create_concurrent_conflict(),
                            2 => create_trust_conflict(),
                            3 => create_consensus_conflict(),
                            _ => create_logical_conflict(),
                        }
                    })
                    .collect();

                b.iter(|| {
                    for conflict in &conflicts {
                        black_box(resolver.resolve(conflict.clone()));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Compare with traditional RAG approach
fn bench_vs_traditional_rag(c: &mut Criterion) {
    let mut group = c.benchmark_group("CMD_vs_RAG");

    // Simulate traditional RAG conflict resolution (always calls LLM)
    group.bench_function("rag_with_llm_call", |b| {
        b.iter(|| {
            // Simulate LLM call latency (mock)
            std::thread::sleep(std::time::Duration::from_micros(100));
            black_box(0.85f32) // Return confidence
        });
    });

    // Our deterministic approach
    group.bench_function("cmd_deterministic", |b| {
        let mut resolver = DeterministicResolver::new();
        let conflict = create_temporal_conflict();

        b.iter(|| {
            black_box(resolver.resolve(conflict.clone()))
        });
    });

    // Memory comparison
    group.bench_function("memory_rag_chunks", |b| {
        b.iter(|| {
            // Traditional RAG: store raw chunks with embeddings
            let mut chunks = Vec::new();
            for i in 0..1000 {
                let chunk = RagChunk {
                    text: format!("Chunk content {}", i),
                    embedding: vec![0.1f32; 1536], // OpenAI embedding size
                    metadata: HashMap::new(),
                };
                chunks.push(chunk);
            }
            black_box(chunks)
        });
    });

    group.bench_function("memory_cmd_facts", |b| {
        b.iter(|| {
            // CMD: structured facts with CRDT
            let mut facts = Vec::new();
            for i in 0..1000 {
                facts.push(create_test_fact(i));
            }
            black_box(facts)
        });
    });

    group.finish();
}

/// Benchmark conflict detection
fn bench_conflict_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("ConflictDetection");

    group.bench_function("detect_temporal", |b| {
        let fact1 = create_test_fact(0);
        let fact2 = create_test_fact(1);

        b.iter(|| {
            let is_conflict = fact1.content.subject == fact2.content.subject
                && fact1.content.predicate == fact2.content.predicate
                && fact1.metadata.created_at != fact2.metadata.created_at;
            black_box(is_conflict)
        });
    });

    group.bench_function("detect_contradiction", |b| {
        let fact1 = create_test_fact(0);
        let fact2 = create_test_fact(1);

        b.iter(|| {
            // Check for semantic contradiction
            let is_contradiction = fact1.content.subject == fact2.content.subject
                && fact1.content.predicate == fact2.content.predicate
                && fact1.content.object != fact2.content.object;
            black_box(is_contradiction)
        });
    });

    group.finish();
}

// Helper functions for creating test data

fn create_temporal_conflict() -> Conflict {
    Conflict {
        id: Uuid::new_v4(),
        conflict_type: ConflictType::TemporalSupersession,
        existing_fact: create_test_fact(0),
        new_fact: create_test_fact(1),
        context: ResolutionContext {
            related_facts: vec![],
            precedents: vec![],
            domain_rules: None,
        },
    }
}

fn create_concurrent_conflict() -> Conflict {
    let mut fact1 = create_test_fact(0);
    let mut fact2 = create_test_fact(1);

    fact1.vector_clock.increment("node1");
    fact2.vector_clock.increment("node2");

    Conflict {
        id: Uuid::new_v4(),
        conflict_type: ConflictType::ConcurrentUpdate,
        existing_fact: fact1,
        new_fact: fact2,
        context: ResolutionContext::default(),
    }
}

fn create_trust_conflict() -> Conflict {
    let mut fact1 = create_test_fact(0);
    let mut fact2 = create_test_fact(1);

    fact1.sources = vec![SourceId::new(); 2];
    fact2.sources = vec![SourceId::new(); 5];

    Conflict {
        id: Uuid::new_v4(),
        conflict_type: ConflictType::SourceDisagreement,
        existing_fact: fact1,
        new_fact: fact2,
        context: ResolutionContext::default(),
    }
}

fn create_consensus_conflict() -> Conflict {
    let mut fact1 = create_test_fact(0);
    let mut fact2 = create_test_fact(1);

    fact1.sources = (0..3).map(|_| SourceId::new()).collect();
    fact2.sources = (0..7).map(|_| SourceId::new()).collect();

    Conflict {
        id: Uuid::new_v4(),
        conflict_type: ConflictType::MultipleVersions,
        existing_fact: fact1,
        new_fact: fact2,
        context: ResolutionContext::default(),
    }
}

fn create_logical_conflict() -> Conflict {
    let mut fact1 = create_test_fact(0);
    let mut fact2 = create_test_fact(1);

    fact1.content.predicate = "is_parent_of".to_string();
    fact2.content.predicate = "is_child_of".to_string();

    Conflict {
        id: Uuid::new_v4(),
        conflict_type: ConflictType::LogicalContradiction,
        existing_fact: fact1,
        new_fact: fact2,
        context: ResolutionContext {
            related_facts: vec![create_test_fact(2), create_test_fact(3)],
            precedents: vec![],
            domain_rules: None,
        },
    }
}

fn create_test_fact(index: usize) -> FactVersion {
    FactVersion {
        id: Uuid::new_v4(),
        content: cmd_core::crdt::Triple {
            subject: format!("subject_{}", index),
            predicate: "has_value".to_string(),
            object: serde_json::json!({"value": index}),
        },
        confidence: 0.85 + (index as f32 * 0.01),
        sources: vec![SourceId::new()],
        vector_clock: VectorClock::new(),
        metadata: cmd_core::crdt::FactMetadata {
            created_at: chrono::Utc::now() + chrono::Duration::seconds(index as i64),
            updated_at: chrono::Utc::now(),
            access_count: 0,
            importance: 0.5,
        },
    }
}

// Mock RAG chunk for comparison
#[derive(Clone)]
struct RagChunk {
    text: String,
    embedding: Vec<f32>,
    metadata: HashMap<String, String>,
}

use std::collections::HashMap;

criterion_group!(
    benches,
    bench_resolution_strategies,
    bench_trust_management,
    bench_resolution_scale,
    bench_vs_traditional_rag,
    bench_conflict_detection
);
criterion_main!(benches);