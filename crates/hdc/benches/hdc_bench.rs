//! HDC performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cmd_hdc::{HyperVector, operations, simd};

fn bench_hypervector_creation(c: &mut Criterion) {
    c.bench_function("hypervector_random_10k", |b| {
        b.iter(|| {
            black_box(HyperVector::random(10_000))
        });
    });

    c.bench_function("hypervector_random_seeded_10k", |b| {
        b.iter(|| {
            black_box(HyperVector::random_seeded(10_000, 42))
        });
    });
}

fn bench_basic_operations(c: &mut Criterion) {
    let hv1 = HyperVector::random_seeded(10_000, 1);
    let hv2 = HyperVector::random_seeded(10_000, 2);

    c.bench_function("xor_10k", |b| {
        b.iter(|| {
            black_box(hv1.xor(&hv2))
        });
    });

    let hvs = vec![
        HyperVector::random_seeded(10_000, 1),
        HyperVector::random_seeded(10_000, 2),
        HyperVector::random_seeded(10_000, 3),
    ];

    c.bench_function("majority_3x10k", |b| {
        b.iter(|| {
            black_box(HyperVector::majority(&hvs))
        });
    });

    c.bench_function("rotate_10k", |b| {
        b.iter(|| {
            black_box(hv1.rotate(100))
        });
    });
}

fn bench_bind_bundle(c: &mut Criterion) {
    let key = HyperVector::random_seeded(10_000, 1);
    let value = HyperVector::random_seeded(10_000, 2);

    c.bench_function("bind_10k", |b| {
        b.iter(|| {
            black_box(operations::bind(&key, &value))
        });
    });

    let hvs = vec![
        HyperVector::random_seeded(10_000, 1),
        HyperVector::random_seeded(10_000, 2),
        HyperVector::random_seeded(10_000, 3),
        HyperVector::random_seeded(10_000, 4),
        HyperVector::random_seeded(10_000, 5),
    ];

    c.bench_function("bundle_5x10k", |b| {
        b.iter(|| {
            black_box(operations::bundle(&hvs))
        });
    });

    c.bench_function("encode_sequence_5x10k", |b| {
        b.iter(|| {
            black_box(operations::encode_sequence(&hvs))
        });
    });
}

fn bench_similarity(c: &mut Criterion) {
    let hv1 = HyperVector::random_seeded(10_000, 1);
    let hv2 = HyperVector::random_seeded(10_000, 2);

    c.bench_function("hamming_distance_10k", |b| {
        b.iter(|| {
            black_box(hv1.hamming_distance(&hv2))
        });
    });

    c.bench_function("similarity_10k", |b| {
        b.iter(|| {
            black_box(hv1.similarity(&hv2))
        });
    });
}

fn bench_simd_operations(c: &mut Criterion) {
    let query = HyperVector::random_seeded(10_000, 1);
    let other = HyperVector::random_seeded(10_000, 2);

    c.bench_function("simd_hamming_distance_10k", |b| {
        b.iter(|| {
            black_box(simd::hamming_distance_simd(&query, &other))
        });
    });

    c.bench_function("simd_similarity_10k", |b| {
        b.iter(|| {
            black_box(simd::similarity_simd(&query, &other))
        });
    });
}

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_similarity");

    for size in [10, 100, 1000] {
        let query = HyperVector::random_seeded(10_000, 42);
        let candidates: Vec<_> = (0..size)
            .map(|i| HyperVector::random_seeded(10_000, i))
            .collect();

        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |b, _| {
            b.iter(|| {
                black_box(simd::batch_similarities(&query, &candidates))
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |b, _| {
            b.iter(|| {
                black_box(simd::parallel_batch_similarities(&query, &candidates))
            });
        });
    }

    group.finish();
}

fn bench_encoders(c: &mut Criterion) {
    use cmd_hdc::encoder::{ScalarEncoder, SymbolEncoder, SequenceEncoder};

    let scalar_encoder = ScalarEncoder::new(10_000, 0.0, 100.0, 100);

    c.bench_function("scalar_encode", |b| {
        b.iter(|| {
            black_box(scalar_encoder.encode(42.5))
        });
    });

    let mut symbol_encoder = SymbolEncoder::new(10_000);
    symbol_encoder.add_symbols(&["apple", "banana", "cherry"]);

    c.bench_function("symbol_encode", |b| {
        b.iter(|| {
            black_box(symbol_encoder.encode("apple"))
        });
    });

    let mut seq_encoder = SequenceEncoder::new(10_000);

    c.bench_function("sequence_encode_5_tokens", |b| {
        b.iter(|| {
            black_box(seq_encoder.encode(&["the", "quick", "brown", "fox", "jumps"]))
        });
    });
}

fn bench_cleanup(c: &mut Criterion) {
    let codebook: Vec<_> = (0..100)
        .map(|i| HyperVector::random_seeded(10_000, i))
        .collect();

    let mut noisy = codebook[0].clone();
    for i in 0..500 {
        noisy.flip(i);
    }

    c.bench_function("cleanup_100_codebook", |b| {
        b.iter(|| {
            black_box(operations::cleanup(&noisy, &codebook))
        });
    });
}

criterion_group!(
    benches,
    bench_hypervector_creation,
    bench_basic_operations,
    bench_bind_bundle,
    bench_similarity,
    bench_simd_operations,
    bench_batch_operations,
    bench_encoders,
    bench_cleanup
);

criterion_main!(benches);
