# Project Morpheus: Cognitive Memory Database

A next-generation memory system for Artificial General Intelligence (AGI) that models human cognitive processes.

## Overview

Cognitive Memory Database (CMD) is a revolutionary approach to AGI memory that combines:
- 🧠 **Biologically-inspired architecture** modeling hippocampus and neocortex
- 🔄 **Active memory consolidation** with mathematical forgetting models
- 🎯 **Hyperdimensional computing** for efficient structural search
- ⚡ **Production-ready performance** with Rust core and deterministic conflict resolution

## Key Features

- **Hybrid Storage**: Combines vector embeddings, knowledge graphs, and HDC representations
- **Active Forgetting**: Implements Ebbinghaus forgetting curve with dynamic stability
- **Deterministic Conflicts**: 95% of conflicts resolved without LLM calls
- **CRDT-based Versioning**: Automatic merge of concurrent updates
- **Native Performance**: Core engine in Rust with SIMD optimizations
- **Trust Management**: Dynamic source credibility scoring and consensus mechanisms

## 📊 Current Implementation Status

### ✅ Completed Modules

#### Core Module (`cmd-core`) - v0.1.0
- **Memory structures**: MemoryUnit with multimodal support
- **CRDT operations**: VectorClock, FactVersion, LWWElementSet for conflict-free merging
- **Retention model**: Mathematical forgetting curve with adaptive parameters
- **Type system**: Strong typing for IDs and entities
- **Tests**: 11/11 passing ✅

#### Resolver Module (`cmd-resolver`) - v0.1.0
- **Deterministic resolver**: 8 conflict resolution strategies
- **Trust management**: SourceTrustManager with performance tracking
- **Domain rules**: Customizable conflict resolution per domain
- **Statistics**: Real-time performance metrics
- **Tests**: 8/8 passing ✅

#### HDC Module (`cmd-hdc`) - v0.1.0
- **Hyperdimensional computing**: 10,000-bit binary vectors for structural representation
- **SIMD optimizations**: AVX2/SSE4.2 acceleration (29x speedup on x86_64)
- **HDC operations**: Bind (XOR), Bundle (majority), Permute (rotation)
- **Encoders**: Scalar, Symbol, Sequence, and Map encoders
- **Performance**: Hamming distance in 65ns, similarity in 66ns (SIMD)
- **Tests**: 26/26 passing ✅
- **Benchmarks**: Complete performance suite with Criterion

#### Search Module (`cmd-search`) - v0.1.0
- **HDC-based indexing**: Fast structural search using hyperdimensional vectors
- **Multiple search modes**: K-nearest neighbors, threshold-based, temporal filtering
- **Weighted ranking**: Combines similarity (70%) and retention score (30%)
- **Batch operations**: Parallel SIMD similarity computations
- **Cache optimization**: DashMap-based similarity caching
- **Tests**: 5/5 passing ✅

### 🚧 In Development

#### Storage Module (`cmd-storage`)
- LanceDB integration for vector storage (requires protobuf-compiler)
- KuzuDB integration for graph database
- Hybrid query optimization

### 📝 Planned

- REST API server (`cmd-api`)
- Python SDK via PyO3
- Memory consolidation engine
- Comprehensive benchmarks

## Architecture

The system uses a three-layer PAD (Persistence, Active consolidation, Decay) model:

```
┌─────────────────────────────────────────────┐
│         Application Layer                   │
│       (REST API / Python SDK)               │
├─────────────────────────────────────────────┤
│         Core Engine                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Memory   │  │ HDC      │  │ Conflict  │ │
│  │ Manager  │  │ Search   │  │ Resolver  │ │
│  └──────────┘  └──────────┘  └──────────┘ │
├─────────────────────────────────────────────┤
│         Storage Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ LanceDB  │  │  KuzuDB  │  │   HDC    │ │
│  │(Vectors) │  │ (Graph)  │  │ (Index)  │ │
│  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/dimaskynet/Project-Morpheus-Cognitive-Memory-Database.git
cd Project-Morpheus-Cognitive-Memory-Database

# Build core modules (Rust stable)
cargo build --package cmd-core --package cmd-resolver

# Build HDC and search modules (Rust stable or nightly for SIMD)
cargo build --package cmd-hdc --package cmd-search

# Run tests
cargo test --package cmd-core --package cmd-resolver --package cmd-hdc --package cmd-search

# Run benchmarks (requires nightly for best performance)
rustup toolchain install nightly
cargo +nightly bench --package cmd-hdc
```

## Usage Example

```rust
use cmd_core::memory::{MemoryUnit, Modality};
use cmd_core::retention::RetentionModel;
use cmd_resolver::deterministic::DeterministicResolver;
use cmd_search::HdcMemoryIndex;

// Create a memory unit
let memory = MemoryUnit::new(
    Modality::Text,
    content_bytes,
    embeddings,
);

// Apply retention model
let mut retention = RetentionModel::new(0.9);
let strength = retention.retention_strength(Utc::now());

// HDC-based search
let mut index = HdcMemoryIndex::new();
index.add_memory(memory)?;

// Search with automatic HDC encoding
let results = index.search_hdc("user preferences", 10);
for result in results {
    println!("Similarity: {:.2}, Retention: {:.2}",
             result.similarity, result.retention_score);
}

// Resolve conflicts deterministically
let mut resolver = DeterministicResolver::new();
let result = resolver.resolve(conflict)?;
println!("Resolution confidence: {:.2}%", result.confidence * 100.0);
```

## Python SDK (Coming Soon)

```python
from cognitive_memory import CMD

# Initialize
cmd = CMD(config={
    "consolidation_interval": 3600,
    "decay_alpha": 0.05,
    "hdc_dimension": 10000,
})

# Write memory
memory_id = await cmd.write(
    content="User prefers dark mode",
    metadata={"source": "chat", "confidence": 0.9}
)

# Search
results = await cmd.retrieve(
    query="What are user's preferences?",
    constraints={"top_k": 10}
)
```

## Performance

| Metric | Target | Current Status |
|--------|--------|----------------|
| HDC Hamming Distance | < 100ns | ✅ **65ns** (SIMD) |
| HDC Similarity | < 100ns | ✅ **66ns** (SIMD) |
| HDC XOR (Bind) | < 2μs | ✅ **1.73μs** |
| HDC Bundle (5 vectors) | < 200μs | ✅ **126μs** |
| Sequence Encoding (5 tokens) | < 200μs | ✅ **164μs** |
| CRDT Merge | < 1ms | ✅ ~0.1ms |
| Conflict Resolution | < 1ms | ✅ Deterministic |
| Memory Efficiency | 3x better than RAG | 🚧 Benchmarking |

**SIMD Acceleration**: 29x speedup over scalar operations on x86_64 (AVX2)

## Testing

```bash
# Run all tests
cargo test --package cmd-core --package cmd-resolver --package cmd-hdc --package cmd-search

# Run HDC benchmarks (requires nightly)
cargo +nightly bench --package cmd-hdc

# Run specific test suite
cargo test --package cmd-hdc operations

# Run with output
cargo test -- --nocapture
```

### Test Coverage

- **cmd-core**: 11 tests covering memory, CRDT, and retention models
- **cmd-resolver**: 8 tests covering conflict resolution and trust management
- **cmd-hdc**: 26 tests covering vectors, operations, encoders, similarity, and SIMD
- **cmd-search**: 5 tests covering HDC-based indexing and search modes
- **Total**: 50 tests, 100% passing ✅

## Documentation

- [Technical Specification](docs/SPECIFICATION.md) - Complete system design (710 lines)
- [Architecture Guide](docs/ARCHITECTURE.md) - System components (368 lines)
- [API Reference](docs/API.md) - Detailed API documentation (742 lines)
- [Contributing Guide](CONTRIBUTING.md) (coming soon)

## Requirements

- **Rust**: 1.75+ (stable for core/resolver, nightly for HDC)
- **Optional**: protobuf-compiler (for storage module)
- **OS**: Linux, macOS, Windows (WSL2 recommended)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 10GB+ for full dataset support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Research & Citations

This project builds upon research in:
- Complementary Learning Systems Theory (McClelland et al., 1995)
- Hyperdimensional Computing (Kanerva, 2009)
- Conflict-free Replicated Data Types (Shapiro et al., 2011)
- Spaced Repetition and Memory Consolidation (Ebbinghaus, 1885)

## Roadmap

- [x] Core memory structures and CRDT
- [x] Deterministic conflict resolver
- [x] Trust management system
- [x] Hyperdimensional computing (HDC) with SIMD
- [x] HDC-based search and indexing
- [x] Performance benchmarking suite
- [ ] Vector storage integration (LanceDB)
- [ ] Graph database integration (KuzuDB)
- [ ] Persistent storage layer
- [ ] REST API server
- [ ] Python SDK
- [ ] Memory consolidation engine
- [ ] End-to-end benchmarks
- [ ] Cloud deployment guide

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/dimaskynet/Project-Morpheus-Cognitive-Memory-Database/issues)
- Discussions: [Join the conversation](https://github.com/dimaskynet/Project-Morpheus-Cognitive-Memory-Database/discussions)

---

*"The mind is not a vessel to be filled, but a fire to be kindled." - Plutarch*