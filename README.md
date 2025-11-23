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
- **Emotional memory**: PAD model (Pleasure-Arousal-Dominance) for affective context
- **Prospective memory**: Goals and intentions with trigger conditions
- **Type system**: Strong typing for IDs and entities
- **Tests**: 26/26 passing ✅

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

#### Storage Module (`cmd-storage`) - v0.1.0
- **Modular architecture**: Trait-based design for episodic and semantic storage
- **Episodic storage**: LanceDB/Parquet for vector search (optional feature)
- **Semantic storage**: KuzuDB for graph relationships (optional feature)
- **In-memory storage**: HashMap-based implementation for testing
- **Async operations**: Full async/await support with tokio
- **Vector search**: Cosine similarity and HDC Hamming distance
- **Graph queries**: Neighbor traversal with depth and relationship filtering
- **Tests**: 8/8 passing ✅

#### Memory Manager (`cmd-manager`) - v0.1.0
- **Unified interface**: Integrates storage, search, and conflict resolution
- **Lifecycle management**: Add, retrieve, update, delete, search operations
- **Auto HDC encoding**: Automatic hyperdimensional vector encoding for text
- **Forgetting process**: Removes weak memories below retention threshold
- **Consolidation engine**: 4 strategies (access frequency, retention, clustering, temporal)
- **Emotional operations**: Search by emotional valence, PAD similarity, and emotion updates
- **Prospective memory**: Manage goals/intentions with trigger conditions and completion tracking
- **Statistics tracking**: Real-time metrics for operations, performance, and emotional states
- **Configurable**: Builder pattern for fine-grained control
- **Tests**: 21/21 passing ✅

### 🚧 In Development

#### Background Processes
- Automatic consolidation scheduler
- Periodic forgetting process
- Memory migration and archival

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

// Memory Manager - unified interface (async)
use cmd_manager::{MemoryManager, ManagerConfig};
use cmd_storage::memory::{InMemoryEpisodicStorage, InMemorySemanticStorage};

let episodic = InMemoryEpisodicStorage::new();
let semantic = InMemorySemanticStorage::new();
let config = ManagerConfig::default()
    .with_forgetting_threshold(0.3)
    .with_auto_hdc_encoding(true);

let manager = MemoryManager::new(episodic, semantic, config);

// Add memories with automatic HDC encoding
let memory_id = manager.add_memory(memory).await?;

// Search with HDC similarity
let results = manager.search("user preferences", 10).await?;

// Get statistics
let stats = manager.get_stats();
println!("Total memories: {}", stats.total_memories);
println!("Total searches: {}", stats.total_searches);

// Run forgetting process
let forgotten = manager.forget_weak_memories(0.3).await?;

// === Emotional Memory ===
use cmd_core::memory::PADVector;

// Create memory with emotional context
let emotion = PADVector::new(0.7, 0.4, 0.3); // Happy (pleasure, arousal, dominance)
let emotional_memory = MemoryUnit::new_text_with_emotion(
    "User is excited about the project".to_string(),
    source,
    emotion,
);
manager.add_memory(emotional_memory).await?;

// Search by emotional valence
let positive_memories = manager.search_by_emotion(EmotionalValence::Positive, 10).await?;

// Search by emotional similarity
let target_emotion = PADVector::new(0.8, 0.5, 0.4);
let similar = manager.search_by_emotional_similarity(target_emotion, 10, 0.5).await?;

// Get emotional statistics
let emotional_stats = manager.get_emotional_stats().await?;
println!("Positive: {}, Negative: {}, Average intensity: {:.2}",
         emotional_stats.positive_count,
         emotional_stats.negative_count,
         emotional_stats.average_intensity);

// === Prospective Memory (Goals/Intentions) ===
use cmd_core::memory::TriggerCondition;

// Create a goal with trigger condition
let goal = MemoryUnit::new_intention(
    "Implement cognitive architecture".to_string(),
    TriggerCondition::TimeBasedAt(deadline),
    0.9, // high priority
    source,
);
let goal_id = manager.add_memory(goal).await?;

// Get active intentions
let active_intentions = manager.get_active_intentions().await?;

// Get intentions that should trigger now
let triggerable = manager.get_triggerable_intentions().await?;

// Complete an intention
manager.complete_intention(&goal_id).await?;
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
cargo test --package cmd-core --package cmd-resolver --package cmd-hdc --package cmd-search --package cmd-storage --package cmd-manager

# Run HDC benchmarks (requires nightly)
cargo +nightly bench --package cmd-hdc

# Run specific test suite
cargo test --package cmd-hdc operations

# Run with output
cargo test -- --nocapture

# Build with optional storage features
cargo build --package cmd-storage --features full  # Requires protoc and kuzu
```

### Test Coverage

- **cmd-core**: 26 tests covering memory, CRDT, retention, emotional (PAD), and prospective memory
- **cmd-resolver**: 8 tests covering conflict resolution and trust management
- **cmd-hdc**: 26 tests covering vectors, operations, encoders, similarity, and SIMD
- **cmd-search**: 5 tests covering HDC-based indexing and search modes
- **cmd-storage**: 8 tests covering episodic storage, semantic graphs, and vector search
- **cmd-manager**: 21 tests covering lifecycle, search, consolidation, emotional operations, and intentions
- **Total**: 94 tests, 100% passing ✅

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
- [x] Storage layer architecture (episodic + semantic)
- [x] In-memory storage implementation
- [x] Memory manager (unified interface)
- [x] Consolidation engine (4 strategies)
- [x] Forgetting process
- [x] Emotional memory (PAD model)
- [x] Prospective memory (goals/intentions)
- [x] Emotional search and statistics
- [ ] Vector storage integration (LanceDB) - optional feature
- [ ] Graph database integration (KuzuDB) - optional feature
- [ ] Background consolidation scheduler
- [ ] REST API server
- [ ] Python SDK
- [ ] End-to-end benchmarks
- [ ] Cloud deployment guide

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/dimaskynet/Project-Morpheus-Cognitive-Memory-Database/issues)
- Discussions: [Join the conversation](https://github.com/dimaskynet/Project-Morpheus-Cognitive-Memory-Database/discussions)

---

*"The mind is not a vessel to be filled, but a fire to be kindled." - Plutarch*