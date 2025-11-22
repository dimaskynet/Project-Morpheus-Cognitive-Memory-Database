# API Documentation - Cognitive Memory Database

## Оглавление
- [Rust Core API](#rust-core-api)
- [Python SDK](#python-sdk)
- [REST API](#rest-api)
- [Примеры использования](#примеры-использования)

## Rust Core API

### Основной трейт CognitiveMemory

```rust
use cmd_core::{MemoryId, Memory, Query, Content, Metadata, Result};

#[async_trait]
pub trait CognitiveMemory {
    /// Записать новую память в систему
    async fn write(
        &self,
        content: Content,
        metadata: Metadata,
    ) -> Result<MemoryId>;

    /// Извлечь релевантные воспоминания
    async fn retrieve(
        &self,
        query: Query,
        constraints: SearchConstraints,
    ) -> Result<Vec<Memory>>;

    /// Запустить процесс консолидации
    async fn consolidate(&self) -> Result<ConsolidationReport>;

    /// Принудительно забыть определенные воспоминания
    async fn forget(
        &self,
        criteria: ForgetCriteria,
    ) -> Result<u32>;

    /// Анализировать недавние воспоминания
    async fn reflect(
        &self,
        window: TimeWindow,
    ) -> Result<Insights>;

    /// Получить статистику системы
    async fn stats(&self) -> Result<MemoryStats>;
}
```

### Структуры данных

#### MemoryUnit
```rust
pub struct MemoryUnit {
    /// Уникальный идентификатор (UUIDv7)
    pub id: MemoryId,

    /// Тип контента
    pub modality: Modality,

    /// Сырые данные
    pub content: Vec<u8>,

    /// Текстовое представление (если применимо)
    pub text: Option<String>,

    /// Векторные представления
    pub embeddings: Embeddings,

    /// Временные метаданные
    pub temporal: TemporalMetadata,

    /// Модель удержания (забывания)
    pub retention: RetentionModel,

    /// Связи с графом знаний
    pub graph_links: Vec<GraphLink>,

    /// Информация об источнике
    pub source: SourceMetadata,

    /// Дополнительные метаданные
    pub metadata: HashMap<String, Value>,
}
```

#### Query
```rust
pub struct Query {
    /// Текст запроса
    pub text: String,

    /// Временное окно для поиска
    pub time_range: Option<TimeRange>,

    /// Минимальная уверенность
    pub min_confidence: Option<f32>,

    /// Фильтры по источникам
    pub source_filters: Vec<SourceFilter>,

    /// Тип поиска
    pub search_type: SearchType,
}

pub enum SearchType {
    /// Семантический поиск по векторам
    Semantic,

    /// Поиск по ключевым словам
    Keyword,

    /// Структурный поиск в графе
    Structural,

    /// Гибридный поиск (комбинация)
    Hybrid { weights: SearchWeights },
}
```

#### SearchConstraints
```rust
pub struct SearchConstraints {
    /// Максимальное количество результатов
    pub top_k: usize,

    /// Вес векторного поиска vs графового (0.0 - 1.0)
    pub hybrid_alpha: f32,

    /// Минимальный retention score
    pub min_retention: f32,

    /// Включить консолидированные факты
    pub include_consolidated: bool,

    /// Применить HDC фильтрацию
    pub use_hdc_filter: bool,
}

impl Default for SearchConstraints {
    fn default() -> Self {
        Self {
            top_k: 10,
            hybrid_alpha: 0.7,
            min_retention: 0.1,
            include_consolidated: true,
            use_hdc_filter: true,
        }
    }
}
```

### Методы работы с CRDT

```rust
use cmd_core::crdt::{FactVersion, VectorClock, LWWElementSet};

// Создание версионированного факта
let fact = FactVersion::new(
    FactPayload {
        subject: "user".to_string(),
        predicate: "prefers".to_string(),
        object: json!("dark_mode"),
    },
    "node1",
    0.9,  // confidence
    vec![source_id],
);

// Слияние конкурентных версий
let merged = fact1.merge(&fact2);

// Работа с векторными часами
let mut clock = VectorClock::new();
clock.increment("node1");
clock.merge(&other_clock);

// LWW-Element-Set для коллекций
let mut set = LWWElementSet::new();
set.add("item1", Utc::now());
set.remove("item1", Utc::now() + Duration::seconds(1));
```

### Модель забывания

```rust
use cmd_core::retention::RetentionModel;

// Создание модели с начальной уверенностью
let mut model = RetentionModel::new(0.9);

// Расчет текущей силы памяти
let strength = model.retention_strength(Utc::now());

// Обновление после успешного воспоминания
model.update_on_recall(true);  // Увеличивает стабильность

// Предсказание времени забывания
let forget_time = model.predict_forget_time(0.3);  // threshold
```

## Python SDK

### Установка

```bash
pip install cognitive-memory-db
```

### Инициализация

```python
from cognitive_memory import CMD, Config

# Создание экземпляра с конфигурацией
cmd = CMD(
    config=Config(
        # Параметры хранилища
        lance_path="./data/episodic",
        kuzu_path="./data/semantic",

        # Параметры консолидации
        consolidation_interval=3600,  # секунды
        consolidation_batch_size=100,

        # Параметры забывания
        decay_alpha=0.05,
        min_retention_threshold=0.1,

        # HDC параметры
        hdc_dimension=10000,
        hdc_similarity_threshold=0.7,

        # Векторизация
        embedding_model="text-embedding-3-large",
        embedding_api_key="your-api-key",
    )
)
```

### Основные операции

#### Запись памяти

```python
import asyncio
from cognitive_memory import SourceType, Modality

async def write_memory():
    # Простая запись текста
    memory_id = await cmd.write(
        content="User mentioned they prefer dark mode UI",
        metadata={
            "source": SourceType.DIRECT_USER_INPUT,
            "confidence": 0.95,
            "tags": ["preferences", "ui"],
        }
    )

    # Запись с явным указанием модальности
    memory_id = await cmd.write(
        content=image_bytes,
        modality=Modality.IMAGE,
        metadata={
            "description": "Screenshot of user's desktop",
            "source": SourceType.TOOL_OUTPUT,
        }
    )

    # Batch запись
    memories = [
        {"content": "Fact 1", "metadata": {...}},
        {"content": "Fact 2", "metadata": {...}},
    ]
    memory_ids = await cmd.write_batch(memories)

    return memory_id
```

#### Поиск памяти

```python
async def search_memories():
    # Простой семантический поиск
    results = await cmd.retrieve("What are user's UI preferences?")

    # Поиск с ограничениями
    results = await cmd.retrieve(
        query="user preferences",
        constraints={
            "top_k": 20,
            "hybrid_alpha": 0.8,  # 80% vector, 20% graph
            "min_confidence": 0.7,
            "time_range": {
                "start": datetime.now() - timedelta(days=7),
                "end": datetime.now(),
            }
        }
    )

    # Структурный поиск в графе
    results = await cmd.graph_search(
        pattern="""
        MATCH (user:Entity)-[:PREFERS]->(pref:Concept)
        WHERE pref.category = 'ui'
        RETURN user, pref
        """,
        limit=10
    )

    # HDC-ускоренный поиск подграфов
    results = await cmd.find_similar_structures(
        reference_subgraph=subgraph,
        similarity_threshold=0.8
    )

    for result in results:
        print(f"Score: {result.score}")
        print(f"Content: {result.content}")
        print(f"Retention: {result.retention_strength}")
```

#### Консолидация и рефлексия

```python
async def consolidate_and_reflect():
    # Запуск консолидации
    report = await cmd.consolidate()
    print(f"Processed: {report.episodes_processed}")
    print(f"Facts created: {report.facts_created}")
    print(f"Conflicts resolved: {report.conflicts_resolved}")

    # Анализ недавней активности
    insights = await cmd.reflect(
        window=TimeWindow.LAST_24_HOURS
    )

    for insight in insights:
        print(f"Pattern: {insight.pattern}")
        print(f"Frequency: {insight.frequency}")
        print(f"Importance: {insight.importance}")

    # Принудительное забывание
    forgotten_count = await cmd.forget(
        criteria={
            "retention_below": 0.1,
            "older_than": timedelta(days=30),
            "source_type": SourceType.INFERENCE,
        }
    )
```

### Работа с событиями

```python
# Подписка на события системы
@cmd.on_consolidation
async def handle_consolidation(report):
    print(f"Consolidation completed: {report}")

@cmd.on_memory_forgotten
async def handle_forget(memory_ids):
    print(f"Memories forgotten: {memory_ids}")

@cmd.on_conflict_detected
async def handle_conflict(conflict):
    print(f"Conflict detected: {conflict}")
    # Можно переопределить стратегию разрешения
    return ResolutionStrategy.KEEP_NEWEST
```

## REST API

### Базовый URL

```
http://localhost:8080/api/v1
```

### Аутентификация

```http
Authorization: Bearer <token>
```

### Endpoints

#### POST /memory
Создать новую память

**Request:**
```json
{
  "content": "User prefers dark mode",
  "modality": "text",
  "metadata": {
    "source": "direct_user_input",
    "confidence": 0.95,
    "tags": ["preferences", "ui"]
  }
}
```

**Response:**
```json
{
  "memory_id": "0192f3a4-5b6c-7d8e-9f0a-1b2c3d4e5f6a",
  "created_at": "2024-11-22T10:30:00Z",
  "status": "stored"
}
```

#### POST /search
Поиск воспоминаний

**Request:**
```json
{
  "query": "What are user's preferences?",
  "constraints": {
    "top_k": 10,
    "hybrid_alpha": 0.7,
    "min_confidence": 0.5,
    "time_range": {
      "start": "2024-11-01T00:00:00Z",
      "end": "2024-11-22T23:59:59Z"
    }
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "memory_id": "0192f3a4-5b6c-7d8e-9f0a-1b2c3d4e5f6a",
      "content": "User prefers dark mode",
      "score": 0.92,
      "retention_strength": 0.85,
      "source": {
        "type": "direct_user_input",
        "confidence": 0.95
      },
      "created_at": "2024-11-22T10:30:00Z"
    }
  ],
  "total": 1,
  "query_time_ms": 35
}
```

#### POST /consolidate
Запустить консолидацию памяти

**Request:**
```json
{
  "mode": "incremental",
  "max_episodes": 1000
}
```

**Response:**
```json
{
  "job_id": "consolidation-2024-11-22-001",
  "status": "started",
  "estimated_duration_seconds": 120
}
```

#### GET /consolidate/{job_id}
Проверить статус консолидации

**Response:**
```json
{
  "job_id": "consolidation-2024-11-22-001",
  "status": "completed",
  "report": {
    "episodes_processed": 847,
    "facts_created": 23,
    "conflicts_resolved": 5,
    "memories_forgotten": 112,
    "duration_seconds": 98
  }
}
```

#### DELETE /memory/{memory_id}
Удалить конкретную память

**Response:**
```json
{
  "memory_id": "0192f3a4-5b6c-7d8e-9f0a-1b2c3d4e5f6a",
  "status": "deleted"
}
```

#### GET /stats
Получить статистику системы

**Response:**
```json
{
  "memory_stats": {
    "total_memories": 45678,
    "episodic_count": 12345,
    "semantic_facts": 33333,
    "active_memories": 40234,
    "forgotten_memories": 5444
  },
  "performance": {
    "avg_write_latency_ms": 8.2,
    "avg_search_latency_ms": 34.5,
    "last_consolidation": "2024-11-22T09:00:00Z",
    "consolidation_rate": 0.87
  },
  "storage": {
    "episodic_size_mb": 234.5,
    "semantic_size_mb": 456.7,
    "total_size_mb": 691.2
  }
}
```

## Примеры использования

### Пример 1: Персональный ассистент

```python
class PersonalAssistant:
    def __init__(self):
        self.cmd = CMD(config=Config(
            consolidation_interval=3600,
            decay_alpha=0.03,  # Медленное забывание
        ))

    async def remember_preference(self, preference: str):
        """Запомнить предпочтение пользователя"""
        memory_id = await self.cmd.write(
            content=f"User preference: {preference}",
            metadata={
                "type": "preference",
                "confidence": 0.9,
                "permanent": True,  # Защита от забывания
            }
        )
        return memory_id

    async def get_user_context(self, topic: str):
        """Получить контекст по теме"""
        results = await self.cmd.retrieve(
            query=topic,
            constraints={
                "top_k": 5,
                "min_confidence": 0.6,
            }
        )

        context = []
        for memory in results:
            if memory.retention_strength > 0.5:
                context.append(memory.content)

        return "\n".join(context)
```

### Пример 2: Исследовательский агент

```python
class ResearchAgent:
    def __init__(self):
        self.cmd = CMD(config=Config(
            hdc_dimension=20000,  # Больше для сложных структур
            consolidation_batch_size=500,
        ))

    async def add_research_finding(self, paper_id: str, finding: dict):
        """Добавить научное открытие"""
        memory_id = await self.cmd.write(
            content=json.dumps(finding),
            modality=Modality.STRUCTURED,
            metadata={
                "paper_id": paper_id,
                "authors": finding.get("authors"),
                "year": finding.get("year"),
                "field": finding.get("field"),
                "confidence": self._calculate_confidence(finding),
            }
        )

        # Создать связи в графе
        await self.cmd.add_graph_link(
            from_node=memory_id,
            to_node=paper_id,
            relationship="EXTRACTED_FROM",
            weight=0.8
        )

        return memory_id

    async def find_related_research(self, topic: str, depth: int = 2):
        """Найти связанные исследования"""
        # Гибридный поиск
        direct_results = await self.cmd.retrieve(
            query=topic,
            constraints={"top_k": 20}
        )

        # Расширение через граф
        expanded_results = []
        for result in direct_results:
            neighbors = await self.cmd.graph_neighbors(
                node_id=result.memory_id,
                depth=depth,
                relationship_filter=["CITES", "EXTENDS", "CONTRADICTS"]
            )
            expanded_results.extend(neighbors)

        return self._deduplicate_and_rank(expanded_results)
```

### Пример 3: Обучающая система

```python
class LearningSystem:
    def __init__(self):
        self.cmd = CMD(config=Config(
            decay_alpha=0.1,  # Быстрое забывание для обучения
        ))

    async def learn_from_feedback(self, action: str, outcome: str, reward: float):
        """Обучение с подкреплением"""
        memory_id = await self.cmd.write(
            content=json.dumps({
                "action": action,
                "outcome": outcome,
                "reward": reward
            }),
            metadata={
                "type": "experience",
                "confidence": abs(reward),
            }
        )

        # Консолидация для выявления паттернов
        if await self._should_consolidate():
            report = await self.cmd.consolidate()

            # Анализ новых инсайтов
            insights = await self.cmd.reflect(TimeWindow.LAST_HOUR)
            for insight in insights:
                if insight.pattern_type == "action_outcome":
                    await self._update_policy(insight)

        return memory_id

    async def get_best_action(self, state: str):
        """Получить лучшее действие для состояния"""
        # Поиск похожих состояний
        similar_experiences = await self.cmd.retrieve(
            query=state,
            constraints={
                "top_k": 10,
                "min_retention": 0.3,  # Учитывать только запомненное
            }
        )

        # Взвешенное голосование по reward и retention
        action_scores = {}
        for exp in similar_experiences:
            data = json.loads(exp.content)
            score = data["reward"] * exp.retention_strength
            action = data["action"]

            if action not in action_scores:
                action_scores[action] = 0
            action_scores[action] += score

        return max(action_scores, key=action_scores.get)
```

## Обработка ошибок

```python
from cognitive_memory import CmdError, MemoryNotFound, ConflictDetected

try:
    result = await cmd.retrieve("query")
except MemoryNotFound as e:
    print(f"Memory not found: {e.memory_id}")
except ConflictDetected as e:
    print(f"Conflict: {e.description}")
    # Ручное разрешение
    resolution = await cmd.resolve_conflict(
        e.conflict_id,
        strategy=ResolutionStrategy.MERGE_ATTRIBUTES
    )
except CmdError as e:
    print(f"General error: {e}")
```

## Best Practices

1. **Используйте batch операции** для массовой записи
2. **Настройте decay_alpha** в зависимости от домена
3. **Запускайте консолидацию** в периоды низкой нагрузки
4. **Мониторьте retention_strength** для критичных данных
5. **Используйте HDC фильтрацию** для структурных запросов
6. **Кешируйте частые запросы** на уровне приложения
7. **Логируйте конфликты** для анализа качества данных

## Производительность

### Рекомендации по оптимизации

- **Batch size**: 100-500 записей
- **Consolidation interval**: 1-4 часа
- **HDC dimension**: 10,000-20,000
- **Top-k limit**: ≤50 для real-time
- **Hybrid alpha**: 0.6-0.8 для баланса

### Бенчмарки

| Операция | Записей | Время | QPS |
|----------|---------|-------|-----|
| Write | 1 | 8ms | 125 |
| Write Batch | 100 | 150ms | 667 |
| Search (vector) | - | 35ms | 28 |
| Search (hybrid) | - | 70ms | 14 |
| Consolidation | 1000 | 5s | - |

## Версионирование API

API следует семантическому версионированию:
- **v1.0.0** - Текущая стабильная версия
- **v1.1.0** - Добавление multimodal поддержки (планируется)
- **v2.0.0** - Breaking changes в CRDT (планируется)