//! REST API for Cognitive Memory Database
//!
//! Provides HTTP endpoints for all memory operations including emotional
//! context and prospective memory (goals/intentions).

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use cmd_core::memory::{
    EmotionalValence, MemoryUnit, PADVector, SourceMetadata, SourceType,
    TriggerCondition,
};
use cmd_core::types::{MemoryId, SourceId};
use uuid::Uuid;
use cmd_manager::{EmotionalStats, ManagerConfig, ManagerStats, MemoryManager};
use cmd_storage::memory::{InMemoryEpisodicStorage, InMemorySemanticStorage};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// API State holding the Memory Manager
#[derive(Clone)]
pub struct ApiState {
    manager: Arc<MemoryManager<InMemoryEpisodicStorage, InMemorySemanticStorage>>,
}

impl ApiState {
    /// Create a new API state with default configuration
    pub fn new() -> Self {
        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();
        let manager = MemoryManager::new(episodic, semantic, config);

        Self {
            manager: Arc::new(manager),
        }
    }

    /// Create API state with custom configuration
    pub fn with_config(config: ManagerConfig) -> Self {
        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let manager = MemoryManager::new(episodic, semantic, config);

        Self {
            manager: Arc::new(manager),
        }
    }
}

impl Default for ApiState {
    fn default() -> Self {
        Self::new()
    }
}

/// Create the main API router
pub fn create_router() -> Router {
    let state = ApiState::new();

    Router::new()
        // Health check
        .route("/health", get(health_check))
        // Basic memory operations
        .route("/memories", post(add_memory))
        .route("/memories/{id}", get(get_memory))
        .route("/memories/{id}", delete(delete_memory))
        .route("/memories/search", post(search_memories))
        .route("/memories/search/temporal", post(search_temporal))
        // Emotional operations
        .route("/emotions/search", post(search_by_emotion))
        .route("/emotions/similarity", post(search_by_emotional_similarity))
        .route("/emotions/update/{id}", put(update_emotion))
        .route("/emotions/stats", get(get_emotional_stats))
        // Prospective memory (intentions)
        .route("/intentions", post(add_intention))
        .route("/intentions/active", get(get_active_intentions))
        .route("/intentions/triggerable", get(get_triggerable_intentions))
        .route("/intentions/{id}/complete", post(complete_intention))
        .route("/intentions/{id}/cancel", post(cancel_intention))
        // System operations
        .route("/system/stats", get(get_stats))
        .route("/system/forget", post(forget_weak_memories))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}

// ============================================================================
// Request/Response DTOs
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct AddMemoryRequest {
    pub text: String,
    pub confidence: f32,
    pub emotion: Option<PADVectorDto>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddIntentionRequest {
    pub goal: String,
    pub trigger: TriggerConditionDto,
    pub priority: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub k: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchTemporalRequest {
    pub query: String,
    pub k: usize,
    pub since: Option<String>, // ISO 8601 datetime
    pub until: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchByEmotionRequest {
    pub valence: EmotionalValenceDto,
    pub k: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchByEmotionalSimilarityRequest {
    pub target_emotion: PADVectorDto,
    pub k: usize,
    pub max_distance: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateEmotionRequest {
    pub emotion: PADVectorDto,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForgetRequest {
    pub threshold: f32,
}

// DTOs for core types
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct PADVectorDto {
    pub pleasure: f32,
    pub arousal: f32,
    pub dominance: f32,
}

impl From<PADVectorDto> for PADVector {
    fn from(dto: PADVectorDto) -> Self {
        PADVector::new(dto.pleasure, dto.arousal, dto.dominance)
    }
}

impl From<PADVector> for PADVectorDto {
    fn from(pad: PADVector) -> Self {
        Self {
            pleasure: pad.pleasure,
            arousal: pad.arousal,
            dominance: pad.dominance,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum EmotionalValenceDto {
    Positive,
    Negative,
    Neutral,
}

impl From<EmotionalValenceDto> for EmotionalValence {
    fn from(dto: EmotionalValenceDto) -> Self {
        match dto {
            EmotionalValenceDto::Positive => EmotionalValence::Positive,
            EmotionalValenceDto::Negative => EmotionalValence::Negative,
            EmotionalValenceDto::Neutral => EmotionalValence::Neutral,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TriggerConditionDto {
    Immediate,
    TimeBasedAt { datetime: String },
    TimeBasedAfter { duration_seconds: u64 },
    EventBased { event: String },
    ContextBased { contexts: Vec<String> },
}

impl TriggerConditionDto {
    fn into_trigger_condition(self) -> Result<TriggerCondition, ApiError> {
        match self {
            TriggerConditionDto::Immediate => Ok(TriggerCondition::Immediate),
            TriggerConditionDto::TimeBasedAt { datetime } => {
                let dt = chrono::DateTime::parse_from_rfc3339(&datetime)
                    .map_err(|_| ApiError::BadRequest("Invalid datetime format".to_string()))?;
                Ok(TriggerCondition::TimeBasedAt(dt.with_timezone(&chrono::Utc)))
            }
            TriggerConditionDto::TimeBasedAfter { duration_seconds } => {
                Ok(TriggerCondition::TimeBasedAfter(std::time::Duration::from_secs(duration_seconds)))
            }
            TriggerConditionDto::EventBased { event } => {
                Ok(TriggerCondition::EventBased(event))
            }
            TriggerConditionDto::ContextBased { contexts } => {
                Ok(TriggerCondition::ContextBased(contexts))
            }
        }
    }
}

// Response types
#[derive(Debug, Serialize)]
pub struct MemoryResponse {
    pub id: String,
    pub text: Option<String>,
    pub created_at: String,
    pub access_count: u32,
    pub retention_strength: f32,
    pub emotional_valence: Option<String>,
    pub is_prospective: bool,
}

impl From<MemoryUnit> for MemoryResponse {
    fn from(memory: MemoryUnit) -> Self {
        let retention_strength = memory.retention_strength();
        let emotional_valence = memory.emotional_valence().map(|v| format!("{:?}", v));
        let is_prospective = memory.is_prospective();

        Self {
            id: memory.id.to_string(),
            text: memory.text,
            created_at: memory.temporal.created_at.to_rfc3339(),
            access_count: memory.temporal.access_count,
            retention_strength,
            emotional_valence,
            is_prospective,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SearchResultResponse {
    pub memory: MemoryResponse,
    pub similarity: f32,
    pub retention_score: f32,
}

#[derive(Debug, Serialize)]
pub struct EmotionalSearchResultResponse {
    pub memory: MemoryResponse,
    pub distance: f32,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub total_memories: u64,
    pub total_searches: u64,
    pub total_conflicts_resolved: u64,
    pub total_consolidations: u64,
    pub total_forgotten: u64,
}

impl From<ManagerStats> for StatsResponse {
    fn from(stats: ManagerStats) -> Self {
        Self {
            total_memories: stats.total_memories,
            total_searches: stats.total_searches,
            total_conflicts_resolved: stats.total_conflicts_resolved,
            total_consolidations: stats.total_consolidations,
            total_forgotten: stats.total_forgotten,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EmotionalStatsResponse {
    pub positive_count: u64,
    pub negative_count: u64,
    pub neutral_count: u64,
    pub average_intensity: f32,
}

impl From<EmotionalStats> for EmotionalStatsResponse {
    fn from(stats: EmotionalStats) -> Self {
        Self {
            positive_count: stats.positive_count,
            negative_count: stats.negative_count,
            neutral_count: stats.neutral_count,
            average_intensity: stats.average_intensity,
        }
    }
}

// ============================================================================
// Error Handling
// ============================================================================

#[derive(Debug)]
pub enum ApiError {
    NotFound(String),
    BadRequest(String),
    InternalError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(serde_json::json!({
            "error": message
        }));

        (status, body).into_response()
    }
}

impl From<cmd_manager::ManagerError> for ApiError {
    fn from(err: cmd_manager::ManagerError) -> Self {
        match err {
            cmd_manager::ManagerError::MemoryNotFound(_) => {
                ApiError::NotFound(err.to_string())
            }
            cmd_manager::ManagerError::InvalidOperation(msg) => ApiError::BadRequest(msg),
            _ => ApiError::InternalError(err.to_string()),
        }
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// Health check endpoint
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "Cognitive Memory Database"
    }))
}

/// Add a new memory
async fn add_memory(
    State(state): State<ApiState>,
    Json(req): Json<AddMemoryRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
    let source = SourceMetadata {
        source_id: SourceId::new(),
        source_type: SourceType::DirectUserInput,
        confidence: req.confidence,
        timestamp: chrono::Utc::now(),
    };

    let memory = if let Some(emotion_dto) = req.emotion {
        let emotion = PADVector::from(emotion_dto);
        MemoryUnit::new_text_with_emotion(req.text, source, emotion)
    } else {
        MemoryUnit::new_text(req.text, source)
    };

    let id = state.manager.add_memory(memory).await?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "id": id.to_string()
        })),
    ))
}

/// Get a memory by ID
async fn get_memory(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<MemoryResponse>, ApiError> {
    let uuid = Uuid::parse_str(&id)
        .map_err(|_| ApiError::BadRequest("Invalid memory ID".to_string()))?;
    let memory_id = MemoryId(uuid);

    let memory = state.manager.get_memory(&memory_id).await?;
    let response = MemoryResponse::from(memory);

    Ok(Json(response))
}

/// Delete a memory
async fn delete_memory(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let uuid = Uuid::parse_str(&id)
        .map_err(|_| ApiError::BadRequest("Invalid memory ID".to_string()))?;
    let memory_id = MemoryId(uuid);

    state.manager.delete_memory(&memory_id).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Search memories
async fn search_memories(
    State(state): State<ApiState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Vec<SearchResultResponse>>, ApiError> {
    let results = state.manager.search(&req.query, req.k).await?;

    let mut response = Vec::new();
    for r in results {
        if let Ok(memory) = state.manager.get_memory(&r.memory_id).await {
            response.push(SearchResultResponse {
                memory: MemoryResponse::from(memory),
                similarity: r.similarity,
                retention_score: r.retention_score,
            });
        }
    }

    Ok(Json(response))
}

/// Search memories with temporal filter
async fn search_temporal(
    State(state): State<ApiState>,
    Json(req): Json<SearchTemporalRequest>,
) -> Result<Json<Vec<SearchResultResponse>>, ApiError> {
    let since = req
        .since
        .map(|s| {
            chrono::DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|_| ApiError::BadRequest("Invalid since datetime".to_string()))
        })
        .transpose()?;

    let until = req
        .until
        .map(|s| {
            chrono::DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|_| ApiError::BadRequest("Invalid until datetime".to_string()))
        })
        .transpose()?;

    let results = state
        .manager
        .search_temporal(&req.query, req.k, since, until)
        .await?;

    let mut response = Vec::new();
    for r in results {
        if let Ok(memory) = state.manager.get_memory(&r.memory_id).await {
            response.push(SearchResultResponse {
                memory: MemoryResponse::from(memory),
                similarity: r.similarity,
                retention_score: r.retention_score,
            });
        }
    }

    Ok(Json(response))
}

/// Search by emotional valence
async fn search_by_emotion(
    State(state): State<ApiState>,
    Json(req): Json<SearchByEmotionRequest>,
) -> Result<Json<Vec<MemoryResponse>>, ApiError> {
    let valence = EmotionalValence::from(req.valence);
    let memories = state.manager.search_by_emotion(valence, req.k).await?;

    let response: Vec<MemoryResponse> = memories.into_iter().map(MemoryResponse::from).collect();

    Ok(Json(response))
}

/// Search by emotional similarity
async fn search_by_emotional_similarity(
    State(state): State<ApiState>,
    Json(req): Json<SearchByEmotionalSimilarityRequest>,
) -> Result<Json<Vec<EmotionalSearchResultResponse>>, ApiError> {
    let target = PADVector::from(req.target_emotion);
    let results = state
        .manager
        .search_by_emotional_similarity(target, req.k, req.max_distance)
        .await?;

    let response: Vec<EmotionalSearchResultResponse> = results
        .into_iter()
        .map(|(memory, distance)| EmotionalSearchResultResponse {
            memory: MemoryResponse::from(memory),
            distance,
        })
        .collect();

    Ok(Json(response))
}

/// Update emotion of a memory
async fn update_emotion(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Json(req): Json<UpdateEmotionRequest>,
) -> Result<StatusCode, ApiError> {
    let uuid = Uuid::parse_str(&id)
        .map_err(|_| ApiError::BadRequest("Invalid memory ID".to_string()))?;
    let memory_id = MemoryId(uuid);

    let emotion = PADVector::from(req.emotion);
    state.manager.update_emotion(&memory_id, emotion).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Get emotional statistics
async fn get_emotional_stats(
    State(state): State<ApiState>,
) -> Result<Json<EmotionalStatsResponse>, ApiError> {
    let stats = state.manager.get_emotional_stats().await?;
    let response = EmotionalStatsResponse::from(stats);

    Ok(Json(response))
}

/// Add a new intention
async fn add_intention(
    State(state): State<ApiState>,
    Json(req): Json<AddIntentionRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
    let source = SourceMetadata {
        source_id: SourceId::new(),
        source_type: SourceType::DirectUserInput,
        confidence: 1.0,
        timestamp: chrono::Utc::now(),
    };

    let trigger = req.trigger.into_trigger_condition()?;
    let memory = MemoryUnit::new_intention(req.goal, trigger, req.priority, source);

    let id = state.manager.add_memory(memory).await?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "id": id.to_string()
        })),
    ))
}

/// Get active intentions
async fn get_active_intentions(
    State(state): State<ApiState>,
) -> Result<Json<Vec<MemoryResponse>>, ApiError> {
    let intentions = state.manager.get_active_intentions().await?;
    let response: Vec<MemoryResponse> = intentions.into_iter().map(MemoryResponse::from).collect();

    Ok(Json(response))
}

/// Get triggerable intentions
async fn get_triggerable_intentions(
    State(state): State<ApiState>,
) -> Result<Json<Vec<MemoryResponse>>, ApiError> {
    let intentions = state.manager.get_triggerable_intentions().await?;
    let response: Vec<MemoryResponse> = intentions.into_iter().map(MemoryResponse::from).collect();

    Ok(Json(response))
}

/// Complete an intention
async fn complete_intention(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let uuid = Uuid::parse_str(&id)
        .map_err(|_| ApiError::BadRequest("Invalid memory ID".to_string()))?;
    let memory_id = MemoryId(uuid);

    state.manager.complete_intention(&memory_id).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Cancel an intention
async fn cancel_intention(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let uuid = Uuid::parse_str(&id)
        .map_err(|_| ApiError::BadRequest("Invalid memory ID".to_string()))?;
    let memory_id = MemoryId(uuid);

    state.manager.cancel_intention(&memory_id).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Get system statistics
async fn get_stats(State(state): State<ApiState>) -> Result<Json<StatsResponse>, ApiError> {
    let stats = state.manager.get_stats();
    let response = StatsResponse::from(stats);

    Ok(Json(response))
}

/// Forget weak memories
async fn forget_weak_memories(
    State(state): State<ApiState>,
    Json(req): Json<ForgetRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let forgotten_count = state.manager.forget_weak_memories(req.threshold).await?;

    Ok(Json(serde_json::json!({
        "forgotten_count": forgotten_count
    })))
}
