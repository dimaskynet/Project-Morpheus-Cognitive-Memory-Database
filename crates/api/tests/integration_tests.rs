//! Integration tests for the REST API
//!
//! These tests verify that all HTTP endpoints work correctly end-to-end.
//!
//! ## Test Status: 18/18 passing ✅
//!
//! ### Test Coverage:
//! - **Health & System**: Health check, system stats, forgetting process
//! - **Memory CRUD**: Add, retrieve, delete, search (text & temporal)
//! - **Emotional Operations**: Search by valence/similarity, update emotions, get stats
//! - **Prospective Memory**: Create, complete, cancel intentions; get active/triggerable
//! - **Error Handling**: Invalid IDs, malformed requests

use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use cmd_api::create_router;
use serde_json::{json, Value};
use tower::Service;

/// Helper function to send a request using a router
async fn send_request_with_app(
    app: &mut Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let request_builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");

    let request = if let Some(body_json) = body {
        request_builder
            .body(Body::from(serde_json::to_string(&body_json).unwrap()))
            .unwrap()
    } else {
        request_builder.body(Body::empty()).unwrap()
    };

    let response = app.call(request).await.unwrap();
    let status = response.status();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap_or(json!({}));

    // Debug output for failed requests
    if status.is_client_error() || status.is_server_error() {
        eprintln!("Request to {} failed with status {}: {}", uri, status, json);
    }

    (status, json)
}

/// Helper for stateless tests
async fn send_request(
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut app = create_router();
    send_request_with_app(&mut app, method, uri, body).await
}

#[tokio::test]
async fn test_health_check() {
    let (status, body) = send_request("GET", "/health", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "healthy");
    assert_eq!(body["service"], "Cognitive Memory Database");
}

#[tokio::test]
async fn test_system_stats() {
    let (status, body) = send_request("GET", "/system/stats", None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.get("total_memories").is_some());
    assert!(body.get("total_searches").is_some());
}

#[tokio::test]
async fn test_add_and_retrieve_memory() {
    let mut app = create_router();

    // Add a memory
    let add_request = json!({
        "text": "User loves dark mode",
        "confidence": 0.9,
        "emotion": {
            "pleasure": 0.7,
            "arousal": 0.5,
            "dominance": 0.6
        }
    });

    let (status, body) = send_request_with_app(&mut app, "POST", "/memories", Some(add_request)).await;
    assert!(status == StatusCode::OK || status == StatusCode::CREATED);

    let memory_id = body["id"].as_str().expect("Expected memory ID");
    assert!(!memory_id.is_empty());

    // Retrieve the memory (using same app instance)
    let uri = format!("/memories/{}", memory_id);
    let (status, body) = send_request_with_app(&mut app, "GET", &uri, None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["id"], memory_id);
}

#[tokio::test]
async fn test_search_memories() {
    let mut app = create_router();

    // First, add a memory
    let add_request = json!({
        "text": "User prefers vim keybindings",
        "confidence": 0.85
    });

    send_request_with_app(&mut app, "POST", "/memories", Some(add_request)).await;

    // Search for it
    let search_request = json!({
        "query": "vim preferences",
        "k": 10
    });

    let (status, body) = send_request_with_app(&mut app, "POST", "/memories/search", Some(search_request)).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.is_array());
}

#[tokio::test]
async fn test_temporal_search() {
    let search_request = json!({
        "query": "user settings",
        "k": 5,
        "time_range": {
            "start": "2024-01-01T00:00:00Z",
            "end": "2025-12-31T23:59:59Z"
        }
    });

    let (status, body) = send_request("POST", "/memories/search/temporal", Some(search_request)).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.is_array());
}

#[tokio::test]
async fn test_emotional_search() {
    let mut app = create_router();

    // Add memory with emotion
    let add_request = json!({
        "text": "User is frustrated with bugs",
        "confidence": 0.8,
        "emotion": {
            "pleasure": -0.6,
            "arousal": 0.7,
            "dominance": -0.3
        }
    });

    send_request_with_app(&mut app, "POST", "/memories", Some(add_request)).await;

    // Search by emotional valence
    let search_request = json!({
        "valence": "negative",
        "k": 10
    });

    let (status, body) = send_request_with_app(&mut app, "POST", "/emotions/search", Some(search_request)).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.is_array());
}

#[tokio::test]
async fn test_emotional_similarity_search() {
    let search_request = json!({
        "target_emotion": {
            "pleasure": 0.8,
            "arousal": 0.6,
            "dominance": 0.5
        },
        "k": 5,
        "max_distance": 0.7
    });

    let (status, body) = send_request("POST", "/emotions/similarity", Some(search_request)).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.is_array());
}

#[tokio::test]
async fn test_emotional_stats() {
    let (status, body) = send_request("GET", "/emotions/stats", None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.get("positive_count").is_some());
    assert!(body.get("negative_count").is_some());
    assert!(body.get("neutral_count").is_some());
}

#[tokio::test]
async fn test_update_emotion() {
    let mut app = create_router();

    // Add a memory first
    let add_request = json!({
        "text": "Test memory for emotion update",
        "confidence": 0.9
    });

    let (_, body) = send_request_with_app(&mut app, "POST", "/memories", Some(add_request)).await;
    let memory_id = body["id"].as_str().unwrap();

    // Update emotion
    let update_request = json!({
        "emotion": {
            "pleasure": 0.5,
            "arousal": 0.4,
            "dominance": 0.3
        }
    });

    let uri = format!("/emotions/update/{}", memory_id);
    let (status, _) = send_request_with_app(&mut app, "PUT", &uri, Some(update_request)).await;

    assert!(status == StatusCode::OK || status == StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn test_create_intention() {
    let intention_request = json!({
        "goal": "Write comprehensive tests",
        "priority": 0.95,
        "trigger": {"type": "immediate"}
    });

    let (status, body) = send_request("POST", "/intentions", Some(intention_request)).await;

    assert!(status == StatusCode::OK || status == StatusCode::CREATED);
    assert!(body.get("id").is_some());
}

#[tokio::test]
async fn test_get_active_intentions() {
    let mut app = create_router();

    // Create an intention first
    let intention_request = json!({
        "goal": "Refactor codebase",
        "priority": 0.8,
        "trigger": {"type": "immediate"}
    });

    send_request_with_app(&mut app, "POST", "/intentions", Some(intention_request)).await;

    // Get active intentions
    let (status, body) = send_request_with_app(&mut app, "GET", "/intentions/active", None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.is_array());
}

#[tokio::test]
async fn test_get_triggerable_intentions() {
    let (status, body) = send_request("GET", "/intentions/triggerable", None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.is_array());
}

#[tokio::test]
async fn test_complete_intention() {
    let mut app = create_router();

    // Create an intention
    let intention_request = json!({
        "goal": "Test completion flow",
        "priority": 0.7,
        "trigger": {"type": "immediate"}
    });

    let (_, body) = send_request_with_app(&mut app, "POST", "/intentions", Some(intention_request)).await;
    let intention_id = body["id"].as_str().unwrap();

    // Complete it
    let uri = format!("/intentions/{}/complete", intention_id);
    let (status, _) = send_request_with_app(&mut app, "POST", &uri, None).await;

    assert!(status == StatusCode::OK || status == StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn test_cancel_intention() {
    let mut app = create_router();

    // Create an intention
    let intention_request = json!({
        "goal": "Test cancellation flow",
        "priority": 0.6,
        "trigger": {"type": "immediate"}
    });

    let (_, body) = send_request_with_app(&mut app, "POST", "/intentions", Some(intention_request)).await;
    let intention_id = body["id"].as_str().unwrap();

    // Cancel it
    let uri = format!("/intentions/{}/cancel", intention_id);
    let (status, _) = send_request_with_app(&mut app, "POST", &uri, None).await;

    assert!(status == StatusCode::OK || status == StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn test_delete_memory() {
    let mut app = create_router();

    // Add a memory first
    let add_request = json!({
        "text": "Memory to be deleted",
        "confidence": 0.9
    });

    let (_, body) = send_request_with_app(&mut app, "POST", "/memories", Some(add_request)).await;
    let memory_id = body["id"].as_str().unwrap();

    // Delete it
    let uri = format!("/memories/{}", memory_id);
    let (status, _) = send_request_with_app(&mut app, "DELETE", &uri, None).await;

    assert!(status == StatusCode::OK || status == StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn test_forgetting_process() {
    let forget_request = json!({
        "threshold": 0.3
    });

    let (status, body) = send_request("POST", "/system/forget", Some(forget_request)).await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.get("forgotten_count").is_some());
}

#[tokio::test]
async fn test_invalid_memory_id() {
    let (status, _) = send_request("GET", "/memories/invalid-id-12345", None).await;

    // Should return error status (404 or 400)
    assert!(status.is_client_error());
}

#[tokio::test]
async fn test_malformed_request() {
    // Try to add memory with missing required fields
    let bad_request = json!({
        "invalid_field": "should fail"
    });

    let (status, _) = send_request("POST", "/memories", Some(bad_request)).await;

    // Should return 400 Bad Request or 422 Unprocessable Entity
    assert!(status.is_client_error());
}
