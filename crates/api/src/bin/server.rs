//! Cognitive Memory Database REST API Server
//!
//! A production-ready HTTP server for the cognitive memory database,
//! providing full access to memory operations, emotional context, and
//! prospective memory (goals/intentions).

use clap::Parser;
use cmd_api::create_router;
use std::net::SocketAddr;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Cognitive Memory Database REST API Server
#[derive(Parser, Debug)]
#[command(
    name = "cmd-server",
    about = "REST API server for Cognitive Memory Database",
    long_about = "A production-ready HTTP server providing access to cognitive memory operations,\n\
                  including memory management, emotional context, and prospective memory (goals/intentions).",
    version
)]
struct Args {
    /// Server host address
    #[arg(
        short = 'H',
        long,
        default_value = "0.0.0.0",
        env = "CMD_HOST",
        help = "Host address to bind the server to"
    )]
    host: String,

    /// Server port
    #[arg(
        short,
        long,
        default_value = "3000",
        env = "CMD_PORT",
        help = "Port number to bind the server to"
    )]
    port: u16,

    /// Logging level
    #[arg(
        short,
        long,
        default_value = "info",
        env = "RUST_LOG",
        help = "Logging level (trace, debug, info, warn, error)"
    )]
    log_level: String,

    /// Enable JSON formatted logs
    #[arg(
        long,
        default_value = "false",
        env = "CMD_JSON_LOGS",
        help = "Output logs in JSON format"
    )]
    json_logs: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize tracing/logging
    init_tracing(&args);

    // Build socket address
    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .expect("Invalid host or port");

    // Create the API router with all endpoints
    let app = create_router();

    // Create TCP listener
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind to {}: {}", addr, e))?;

    // Print startup information
    print_banner(&addr);

    // Start the server
    info!("Server starting on http://{}", addr);
    info!("Health check available at http://{}/health", addr);

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Server error: {}", e))?;

    Ok(())
}

/// Initialize tracing subscriber with appropriate configuration
fn init_tracing(args: &Args) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&args.log_level));

    if args.json_logs {
        // JSON formatted logs for production
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        // Human-readable logs for development
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().pretty())
            .init();
    }
}

/// Print startup banner with server information
fn print_banner(addr: &SocketAddr) {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                                                               ║");
    println!("║          Cognitive Memory Database - REST API Server         ║");
    println!("║                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("🚀 Server Address:    http://{}", addr);
    println!("💚 Health Check:      http://{}/health", addr);
    println!();
    println!("📚 Available Endpoints:");
    println!();
    println!("   Memory Operations:");
    println!("     POST   /memories              - Add new memory");
    println!("     GET    /memories/:id          - Retrieve memory");
    println!("     DELETE /memories/:id          - Delete memory");
    println!("     POST   /memories/search       - Search memories");
    println!("     POST   /memories/search/temporal - Temporal search");
    println!();
    println!("   Emotional Operations:");
    println!("     POST   /emotions/search       - Search by emotion");
    println!("     POST   /emotions/similarity   - Emotional similarity");
    println!("     PUT    /emotions/update/:id   - Update emotion");
    println!("     GET    /emotions/stats        - Emotional statistics");
    println!();
    println!("   Prospective Memory (Intentions):");
    println!("     POST   /intentions            - Create intention");
    println!("     GET    /intentions/active     - Active intentions");
    println!("     GET    /intentions/triggerable - Triggerable intentions");
    println!("     POST   /intentions/:id/complete - Complete intention");
    println!("     POST   /intentions/:id/cancel - Cancel intention");
    println!();
    println!("   System Operations:");
    println!("     GET    /health                - Health check");
    println!("     GET    /system/stats          - System statistics");
    println!("     POST   /system/forget         - Run forgetting process");
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}
