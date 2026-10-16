use anyhow::Result;
use aries_image_studio::api::build_http_router;
use aries_image_studio::config::BackendConfig;
use aries_image_studio::db::Database;
use aries_image_studio::grpc::AgentGatewayService;
use aries_image_studio::proto::agent_gateway_server::AgentGatewayServer;
use aries_image_studio::scheduler::AppState;
use aries_image_studio::scraper::spawn_scraper_and_gc_loop;
use aries_image_studio::storage::BlobStore;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tonic::transport::Server as GrpcServer;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "aries_image_studio=info,tower_http=info".into()),
        )
        .init();

    let config = BackendConfig::from_env();
    if config.agent_token.is_empty() {
        anyhow::bail!("ARIES_AGENT_TOKEN must be set: agents and artifact transfers authenticate with it");
    }
    let db_path = config.data_dir.join("studio.db");
    let db = Database::connect(&db_path).await?;
    let blobs = BlobStore::new(&config.data_dir)?;
    let state = AppState::new(config.clone(), db, blobs);

    spawn_scraper_and_gc_loop(state.clone());

    let http_addr: SocketAddr = config.http_addr.parse()?;
    let grpc_addr: SocketAddr = config.grpc_addr.parse()?;

    let http_router = build_http_router(state.clone());
    let http_listener = TcpListener::bind(http_addr).await?;
    info!(%http_addr, "Aries Image Studio HTTP server listening");

    let grpc_svc = AgentGatewayServer::new(AgentGatewayService::new(state.clone()))
        .max_decoding_message_size(16 * 1024 * 1024)
        .max_encoding_message_size(16 * 1024 * 1024);
    info!(%grpc_addr, "Aries Image Studio AgentGateway gRPC server listening");

    let http_task = tokio::spawn(async move {
        axum::serve(http_listener, http_router).await
    });

    let grpc_task = tokio::spawn(async move {
        GrpcServer::builder()
            .add_service(grpc_svc)
            .serve(grpc_addr)
            .await
    });

    tokio::select! {
        res = http_task => {
            res??;
        }
        res = grpc_task => {
            res??;
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Shutting down Aries Image Studio backend");
        }
    }

    Ok(())
}
