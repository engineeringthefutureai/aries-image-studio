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

    let config = BackendConfig::from_env()?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("login-link") {
        return login_link(&config, args.get(1).map(String::as_str)).await;
    }
    if config.agent_token.is_empty() {
        anyhow::bail!(
            "ARIES_AGENT_TOKEN must be set: agents and artifact transfers authenticate with it"
        );
    }
    let db_path = config.data_dir.join("studio.db");
    let db = Database::connect(&db_path).await?;
    let blobs = BlobStore::new(&config.data_dir)?;
    let state = AppState::new(config.clone(), db, blobs);
    // Cloud enhances run inside Studio: any left in flight ended with the last process.
    let reclaimed = state.db.reclaim_cloud_tasks().await?;
    if !reclaimed.is_empty() {
        info!(
            jobs = reclaimed.len(),
            "Requeued cloud enhances left from the last run"
        );
    }
    if let Some(auth) = &config.auth {
        // What was made before sign-in existed belongs to the owner.
        let claimed = state.db.claim_unowned(&auth.owner).await?;
        if claimed > 0 {
            info!(jobs = claimed, owner = %auth.owner, "Earlier jobs now belong to the owner");
        }
        info!(
            issuer = %auth.issuer,
            accounts = auth.allowed_emails.len(),
            "Sign-in required"
        );
    }
    for p in state.cloud.infos() {
        info!(enhancer = %p.id, model = %p.model, "Cloud enhancer on");
    }

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

    let http_task = tokio::spawn(async move { axum::serve(http_listener, http_router).await });

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

/// `aries-image-studio login-link <email>`: prints a single-use sign-in link that skips
/// Google, for when it cannot be reached (docs/AUTH.md §2). Run it where Studio runs, with
/// Studio's environment (`kubectl exec`, `docker compose exec`).
async fn login_link(config: &BackendConfig, email: Option<&str>) -> Result<()> {
    let Some(email) = email else {
        anyhow::bail!("usage: aries-image-studio login-link <email>");
    };
    let Some(auth) = &config.auth else {
        anyhow::bail!("sign-in is off (ARIES_OIDC_CLIENT_ID is not set): no link needed");
    };
    let db = Database::connect(config.data_dir.join("studio.db")).await?;
    // One per address Studio is reached on; the same token, so any one of them, once.
    for link in aries_image_studio::auth::create_login_link(&db, auth, email).await? {
        println!("{link}");
    }
    eprintln!(
        "Open one within {} minutes; it works once.",
        aries_image_studio::auth::LOGIN_LINK_TTL_S as u64 / 60
    );
    Ok(())
}
