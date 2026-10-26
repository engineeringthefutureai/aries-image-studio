use crate::db::{now_s, AgentRecord};
use crate::metrics::parse_agent_prometheus_text;
use crate::scheduler::{tick, AppState};
use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tracing::debug;

pub async fn scrape_single_agent(state: &Arc<AppState>, agent: &AgentRecord) -> Result<()> {
    if !agent.online || agent.metrics_url.trim().is_empty() {
        return Ok(());
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(800))
        .build()?;
    let resp = client.get(&agent.metrics_url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("metrics scrape returned HTTP {}", resp.status());
    }
    let text = resp.text().await?;
    let parsed = parse_agent_prometheus_text(&text, now_s());
    if let Ok(mut map) = state.live_metrics.write() {
        map.insert(agent.id.clone(), parsed);
    }
    Ok(())
}

pub fn spawn_scraper_and_gc_loop(state: Arc<AppState>) {
    let interval_ms = state.config.scrape_interval_ms.max(100);
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_millis(interval_ms));
        let mut ticks = 0u64;
        loop {
            ticker.tick().await;
            ticks += 1;

            if let Ok(agents) = state.db.list_agents().await {
                for agent in agents.into_iter().filter(|a| a.online) {
                    if let Err(e) = scrape_single_agent(&state, &agent).await {
                        debug!(agent = %agent.id, error = %e, "Failed to scrape agent /metrics");
                    }
                }
            }

            tick(&state).await;

            // Every 30 ticks (~30s), run intermediate artifact GC (§7.4)
            if ticks % 30 == 0 {
                let emergency = state
                    .blobs
                    .disk_usage_pct()
                    .map(|pct| pct >= state.config.disk_watermark_pct)
                    .unwrap_or(false);
                let _collecting = state.blobs.collecting().await;
                if let Ok(orphans) = state
                    .db
                    .gc_intermediate_artifacts(86400.0, emergency, None)
                    .await
                {
                    for sha in orphans {
                        let _ = state.blobs.delete_blob(&sha);
                    }
                }
                // Expired sessions and sign-ins never finished (docs/AUTH.md).
                let _ = state.db.purge_auth().await;
                // Reference uploads no job uses, a day after upload (docs/REFERENCE_IMAGES.md §4).
                if let Ok(orphans) = state.db.gc_uploads(86400.0).await {
                    for sha in orphans {
                        let _ = state.blobs.delete_blob(&sha);
                        state
                            .blobs
                            .delete_thumbnails(&crate::storage::upload_thumb_key(&sha));
                    }
                }
            }
        }
    });
}
