use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub http_addr: String,
    pub grpc_addr: String,
    pub data_dir: PathBuf,
    pub static_dir: PathBuf,
    pub agent_token: String,
    pub artifact_base_url: String,
    pub max_wait_s: f64,
    pub idle_grace_s: f64,
    pub scrape_interval_ms: u64,
    pub scrape_stale_s: f64,
    pub reconnect_grace_s: f64,
    pub disk_watermark_pct: f64,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            http_addr: "0.0.0.0:8080".to_string(),
            grpc_addr: "0.0.0.0:50051".to_string(),
            data_dir: PathBuf::from("./data"),
            static_dir: PathBuf::from("./web/dist"),
            agent_token: String::new(), // required: ARIES_AGENT_TOKEN
            artifact_base_url: "http://127.0.0.1:8080".to_string(),
            max_wait_s: 600.0,
            idle_grace_s: 45.0,
            scrape_interval_ms: 1000,
            scrape_stale_s: 5.0,
            reconnect_grace_s: 30.0,
            disk_watermark_pct: 80.0,
        }
    }
}

impl BackendConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = env::var("ARIES_HTTP_ADDR") {
            cfg.http_addr = v;
        }
        if let Ok(v) = env::var("ARIES_GRPC_ADDR") {
            cfg.grpc_addr = v;
        }
        if let Ok(v) = env::var("ARIES_DATA_DIR") {
            cfg.data_dir = PathBuf::from(v);
        }
        if let Ok(v) = env::var("ARIES_STATIC_DIR") {
            cfg.static_dir = PathBuf::from(v);
        }
        if let Ok(v) = env::var("ARIES_AGENT_TOKEN") {
            cfg.agent_token = v.trim().to_string();
        }
        if let Ok(v) = env::var("ARIES_ARTIFACT_BASE_URL") {
            cfg.artifact_base_url = v.trim_end_matches('/').to_string();
        }
        if let Ok(v) = env::var("ARIES_MAX_WAIT_S") {
            if let Ok(n) = v.parse() {
                cfg.max_wait_s = n;
            }
        }
        if let Ok(v) = env::var("ARIES_IDLE_GRACE_S") {
            if let Ok(n) = v.parse() {
                cfg.idle_grace_s = n;
            }
        }
        if let Ok(v) = env::var("ARIES_SCRAPE_INTERVAL_MS") {
            if let Ok(n) = v.parse() {
                cfg.scrape_interval_ms = n;
            }
        }
        if let Ok(v) = env::var("ARIES_SCRAPE_STALE_S") {
            if let Ok(n) = v.parse() {
                cfg.scrape_stale_s = n;
            }
        }
        if let Ok(v) = env::var("ARIES_RECONNECT_GRACE_S") {
            if let Ok(n) = v.parse() {
                cfg.reconnect_grace_s = n;
            }
        }
        if let Ok(v) = env::var("ARIES_DISK_WATERMARK_PCT") {
            if let Ok(n) = v.parse() {
                cfg.disk_watermark_pct = n;
            }
        }
        cfg
    }
}

/// Compares a request's `Authorization` header with `Bearer <token>` in constant time.
/// An empty configured token never matches.
pub fn bearer_matches(header_value: &str, token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let expected = format!("Bearer {token}");
    let (a, b) = (header_value.as_bytes(), expected.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}
