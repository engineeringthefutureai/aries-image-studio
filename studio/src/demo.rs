//! The read-only demo (docs/DEMO.md): with `ARIES_DEMO` and sign-in on, someone who accepts the
//! terms on the sign-in screen gets a guest id, and is a visitor. Visitors reach only the routes in
//! [`visitor_may`], see only the images the owner marked, and are held to per-IP quotas: an id can
//! be thrown away, an address less easily. Everything they are sent goes
//! through the `scrub_*` functions here: what is not named there does not reach them.

use crate::auth::User;
use crate::db::JobRecord;
use crate::scheduler::{AppState, CapabilitiesResponse};
use axum::extract::connect_info::ConnectInfo;
use axum::extract::{MatchedPath, Query, Request};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

/// Searches embedding at once for all visitors: each takes a CPU core for a while.
const SEARCH_SLOTS: usize = 2;
/// A visitor's search query, in bytes.
pub const MAX_QUERY_BYTES: usize = 200;

#[derive(Debug, Clone)]
pub struct DemoConfig {
    /// `ARIES_DEMO_TRUSTED_PROXIES`: how many proxies in front of Studio add to
    /// `X-Forwarded-For`; 0 uses the connection's address and ignores the header.
    pub trusted_proxies: usize,
    pub requests_per_min: u32,
    pub searches_per_min: u32,
    pub downloads_per_min: u32,
    /// New guest ids, each a row in the database.
    pub guests_per_min: u32,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            trusted_proxies: 0,
            requests_per_min: 600,
            searches_per_min: 30,
            downloads_per_min: 120,
            guests_per_min: 10,
        }
    }
}

impl DemoConfig {
    /// `None` unless `ARIES_DEMO` is on.
    pub fn from_env() -> Option<Self> {
        let on = env::var("ARIES_DEMO").is_ok_and(|v| matches!(v.trim(), "1" | "true" | "on"));
        if !on {
            return None;
        }
        let d = Self::default();
        let get = |k: &str, default: u32| {
            env::var(k)
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(default)
        };
        Some(Self {
            trusted_proxies: get("ARIES_DEMO_TRUSTED_PROXIES", d.trusted_proxies as u32) as usize,
            requests_per_min: get("ARIES_DEMO_REQUESTS_PER_MIN", d.requests_per_min),
            searches_per_min: get("ARIES_DEMO_SEARCHES_PER_MIN", d.searches_per_min),
            downloads_per_min: get("ARIES_DEMO_DOWNLOADS_PER_MIN", d.downloads_per_min),
            guests_per_min: get("ARIES_DEMO_GUESTS_PER_MIN", d.guests_per_min),
        })
    }
}

/// The only routes a visitor reaches, and only to read. A route added later stays closed
/// until it is named here.
pub fn visitor_may(method: &Method, route: &str) -> bool {
    matches!(*method, Method::GET | Method::HEAD)
        && matches!(
            route,
            "/api/jobs"
                | "/api/jobs/:id"
                | "/api/gallery/facets"
                | "/api/images/:id"
                | "/api/images/:id/thumb"
                | "/api/uploads/:sha"
                | "/api/uploads/:sha/thumb"
                | "/api/system"
                | "/api/capabilities"
                | "/api/preferences"
        )
}

/// Whose quota a request counts against: the address `trusted_proxies` hops back, an IPv6
/// address by its /64 (one household has all of it).
fn client_key(headers: &HeaderMap, peer: Option<IpAddr>, trusted_proxies: usize) -> String {
    let forwarded = (trusted_proxies > 0)
        .then(|| {
            let all = headers
                .get_all("x-forwarded-for")
                .iter()
                .filter_map(|v| v.to_str().ok())
                .flat_map(|v| v.split(','))
                .map(str::trim)
                .collect::<Vec<_>>();
            let at = all.len().checked_sub(trusted_proxies)?;
            all.get(at)?.parse::<IpAddr>().ok()
        })
        .flatten();
    match forwarded.or(peer).map(|ip| ip.to_canonical()) {
        Some(IpAddr::V6(ip)) => {
            let s = ip.segments();
            format!("{:x}:{:x}:{:x}:{:x}::/64", s[0], s[1], s[2], s[3])
        }
        Some(ip) => ip.to_string(),
        None => "unknown".to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Request,
    Search,
    Download,
    /// Opening the demo: a new guest id.
    Start,
}

#[derive(Default)]
struct Counts {
    requests: u32,
    searches: u32,
    downloads: u32,
    starts: u32,
    /// Going over a limit has been logged, once a minute for an address.
    logged: bool,
}

/// This minute's per-IP counts, kept in memory only.
#[derive(Default)]
struct Minute {
    minute: u64,
    clients: HashMap<String, Counts>,
    over_quota: u32,
    closed: u32,
    busy: u32,
    guests: u32,
}

impl Minute {
    /// A new minute forgets every address; the log keeps only how many there were.
    fn roll(&mut self, minute: u64) {
        if self.minute == minute {
            return;
        }
        if !self.clients.is_empty() || self.closed > 0 || self.guests > 0 {
            let total = |f: fn(&Counts) -> u32| self.clients.values().map(f).sum::<u32>();
            tracing::info!(
                addresses = self.clients.len(),
                requests = total(|c| c.requests),
                searches = total(|c| c.searches),
                downloads = total(|c| c.downloads),
                new_guests = self.guests,
                over_quota = self.over_quota,
                closed_routes = self.closed,
                search_busy = self.busy,
                "Demo visitors in the last minute"
            );
        }
        *self = Minute {
            minute,
            ..Default::default()
        };
    }
}

const REGISTERS: usize = 4096;

/// How many different guests came in an hour or a day, as a HyperLogLog: a fixed 4 KB however
/// many come, within about 2% (exact-ish while few), and no id can be read back. Each period
/// has a new random salt, so one period's cannot be compared with another's.
struct Distinct {
    period: u64,
    salt: [u8; 16],
    registers: Vec<u8>,
}

impl Distinct {
    fn new() -> Self {
        Self {
            period: u64::MAX,
            salt: [0; 16],
            registers: vec![0; REGISTERS],
        }
    }

    fn note(&mut self, key: &str, period: u64) {
        if self.period != period {
            self.period = period;
            self.registers.fill(0);
            if getrandom::getrandom(&mut self.salt).is_err() {
                self.salt = [0; 16];
            }
        }
        let digest = Sha256::new()
            .chain_update(self.salt)
            .chain_update(key)
            .finalize();
        let mut first = [0u8; 8];
        first.copy_from_slice(&digest[..8]);
        let hash = u64::from_be_bytes(first);
        let at = (hash >> 52) as usize;
        let rank = (hash << 12).leading_zeros().min(52) as u8 + 1;
        self.registers[at] = self.registers[at].max(rank);
    }

    fn count(&self, period: u64) -> u64 {
        if self.period != period {
            return 0;
        }
        let m = REGISTERS as f64;
        let empty = self.registers.iter().filter(|r| **r == 0).count();
        let sum: f64 = self.registers.iter().map(|r| 2f64.powi(-(*r as i32))).sum();
        let raw = 0.7213 / (1.0 + 1.079 / m) * m * m / sum;
        let estimate = if raw <= 2.5 * m && empty > 0 {
            m * (m / empty as f64).ln()
        } else {
            raw
        };
        estimate.round() as u64
    }
}

/// Limits on visitors, and what is counted of them.
pub struct Quotas {
    window: Mutex<Minute>,
    search_slots: Semaphore,
    /// This hour's and today's different guests (UTC).
    visitors: Mutex<(Distinct, Distinct)>,
    requests: [AtomicU64; 4],
    guests: AtomicU64,
    refused_closed: AtomicU64,
    refused_quota: AtomicU64,
    refused_busy: AtomicU64,
    proxy_warned: AtomicBool,
}

impl Default for Quotas {
    fn default() -> Self {
        Self {
            window: Mutex::new(Minute::default()),
            search_slots: Semaphore::new(SEARCH_SLOTS),
            visitors: Mutex::new((Distinct::new(), Distinct::new())),
            requests: Default::default(),
            guests: AtomicU64::new(0),
            refused_closed: AtomicU64::new(0),
            refused_quota: AtomicU64::new(0),
            refused_busy: AtomicU64::new(0),
            proxy_warned: AtomicBool::new(false),
        }
    }
}

impl Quotas {
    /// Counts one request; `false` when it is over the overall limit for `now_s`'s minute, or
    /// its own kind's: too many searches must not stop someone browsing.
    pub fn allow(&self, cfg: &DemoConfig, key: &str, kind: Kind, now_s: u64) -> bool {
        self.requests[kind as usize].fetch_add(1, Ordering::Relaxed);
        let Ok(mut window) = self.window.lock() else {
            return false;
        };
        window.roll(now_s / 60);
        let counts = window.clients.entry(key.to_string()).or_default();
        counts.requests += 1;
        let within_kind = match kind {
            Kind::Request => true,
            Kind::Search => {
                counts.searches += 1;
                counts.searches <= cfg.searches_per_min
            }
            Kind::Download => {
                counts.downloads += 1;
                counts.downloads <= cfg.downloads_per_min
            }
            Kind::Start => {
                counts.starts += 1;
                counts.starts <= cfg.guests_per_min
            }
        };
        let allowed = counts.requests <= cfg.requests_per_min && within_kind;
        if !allowed {
            let first = !std::mem::replace(&mut counts.logged, true);
            window.over_quota += 1;
            self.refused_quota.fetch_add(1, Ordering::Relaxed);
            // Never the address: /privacy says Studio does not store it.
            if first {
                tracing::info!(kind = ?kind, "A demo visitor went over a quota");
            }
        }
        allowed
    }

    /// Counts a request against its address's quotas, or gives the refusal to send.
    pub fn admit(
        &self,
        cfg: &DemoConfig,
        headers: &HeaderMap,
        peer: Option<IpAddr>,
        kind: Kind,
        now_s: u64,
    ) -> Result<(), Box<Response>> {
        self.check_proxy_setup(cfg, headers);
        let key = client_key(headers, peer, cfg.trusted_proxies);
        if self.allow(cfg, &key, kind, now_s) {
            Ok(())
        } else {
            Err(Box::new(too_many(
                60 - now_s % 60,
                "too many requests: try again in a minute",
            )))
        }
    }

    /// Forgets the addresses of a minute that has ended, with or without a later visitor.
    pub fn forget_past(&self, now_s: u64) {
        if let Ok(mut window) = self.window.lock() {
            window.roll(now_s / 60);
        }
    }

    fn note_visitor(&self, guest: &str, now_s: u64) {
        if let Ok(mut visitors) = self.visitors.lock() {
            visitors.0.note(guest, now_s / 3600);
            visitors.1.note(guest, now_s / 86400);
        }
    }

    /// Someone accepted the terms and has a new id.
    pub fn note_guest(&self) {
        self.guests.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut window) = self.window.lock() {
            window.guests += 1;
        }
    }

    fn note_closed(&self) {
        self.refused_closed.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut window) = self.window.lock() {
            window.closed += 1;
        }
    }

    fn note_busy(&self) {
        self.refused_busy.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut window) = self.window.lock() {
            window.busy += 1;
        }
    }

    /// A setting that doesn't match the requests gives everyone one quota, or lets them dodge it.
    fn check_proxy_setup(&self, cfg: &DemoConfig, headers: &HeaderMap) {
        let forwarded = headers
            .get_all("x-forwarded-for")
            .iter()
            .filter_map(|v| v.to_str().ok())
            .flat_map(|v| v.split(','))
            .count();
        let wrong = if cfg.trusted_proxies == 0 {
            forwarded > 0
        } else {
            forwarded < cfg.trusted_proxies
        };
        if wrong && !self.proxy_warned.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                trusted_proxies = cfg.trusted_proxies,
                forwarded_addresses = forwarded,
                "A demo visitor's X-Forwarded-For does not match ARIES_DEMO_TRUSTED_PROXIES: \
                 quotas may treat visitors as one address (docs/DEMO.md)"
            );
        }
    }

    /// The demo's part of `/metrics`.
    pub fn render_metrics(&self, now_s: u64) -> String {
        let (hour, day) = self
            .visitors
            .lock()
            .map(|v| (v.0.count(now_s / 3600), v.1.count(now_s / 86400)))
            .unwrap_or((0, 0));
        let load = |c: &AtomicU64| c.load(Ordering::Relaxed);
        let [request, search, download, start] = &self.requests;
        format!(
            "# HELP aries_demo_unique_visitors Different guest ids that used the demo this UTC hour or day (estimate; starts over with Studio)\n\
             # TYPE aries_demo_unique_visitors gauge\n\
             aries_demo_unique_visitors{{window=\"hour\"}} {hour}\n\
             aries_demo_unique_visitors{{window=\"day\"}} {day}\n\
             # HELP aries_demo_guests_total Times someone accepted the terms and opened the demo\n\
             # TYPE aries_demo_guests_total counter\n\
             aries_demo_guests_total {}\n\
             # HELP aries_demo_requests_total Requests to the routes open to demo visitors, and to open the demo\n\
             # TYPE aries_demo_requests_total counter\n\
             aries_demo_requests_total{{kind=\"browse\"}} {}\n\
             aries_demo_requests_total{{kind=\"search\"}} {}\n\
             aries_demo_requests_total{{kind=\"download\"}} {}\n\
             aries_demo_requests_total{{kind=\"start\"}} {}\n\
             # HELP aries_demo_refused_total Requests from demo visitors that were refused\n\
             # TYPE aries_demo_refused_total counter\n\
             aries_demo_refused_total{{reason=\"closed_route\"}} {}\n\
             aries_demo_refused_total{{reason=\"quota\"}} {}\n\
             aries_demo_refused_total{{reason=\"search_busy\"}} {}\n",
            load(&self.guests),
            load(request),
            load(search),
            load(download),
            load(start),
            load(&self.refused_closed),
            load(&self.refused_quota),
            load(&self.refused_busy),
        )
    }
}

/// Keeps the promise on `/privacy`: addresses stay in memory for up to a minute.
pub fn spawn_forgetting(state: Arc<AppState>, cfg: &DemoConfig) {
    tracing::info!(
        trusted_proxies = cfg.trusted_proxies,
        requests_per_min = cfg.requests_per_min,
        searches_per_min = cfg.searches_per_min,
        downloads_per_min = cfg.downloads_per_min,
        guests_per_min = cfg.guests_per_min,
        "The read-only demo is on"
    );
    tokio::spawn(async move {
        loop {
            let now = crate::db::now_s() as u64;
            tokio::time::sleep(std::time::Duration::from_secs(60 - now % 60)).await;
            state.visitor_quotas.forget_past(crate::db::now_s() as u64);
        }
    });
}

fn too_many(retry_after_s: u64, error: &str) -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(header::RETRY_AFTER, retry_after_s.to_string())],
        Json(json!({ "error": error, "demo": true })),
    )
        .into_response()
}

fn kind_of(req: &Request, route: &str) -> Kind {
    // Decoded as the handler will: `?%71=cat` is a search too.
    let searches = route == "/api/jobs"
        && Query::<Vec<(String, String)>>::try_from_uri(req.uri()).is_ok_and(|Query(pairs)| {
            pairs
                .iter()
                .any(|(name, value)| matches!(name.as_str(), "q" | "like") && !value.is_empty())
        });
    if searches {
        Kind::Search
    } else if route == "/api/images/:id" {
        Kind::Download
    } else {
        Kind::Request
    }
}

/// On every answer to a visitor, so a tab that thinks it is signed in learns it is a guest now: its
/// session cookie has gone, and this browser opened the demo in another tab.
pub const VISITOR_HEADER: &str = "x-aries-visitor";

/// A guest's request: closed routes refused, quotas applied, then the handler as a
/// [`User::visitor`]. `guest` is the hash of the id.
pub async fn serve_visitor(
    state: Arc<AppState>,
    cfg: &DemoConfig,
    guest: &str,
    req: Request,
    next: Next,
) -> Response {
    let mut res = answer_visitor(state, cfg, guest, req, next).await;
    res.headers_mut()
        .insert(VISITOR_HEADER, HeaderValue::from_static("1"));
    res
}

async fn answer_visitor(
    state: Arc<AppState>,
    cfg: &DemoConfig,
    guest: &str,
    mut req: Request,
    next: Next,
) -> Response {
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_default();
    if !visitor_may(req.method(), &route) {
        state.visitor_quotas.note_closed();
        tracing::debug!(method = %req.method(), route = %route, "Demo visitor refused: closed route");
        return (
            StatusCode::FORBIDDEN,
            Json(json!({ "error": "this is a read-only demo: sign in to do that", "demo": true })),
        )
            .into_response();
    }
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip());
    let kind = kind_of(&req, &route);
    let now = crate::db::now_s() as u64;
    state.visitor_quotas.note_visitor(guest, now);
    if let Err(refusal) = state
        .visitor_quotas
        .admit(cfg, req.headers(), peer, kind, now)
    {
        return *refusal;
    }
    // Held until the answer is ready: a search embeds its query on the CPU.
    let _slot = if kind == Kind::Search {
        match state.visitor_quotas.search_slots.try_acquire() {
            Ok(slot) => Some(slot),
            Err(_) => {
                state.visitor_quotas.note_busy();
                return too_many(2, "search is busy: try again in a moment");
            }
        }
    } else {
        None
    };
    req.extensions_mut().insert(User::visitor());
    next.run(req).await
}

const JOB_FIELDS: &[&str] = &[
    "id",
    "status",
    "prompt",
    "enhanced_prompt",
    "thinking",
    "has_thinking",
    "negative_prompt",
    "width",
    "height",
    "steps",
    "seed",
    "true_cfg_scale",
    "enhance",
    "enhance_max_tokens",
    "enhancer",
    "tiling",
    "wh_ratio",
    "ratio_follow",
    "model_id",
    "model_rev",
    "demo",
    "references",
    "version",
    "created_at",
    "started_at",
    "finished_at",
    "search_match",
    "search_strength",
    "search_words",
    "search_phrase",
];
const META_FIELDS: &[&str] = &["batch_id", "aspect", "size_multiplier", "quality", "index"];
const TASK_FIELDS: &[&str] = &["seq", "stage", "status", "step", "steps"];
const PROVENANCE_FIELDS: &[&str] = &["executor", "model", "timings", "peak_mem_bytes"];

/// What a visitor is sent of a job: the picture's own story (prompts, settings, references,
/// timings), not where it ran, nor anything about the account. Built from these lists, so a
/// field added to the record stays out until it is named here.
pub fn visitor_job(job: &JobRecord) -> Value {
    let all = serde_json::to_value(job).unwrap_or_default();
    let mut out = picked(&all, JOB_FIELDS);
    out["favorite"] = json!(false);
    out["meta"] = picked(&all["meta"], META_FIELDS);
    out["tasks"] = all["tasks"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| picked(t, TASK_FIELDS))
        .collect();
    let provenance: serde_json::Map<String, Value> = all["provenance"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(stage, p)| {
            let mut named = picked(p, PROVENANCE_FIELDS);
            named["agent"] = json!("gpu-host");
            (stage.clone(), named)
        })
        .collect();
    out["provenance"] = provenance.into();
    out
}

fn picked(from: &Value, keys: &[&str]) -> Value {
    keys.iter()
        .filter_map(|k| Some((k.to_string(), from.get(*k)?.clone())))
        .collect::<serde_json::Map<_, _>>()
        .into()
}

/// The status page for a visitor (`/api/system`'s shape, with the same field names): cards,
/// memory and the queue's length; hosts as `gpu-host-N`, with no addresses, process ids,
/// versions, job ids or free-text reasons.
pub fn scrub_system(system: &Value) -> Value {
    let agents: Vec<Value> = system["agents"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, agent)| {
            let host = format!("gpu-host-{}", i + 1);
            let executors: Vec<Value> = agent["executors"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|e| {
                    let mut out = picked(
                        e,
                        &[
                            "id",
                            "mem_total_bytes",
                            "mem_used_bytes",
                            "mem_available_bytes",
                            "overflow_host_bytes",
                            "stages",
                            "state",
                            "current_stage",
                            "current_model_id",
                            "current_model_rev",
                            "load_s",
                            "lent",
                        ],
                    );
                    out["key"] = json!(format!("{host}:{}", e["id"].as_str().unwrap_or("")));
                    out["agent_id"] = json!(host);
                    out
                })
                .collect();
            let mut out = picked(
                agent,
                &[
                    "online",
                    "host_ram_total_bytes",
                    "host_ram_available_bytes",
                    "scrape_age_s",
                    "scrape_stale",
                    "models",
                ],
            );
            out["id"] = json!(host);
            out["executors"] = json!(executors);
            out
        })
        .collect();
    let queues: Vec<Value> = system["queues"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|q| {
            let mut out = picked(
                q,
                &["stage", "queued", "running", "blocked", "oldest_wait_s"],
            );
            out["blocked_reasons"] = json!([]);
            out
        })
        .collect();
    json!({ "agents": agents, "queues": queues })
}

/// Reasons name hosts and cards; a visitor cannot send anything anyway.
pub fn visitor_capabilities(caps: &CapabilitiesResponse) -> Value {
    let all = serde_json::to_value(caps).unwrap_or_default();
    let each = |list: &str, fields: &[&str]| -> Value {
        all[list]
            .as_array()
            .into_iter()
            .flatten()
            .map(|v| picked(v, fields))
            .collect()
    };
    let mut out = picked(
        &all,
        &[
            "max_px_now",
            "max_px_ever",
            "max_references_now",
            "max_references_ever",
            "enhance_now",
            "enhance_ever",
            "enhance_max_references_now",
            "enhance_max_references_ever",
        ],
    );
    out["sizes"] = each(
        "sizes",
        &["width", "height", "px", "servable_now", "servable_ever"],
    );
    out["stages"] = each("stages", &["stage", "servable_now", "servable_ever"]);
    out["cloud_enhancers"] = each(
        "cloud_enhancers",
        &["id", "provider", "label", "model", "max_references"],
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;

    #[test]
    fn visitors_only_read_the_named_routes() {
        assert!(visitor_may(&Method::GET, "/api/jobs"));
        assert!(visitor_may(&Method::HEAD, "/api/images/:id"));
        assert!(!visitor_may(&Method::POST, "/api/jobs"));
        assert!(!visitor_may(&Method::PATCH, "/api/jobs/:id"));
        assert!(!visitor_may(&Method::DELETE, "/api/jobs/:id"));
        assert!(!visitor_may(&Method::PUT, "/api/preferences"));
        assert!(!visitor_may(&Method::POST, "/api/uploads"));
        assert!(!visitor_may(&Method::GET, "/api/tokens"));
        assert!(!visitor_may(&Method::GET, "/api/jobs/:id/preview"));
        assert!(!visitor_may(&Method::GET, "/api/some/later/route"));
        assert!(!visitor_may(&Method::GET, ""));
    }

    #[test]
    fn the_client_is_found_behind_the_trusted_proxies_only() {
        let mut h = HeaderMap::new();
        h.insert(
            "x-forwarded-for",
            "6.6.6.6, 1.2.3.4, 10.0.0.1".parse().unwrap(),
        );
        let peer = Some("10.0.0.2".parse().unwrap());
        assert_eq!(client_key(&h, peer, 0), "10.0.0.2");
        assert_eq!(client_key(&h, peer, 1), "10.0.0.1");
        assert_eq!(client_key(&h, peer, 2), "1.2.3.4");
        // Fewer entries than proxies, or nonsense: the connection's address.
        assert_eq!(client_key(&h, peer, 9), "10.0.0.2");
        h.insert("x-forwarded-for", "nonsense".parse().unwrap());
        assert_eq!(client_key(&h, peer, 1), "10.0.0.2");
        assert_eq!(client_key(&HeaderMap::new(), None, 0), "unknown");
    }

    #[test]
    fn an_ipv6_client_counts_by_its_64() {
        let h = HeaderMap::new();
        let a = Some("2001:db8:1:2:aaaa::1".parse().unwrap());
        let b = Some("2001:db8:1:2:bbbb::9".parse().unwrap());
        assert_eq!(client_key(&h, a, 0), client_key(&h, b, 0));
        let mapped = Some("::ffff:1.2.3.4".parse().unwrap());
        assert_eq!(client_key(&h, mapped, 0), "1.2.3.4");
    }

    #[test]
    fn quotas_are_per_client_and_per_minute() {
        let cfg = DemoConfig {
            requests_per_min: 3,
            searches_per_min: 1,
            downloads_per_min: 1,
            ..Default::default()
        };
        let q = Quotas::default();
        assert!(q.allow(&cfg, "a", Kind::Search, 60));
        assert!(!q.allow(&cfg, "a", Kind::Search, 61));
        // Another client is not held back, nor another minute.
        assert!(q.allow(&cfg, "b", Kind::Search, 62));
        assert!(q.allow(&cfg, "a", Kind::Search, 120));
        assert!(q.allow(&cfg, "c", Kind::Request, 180));
        assert!(q.allow(&cfg, "c", Kind::Request, 181));
        assert!(q.allow(&cfg, "c", Kind::Request, 182));
        assert!(!q.allow(&cfg, "c", Kind::Request, 183));
    }

    #[test]
    fn over_one_limit_does_not_stop_the_rest() {
        let cfg = DemoConfig {
            requests_per_min: 5,
            searches_per_min: 1,
            downloads_per_min: 1,
            ..Default::default()
        };
        let q = Quotas::default();
        assert!(q.allow(&cfg, "a", Kind::Download, 60));
        assert!(!q.allow(&cfg, "a", Kind::Download, 60));
        assert!(q.allow(&cfg, "a", Kind::Search, 60));
        assert!(!q.allow(&cfg, "a", Kind::Search, 60));
        assert!(q.allow(&cfg, "a", Kind::Request, 60));
        // The overall limit still holds, for every kind.
        assert!(!q.allow(&cfg, "a", Kind::Request, 60));
    }

    #[test]
    fn addresses_go_when_the_minute_ends_without_another_visitor() {
        let q = Quotas::default();
        assert!(q.allow(&DemoConfig::default(), "a", Kind::Request, 60));
        q.forget_past(119);
        assert_eq!(q.window.lock().unwrap().clients.len(), 1);
        q.forget_past(120);
        assert!(q.window.lock().unwrap().clients.is_empty());
    }

    #[test]
    fn a_search_is_a_search_however_it_is_spelled() {
        let kind = |uri: &str| {
            let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
            kind_of(&req, "/api/jobs")
        };
        for search in [
            "/api/jobs?q=fox",
            "/api/jobs?like=01A",
            "/api/jobs?%71=fox",
            "/api/jobs?limit=1&%6Cike=01A",
        ] {
            assert!(matches!(kind(search), Kind::Search), "{search}");
        }
        for plain in [
            "/api/jobs",
            "/api/jobs?limit=1",
            "/api/jobs?q=",
            "/api/jobs?%71=",
        ] {
            assert!(matches!(kind(plain), Kind::Request), "{plain}");
        }
    }

    #[test]
    fn a_job_says_nothing_of_where_it_ran() {
        let mut job: JobRecord = serde_json::from_str(
            r#"{
            "id": "j", "prompt": "a fox", "enhance": false, "width": 64, "height": 64,
            "steps": 4, "seed": 1, "true_cfg_scale": 4.0, "tiling": false, "status": "done",
            "meta": { "aspect": "1:1", "note": "private" }, "favorite": true, "version": 1,
            "negative_prompt": "", "created_at": 1.0, "has_thinking": false, "started_at": null, "finished_at": null,
            "enhanced_prompt": null, "thinking": null, "enhance_max_tokens": null, "enhancer": null, "wh_ratio": null, "ratio_follow": null, "model_id": null, "model_rev": null, "error_class": null, "error_detail": null, "blocked_reason": null,
            "provenance": { "decode": {
                "agent": "my-private-host", "executor": "xpu0", "device": "0000:08:00.0",
                "backend_kind": "xpu", "torch_version": "t", "worker_version": "w",
                "model": { "id": "m", "rev": "r", "dtype": "d" }, "peak_mem_bytes": 5,
                "timings": { "decode_s": 2.0 }
            }}
        }"#,
        )
        .unwrap();
        job.tasks = serde_json::from_value(json!([{
            "id": "task-private", "job_id": "j", "seq": 3, "stage": "decode", "status": "done",
            "attempts": 1, "current_attempt": 1, "assigned_agent": "my-private-host",
            "assigned_executor": "xpu0", "blocked_reason": null, "step": 4, "steps": 4,
            "s_per_step": 1.0, "sub_stage": null, "stats": null, "error_class": null,
            "error_detail": "private", "queued_at": null, "started_at": null, "finished_at": null
        }]))
        .unwrap();
        let seen = visitor_job(&job);
        let text = seen.to_string();
        for private in ["my-private-host", "0000:08", "private", "torch_version"] {
            assert!(!text.contains(private), "{private} in {text}");
        }
        assert_eq!(seen["favorite"], false);
        assert_eq!(seen["prompt"], "a fox");
        assert_eq!(seen["meta"], json!({ "aspect": "1:1" }));
        assert_eq!(seen["tasks"][0]["stage"], "decode");
        let decode = &seen["provenance"]["decode"];
        assert_eq!(decode["executor"], "xpu0");
        assert_eq!(decode["peak_mem_bytes"], 5);
    }

    #[test]
    fn unique_visitors_are_counted_without_keeping_ids() {
        let mut d = Distinct::new();
        assert_eq!(d.count(7), 0);
        for _ in 0..50 {
            d.note("guest-1", 7);
        }
        assert_eq!(d.count(7), 1);
        d.note("guest-2", 7);
        d.note("guest-3", 7);
        assert_eq!(d.count(7), 3);
        // The period ended: nothing to report, with or without another visitor.
        assert_eq!(d.count(8), 0);
        d.note("guest-1", 8);
        assert_eq!(d.count(8), 1);

        let mut many = Distinct::new();
        for i in 0..20_000 {
            many.note(&format!("guest-{i}"), 1);
        }
        let n = many.count(1) as f64;
        assert!((n - 20_000.0).abs() < 20_000.0 * 0.06, "{n}");
        assert_eq!(many.registers.len(), REGISTERS);
    }

    fn text_has(text: &str, line: &str) -> bool {
        text.lines().any(|l| l == line)
    }

    #[test]
    fn new_guests_have_their_own_limit() {
        let cfg = DemoConfig {
            guests_per_min: 1,
            ..Default::default()
        };
        let q = Quotas::default();
        assert!(q.allow(&cfg, "a", Kind::Start, 60));
        assert!(!q.allow(&cfg, "a", Kind::Start, 60));
        assert!(q.allow(&cfg, "a", Kind::Request, 60), "browsing goes on");
        assert!(q.allow(&cfg, "b", Kind::Start, 60));
        assert!(q.allow(&cfg, "a", Kind::Start, 120));
    }

    #[test]
    fn metrics_count_visitors_and_refusals() {
        let cfg = DemoConfig {
            downloads_per_min: 1,
            ..Default::default()
        };
        let q = Quotas::default();
        q.note_visitor("a", 3600);
        q.note_visitor("b", 3600);
        q.note_closed();
        q.note_guest();
        assert!(text_has(
            &q.render_metrics(3600),
            "aries_demo_guests_total 1"
        ));
        assert!(q.allow(&cfg, "a", Kind::Download, 3600));
        assert!(!q.allow(&cfg, "a", Kind::Download, 3600));
        let text = q.render_metrics(3600);
        assert!(
            text.contains("aries_demo_unique_visitors{window=\"hour\"} 2"),
            "{text}"
        );
        assert!(
            text.contains("aries_demo_unique_visitors{window=\"day\"} 2"),
            "{text}"
        );
        assert!(
            text.contains("aries_demo_requests_total{kind=\"download\"} 2"),
            "{text}"
        );
        assert!(
            text.contains("aries_demo_refused_total{reason=\"quota\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("aries_demo_refused_total{reason=\"closed_route\"} 1"),
            "{text}"
        );
        // A new hour starts over, a new day too.
        let later = q.render_metrics(3600 * 2);
        assert!(
            later.contains("aries_demo_unique_visitors{window=\"hour\"} 0"),
            "{later}"
        );
        assert!(
            later.contains("aries_demo_unique_visitors{window=\"day\"} 2"),
            "{later}"
        );
    }

    #[test]
    fn the_status_page_names_no_host() {
        let system = json!({
            "agents": [{
                "id": "my-private-host", "agent_version": "1", "backend_kind": "xpu",
                "metrics_url": "http://192.0.2.1:9100/metrics", "online": true,
                "host_ram_total_bytes": 1, "models": [], "limits": {},
                "executors": [{
                    "key": "my-private-host:xpu0", "id": "xpu0", "agent_id": "my-private-host",
                    "device": "0000:08:00.0", "state": "busy", "worker_pid": 123,
                    "current_task_id": "t1", "disabled_reason": "private note",
                    "mem_total_bytes": 2
                }]
            }],
            "queues": [{ "stage": "denoise", "queued": 1, "running": 1, "blocked": 1,
                         "oldest_wait_s": 3.0, "blocked_reasons": ["my-private-host is busy"] }]
        });
        let out = scrub_system(&system);
        let text = out.to_string();
        for private in [
            "my-private-host",
            "192.0.2.1",
            "0000:08",
            "123",
            "t1",
            "private note",
        ] {
            assert!(!text.contains(private), "{private} in {text}");
        }
        assert_eq!(out["agents"][0]["executors"][0]["state"], "busy");
        assert_eq!(out["agents"][0]["executors"][0]["key"], "gpu-host-1:xpu0");
        assert_eq!(out["queues"][0]["queued"], 1);
    }
}
