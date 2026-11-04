//! Sign-in with Google (or another OpenID Connect provider), off unless configured
//! (docs/AUTH.md). No passwords: the provider vouches for an email address, and only the
//! addresses on the allowlist get a session. Sessions live in SQLite; the browser holds an
//! opaque token in an HttpOnly cookie.
//!
//! With auth on, every `/api/*` request carries a [`User`]; jobs, images, uploads and
//! preferences belong to the account that made them. With it off, the one [`User`] sees
//! everything, as before.

use crate::scheduler::AppState;
use anyhow::{anyhow, bail, Context, Result};
use axum::extract::connect_info::ConnectInfo;
use axum::extract::{Query, Request, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Redirect, Response};
use axum::Json;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{info, warn};

pub const SESSION_COOKIE: &str = "aries_session";
/// Ties a sign-in to the browser that started it: the callback must come with this.
pub const LOGIN_COOKIE: &str = "aries_login";
/// A demo guest's id (docs/DEMO.md). Its own cookie and table: never a session.
pub const GUEST_COOKIE: &str = "aries_guest";
/// How long a guest stays one before accepting the terms again. Not renewed by use.
pub const GUEST_LIFE_S: u64 = 30 * 86400;
/// Guest rows kept at most, the oldest going first: anyone can make them, so the table must not
/// grow without limit, and refusing new ones would let someone close the demo. A guest who loses
/// theirs clicks again.
pub const MAX_GUESTS: u32 = 100_000;
/// How long a sign-in may take between the redirect to the provider and the callback.
pub const LOGIN_STATE_TTL_S: f64 = 600.0;
/// How long a link from `aries-image-studio login-link` stays usable.
pub const LOGIN_LINK_TTL_S: f64 = 600.0;
/// A session's expiry moves forward at most this often, so requests rarely write.
const RENEW_EVERY_S: f64 = 3600.0;
const GOOGLE_ISSUER: &str = "https://accounts.google.com";

#[derive(Clone)]
pub struct AuthConfig {
    /// The provider's issuer URL; its `/.well-known/openid-configuration` names the rest.
    pub issuer: String,
    pub client_id: String,
    pub client_secret: String,
    /// Lowercase email addresses that may sign in.
    pub allowed_emails: Vec<String>,
    /// The account that owns what was made before auth was switched on, and the only one
    /// that may lend, disable or enable cards.
    pub owner: String,
    /// Studio's addresses as browsers see them (`https://studio.example`), one per way in
    /// (say an intranet and an internet name). Each request uses the one its `Host` names,
    /// else the first: the callback is `{that}/auth/callback`, and the cookie is `Secure`
    /// on https.
    pub public_urls: Vec<String>,
    pub session_days: u32,
}

impl std::fmt::Debug for AuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthConfig")
            .field("issuer", &self.issuer)
            .field("client_id", &self.client_id)
            .field("client_secret", &"<redacted>")
            .field("allowed_emails", &self.allowed_emails)
            .field("owner", &self.owner)
            .field("public_urls", &self.public_urls)
            .field("session_days", &self.session_days)
            .finish()
    }
}

impl AuthConfig {
    /// `None` unless `ARIES_OIDC_CLIENT_ID` is set; then the rest must be too.
    pub fn from_env() -> Result<Option<Self>> {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Result<Option<Self>> {
        let get = |k: &str| {
            var(k)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let Some(client_id) = get("ARIES_OIDC_CLIENT_ID") else {
            return Ok(None);
        };
        let client_secret = get("ARIES_OIDC_CLIENT_SECRET")
            .context("ARIES_OIDC_CLIENT_SECRET must be set with ARIES_OIDC_CLIENT_ID")?;
        let allowed_emails: Vec<String> = get("ARIES_AUTH_ALLOWED_EMAILS")
            .unwrap_or_default()
            .split(',')
            .map(|e| e.trim().to_lowercase())
            .filter(|e| !e.is_empty())
            .collect();
        if allowed_emails.is_empty() {
            bail!("ARIES_AUTH_ALLOWED_EMAILS must list who may sign in");
        }
        let owner = get("ARIES_AUTH_OWNER")
            .map(|o| o.to_lowercase())
            .unwrap_or_else(|| allowed_emails[0].clone());
        if !allowed_emails.contains(&owner) {
            bail!("ARIES_AUTH_OWNER ({owner}) must be in ARIES_AUTH_ALLOWED_EMAILS");
        }
        let public_urls: Vec<String> = get("ARIES_PUBLIC_URL")
            .context(
                "ARIES_PUBLIC_URL must be set with ARIES_OIDC_CLIENT_ID (the callback address)",
            )?
            .split(',')
            .map(|u| u.trim().trim_end_matches('/').to_string())
            .filter(|u| !u.is_empty())
            .collect();
        for u in &public_urls {
            reqwest::Url::parse(u)
                .with_context(|| format!("ARIES_PUBLIC_URL: {u} is not a URL"))?;
        }
        if public_urls.is_empty() {
            bail!("ARIES_PUBLIC_URL must name at least one address");
        }
        Ok(Some(Self {
            issuer: get("ARIES_OIDC_ISSUER")
                .unwrap_or_else(|| GOOGLE_ISSUER.to_string())
                .trim_end_matches('/')
                .to_string(),
            client_id,
            client_secret,
            allowed_emails,
            owner,
            public_urls,
            session_days: get("ARIES_SESSION_DAYS")
                .and_then(|d| d.parse().ok())
                .filter(|d| *d > 0)
                .unwrap_or(30),
        }))
    }

    /// The public URL this request came in on: the one whose `host[:port]` its
    /// `X-Forwarded-Host` or `Host` names, else the first. Only configured addresses are
    /// ever chosen, so a forged header can at worst pick another of them.
    pub fn public_url(&self, headers: &HeaderMap) -> &str {
        let wanted: Vec<String> = [HeaderName::from_static("x-forwarded-host"), header::HOST]
            .iter()
            .filter_map(|h| headers.get(h)?.to_str().ok())
            .map(|h| {
                h.split(',')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_ascii_lowercase()
            })
            .collect();
        self.public_urls
            .iter()
            .find(|u| {
                wanted
                    .iter()
                    .any(|w| authority(u).as_deref() == Some(w.as_str()))
            })
            .unwrap_or(&self.public_urls[0])
    }

    fn redirect_uri(&self, headers: &HeaderMap) -> String {
        format!("{}/auth/callback", self.public_url(headers))
    }

    /// `scheme://host[:port]` of each public URL, what browsers send as `Origin`.
    fn origins(&self) -> Vec<String> {
        self.public_urls
            .iter()
            .filter_map(|u| reqwest::Url::parse(u).ok())
            .map(|u| u.origin().ascii_serialization())
            .collect()
    }

    fn secure_cookie(&self, headers: &HeaderMap) -> bool {
        self.public_url(headers).starts_with("https://")
    }
}

/// `host[:port]` of a URL, as a `Host` header writes it (the port only when not the
/// scheme's default).
fn authority(url: &str) -> Option<String> {
    let u = reqwest::Url::parse(url).ok()?;
    let host = u.host_str()?.to_ascii_lowercase();
    Some(match u.port() {
        Some(p) => format!("{host}:{p}"),
        None => host,
    })
}

/// The provider's endpoints, from its discovery document.
#[derive(Debug, Clone, Deserialize)]
struct Discovery {
    issuer: String,
    authorization_endpoint: String,
    token_endpoint: String,
}

/// Auth at runtime: the config and the provider's endpoints, fetched on first use.
pub struct Auth {
    pub config: AuthConfig,
    discovery: tokio::sync::OnceCell<Discovery>,
    http: reqwest::Client,
}

impl Auth {
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config,
            discovery: tokio::sync::OnceCell::new(),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_default(),
        }
    }

    /// A failed fetch is not kept: the next sign-in tries again.
    async fn discovery(&self) -> Result<&Discovery> {
        self.discovery
            .get_or_try_init(|| async {
                let url = format!("{}/.well-known/openid-configuration", self.config.issuer);
                let d: Discovery = self
                    .http
                    .get(&url)
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await
                    .with_context(|| format!("bad discovery document at {url}"))?;
                Ok::<_, anyhow::Error>(d)
            })
            .await
    }
}

/// What a user sees. A visitor is its own case and not "no account", which sees everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope<'a> {
    /// Sign-in is off.
    Everyone,
    Account(&'a str),
    /// The images marked for the demo (docs/DEMO.md).
    Demo,
}

/// Who is asking. With auth off there is one user who owns everything (`email: None`).
#[derive(Debug, Clone)]
pub struct User {
    pub email: Option<String>,
    pub is_owner: bool,
    /// Came with an access token (the CLI, an AI tool), not a browser session.
    pub via_token: bool,
    /// A demo guest (docs/DEMO.md): no session, no token, read-only.
    pub visitor: bool,
}

impl User {
    pub fn everyone() -> Self {
        Self {
            email: None,
            is_owner: true,
            via_token: false,
            visitor: false,
        }
    }

    pub fn visitor() -> Self {
        Self {
            email: None,
            is_owner: false,
            via_token: false,
            visitor: true,
        }
    }

    /// Whose things this user sees.
    pub fn scope(&self) -> Scope<'_> {
        match (self.visitor, self.email.as_deref()) {
            (true, _) => Scope::Demo,
            (false, Some(email)) => Scope::Account(email),
            (false, None) => Scope::Everyone,
        }
    }

    /// The account that what this user makes belongs to: `None` with sign-in off. For writes,
    /// which visitors never reach (demo.rs).
    pub fn account(&self) -> Option<&str> {
        self.email.as_deref()
    }
}

/// A signed-in session, as stored.
#[derive(Debug, Clone)]
pub struct Session {
    pub email: String,
    pub name: String,
    pub picture: String,
    pub expires_at: f64,
}

fn random_token() -> String {
    let mut buf = [0u8; 32];
    getrandom::getrandom(&mut buf).expect("the OS random source failed");
    URL_SAFE_NO_PAD.encode(buf)
}

/// Sessions are stored by the hash of their token: the database never holds a usable one.
pub fn token_hash(token: &str) -> String {
    hex::encode(Sha256::digest(token.as_bytes()))
}

fn session_token(headers: &HeaderMap) -> Option<String> {
    cookie_value(headers, SESSION_COOKIE)
}

fn guest_token(headers: &HeaderMap) -> Option<String> {
    cookie_value(headers, GUEST_COOKIE)
}

fn cookie_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .filter_map(|c| c.trim().split_once('='))
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v.to_string())
        .filter(|v| !v.is_empty())
}

/// The database could not say whether a session exists. Not the same as there being none: the
/// cookie stays, and the request can be tried again.
struct SessionUnknown;

const SESSION_UNKNOWN: &str = "could not check the session: try again";

/// The browser's session, if its address is still on the allowlist, its expiry moved
/// forward when due: one in use does not run out, so a device signed in while online keeps
/// working through an outage. With the renewed cookie to send back, when it was renewed.
async fn current_session(
    state: &AppState,
    auth: &Auth,
    headers: &HeaderMap,
) -> Result<Option<(Session, Option<HeaderValue>)>, SessionUnknown> {
    let Some(token) = session_token(headers) else {
        return Ok(None);
    };
    let hash = token_hash(&token);
    let session = match state.db.get_session(&hash).await {
        Ok(Some(session)) => session,
        Ok(None) => return Ok(None),
        Err(e) => {
            warn!(error = %e, "Could not look up a session");
            return Err(SessionUnknown);
        }
    };
    // Taken off the allowlist since signing in: signed out, here and on every device.
    if !auth.config.allowed_emails.contains(&session.email) {
        let _ = state.db.delete_session(&hash).await;
        return Ok(None);
    }
    let life_s = auth.config.session_days as u64 * 86400;
    let renewed = crate::db::now_s() + life_s as f64;
    if session.expires_at < renewed - RENEW_EVERY_S
        && state.db.renew_session(&hash, renewed).await.is_ok()
    {
        let cookie =
            HeaderValue::from_str(&session_cookie(&auth.config, headers, &token, life_s)).ok();
        return Ok(Some((session, cookie)));
    }
    Ok(Some((session, None)))
}

/// The hash of this browser's guest id, if it has one still good: what visitors are counted by.
async fn current_guest(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<Option<String>, SessionUnknown> {
    let Some(token) = guest_token(headers) else {
        return Ok(None);
    };
    let hash = token_hash(&token);
    match state.db.guest_exists(&hash).await {
        Ok(true) => Ok(Some(hash)),
        Ok(false) => Ok(None),
        Err(e) => {
            warn!(error = %e, "Could not look up a guest");
            Err(SessionUnknown)
        }
    }
}

fn session_cookie(cfg: &AuthConfig, headers: &HeaderMap, token: &str, max_age_s: u64) -> String {
    auth_cookie(cfg, headers, SESSION_COOKIE, token, max_age_s)
}

fn guest_cookie(cfg: &AuthConfig, headers: &HeaderMap, token: &str, max_age_s: u64) -> String {
    auth_cookie(cfg, headers, GUEST_COOKIE, token, max_age_s)
}

fn auth_cookie(
    cfg: &AuthConfig,
    headers: &HeaderMap,
    name: &str,
    token: &str,
    max_age_s: u64,
) -> String {
    format!(
        "{name}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age_s}{}",
        if cfg.secure_cookie(headers) {
            "; Secure"
        } else {
            ""
        }
    )
}

fn json_error(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({ "error": msg }))).into_response()
}

/// Access tokens (docs/INTEGRATIONS.md) start with this, so they are easy to spot in a
/// leaked file and to tell from a session cookie.
pub const API_TOKEN_PREFIX: &str = "aries_";
/// A token's "last used" moves forward at most this often, so requests rarely write.
const TOUCH_EVERY_S: f64 = 600.0;

/// A new access token: the prefix and 256 random bits.
pub fn new_api_token() -> String {
    format!("{API_TOKEN_PREFIX}{}", random_token())
}

/// `Authorization: Bearer <token>`, if the request has one.
fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let v = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = v.trim().split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| token.trim())
        .filter(|t| !t.is_empty())
}

/// The account of an access token, if the token exists and its account may still sign in.
async fn token_user(state: &AppState, auth: &Auth, token: &str) -> Option<User> {
    let (id, email, last_used) = state
        .db
        .get_api_token(&token_hash(token))
        .await
        .ok()
        .flatten()?;
    // Taken off the allowlist: its tokens stop working with its sessions.
    if !auth.config.allowed_emails.contains(&email) {
        return None;
    }
    if last_used.is_none_or(|t| t < crate::db::now_s() - TOUCH_EVERY_S) {
        let _ = state.db.touch_api_token(&id).await;
    }
    Some(User {
        is_owner: email == auth.config.owner,
        email: Some(email),
        via_token: true,
        visitor: false,
    })
}

fn from_another_site(auth: &Auth, headers: &HeaderMap) -> bool {
    headers.get(header::ORIGIN).is_some_and(|origin| {
        !origin
            .to_str()
            .is_ok_and(|o| auth.config.origins().iter().any(|a| a == o))
    })
}

/// Who a request comes from: an access token when it carries one (a bad one is refused,
/// never passed over for the cookie), else the browser's session, with its renewed cookie
/// when it was renewed. `Err` is the answer to send instead.
pub async fn request_user(
    state: &AppState,
    headers: &HeaderMap,
    method: &Method,
) -> Result<(User, Option<HeaderValue>), Box<Response>> {
    let Some(auth) = &state.auth else {
        return Ok((User::everyone(), None));
    };
    if let Some(token) = bearer_token(headers) {
        return match token_user(state, auth, token).await {
            Some(user) => Ok((user, None)),
            None => Err(Box::new(json_error(
                StatusCode::UNAUTHORIZED,
                "the access token is not valid (revoked?)",
            ))),
        };
    }
    // Writes from another site's page are refused (SameSite=Lax already keeps the cookie
    // off them). Token requests carry no cookie, so they need no such check.
    if !matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS)
        && from_another_site(auth, headers)
    {
        return Err(Box::new(json_error(
            StatusCode::FORBIDDEN,
            "cross-site request refused",
        )));
    }
    let (session, renewed) = match current_session(state, auth, headers).await {
        Ok(Some(found)) => found,
        Ok(None) => {
            return Err(Box::new(json_error(
                StatusCode::UNAUTHORIZED,
                "sign in first",
            )))
        }
        Err(SessionUnknown) => {
            return Err(Box::new(json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                SESSION_UNKNOWN,
            )))
        }
    };
    let is_owner = session.email == auth.config.owner;
    Ok((
        User {
            email: Some(session.email),
            is_owner,
            via_token: false,
            visitor: false,
        },
        renewed,
    ))
}

/// Middleware on `/api/*`: puts the [`User`] on the request, or answers 401.
pub async fn require_user(
    State(state): State<Arc<AppState>>,
    mut req: Request,
    next: Next,
) -> Response {
    // A guest who accepted the terms is a visitor, unless a session or token says otherwise.
    // Without an id it is like any request with nothing to identify it. A cookie that ended is
    // told so and forgotten, so the app asks Studio who this is now.
    let demo = state.config.demo.clone().filter(|_| state.auth.is_some());
    if let Some(cfg) = &demo {
        if bearer_token(req.headers()).is_none() && session_token(req.headers()).is_none() {
            match current_guest(&state, req.headers()).await {
                Ok(Some(guest)) => {
                    return crate::demo::serve_visitor(state.clone(), cfg, &guest, req, next).await
                }
                Ok(None) => {}
                Err(SessionUnknown) => {
                    return json_error(StatusCode::SERVICE_UNAVAILABLE, SESSION_UNKNOWN)
                }
            }
        }
    }
    let (user, renewed) = match request_user(&state, req.headers(), req.method()).await {
        Ok(u) => u,
        Err(res) => {
            let mut res = *res;
            if demo.is_some()
                && res.status() == StatusCode::UNAUTHORIZED
                && bearer_token(req.headers()).is_none()
            {
                if let Some(auth) = &state.auth {
                    let ended = if session_token(req.headers()).is_some() {
                        info!("A session ended: its cookie is cleared");
                        Some(session_cookie(&auth.config, req.headers(), "", 0))
                    } else {
                        guest_token(req.headers())
                            .map(|_| guest_cookie(&auth.config, req.headers(), "", 0))
                    };
                    if let Some(cookie) = ended.and_then(|c| HeaderValue::from_str(&c).ok()) {
                        res.headers_mut().append(header::SET_COOKIE, cookie);
                    }
                }
            }
            return res;
        }
    };
    req.extensions_mut().insert(user);
    let mut res = next.run(req).await;
    if let Some(cookie) = renewed {
        res.headers_mut().append(header::SET_COOKIE, cookie);
    }
    res
}

/// `GET /api/me`: who is signed in (or whose the access token is); `{auth: false}` when auth is off, 401 when signed out.
pub async fn me(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let Some(auth) = &state.auth else {
        return Json(json!({ "auth": false })).into_response();
    };
    // An access token (the CLI checking its login): whose it is.
    if bearer_token(&headers).is_some() {
        return match request_user(&state, &headers, &Method::GET).await {
            Ok((user, _)) => Json(json!({
                "auth": true,
                "email": user.email,
                "owner": user.is_owner,
                "token": true,
            }))
            .into_response(),
            Err(res) => *res,
        };
    }
    let session = match current_session(&state, auth, &headers).await {
        Ok(session) => session,
        Err(SessionUnknown) => return json_error(StatusCode::SERVICE_UNAVAILABLE, SESSION_UNKNOWN),
    };
    match session {
        Some((s, renewed)) => {
            let mut res = Json(json!({
                "auth": true,
                "email": s.email,
                "name": s.name,
                "picture": s.picture,
                "owner": s.email == auth.config.owner,
                "demo_mode": state.config.demo.is_some(),
            }))
            .into_response();
            if let Some(cookie) = renewed {
                res.headers_mut().append(header::SET_COOKIE, cookie);
            }
            res
        }
        // In a Studio with a demo: a guest who accepted the terms, or someone who can choose to.
        // A cookie that no longer means anything is forgotten.
        None if state.config.demo.is_some() => {
            let guest = match current_guest(&state, &headers).await {
                Ok(guest) => guest,
                Err(SessionUnknown) => {
                    return json_error(StatusCode::SERVICE_UNAVAILABLE, SESSION_UNKNOWN)
                }
            };
            let mut res = if guest.is_some() {
                Json(json!({
                    "auth": true,
                    "visitor": true,
                    "owner": false,
                    "demo_mode": true,
                }))
                .into_response()
            } else {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({ "auth": true, "demo_mode": true, "error": "sign in first" })),
                )
                    .into_response()
            };
            let mut forget = Vec::new();
            if session_token(&headers).is_some() {
                forget.push(session_cookie(&auth.config, &headers, "", 0));
            }
            if guest.is_none() && guest_token(&headers).is_some() {
                forget.push(guest_cookie(&auth.config, &headers, "", 0));
            }
            for cookie in forget {
                if let Ok(cookie) = HeaderValue::from_str(&cookie) {
                    res.headers_mut().append(header::SET_COOKIE, cookie);
                }
            }
            res
        }
        None => (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "auth": true, "error": "sign in first" })),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct GuestStart {
    pub accept_terms: bool,
}

/// `POST /auth/guest`: the click on the sign-in screen that accepts the terms. A new id each time;
/// Studio keeps its hash and when it was made, not the address or anything the guest looks at.
pub async fn start_guest(
    State(state): State<Arc<AppState>>,
    peer: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    Json(body): Json<GuestStart>,
) -> Response {
    let (Some(auth), Some(cfg)) = (&state.auth, &state.config.demo) else {
        return json_error(StatusCode::NOT_FOUND, "this Studio has no demo");
    };
    if from_another_site(auth, &headers) {
        return json_error(StatusCode::FORBIDDEN, "cross-site request refused");
    }
    if !body.accept_terms {
        return json_error(StatusCode::BAD_REQUEST, "the demo needs the terms accepted");
    }
    // Signed in meanwhile (a sign-in screen left open in another tab): no id, or it would take
    // over when the session ends. The app asks who this is next, and finds the account.
    match current_session(&state, auth, &headers).await {
        Ok(Some((_, renewed))) => {
            let mut res = StatusCode::NO_CONTENT.into_response();
            if let Some(cookie) = renewed {
                res.headers_mut().append(header::SET_COOKIE, cookie);
            }
            end_guest(&state, auth, &headers, &mut res).await;
            return res;
        }
        Ok(None) => {}
        Err(SessionUnknown) => return json_error(StatusCode::SERVICE_UNAVAILABLE, SESSION_UNKNOWN),
    }
    let quotas = &state.visitor_quotas;
    let now = crate::db::now_s();
    if let Err(refusal) = quotas.admit(
        cfg,
        &headers,
        peer.map(|p| p.0.ip()),
        crate::demo::Kind::Start,
        now as u64,
    ) {
        return *refusal;
    }
    let token = random_token();
    if let Err(e) = state
        .db
        .create_guest(&token_hash(&token), now + GUEST_LIFE_S as f64, MAX_GUESTS)
        .await
    {
        warn!(error = %e, "Could not start a guest");
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "could not open the demo: try again",
        );
    }
    if let Some(old) = guest_token(&headers) {
        delete_guest(&state, &old).await;
    }
    quotas.note_guest();
    (
        StatusCode::NO_CONTENT,
        [(
            header::SET_COOKIE,
            guest_cookie(&auth.config, &headers, &token, GUEST_LIFE_S),
        )],
    )
        .into_response()
}

#[derive(Debug, Deserialize)]
pub struct LoginQuery {
    pub next: Option<String>,
}

/// Only paths of this site: `/gallery`, not `//evil.example` or `https://…`. Browsers drop tabs and
/// newlines from a URL, so with those `/<TAB>/evil.example` would be another site.
fn safe_next(next: Option<&str>) -> String {
    match next {
        Some(n)
            if n.starts_with('/')
                && !n.starts_with("//")
                && !n.contains('\\')
                && !n.chars().any(char::is_control) =>
        {
            n.to_string()
        }
        _ => "/".to_string(),
    }
}

fn pkce_challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

/// `GET /auth/login?next=/path`: off to the provider's sign-in page.
pub async fn login(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<LoginQuery>,
) -> Response {
    let Some(auth) = &state.auth else {
        return Redirect::to("/").into_response();
    };
    let discovery = match auth.discovery().await {
        Ok(d) => d,
        Err(e) => {
            warn!(error = %e, "OIDC discovery failed");
            return json_error(
                StatusCode::BAD_GATEWAY,
                "the sign-in provider cannot be reached",
            );
        }
    };
    let (login_state, nonce, verifier) = (random_token(), random_token(), random_token());
    let next = safe_next(q.next.as_deref());
    if let Err(e) = state
        .db
        .save_login_state(&login_state, &nonce, &verifier, &next)
        .await
    {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
    }
    let challenge = pkce_challenge(&verifier);
    let redirect_uri = auth.config.redirect_uri(&headers);
    let url = reqwest::Url::parse_with_params(
        &discovery.authorization_endpoint,
        &[
            ("response_type", "code"),
            ("client_id", auth.config.client_id.as_str()),
            ("redirect_uri", redirect_uri.as_str()),
            ("scope", "openid email profile"),
            ("state", login_state.as_str()),
            ("nonce", nonce.as_str()),
            ("code_challenge", challenge.as_str()),
            ("code_challenge_method", "S256"),
            ("prompt", "select_account"),
        ],
    );
    match url {
        Ok(u) => {
            let mut res = Redirect::to(u.as_str()).into_response();
            let cookie = auth_cookie(
                &auth.config,
                &headers,
                LOGIN_COOKIE,
                &login_state,
                LOGIN_STATE_TTL_S as u64,
            );
            if let Ok(cookie) = HeaderValue::from_str(&cookie) {
                res.headers_mut().append(header::SET_COOKIE, cookie);
            }
            res
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
pub struct CallbackQuery {
    pub code: Option<String>,
    pub state: Option<String>,
    pub error: Option<String>,
}

/// The claims Studio uses from the ID token.
#[derive(Debug, Deserialize)]
struct IdClaims {
    iss: String,
    aud: Value,
    exp: f64,
    nonce: Option<String>,
    email: Option<String>,
    #[serde(default)]
    email_verified: Value,
    #[serde(default)]
    name: String,
    #[serde(default)]
    picture: String,
}

/// The ID token's claims, checked. Its signature is not: the token comes straight from the
/// provider's token endpoint over TLS, in exchange for our client secret (OpenID Connect
/// Core §3.1.3.7 allows TLS to stand in for it).
fn check_id_token(
    id_token: &str,
    discovery: &Discovery,
    cfg: &AuthConfig,
    nonce: &str,
) -> Result<IdClaims> {
    let payload = id_token
        .split('.')
        .nth(1)
        .ok_or_else(|| anyhow!("malformed ID token"))?;
    let claims: IdClaims =
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(payload.trim_end_matches('='))?)?;
    // Google writes its issuer with or without the scheme (`accounts.google.com`).
    let bare = |i: &str| {
        i.trim_start_matches("https://")
            .trim_end_matches('/')
            .to_string()
    };
    if bare(&claims.iss) != bare(&discovery.issuer) {
        bail!("ID token from another issuer ({})", claims.iss);
    }
    let aud_ok = match &claims.aud {
        Value::String(a) => a == &cfg.client_id,
        Value::Array(a) => a.iter().any(|x| x.as_str() == Some(cfg.client_id.as_str())),
        _ => false,
    };
    if !aud_ok {
        bail!("ID token for another client");
    }
    if claims.exp < crate::db::now_s() {
        bail!("ID token expired");
    }
    if claims.nonce.as_deref() != Some(nonce) {
        bail!("ID token nonce does not match");
    }
    Ok(claims)
}

fn signin_error(reason: &str) -> Response {
    Redirect::to(&format!("/?auth_error={reason}")).into_response()
}

/// `GET /auth/callback`: the provider sends the browser back here with a code.
pub async fn callback(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<CallbackQuery>,
) -> Response {
    let Some(auth) = &state.auth else {
        return Redirect::to("/").into_response();
    };
    // Only the browser that started this sign-in may finish it: otherwise someone could send you
    // a link that signs you in to their account, where what you make next is theirs to read. Before
    // the state is used, so such a link can't spoil a sign-in in progress either.
    if q.state.is_none() || cookie_value(&headers, LOGIN_COOKIE) != q.state {
        info!("Sign-in refused: it was started in another browser");
        return signin_error("expired");
    }
    let Some(login) = (match q.state.as_deref() {
        Some(s) => state.db.take_login_state(s).await.ok().flatten(),
        None => None,
    }) else {
        // Expired, used already, or not ours.
        return signin_error("expired");
    };
    if q.error.is_some() {
        return signin_error("cancelled");
    }
    let Some(code) = q.code else {
        return signin_error("failed");
    };
    // The provider sent the browser to the callback on the address the sign-in started
    // from, so this request's host names the same redirect URI.
    let redirect_uri = auth.config.redirect_uri(&headers);
    let claims =
        match exchange_code(auth, &code, &redirect_uri, &login.verifier, &login.nonce).await {
            Ok(c) => c,
            Err(e) => {
                warn!(error = %e, "Sign-in failed");
                return signin_error("failed");
            }
        };
    let email = claims.email.unwrap_or_default().trim().to_lowercase();
    let verified = matches!(&claims.email_verified, Value::Bool(true))
        || claims.email_verified.as_str() == Some("true");
    if email.is_empty() || !verified || !auth.config.allowed_emails.contains(&email) {
        info!(email = %email, "Sign-in refused: not on the allowlist");
        return signin_error("not_allowed");
    }
    info!(email = %email, "Signed in");
    let mut res = start_session(
        &state,
        auth,
        &headers,
        &email,
        &claims.name,
        &claims.picture,
        login.next,
    )
    .await;
    let spent = auth_cookie(&auth.config, &headers, LOGIN_COOKIE, "", 0);
    if let Ok(cookie) = HeaderValue::from_str(&spent) {
        res.headers_mut().append(header::SET_COOKIE, cookie);
    }
    res
}

/// Creates a session for `email` and sends the browser on to `next` with its cookie.
async fn start_session(
    state: &AppState,
    auth: &Auth,
    headers: &HeaderMap,
    email: &str,
    name: &str,
    picture: &str,
    next: String,
) -> Response {
    let token = random_token();
    let max_age_s = auth.config.session_days as u64 * 86400;
    if let Err(e) = state
        .db
        .create_session(
            &token_hash(&token),
            email,
            name,
            picture,
            crate::db::now_s() + max_age_s as f64,
        )
        .await
    {
        return json_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
    }
    let mut res = (
        StatusCode::FOUND,
        [
            (header::LOCATION, next),
            (
                header::SET_COOKIE,
                session_cookie(&auth.config, headers, &token, max_age_s),
            ),
        ],
    )
        .into_response();
    end_guest(state, auth, headers, &mut res).await;
    res
}

/// Signed in or signed out, this browser is no longer a guest: when the session ends it comes
/// back to the sign-in screen, not to the demo.
async fn end_guest(state: &AppState, auth: &Auth, headers: &HeaderMap, res: &mut Response) {
    let Some(token) = guest_token(headers) else {
        return;
    };
    delete_guest(state, &token).await;
    if let Ok(cookie) = HeaderValue::from_str(&guest_cookie(&auth.config, headers, "", 0)) {
        res.headers_mut().append(header::SET_COOKIE, cookie);
    }
}

/// The browser forgets the id either way, so a failure is only for the log: the row stays until
/// the id would have run out, and /privacy says it goes when the id ends.
async fn delete_guest(state: &AppState, token: &str) {
    if let Err(e) = state.db.delete_guest(&token_hash(token)).await {
        warn!(error = %e, "Could not delete a guest id that ended");
    }
}

/// Makes a single-use sign-in link for `email` that skips the provider, for when it cannot
/// be reached (no internet): `aries-image-studio login-link`. Whoever can run that on the
/// server can read everything already, so it proves enough. One link per public URL, all
/// with the same token: whichever is opened first uses it up.
pub async fn create_login_link(
    db: &crate::db::Database,
    cfg: &AuthConfig,
    email: &str,
) -> Result<Vec<String>> {
    let email = email.trim().to_lowercase();
    if !cfg.allowed_emails.contains(&email) {
        bail!("{email} is not in ARIES_AUTH_ALLOWED_EMAILS");
    }
    let token = random_token();
    db.save_login_link(
        &token_hash(&token),
        &email,
        crate::db::now_s() + LOGIN_LINK_TTL_S,
    )
    .await?;
    Ok(cfg
        .public_urls
        .iter()
        .map(|u| format!("{u}/auth/link?token={token}"))
        .collect())
}

#[derive(Debug, Deserialize)]
pub struct LinkQuery {
    pub token: Option<String>,
}

/// `GET /auth/link?token=`: a link from `aries-image-studio login-link`, once.
pub async fn link(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<LinkQuery>,
) -> Response {
    let Some(auth) = &state.auth else {
        return Redirect::to("/").into_response();
    };
    let email = match q.token.as_deref() {
        Some(t) => state
            .db
            .take_login_link(&token_hash(t))
            .await
            .ok()
            .flatten(),
        None => None,
    };
    let Some(email) = email else {
        return signin_error("expired");
    };
    // Taken off the list since the link was made.
    if !auth.config.allowed_emails.contains(&email) {
        return signin_error("not_allowed");
    }
    let (name, picture) = state
        .db
        .last_profile(&email)
        .await
        .ok()
        .flatten()
        .unwrap_or_default();
    info!(email = %email, "Signed in with a login link");
    start_session(&state, auth, &headers, &email, &name, &picture, "/".into()).await
}

async fn exchange_code(
    auth: &Auth,
    code: &str,
    redirect_uri: &str,
    verifier: &str,
    nonce: &str,
) -> Result<IdClaims> {
    let discovery = auth.discovery().await?;
    let res = auth
        .http
        .post(&discovery.token_endpoint)
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("client_id", auth.config.client_id.as_str()),
            ("client_secret", auth.config.client_secret.as_str()),
            ("code_verifier", verifier),
        ])
        .send()
        .await?;
    if !res.status().is_success() {
        bail!("token endpoint answered {}", res.status());
    }
    let body: Value = res.json().await?;
    let id_token = body["id_token"]
        .as_str()
        .ok_or_else(|| anyhow!("no ID token in the token response"))?;
    check_id_token(id_token, discovery, &auth.config, nonce)
}

/// `POST /auth/logout`: ends this browser's session, or its visit as a guest.
pub async fn logout(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let Some(auth) = &state.auth else {
        return StatusCode::NO_CONTENT.into_response();
    };
    // The cookie is sent with the answer, so another site's form could still sign you out.
    if from_another_site(auth, &headers) {
        return json_error(StatusCode::FORBIDDEN, "cross-site request refused");
    }
    if let Some(token) = session_token(&headers) {
        let _ = state.db.delete_session(&token_hash(&token)).await;
    }
    let mut res = (
        StatusCode::NO_CONTENT,
        [(
            header::SET_COOKIE,
            session_cookie(&auth.config, &headers, "", 0),
        )],
    )
        .into_response();
    end_guest(&state, auth, &headers, &mut res).await;
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn vars(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| m.get(k).cloned()
    }

    #[test]
    fn off_unless_configured_and_complete_when_on() {
        assert!(AuthConfig::from_vars(vars(&[])).unwrap().is_none());
        assert!(AuthConfig::from_vars(vars(&[("ARIES_OIDC_CLIENT_ID", "c")])).is_err());
        let full = [
            ("ARIES_OIDC_CLIENT_ID", "c"),
            ("ARIES_OIDC_CLIENT_SECRET", "s"),
            (
                "ARIES_AUTH_ALLOWED_EMAILS",
                " Me@Example.com, you@example.com ",
            ),
            ("ARIES_PUBLIC_URL", "https://studio.example/"),
        ];
        let cfg = AuthConfig::from_vars(vars(&full)).unwrap().unwrap();
        assert_eq!(cfg.allowed_emails, ["me@example.com", "you@example.com"]);
        assert_eq!(cfg.owner, "me@example.com", "the first address by default");
        assert_eq!(cfg.issuer, GOOGLE_ISSUER);
        let none = HeaderMap::new();
        assert_eq!(
            cfg.redirect_uri(&none),
            "https://studio.example/auth/callback"
        );
        assert_eq!(cfg.origins(), ["https://studio.example"]);
        assert!(
            !format!("{cfg:?}").contains("\"s\""),
            "Debug leaves the secret out"
        );

        let mut stranger = full.to_vec();
        stranger.push(("ARIES_AUTH_OWNER", "stranger@example.com"));
        assert!(AuthConfig::from_vars(vars(&stranger)).is_err());
    }

    #[test]
    fn several_public_urls_follow_the_host() {
        let cfg = AuthConfig::from_vars(vars(&[
            ("ARIES_OIDC_CLIENT_ID", "c"),
            ("ARIES_OIDC_CLIENT_SECRET", "s"),
            ("ARIES_AUTH_ALLOWED_EMAILS", "me@example.com"),
            (
                "ARIES_PUBLIC_URL",
                "https://studio.example/, http://studio.lan:8080",
            ),
        ]))
        .unwrap()
        .unwrap();
        let host = |h: &[(&'static str, &str)]| {
            let mut m = HeaderMap::new();
            for (k, v) in h {
                m.insert(*k, HeaderValue::from_str(v).unwrap());
            }
            m
        };
        assert_eq!(
            cfg.origins(),
            ["https://studio.example", "http://studio.lan:8080"]
        );
        let lan = host(&[("host", "Studio.LAN:8080")]);
        assert_eq!(
            cfg.redirect_uri(&lan),
            "http://studio.lan:8080/auth/callback"
        );
        assert!(!cfg.secure_cookie(&lan));
        let proxied = host(&[
            ("host", "127.0.0.1:8080"),
            ("x-forwarded-host", "studio.example"),
        ]);
        assert_eq!(
            cfg.redirect_uri(&proxied),
            "https://studio.example/auth/callback"
        );
        assert!(cfg.secure_cookie(&proxied));
        let other = host(&[("host", "evil.example")]);
        assert_eq!(
            cfg.redirect_uri(&other),
            "https://studio.example/auth/callback",
            "the first"
        );
        assert!(AuthConfig::from_vars(vars(&[
            ("ARIES_OIDC_CLIENT_ID", "c"),
            ("ARIES_OIDC_CLIENT_SECRET", "s"),
            ("ARIES_AUTH_ALLOWED_EMAILS", "me@example.com"),
            ("ARIES_PUBLIC_URL", " , "),
        ]))
        .is_err());
    }

    #[test]
    fn id_token_checks() {
        let cfg = AuthConfig::from_vars(vars(&[
            ("ARIES_OIDC_CLIENT_ID", "client"),
            ("ARIES_OIDC_CLIENT_SECRET", "s"),
            ("ARIES_AUTH_ALLOWED_EMAILS", "me@example.com"),
            ("ARIES_PUBLIC_URL", "https://studio.example"),
        ]))
        .unwrap()
        .unwrap();
        let d = Discovery {
            issuer: GOOGLE_ISSUER.into(),
            authorization_endpoint: String::new(),
            token_endpoint: String::new(),
        };
        let token = |claims: Value| {
            format!(
                "{}.{}.sig",
                URL_SAFE_NO_PAD.encode("{}"),
                URL_SAFE_NO_PAD.encode(claims.to_string())
            )
        };
        let ok = json!({"iss": "accounts.google.com", "aud": "client", "exp": crate::db::now_s() + 60.0, "nonce": "n", "email": "me@example.com"});
        assert!(
            check_id_token(&token(ok.clone()), &d, &cfg, "n").is_ok(),
            "issuer without the scheme"
        );
        for (k, v) in [
            ("iss", json!("https://evil.example")),
            ("aud", json!(["other"])),
            ("exp", json!(1.0)),
            ("nonce", json!("m")),
        ] {
            let mut bad = ok.clone();
            bad[k] = v;
            assert!(check_id_token(&token(bad), &d, &cfg, "n").is_err(), "{k}");
        }
    }

    #[test]
    fn next_stays_on_this_site() {
        assert_eq!(safe_next(Some("/gallery?fav=1")), "/gallery?fav=1");
        assert_eq!(safe_next(Some("//evil.example")), "/");
        assert_eq!(safe_next(Some("https://evil.example")), "/");
        assert_eq!(safe_next(Some("/\\evil.example")), "/");
        for sneaky in [
            "/\t/evil.example",
            "/\n/evil.example",
            "/\r/evil.example",
            "/\x7f/x",
        ] {
            assert_eq!(safe_next(Some(sneaky)), "/", "{sneaky:?}");
        }
        assert_eq!(safe_next(None), "/");
    }

    #[test]
    fn session_cookie_is_read_among_others() {
        let mut h = HeaderMap::new();
        h.insert(
            header::COOKIE,
            "theme=dark; aries_session=abc123; x=1".parse().unwrap(),
        );
        assert_eq!(session_token(&h).as_deref(), Some("abc123"));
    }

    #[test]
    fn pkce_challenge_is_base64url_sha256() {
        // Expected value from Python: urlsafe_b64encode(sha256(verifier)) without padding.
        assert_eq!(
            pkce_challenge("dBjftJeZ4CVP-mJ92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "ngF5GsXcbwljx6u133FFr3Xht9xooA_DuaX_3QwODtc"
        );
    }
}
