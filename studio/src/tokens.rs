//! Access tokens for the CLI and AI tools (docs/INTEGRATIONS.md): made and revoked in the
//! browser, each acts as the account that made it. Studio keeps only their hashes.

use crate::auth::{new_api_token, token_hash, User};
use crate::scheduler::AppState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

/// Longest token name kept.
const NAME_MAX_CHARS: usize = 80;
/// Most tokens an account may have.
const MAX_TOKENS: usize = 50;

fn error(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({ "error": msg }))).into_response()
}

/// Tokens are managed from a browser session, by a signed-in account: with sign-in off
/// anyone can call the API without one, and a token cannot make or revoke tokens (a
/// leaked one cannot make itself another).
fn account(state: &AppState, user: &User) -> Result<String, Box<Response>> {
    if state.auth.is_none() {
        return Err(Box::new(error(
            StatusCode::NOT_FOUND,
            "sign-in is off: the API needs no token",
        )));
    }
    if user.via_token {
        return Err(Box::new(error(
            StatusCode::FORBIDDEN,
            "tokens are managed in the browser",
        )));
    }
    user.email
        .clone()
        .ok_or_else(|| Box::new(error(StatusCode::UNAUTHORIZED, "sign in first")))
}

/// `GET /api/tokens`: the account's tokens, newest first (names and dates, never tokens).
pub async fn list_tokens(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
) -> Response {
    let email = match account(&state, &user) {
        Ok(e) => e,
        Err(res) => return *res,
    };
    match state.db.list_api_tokens(&email).await {
        Ok(tokens) => Json(json!({ "tokens": tokens })).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateTokenBody {
    pub name: String,
}

/// `POST /api/tokens {name}`: a new token, shown this once.
pub async fn create_token(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Json(body): Json<CreateTokenBody>,
) -> Response {
    let email = match account(&state, &user) {
        Ok(e) => e,
        Err(res) => return *res,
    };
    let name: String = body.name.trim().chars().take(NAME_MAX_CHARS).collect();
    if name.is_empty() {
        return error(StatusCode::BAD_REQUEST, "name the token");
    }
    match state.db.list_api_tokens(&email).await {
        Ok(t) if t.len() >= MAX_TOKENS => {
            return error(
                StatusCode::CONFLICT,
                &format!("at most {MAX_TOKENS} tokens: revoke one first"),
            )
        }
        Ok(_) => {}
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
    let token = new_api_token();
    let id = ulid::Ulid::new().to_string();
    match state
        .db
        .create_api_token(&id, &token_hash(&token), &email, &name)
        .await
    {
        Ok(record) => {
            tracing::info!(email = %email, token = %id, "Access token created");
            (
                StatusCode::CREATED,
                Json(json!({ "token": token, "record": record })),
            )
                .into_response()
        }
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `DELETE /api/tokens/{id}`: revokes one of the account's tokens, at once.
pub async fn delete_token(
    State(state): State<Arc<AppState>>,
    Extension(user): Extension<User>,
    Path(id): Path<String>,
) -> Response {
    let email = match account(&state, &user) {
        Ok(e) => e,
        Err(res) => return *res,
    };
    match state.db.delete_api_token(&id, &email).await {
        Ok(true) => {
            tracing::info!(email = %email, token = %id, "Access token revoked");
            StatusCode::NO_CONTENT.into_response()
        }
        Ok(false) => error(StatusCode::NOT_FOUND, "token not found"),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}
