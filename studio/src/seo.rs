//! What search engines and link previews see (docs/SEO.md). Only the page's own description and
//! Studio's configured addresses: nothing a visitor has to sign in for, and no job or image.

use crate::scheduler::AppState;
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

/// In `web/index.html`, replaced with the tags that need Studio's address.
const MARKER: &str = "<!-- aries:head -->";
const TITLE: &str = "Aries Image Studio";
const DESCRIPTION: &str = "A self-hosted AI image generator: open models on the host's own \
                           computers, by invitation.";
const DESCRIPTION_DEMO: &str = "A self-hosted AI image generator: open models on the host's own \
                                computers. Open the read-only guest demo to browse what it has \
                                made, with the prompts.";
const IMAGE_PATH: &str = "/og-image.jpg";
const IMAGE_ALT: &str = "Aries Image Studio, beside a collage of images made with open models: \
                         a spell duel, a small wizard, robots and a dwarf with his treasure.";

fn escaped(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Link previews and search engines need absolute addresses, so these need `ARIES_PUBLIC_URL`.
fn head_tags(base: &str, demo: bool) -> String {
    let base = escaped(base);
    let description = if demo { DESCRIPTION_DEMO } else { DESCRIPTION };
    let description = escaped(description);
    let alt = escaped(IMAGE_ALT);
    format!(
        r#"<link rel="canonical" href="{base}/" />
    <meta property="og:type" content="website" />
    <meta property="og:site_name" content="{TITLE}" />
    <meta property="og:title" content="{TITLE}" />
    <meta property="og:description" content="{description}" />
    <meta property="og:url" content="{base}/" />
    <meta property="og:image" content="{base}{IMAGE_PATH}" />
    <meta property="og:image:type" content="image/jpeg" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta property="og:image:alt" content="{alt}" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="{TITLE}" />
    <meta name="twitter:description" content="{description}" />
    <meta name="twitter:image" content="{base}{IMAGE_PATH}" />
    <meta name="twitter:image:alt" content="{alt}" />"#
    )
}

/// The web app's page, for this request's address when Studio knows it.
pub fn render_index(html: &str, base: Option<&str>, demo: bool) -> String {
    html.replace(
        MARKER,
        &base.map(|b| head_tags(b, demo)).unwrap_or_default(),
    )
}

/// One of the configured addresses, so a forged `Host` can't choose what is written.
fn public_url<'a>(state: &'a AppState, headers: &HeaderMap) -> Option<&'a str> {
    state.auth.as_ref().map(|a| a.config.public_url(headers))
}

fn has_demo(state: &AppState) -> bool {
    state.auth.is_some() && state.config.demo.is_some()
}

/// Every page of the web app, whatever its path. Not cached, since it depends on the address.
pub async fn serve_index(state: &AppState, headers: &HeaderMap) -> Response {
    let path = state.config.static_dir.join("index.html");
    match tokio::fs::read_to_string(path).await {
        Ok(html) => (
            [
                (header::CONTENT_TYPE, "text/html; charset=utf-8"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            render_index(&html, public_url(state, headers), has_demo(state)),
        )
            .into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

/// `GET /robots.txt`: a private Studio asks not to be listed. With a demo, its public pages may be,
/// and the routes that answer nothing to a crawler are left out.
pub async fn robots(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let text = match public_url(&state, &headers).filter(|_| has_demo(&state)) {
        Some(base) => format!(
            "User-agent: *\n\
             Disallow: /api/\n\
             Disallow: /auth/\n\
             Disallow: /agent/\n\
             Disallow: /mcp\n\
             Disallow: /metrics\n\
             Allow: /\n\n\
             Sitemap: {base}/sitemap.xml\n"
        ),
        None => "User-agent: *\nDisallow: /\n".to_string(),
    };
    ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response()
}

/// `GET /sitemap.xml`: the pages anyone can read, with a demo only.
pub async fn sitemap(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let Some(base) = public_url(&state, &headers).filter(|_| has_demo(&state)) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let base = escaped(base);
    let urls: String = ["/", "/terms", "/privacy"]
        .iter()
        .map(|path| format!("  <url><loc>{base}{path}</loc></url>\n"))
        .collect();
    (
        [(header::CONTENT_TYPE, "application/xml; charset=utf-8")],
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n{urls}</urlset>\n"
        ),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_marker_becomes_tags_or_nothing() {
        let page = format!("<head>{MARKER}</head>");
        let open = render_index(&page, None, false);
        assert_eq!(open, "<head></head>");
        let tags = render_index(&page, Some("https://studio.example"), false);
        assert!(tags.contains(r#"content="https://studio.example/og-image.jpg""#));
        assert!(tags.contains(r#"href="https://studio.example/""#));
        assert!(!tags.contains(MARKER));
        assert!(!tags.contains("guest demo"), "no demo to speak of");
        assert!(render_index(&page, Some("https://s"), true).contains("guest demo"));
    }

    #[test]
    fn an_address_cannot_break_out_of_a_tag() {
        let tags = head_tags("https://s/\"><script>", false);
        assert!(!tags.contains("<script>"));
    }
}
