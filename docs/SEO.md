# Search engines and link previews

Status: **since 0.8.0**. Tests: `studio/tests/v3_backend_test.rs` (`test_search_engines_and_link_previews`) and
`studio/src/seo.rs`.

These pages describe Studio itself, never a job or an image: everything else stays behind sign-in
or a guest id, whatever link a crawler follows.

| | Sign-in off | Sign-in on, no demo | Sign-in and demo |
|---|---|---|---|
| Title and description | yes | yes | yes |
| Canonical address, Open Graph and Twitter card tags | no: no known address | yes | yes, and they mention the guest demo |
| `/robots.txt` | asks not to be crawled | asks not to be crawled | allows the public pages, closes `/api/`, `/auth/`, `/agent/`, `/mcp`, `/metrics` |
| `/sitemap.xml` | 404 | 404 | `/`, `/terms`, `/privacy` |

**Addresses come from `ARIES_PUBLIC_URL`**, whichever of them the request's `Host` or
`X-Forwarded-Host` names, else the first, so a forged header cannot write another site into the page.
Studio fills them in where `web/index.html` has the marker `<!-- aries:head -->`. Every path of the web
app gets the same tags, with the canonical address at `/`, and is sent with `Cache-Control: no-cache`.

**The preview image** is `web/public/og-image.jpg`: 1200×630, with no metadata, public like the
app's other files. To replace it, keep the size and update `IMAGE_ALT` in `studio/src/seo.rs`.

**Your reverse proxy** must let crawlers reach `/`, `/og-image.jpg`, `/robots.txt`, `/sitemap.xml`,
`/terms` and `/privacy`. Some preview services obey `robots.txt`, so a private Studio's links may not
get a card on them. Search listings and cached previews take time to update; link-preview debuggers
can refresh a cached card.
