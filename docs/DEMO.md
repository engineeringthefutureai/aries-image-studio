# The read-only demo

Status: **since 0.8.0**. Tests: `studio/tests/v3_backend_test.rs` (`test_demo_*`) and
`studio/src/demo.rs`.

With `ARIES_DEMO=1` and sign-in on, a Studio has a public, read-only face: anyone can open a guest
demo from the sign-in screen and browse the images the owner chose, and nothing else. Off by default.
Without sign-in the setting is ignored (with a warning), since everything is already open.

## 1. What a visitor gets

**Getting in.** Everyone arrives at the sign-in screen, as in a Studio with no demo: without an id,
a session or a token, `/api/*` answers 401 and shows nothing, whatever the link. Beside
**Sign in with Google** is **Open guest demo**, with a note that doing either accepts the terms and
privacy policy. The click (`POST /auth/guest {"accept_terms": true}`) gives the browser a new random
id in the cookie `aries_guest` (HttpOnly, SameSite=Lax, 30 days, not renewed by use). Studio keeps
its hash and when it was made, in the table `guests`: not the address, and nothing the guest looks
at. Refused from another site's page, and limited per address (§2). The table keeps at most 100,000
ids, the oldest going first, so it cannot grow without limit and nobody can close the demo by filling
it: a guest who loses theirs clicks again. The page stays where it was, so a link to an image opens
that image after the click. A browser that does not keep the cookie is told the demo needs one. A
browser already signed in gets no id, and the app shows the account.

With a good id and no session or token, a request is a **visitor**:

- **Sees:** the timeline, gallery, viewer, search and "more like this", and a reduced status
  page, over the marked images only. The composer can be opened and filled (Vary and Rerun
  too), so its controls can be tried, but Send is disabled. Preferences last for the visit.
- **Cannot:** send, upload, favorite, delete, cancel, retry, change cards, or manage tokens. The
  buttons are not shown, and Studio refuses the requests anyway (§2). `/mcp` still needs a token.
- **Leaving, signing in:** a banner under the header has **Leave demo**, which deletes the id, and
  **Sign in**, which does too once it succeeds, so a later sign-out comes back to the sign-in
  screen and not the demo. A session wins over an id.
- **When something ends:** an id or session that has ended gets 401 and its cookie is cleared, and
  the app asks Studio who this is now, which is the sign-in screen. If a signed-in tab finds this
  browser is now a guest (`X-Aries-Visitor: 1`, on every visitor's answer), it does the same, and so
  does a visitor's tab that gets an answer without that header (signed in from another tab). If
  Studio can't say (a failed lookup gets 503, and the cookie stays), the page stays as it is and the
  app keeps asking.

## 2. How it is enforced

`auth::require_user` hands visitors to `demo::serve_visitor`, before any handler:

- **An allowlist of routes.** `demo::visitor_may` names the routes a visitor reaches, `GET` and
  `HEAD` only: `/api/jobs`, `/api/jobs/{id}`, `/api/gallery/facets`, `/api/images/{id}` and its
  `thumb`, `/api/uploads/{sha}` and its `thumb`, `/api/system`, `/api/capabilities`,
  `/api/preferences`. Everything else is 403 with `{"demo": true}`, so a route added later is
  closed until it is named.
- **Its own scope.** `User::scope()` returns `Scope::Demo`, not the "no account" that means
  "sees everything" with sign-in off. Reads match on it, so each has to handle a visitor.
- **What is sent is named, not filtered out** (`demo::visitor_job`, `visitor_capabilities`,
  `scrub_system`, and PNG chunks in `strip_png_metadata`), so a field added later stays out:

  | | Visitors get | Not |
  |---|---|---|
  | Job | prompts, enhanced prompt and reasoning, settings, seed, references, timings, peak memory, the UI's batch, aspect, size and quality | favorites, other `meta`, errors, host, device, software versions, assignments |
  | `/api/system` | hosts as `gpu-host-N`, cards' state, stage, memory and model, queue lengths | host names, addresses, versions, process ids, job ids, limits, free-text reasons |
  | `/api/capabilities` | sizes and stages available | reasons, which name hosts |
  | Full PNG | the pixels | the provenance chunk and any other text or EXIF (`strip_png_metadata`) |
  | A reference | a JPEG of up to 1024 px, made by Studio, at both upload URLs | the original file and its EXIF (where a photo was taken) |

  Only references of marked images exist for a visitor. What was behind them does not.

**Quotas**, per IP and per minute, for visitors only, kept in memory and forgotten each minute.
They go by address and not by id, since an id can be thrown away for a new one:

| Variable | Default | |
|---|---|---|
| `ARIES_DEMO_REQUESTS_PER_MIN` | 600 | Every request (a gallery page is dozens of thumbnails) |
| `ARIES_DEMO_SEARCHES_PER_MIN` | 30 | `q` or `like` on `/api/jobs` |
| `ARIES_DEMO_DOWNLOADS_PER_MIN` | 120 | Full PNGs, which is also what the viewer shows: every image opened counts |
| `ARIES_DEMO_GUESTS_PER_MIN` | 10 | Clicks on **Open guest demo**, each a row in the database until it ends |
| `ARIES_DEMO_TRUSTED_PROXIES` | 0 | How many proxies in front of Studio add to `X-Forwarded-For` |

Over a limit: 429 with `Retry-After`. The search, download and new-id limits refuse only their own kind,
so browsing goes on; the overall limit refuses everything. At most two visitor searches run at once, across all
addresses, since each embeds its query on the CPU; a third gets 429. A search is at most 200
bytes. An IPv6 address counts by its /64.

**Set `ARIES_DEMO_TRUSTED_PROXIES` to match your setup.** Behind one reverse proxy, use 1: Studio
takes the last address in `X-Forwarded-For`, the one the proxy added. With the default 0 behind a
proxy, all visitors share the proxy's address and one quota. Set too high, a client can write its
own address and dodge its quota. The proxy must add to the header, as Caddy and nginx do.

## 3. Choosing what is shown

The owner only, signed in with a browser: `PATCH /api/jobs/{id} {"demo": true|false}` is refused
for other accounts (403) and for access tokens, so no AI tool or script can publish an image.
Finished images only (409 otherwise).

- **In the app:** the globe button in the viewer and on a gallery tile for one image, and on a
  turn in the timeline for its whole group, loaded in the page or not (`GET /api/jobs?batch=`).
  The app asks every time, showing what would be shared with its references, with **Cancel**
  focused; a group is asked about once, as a whole. If a check or a change fails, the app says so
  and what state it left. The gallery's **In demo** filter (`GET /api/jobs?demo=true`) shows what is public.
- **Taking one out** needs no asking and works at once, also on pages open at the time (`deleted` in the
  `changed_since` poll, from the table `demo_removed`). Deleting a marked image does the same.
  Copies visitors already made stay theirs, and browsers keep images they fetched.

**Anything shown is public: its prompt, enhanced prompt, reasoning, seed and references.** That
includes a generated reference from an image you never marked: visitors can fetch it at up to
1024 px, so the question shows it.

**Images made from uploads.** Sharing one is not recommended. Studio refuses it (409, with the
hashes in `uploaded_references`) unless the request says `"accept_reference_risk": true`, and the
app asks for that only through a warning, in place of the plain question: the uploads shown, what
can go wrong, and a box to tick before the button works. It knows which to ask from
`GET /api/jobs/{id}/uploads` (same access as marking); if that fails it asks nothing and shares
nothing. An upload counts when no job in Studio made it, following generated
references back to theirs (an edit of an edit of a photo still shows the photo); a reference
whose job was deleted counts too. Images generated from scratch, and edits of them, get the plain
question. In a group with any such image, the warning covers the group: cancelling shares none.

## 4. Watching it

`/metrics` (keep it blocked at the proxy, §5), only with the demo on:

| Metric | |
|---|---|
| `aries_demo_unique_visitors{window="hour"\|"day"}` | Different guest ids using the demo this UTC hour or day. An estimate (within about 2%, exact-ish while few). Starts over each period and with Studio, so read a day's total with `max_over_time`. |
| `aries_demo_guests_total` | Times someone accepted the terms and got an id |
| `aries_demo_requests_total{kind="browse"\|"search"\|"download"\|"start"}` | Requests to the routes open to visitors, and to open the demo |
| `aries_demo_refused_total{reason="closed_route"\|"quota"\|"search_busy"}` | Refused requests |

A unique visitor is an id that reached an open `/api` route, so people signed in are not counted,
nor crawlers, which stop at the sign-in screen. People behind one address count separately. Someone
who opens the demo again, clears cookies or uses another browser counts again. The count is a
HyperLogLog with a random salt replaced each period: 4 KB in memory, no ids in it, and nothing to
compare between periods. `/privacy` says so.

**Logs never hold a visitor's address or id.** A summary each minute with visitors in it (how many
addresses, requests, searches, downloads, new guests and refusals); once a minute per address, that one went
over a quota; once per run, if `X-Forwarded-For` doesn't match `ARIES_DEMO_TRUSTED_PROXIES`.
Sharing and taking out are logged with the job, and a warning accepted is logged as a warning.
Closed routes are logged at `debug`. The Grafana dashboard has a **Public demo** row.

## 5. Before you turn it on

- AUTH.md advises against opening Studio to the public. The demo is the one exception: visitors
  reach only what is above, and the rest stays behind sign-in. It does mean Studio's address, and
  the read-only part of its code, face the internet without a login.
- `/privacy` and `/terms` gain a section on the demo, and the click accepts them. Read them (AUTH.md
  §3, step 5). The terms have no version: someone already in the demo is not asked again when they
  change, until their id runs out. The
  Qwen-Image 2.1 models are non-commercial, so keep the demo non-commercial too.
- The reverse proxy should keep blocking `/metrics` and `/agent/*` (AUTH.md §3, step 4). That only
  helps if the internet can't reach Studio without it: `deploy/studio/compose.yaml` publishes 8080
  and 50051 on every interface, so keep your firewall or router from forwarding them.
- With the demo on, search engines may list its public pages (SEO.md).
- The status page shows visitors when your cards are busy, and how long the queue is.
- `/api/jobs` reports Studio's overall change counter
  (`version`), which goes up with any account's activity.
