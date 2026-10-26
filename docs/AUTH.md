# Sign-in and accounts

Status: **since 0.6.0**. Tests: `studio/tests/v3_backend_test.rs` (`test_sign_in_*`,
`test_jobs_images_and_uploads_*`), against a stand-in provider.

## 1. What it is

- **Sign in with Google, nothing else.** No passwords, no sign-up: Google vouches for an
  email address, and only the addresses on an allowlist get in. Any OpenID Connect
  provider works the same way (`ARIES_OIDC_ISSUER`); Google is the default.
- **Off unless configured.** Without `ARIES_OIDC_CLIENT_ID`, Studio is open as before and
  everyone sees everything. Set the client ID and the rest must be set too, or Studio does
  not start.
- **Each account has its own history.** Jobs, images, favorites, reference uploads and
  preferences belong to the account that made them. Someone else's job or image is "not
  found" (404), not "forbidden". The queue and the cards are shared: jobs run first come,
  first served, whoever sent them.
- **The owner** (`ARIES_AUTH_OWNER`, by default the first address on the list) gets
  everything made before sign-in was switched on, and is the only account that may lend,
  disable or enable cards. Everyone signed in sees the system page.
- **For people you know.** Studio is meant for a few invited people, not the public: keep
  the OAuth app in Testing (§3) and the allowlist short.

## 2. How it works

Studio is the OpenID Connect client (`studio/src/auth.rs`); no proxy in front is needed.

1. **`GET /auth/login?next=/path`** looks up the provider's endpoints
   (`{issuer}/.well-known/openid-configuration`, fetched once), stores a random `state`,
   `nonce` and PKCE verifier (table `login_states`, 10 minutes), and redirects to the
   provider's sign-in page (`scope=openid email profile`, `prompt=select_account`). The
   redirect URI is `{address}/auth/callback`, for the address in `ARIES_PUBLIC_URL` that the
   request's `X-Forwarded-Host` or `Host` names (else the first), so a sign-in started on
   the intranet name comes back to it.
2. The provider sends the browser back to **`GET /auth/callback?code&state`**. Studio
   takes the stored state (once), exchanges the code at the token endpoint with its client
   secret, the PKCE verifier and the same redirect URI (from the callback's own host), and
   checks the ID token: issuer, audience (our client ID), expiry and nonce. Its signature is
   not checked: the token comes straight from the provider over TLS in exchange for the
   client secret, which OpenID Connect Core §3.1.3.7 allows instead.
3. The email must be verified and on the allowlist (compared in lowercase). Then Studio
   creates a session: 32 random bytes in the cookie `aries_session` (HttpOnly,
   SameSite=Lax, Secure when the address is https), stored as its SHA-256 (table
   `sessions`), so the database never holds a usable token. The cookie belongs to the
   address it was set on: each address has its own sessions. The browser lands on `next`
   (paths of this site only). Refused: back to `/?auth_error=not_allowed` (or `expired`,
   `cancelled`, `failed`), and the sign-in screen says why.
4. **Every request checks the session**: it must exist, be unexpired, and its address must
   still be on the allowlist. Taking an address off the list (and restarting Studio) signs
   it out everywhere on its next request; its sessions are deleted. A session in use does
   not run out: a request more than an hour after the last renewal moves its expiry to
   `ARIES_SESSION_DAYS` (30) days from now and sends the cookie again, so that is how long
   an unused session lasts.
5. **`POST /auth/logout`** deletes the session and clears the cookie.

**Without Google (no internet).** Only signing in needs the provider: a session is checked
against Studio's own database, so signed-in devices keep working offline, and keep
renewing while used. To sign in a new device, or one whose session ran out, run on the
server, with Studio's environment:

```
kubectl exec deploy/<studio> -- aries-image-studio login-link you@example.com
docker compose exec studio aries-image-studio login-link you@example.com
```

It prints `{address}/auth/link?token=…`, one line per address in `ARIES_PUBLIC_URL`, all
with the same token: a single-use link, good for 10 minutes, for an allowlisted address
only (table `login_links`, stored by hash). **`GET /auth/link`** takes it once and starts a
normal session with the name and picture the provider last gave for that address; used,
expired or made up, it lands on the sign-in screen ("expired"). No password is involved:
being able to run commands where Studio runs is the proof, and whoever can already reads
the database. A link is a session in waiting: open it yourself rather than pasting it
where previews might fetch it.

Every `/api/*` route except `/api/me` needs a session (401 without). Writes (`POST`,
`PUT`, `PATCH`, `DELETE`) whose `Origin` is not one of `ARIES_PUBLIC_URL`'s addresses are
refused (403); SameSite=Lax already keeps the cookie off them. Images and uploads are sent
`Cache-Control: private` instead of `public`. Open as before: `/healthz` (for probes),
`/metrics`, the agents' routes (`/agent/v1/*` and gRPC, with the agent token), the web
app's files, `/auth/*`, and the privacy policy and terms (`/privacy`, `/terms`).

**`GET /api/me`**: `{auth: false}` with sign-in off; `{auth: true, email, name, picture,
owner}` signed in; 401 signed out. The web app asks it first and shows the sign-in screen
or the app; any later 401 from the API (the session expired) brings the sign-in screen
back. Signed in, the header shows the account's picture with a menu to sign out.

**Ownership in the database:** `jobs.owner` and `deleted_jobs.owner` (the email, NULL
with sign-in off), `upload_owners (sha256, owner)` (an upload is stored once by its hash
and may belong to several accounts), `user_preferences (owner, json)`. On start with
sign-in on, Studio gives the owner every job, tombstone and upload without one, and copies
the old shared preferences to the owner. Expired sessions and abandoned sign-ins are
deleted every 30 s.

## 3. Setting it up with Google

**Keep it private.** It is highly advised to leave the OAuth app in **Testing** and never
open Studio to the public. Testing allows up to 100 test users, which suits a Studio for
people you know, and works indefinitely: Studio uses Google only at sign-in and keeps its
own sessions, so Testing's 7-day expiry of refresh tokens does not apply (Studio asks for
none). The allowlist still decides who gets in; the test-user list is a second gate.

1. **Google Cloud console → Google Auth Platform** (APIs & Services → OAuth consent screen):
   - **Audience:** user type **External**, publishing status **Testing**; add each
     allowlisted address as a **test user**.
   - **Data access (scopes):** `openid`, `.../auth/userinfo.email`,
     `.../auth/userinfo.profile`. All three are non-sensitive: no security assessment
     (CASA), and no verification is required to use them.
   - **Branding:** app name, support email, the home page `https://<studio host>/`, the
     privacy policy `https://<studio host>/privacy` and the terms
     `https://<studio host>/terms`; under **Authorized domains**, the registrable domain of
     each address Studio is reached on.
2. **Clients → Create client**, type **Web application**:
   - **Authorized redirect URIs:** `https://<studio host>/auth/callback`, one for each
     address Studio is reached on (say an intranet and an internet name), all in the same
     client. Exact match: https, no trailing slash, no port unless the address has one.
     Google requires https (or `http://localhost`).
   - Changes can take from minutes to a few hours to apply.
3. **Set Studio's environment**; keep the secret out of the repo (a Kubernetes Secret, or
   `deploy/studio/.env`):

   | Variable | | |
   |---|---|---|
   | `ARIES_OIDC_CLIENT_ID` | required to switch it on | the client ID |
   | `ARIES_OIDC_CLIENT_SECRET` | required, secret | the client secret |
   | `ARIES_AUTH_ALLOWED_EMAILS` | required | comma-separated addresses that may sign in; removing one signs it out (§2, step 4) |
   | `ARIES_PUBLIC_URL` | required | Studio's address as the browser sees it, e.g. `https://studio.example`; several, comma-separated, when it has more than one way in (§2, step 1). Login links use each |
   | `ARIES_AUTH_OWNER` | optional | the owner; default the first allowed address |
   | `ARIES_OIDC_ISSUER` | optional | default `https://accounts.google.com` |
   | `ARIES_SESSION_DAYS` | optional | how long an unused session lasts; default 30 |
   | `ARIES_LEGAL_OPERATOR` | advised | who runs Studio, named on `/privacy` and `/terms` |
   | `ARIES_LEGAL_CONTACT` | advised | how to reach them: an email address or a URL |

4. **Network.** Studio needs outbound HTTPS to `accounts.google.com` and
   `oauth2.googleapis.com`. Google never connects to Studio: the browser carries the
   redirects, so a Studio reachable only on the LAN works, as long as browsers trust its
   certificate. A reverse proxy in front must pass the original `Host` (Caddy and most
   proxies do) or set `X-Forwarded-Host`, and should block `/metrics` and `/agent/*` from
   outside (the README's "Optional: sign-in with Google" has a Caddy example). Probes that call `/api/*` get
   401: point them at `/healthz`.
5. **Privacy policy and terms.** Google's consent screen asks for both. Studio serves them
   at `/privacy` and `/terms` (`studio/src/legal.rs`), public, written from its own
   settings: the session length, the cloud enhancers switched on, and the models the agents
   declare, each with its license (the Qwen-Image 2.1 models are non-commercial:
   `NON_COMMERCIAL`).
   - **The pages speak for the host.** By running Studio with Google sign-in, you make
     their commitments to your users and to Google, among them Google's Limited Use rules
     for account data. The software's authors are not party to them (MIT license, no
     warranty), and the pages say so. Read them first; if the wording does not fit how you
     run Studio, change it in `legal.rs` and move `LAST_UPDATED`.
   - Set `ARIES_LEGAL_OPERATOR` and `ARIES_LEGAL_CONTACT`: Google expects the privacy
     policy to say who runs the app and how to reach them. Without them the pages say "the
     person who runs this Studio" and "the person who invited you".
   - The sign-in screen and the account menu link to both.
6. **Brand verification** (optional) shows the app's name and logo on the consent screen.
   Signed out, every page shows the sign-in screen, with one line of small print under the
   card saying what Studio is (`index.html` says the same in its meta description and a
   `<noscript>` block); the terms prohibit non-consensual intimate imagery and sexual
   content involving minors ("Intimate images and minors: zero tolerance").

**Troubleshooting.**
- `Error 400: redirect_uri_mismatch`: open "error details" on Google's page; the
  `redirect_uri` there must be in the client's list exactly. If it is and was just added,
  wait for it to apply.
- Signed in on one address but not the other: sessions are per address; sign in on each.
- "That Google account doesn't have access to this Studio": the address is not on
  `ARIES_AUTH_ALLOWED_EMAILS`.
- Google's own "Access blocked" page, in Testing: the account is not a test user.

## 4. Open

- Accounts are by email: changing an account's address on the allowlist leaves its
  history under the old one (no rename yet).
- No per-account queue fairness or quotas; cloud enhancer limits are shared.
- Sessions renew while used, with no upper limit: a session stays until it goes unused for
  `ARIES_SESSION_DAYS`, its device signs out, or its address leaves the allowlist. Passkeys
  would let anyone on the list sign in offline without the operator's help.
