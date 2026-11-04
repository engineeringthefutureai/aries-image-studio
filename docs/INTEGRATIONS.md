# CLI and AI tools

Status: **since 0.8.0**. Access tokens, the MCP server, the `aries` CLI and the skill.
Tests: `studio/tests/v3_backend_test.rs` (`test_access_token*`, `test_mcp_*`).

AI tools and scripts use Studio the way the web app does, as an account: they make images,
wait for them, look at them, and search the gallery. Four parts:

- **Access tokens** (sign-in on): made on Studio's **Connections** page, each one acts as
  the account that made it.
- **An MCP server in Studio** at `/mcp`, for AI tools that speak the Model Context
  Protocol, such as Antigravity. Its tools return the images as pictures, so the model sees
  what it made.
- **The `aries` CLI** (`cli/aries.py`): one Python file with no dependencies, to make
  images and save the PNGs, search, and upload reference images, from a terminal or an AI
  coding tool.
- **A skill** (`skills/aries-studio-mcp/SKILL.md`, the Agent Skills format) that teaches an AI
  tool how to use the MCP tools: choosing sizes and aspect ratios, waiting for jobs that take
  minutes, showing the results with their Studio links.

**Claude (claude.ai) is not yet supported**: its custom connectors sign in with OAuth,
which Studio does not offer yet (§6).

## 1. Access tokens

- **Making one:** the account menu → **Connections** (`/connections`) → name it → Create.
  The token (`aries_` and 43 more characters) is shown once; Studio keeps only its SHA-256
  (table `api_tokens`), like sessions. Give each tool its own.
- **Using one:** `Authorization: Bearer <token>` on any `/api/*` route and on `/mcp`. It
  acts as its account: the same jobs, images, uploads, preferences, and, for the owner,
  the card controls. A request with a token is judged by the token alone: a wrong or
  revoked one is refused (401) even next to a valid session cookie. Token requests are not
  subject to the `Origin` check (a browser never attaches a token on its own).
- **Revoking one:** the bin on the Connections page; it stops at once. Taking an account
  off `ARIES_AUTH_ALLOWED_EMAILS` stops its tokens with its sessions.
- Tokens are made, listed and revoked from a browser session only: a token cannot make
  another token. They do not expire; the page shows when each was last used (updated at
  most every 10 minutes). An account may have 50.
- Tokens cannot change what the read-only demo shows (DEMO.md §3). The demo is for browsers: a
  CLI needs a token as in any Studio with sign-in.
- **With sign-in off** there are no tokens: anyone who can reach Studio uses the API and
  `/mcp` without one.

API: `GET /api/tokens` (`{tokens: [{id, name, created_at, last_used_at}]}`, newest first),
`POST /api/tokens {name}` (201, `{token, record}`), `DELETE /api/tokens/{id}` (204). With a
token, `GET /api/me` answers `{auth: true, email, owner, token: true}`.

## 2. The MCP server

`POST /mcp`: the Streamable HTTP transport of the Model Context Protocol, the simplest
way: each JSON-RPC request is answered with one JSON response, with no session and no
event stream (`GET` and `DELETE` answer 405; notifications 202). Protocol versions
2025-03-26, 2025-06-18 and 2025-11-25. With sign-in on it needs a token (401 with
`WWW-Authenticate: Bearer` without one); the tools run through the same handlers as the
web app's API, so ownership, limits and validation are the same.

| Tool | Does |
|---|---|
| `generate_images` | Starts 1–4 images: prompt, aspect (the composer's seven), size (full, half, quarter), quality (draft, standard, high), seed, enhance and enhancer, negative prompt, up to 4 references (image ids, or upload hashes). Returns the job ids at once. The jobs look like the web app's: one batch in the timeline. |
| `get_jobs` | Where jobs are (queued and why, enhancing, drawing step n of m and time left, decoding, done, failed and why); with `wait_seconds` (up to 50), waits for them to finish. Finished images come back as 512 px JPEGs. |
| `find_images` | The finished images: newest first, by a search (SEARCH.md), or "more like" an image; favorites only; paged. Each with how it matched, and as a picture unless `images: false`. |
| `view_image` | One image as a 512, 768 or 1024 px JPEG, with its prompt, enhanced prompt, seed, size, steps and links. |
| `set_favorite`, `cancel_jobs` | As in the web app. |
| `studio_status` | The GPU host and its cards (online, busy, offline), the queue, the largest image possible now, the prompt enhancers. |

Links in answers (`…/images/{id}`, the PNG at `…/api/images/{id}`) use the address in
`ARIES_PUBLIC_URL` that the request came in on (the `Host` header without sign-in). Behind
a reverse proxy, `/mcp` must be passed through like `/api/*`. Each call is logged
(`MCP tool call`, the tool and whether it failed; never the arguments).

## 3. The `aries` CLI

`cli/aries.py`: Python 3.9 or later, standard library only. Copy it anywhere on the `PATH`:

```bash
install -m 755 cli/aries.py ~/.local/bin/aries
aries login https://studio.example --token aries_...   # asked for when left out
```

`aries login` checks the token and saves the address and token in
`~/.config/aries/config.json` (`%APPDATA%\aries` on Windows), readable by the user only.
`ARIES_URL` and `ARIES_TOKEN` override it.

| Command | Does |
|---|---|
| `aries gen "<prompt>" [-a 16:9] [-s half] [-q draft] [-n 4] [--seed N] [--enhance] [--enhancer ID] [--negative TEXT] [-r FILE\|ID ...] [-o DIR\|FILE.png]` | Makes images, prints progress to stderr, waits, and saves the PNGs (a folder gets `<id>.png`). `--no-wait` prints the ids instead. A file given to `-r` is uploaded first. |
| `aries wait <ids> [-o DIR]` | Waits for jobs and saves their PNGs. |
| `aries show <ids>` | State, prompt, enhanced prompt, settings, link. |
| `aries get <id> [-o DIR\|FILE] [--thumb 512\|768\|1024]` | Downloads an image (or a JPEG thumbnail). |
| `aries search ["<query>"] [--like ID] [--fav] [-n 10]` | The gallery, best match first, with how each matched. |
| `aries upload <files>` | Uploads reference images and prints their hashes. |
| `aries fav <ids> [--off]`, `aries cancel <ids>`, `aries status` | Favorites, cancelling, the cards and queue. |

`--json` prints machine-readable output. Exit codes: 0 done, 1 an error (Studio's message on
stderr), 3 a job failed or was cancelled, 4 `--timeout` passed, 130 interrupted (the jobs
go on; `aries wait` picks them up).

## 4. The skill

`skills/aries-studio-mcp/` is an [Agent Skill](https://agentskills.io): `SKILL.md` with a name
and a description that tells the AI tool when to use it, and instructions for doing it
well. It uses the MCP tools only, through a server named `aries-studio` (the name the
Connections page uses), and holds no address or token: links come from the tools' answers.
It is written for Antigravity: it names Antigravity's `call_mcp_tool` and where Antigravity
keeps the images the tools return, and tells the agent not to look for a CLI or probe its
configuration. The CLI
is for people and scripts: an AI tool that runs commands can use it too, without the skill.

## 5. Setting up Antigravity

1. Add Studio's MCP server (the Connections page shows this command with the address
   filled in):

   ```bash
   agy mcp add aries-studio https://studio.example/mcp
   ```

2. In Studio, **Connections** → make a token for Antigravity.
3. Put the token in the server's entry in `~/.gemini/config/mcp_config.json`; the
   Connections page shows the entry with the address and new token filled in:

   ```json
   {
     "mcpServers": {
       "aries-studio": {
         "serverUrl": "https://studio.example/mcp",
         "headers": { "Authorization": "Bearer aries_..." }
       }
     }
   }
   ```

4. Install the skill, for one project or for every project:

   ```bash
   # this project only (from the project's folder)
   mkdir -p .agents/skills && cp -r <aries-studio>/skills/aries-studio-mcp .agents/skills/
   # every project
   mkdir -p ~/.gemini/config/skills && cp -r <aries-studio>/skills/aries-studio-mcp ~/.gemini/config/skills/
   ```

   Each skill is a folder with `SKILL.md` directly inside it (`…/skills/aries-studio-mcp/SKILL.md`).
   Start a new conversation to load it.
5. Optionally, install the CLI (§3): the agent can then run it to save images into the
   project or use local files as references (the skill does not cover it; ask for it).

## 6. Later: Claude (claude.ai)

Claude's custom connectors are called from Anthropic's servers, so Studio must be reachable
from the internet (it is, when `ARIES_PUBLIC_URL` names an internet address), and they sign
in with OAuth: Studio would be an OAuth 2.1 authorization server for `/mcp` (protected
resource and server metadata, dynamic client registration, `/authorize` that signs in
with Google as now and asks the user to allow the connection, `/token` with PKCE and
refresh tokens), with redirects limited to Claude's. The connection would then show on the
Connections page next to the tokens, to revoke the same way. The skill is uploaded to
Claude as a zip of `skills/aries-studio-mcp`.
