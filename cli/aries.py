#!/usr/bin/env python3
"""aries: Aries Image Studio from the command line (docs/INTEGRATIONS.md).

Make images, wait for them and save the PNGs; search the gallery; upload reference
images. One file, Python 3.9+ and its standard library only: copy it anywhere.

    aries login https://studio.example --token aries_...
    aries gen "a lighthouse in fog at dawn" --aspect 16:9 -o out/
    aries search "rainy street at night"

The address and token come from `aries login` (kept in the user's config folder, readable
by the user only), or from ARIES_URL and ARIES_TOKEN.
"""

import argparse
import getpass
import json
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

__version__ = "0.8.0"

# The composer's choices (web/src/types.ts; the same tables as studio/src/mcp.rs).
ASPECTS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
SIZES = {"full": 1.0, "half": 0.5, "quarter": 0.25}
QUALITIES = {"draft": 12, "standard": 20, "high": 40}
FINISHED = ("done", "failed", "cancelled", "rejected")
POLL_S = 2.0


class Fail(Exception):
    """An error to print, then exit with `code`."""

    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def dimensions(aspect, size):
    w, h = ASPECTS[aspect]
    f = SIZES[size]
    return round(w * f / 32) * 32, round(h * f / 32) * 32


def new_id():
    """A ULID, as the web app makes job ids: 48 bits of milliseconds, 80 random."""
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    n = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    return "".join(alphabet[(n >> (5 * i)) & 31] for i in reversed(range(26)))


# --- configuration ---------------------------------------------------------------------


def config_path():
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "aries", "config.json")


def load_config():
    cfg = {}
    try:
        with open(config_path(), encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        raise Fail(f"cannot read {config_path()}: {e}")
    url = os.environ.get("ARIES_URL") or cfg.get("url")
    token = os.environ.get("ARIES_TOKEN") or cfg.get("token")
    if not url:
        raise Fail("not logged in: run `aries login <studio address>` (or set ARIES_URL)")
    return url.rstrip("/"), token or None


def save_config(url, token):
    path = config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Created readable by the user only: it holds the token.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"url": url, "token": token}, f, indent=2)
    return path


# --- HTTP ------------------------------------------------------------------------------


class KeepTokenHome(urllib.request.HTTPRedirectHandler):
    """A redirect keeps the token only on the same site, or moving from http to https on the same
    host: urllib would otherwise send it wherever a server or proxy points."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not same_site(req.full_url, newurl):
            new.remove_header("Authorization")
        return new


def same_site(old, new):
    a, b = urllib.parse.urlsplit(old), urllib.parse.urlsplit(new)
    if (a.scheme, a.netloc) == (b.scheme, b.netloc):
        return True
    return a.hostname == b.hostname and a.scheme == "http" and b.scheme == "https"


OPENER = urllib.request.build_opener(KeepTokenHome)


class Studio:
    def __init__(self, url, token):
        self.url = url
        self.token = token

    def request(self, method, path, body=None, raw=None, content_type=None):
        """The answer's bytes; an error answer raises Fail with Studio's message."""
        headers = {"User-Agent": f"aries-cli/{__version__}", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        elif raw is not None:
            data = raw
            headers["Content-Type"] = content_type or "application/octet-stream"
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with OPENER.open(req, timeout=120) as res:
                return res.read()
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            try:
                v = json.loads(text)
                msg = v.get("rejected") or v.get("error") or text
            except ValueError:
                msg = text.strip()[:300] or e.reason
            if e.code == 401 and self.token:
                msg += " (make a new token in Studio's Connections page, then `aries login`)"
            raise Fail(f"{method} {path}: {e.code} {msg}")
        except urllib.error.URLError as e:
            raise Fail(f"cannot reach {self.url}: {e.reason}")

    def json(self, method, path, body=None):
        out = self.request(method, path, body=body)
        return json.loads(out) if out else None

    def job(self, job_id):
        return self.json("GET", f"/api/jobs/{urllib.parse.quote(job_id)}")


# --- jobs ------------------------------------------------------------------------------


def job_state(job):
    """Where a job is, in words (the same as the MCP tools say)."""
    status = job.get("status")
    if status in ("done", "cancelled"):
        return status
    if status in ("failed", "rejected"):
        return f"{status}: {job.get('error_detail') or job.get('error_class') or 'no reason given'}"
    tasks = job.get("tasks") or []
    active = next((t for t in tasks if t.get("status") in ("assigned", "running", "uploading")), None)
    if active is None:
        blocked = next((t["blocked_reason"] for t in tasks if t.get("blocked_reason")), None)
        blocked = blocked or job.get("blocked_reason")
        return f"queued ({blocked})" if blocked else "queued"
    stage = active.get("stage")
    if stage == "denoise":
        step, steps = active.get("step") or 0, active.get("steps") or 0
        left = max(0, steps - step) * (active.get("s_per_step") or 0)
        eta = f", about {round(left)} s left" if step and left else ""
        return f"drawing (step {step}/{steps}{eta})"
    return {
        "enhance": "enhancing the prompt",
        "encode": "encoding the prompt",
        "decode": "decoding the image",
    }.get(stage, f"running {stage}")


SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def save_image(studio, job_id, out, single):
    """Downloads the PNG to `out`: a file name when it ends in .png and there is one image,
    else a folder (made when missing) that gets `<id>.png`."""
    if single and out.lower().endswith(".png"):
        path = out
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        # From the server's answer, and about to be a file name.
        if not SAFE_ID.fullmatch(job_id):
            raise Fail(f"Not saving an image with the odd id {job_id!r}")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"{job_id}.png")
    data = studio.request("GET", f"/api/images/{urllib.parse.quote(job_id)}")
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def wait_for(studio, ids, timeout, quiet):
    """Polls the jobs until all have finished (or `timeout` seconds pass), printing each
    change of state to stderr. The jobs as they ended."""
    deadline = time.monotonic() + timeout if timeout else None
    said = {}
    while True:
        jobs = [studio.job(i) for i in ids]
        for j in jobs:
            state = job_state(j)
            if not quiet and said.get(j["id"]) != state:
                print(f"{j['id']}: {state}", file=sys.stderr)
                said[j["id"]] = state
        if all(j["status"] in FINISHED for j in jobs):
            return jobs
        if deadline and time.monotonic() > deadline:
            raise Fail(f"still running after {timeout} s; later: aries wait {' '.join(ids)}", 4)
        time.sleep(POLL_S)


def finish(studio, args, ids):
    """Waits for the jobs, saves the finished images, and reports them."""
    try:
        jobs = wait_for(studio, ids, args.timeout, args.quiet)
    except KeyboardInterrupt:
        print(f"\nStopped waiting; the jobs go on. Later: aries wait {' '.join(ids)}", file=sys.stderr)
        raise SystemExit(130)
    failed = 0
    for j in jobs:
        if j["status"] == "done":
            j["file"] = save_image(studio, j["id"], args.output, len(jobs) == 1)
        else:
            failed += 1
    if args.json:
        print(json.dumps(jobs, indent=2))
    else:
        for j in jobs:
            print(j.get("file") or f"{j['id']}: {job_state(j)}")
    if failed:
        raise Fail(f"{failed} of {len(jobs)} did not finish", 3)


def upload(studio, path):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise Fail(f"cannot read {path}: {e.strerror}")
    return json.loads(studio.request("POST", "/api/uploads", raw=data))["sha256"]


def is_sha256(s):
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s)


# --- commands --------------------------------------------------------------------------


def cmd_login(args):
    url = args.url.rstrip("/")
    if "://" not in url:
        url = "https://" + url
    token = args.token or None
    me = json.loads(Studio(url, token).request("GET", "/api/me") or b"{}") if token else None
    if token is None:
        # Without sign-in Studio needs no token; with it, ask for one.
        try:
            me = json.loads(Studio(url, None).request("GET", "/api/me"))
            needs_token = False
        except Fail as e:
            if "401" not in str(e):
                raise
            needs_token = True
        if needs_token:
            print(f"Make a token in Studio: {url}/connections", file=sys.stderr)
            token = getpass.getpass("Token: ").strip()
            if not token:
                raise Fail("No token given: not logged in")
            me = json.loads(Studio(url, token).request("GET", "/api/me") or b"{}")
    # And the API itself answers (an old Studio without tokens would not).
    Studio(url, token).request("GET", "/api/jobs?limit=1")
    path = save_config(url, token)
    who = me.get("email") if me and me.get("auth") else None
    print(f"Logged in to {url}" + (f" as {who}" if who else "") + f" (saved in {path})")


def cmd_logout(_args):
    try:
        os.remove(config_path())
        print(f"Removed {config_path()}")
    except FileNotFoundError:
        print("Not logged in")


def cmd_gen(studio, args):
    width, height = dimensions(args.aspect, args.size)
    refs = []
    for r in args.ref or []:
        if os.path.isfile(r):
            refs.append({"upload": upload(studio, r)})
        elif is_sha256(r):
            refs.append({"upload": r})
        else:
            refs.append({"image": r})
    enhancer = args.enhancer
    if enhancer == "local":
        enhancer = "builtin"
    batch = new_id()
    ids = []
    for i in range(args.count):
        job_id = batch if i == 0 else new_id()
        body = {
            "id": job_id,
            "prompt": args.prompt,
            "negative_prompt": args.negative or "",
            "width": width,
            "height": height,
            "steps": QUALITIES[args.quality],
            "enhance": bool(args.enhance or enhancer),
            "tiling": False,
            "references": refs,
            "meta": {
                "batch_id": batch,
                "aspect": args.aspect,
                "size_multiplier": args.size,
                "quality": args.quality,
                "index": i,
                "client": "cli",
            },
        }
        if args.seed is not None:
            body["seed"] = args.seed + i
        if enhancer:
            body["enhancer"] = enhancer
        studio.json("POST", "/api/jobs", body)
        ids.append(job_id)
    if args.no_wait:
        print(json.dumps(ids) if args.json else "\n".join(ids))
        return
    if not args.quiet:
        print(f"Started {len(ids)} at {width}×{height}, {QUALITIES[args.quality]} steps", file=sys.stderr)
    finish(studio, args, ids)


def cmd_wait(studio, args):
    finish(studio, args, args.ids)


def cmd_show(studio, args):
    jobs = [studio.job(i) for i in args.ids]
    if args.json:
        print(json.dumps(jobs, indent=2))
        return
    for j in jobs:
        print(f"{j['id']}: {job_state(j)}")
        print(
            f"  {j['width']}×{j['height']}, {j['steps']} steps, seed {j['seed']}"
            + (", favorite" if j.get("favorite") else "")
        )
        print(f"  prompt: {j['prompt']}")
        if j.get("enhanced_prompt"):
            print(f"  enhanced: {j['enhanced_prompt']}")
        if j["status"] == "done":
            print(f"  {studio.url}/images/{j['id']}")


def cmd_get(studio, args):
    if args.thumb:
        data = studio.request("GET", f"/api/images/{urllib.parse.quote(args.id)}/thumb?size={args.thumb}")
        path = args.output if args.output.lower().endswith(".jpg") else os.path.join(args.output, f"{args.id}.jpg")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    else:
        path = save_image(studio, args.id, args.output, True)
    print(path)


def cmd_search(studio, args):
    q = {"limit": args.limit, "status": "done"}
    if args.query:
        q["q"] = args.query
    if args.like:
        q["like"] = args.like
    if args.fav:
        q["favorite"] = "true"
    if args.cursor:
        q["before"] = args.cursor
    page = studio.json("GET", "/api/jobs?" + urllib.parse.urlencode(q))
    jobs = page.get("jobs") or []
    if args.json:
        print(json.dumps(page, indent=2))
        return
    if not jobs:
        print("No images found.")
    for j in jobs:
        how = []
        if j.get("search_strength"):
            how.append(f"{j['search_strength']} look")
        if j.get("search_words"):
            how.append(("phrase in " if j.get("search_phrase") else "words in ") + j["search_words"])
        prompt = j["prompt"] if len(j["prompt"]) <= 100 else j["prompt"][:100] + "…"
        print(
            f"{j['id']}  {j['width']}×{j['height']}{'  ★' if j.get('favorite') else ''}"
            + (f"  [{', '.join(how)}]" if how else "")
        )
        print(f"  {prompt}")
    if page.get("next"):
        print(f"More: aries search ... --cursor {page['next']}", file=sys.stderr)


def cmd_upload(studio, args):
    for path in args.files:
        print(f"{upload(studio, path)}  {path}")


def cmd_fav(studio, args):
    for i in args.ids:
        studio.json("PATCH", f"/api/jobs/{urllib.parse.quote(i)}", {"favorite": not args.off})


def cmd_cancel(studio, args):
    for i in args.ids:
        j = studio.json("POST", f"/api/jobs/{urllib.parse.quote(i)}/cancel")
        print(f"{i}: {job_state(j)}")


def cmd_status(studio, args):
    system = studio.json("GET", "/api/system")
    caps = studio.json("GET", "/api/capabilities")
    if args.json:
        print(json.dumps({"system": system, "capabilities": caps}, indent=2))
        return
    if not system.get("agents"):
        print("No GPU host has connected.")
    for a in system.get("agents") or []:
        cards = ", ".join(
            f"{e['state']} ({e['current_stage']})" if e["state"] == "busy" and e.get("current_stage") else e["state"]
            for e in a.get("executors") or []
        )
        print(f"GPU host {a['id']}: {'online' if a.get('online') else 'offline'}; cards: {cards or 'none'}")
    queued = sum(q.get("queued", 0) for q in system.get("queues") or [])
    running = sum(q.get("running", 0) for q in system.get("queues") or [])
    print(f"Stages queued: {queued}, running: {running}")
    enhancers = (["local"] if caps.get("enhance_ever") else []) + [c["id"] for c in caps.get("cloud_enhancers") or []]
    print(f"Enhancers: {', '.join(enhancers) or 'none'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="aries", description="Aries Image Studio from the command line.")
    p.add_argument("--version", action="version", version=f"aries {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    s = sub.add_parser("login", help="save Studio's address and an access token")
    s.add_argument("url", help="Studio's address, e.g. https://studio.example")
    s.add_argument("--token", help="an access token from Studio's Connections page (asked for when needed)")
    sub.add_parser("logout", help="forget the address and token")

    def waiting(s):
        s.add_argument(
            "-o", "--output", default=".", help="folder for the PNGs, or a .png name for one image (default: .)"
        )
        s.add_argument(
            "--timeout", type=float, default=0, help="give up waiting after this many seconds (default: never)"
        )
        s.add_argument("--quiet", action="store_true", help="no progress on stderr")
        s.add_argument("--json", action="store_true", help="print the jobs as JSON")

    s = sub.add_parser("gen", help="make images, wait for them and save the PNGs")
    s.add_argument("prompt")
    s.add_argument("-a", "--aspect", choices=ASPECTS, default="1:1")
    s.add_argument(
        "-s", "--size", choices=SIZES, default="full", help="full is about 4 MP; half and quarter are faster"
    )
    s.add_argument(
        "-q", "--quality", choices=QUALITIES, default="standard", help="steps: draft 12, standard 20, high 40"
    )
    s.add_argument("-n", "--count", type=int, choices=range(1, 5), default=1, metavar="1-4")
    s.add_argument("--seed", type=int, help="repeat an image (with -n: seed, seed+1, ...)")
    s.add_argument("--enhance", action="store_true", help="let Studio rewrite the prompt first")
    s.add_argument("--enhancer", help="'local' or a cloud enhancer's id (implies --enhance)")
    s.add_argument("--negative", help="what to keep out of the image")
    s.add_argument(
        "-r",
        "--ref",
        action="append",
        metavar="FILE|ID",
        help="a reference image: a file, an image id or an upload's hash; repeat in order",
    )
    s.add_argument("--no-wait", action="store_true", help="print the job ids and return")
    waiting(s)

    s = sub.add_parser("wait", help="wait for jobs and save their PNGs")
    s.add_argument("ids", nargs="+")
    waiting(s)

    s = sub.add_parser("show", help="jobs' state, prompts and settings")
    s.add_argument("ids", nargs="+")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("get", help="download an image")
    s.add_argument("id")
    s.add_argument("-o", "--output", default=".", help="folder, or a file name")
    s.add_argument("--thumb", type=int, choices=(512, 768, 1024), help="a JPEG thumbnail of this size instead")

    s = sub.add_parser("search", help="the gallery: newest first, or by a search")
    s.add_argument("query", nargs="?", help="what the images show, or their prompts' words")
    s.add_argument("--like", metavar="ID", help="images that look like this one")
    s.add_argument("--fav", action="store_true", help="favorites only")
    s.add_argument("-n", "--limit", type=int, default=10)
    s.add_argument("--cursor", help="the next page, from the previous search")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("upload", help="upload reference images; prints their hashes")
    s.add_argument("files", nargs="+")

    s = sub.add_parser("fav", help="mark images as favorites")
    s.add_argument("ids", nargs="+")
    s.add_argument("--off", action="store_true", help="unmark them")

    s = sub.add_parser("cancel", help="stop jobs")
    s.add_argument("ids", nargs="+")

    s = sub.add_parser("status", help="the GPU cards, the queue and the enhancers")
    s.add_argument("--json", action="store_true")

    args = p.parse_args(argv)
    try:
        if args.cmd == "login":
            return cmd_login(args)
        if args.cmd == "logout":
            return cmd_logout(args)
        studio = Studio(*load_config())
        globals()["cmd_" + args.cmd](studio, args)
    except Fail as e:
        print(f"aries: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
