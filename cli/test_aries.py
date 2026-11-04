"""Tests for the aries CLI: `python -m pytest cli` (no Studio needed)."""

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("aries", HERE / "aries.py")
aries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aries)


def test_sizes_match_the_web_app():
    """The CLI's tables are copies of web/src/types.ts: they must not drift."""
    types = (HERE.parent / "web/src/types.ts").read_text()
    for label, (w, h) in aries.ASPECTS.items():
        assert re.search(rf"'{re.escape(label)}': {{ width: {w}, height: {h},", types), label
    for name, factor in aries.SIZES.items():
        assert re.search(rf"{name}: {{\s*value: '{name}',[^}}]*factor: {factor},", types), name
    for name, steps in aries.QUALITIES.items():
        assert f"  {name}: {steps}," in types, name


def test_dimensions():
    assert aries.dimensions("1:1", "full") == (2048, 2048)
    assert aries.dimensions("16:9", "half") == (1376, 768)
    assert aries.dimensions("3:2", "quarter") == (640, 416)


def test_new_id_is_a_ulid():
    a, b = aries.new_id(), aries.new_id()
    assert re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", a)
    assert a != b


@pytest.mark.parametrize(
    "job, said",
    [
        ({"status": "done"}, "done"),
        ({"status": "failed", "error_detail": "out of memory"}, "failed: out of memory"),
        (
            {"status": "submitted", "tasks": [{"status": "queued", "blocked_reason": "no card free"}]},
            "queued (no card free)",
        ),
        (
            {
                "status": "running",
                "tasks": [{"stage": "denoise", "status": "running", "step": 5, "steps": 20, "s_per_step": 2.0}],
            },
            "drawing (step 5/20, about 30 s left)",
        ),
        ({"status": "running", "tasks": [{"stage": "decode", "status": "running"}]}, "decoding the image"),
    ],
)
def test_job_state(job, said):
    assert aries.job_state(job) == said


def test_config_is_private(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("ARIES_URL", raising=False)
    monkeypatch.delenv("ARIES_TOKEN", raising=False)
    path = aries.save_config("https://studio.example", "aries_x")
    assert (Path(path).stat().st_mode & 0o777) == 0o600
    assert aries.load_config() == ("https://studio.example", "aries_x")
    monkeypatch.setenv("ARIES_TOKEN", "aries_y")
    assert aries.load_config() == ("https://studio.example", "aries_y")


def test_login_refuses_an_empty_token(tmp_path, monkeypatch):
    """Signed out, Studio asks for a token, and nothing typed is not one."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    class SignedOut:
        def __init__(self, _url, token):
            assert not token, "no token was given"

        def request(self, _method, _path):
            raise aries.Fail("401: sign in first")

    monkeypatch.setattr(aries, "Studio", SignedOut)
    monkeypatch.setattr(aries.getpass, "getpass", lambda _prompt: "  ")
    for token in (None, ""):
        with pytest.raises(aries.Fail, match="No token"):
            aries.cmd_login(SimpleNamespace(url="https://studio.example", token=token))
    assert not Path(aries.config_path()).exists()


@pytest.mark.parametrize("job_id", ["../../evil", "a/b", "..", "", "a\\b", "a.png"])
def test_a_server_cannot_choose_where_an_image_is_saved(tmp_path, job_id):
    class Studio:
        def request(self, _method, _path):
            raise AssertionError("nothing is fetched for such an id")

    with pytest.raises(aries.Fail, match="odd id"):
        aries.save_image(Studio(), job_id, str(tmp_path / "out"), False)
    assert not (tmp_path / "out").exists()


def test_an_ordinary_id_is_saved_in_the_folder(tmp_path):
    class Studio:
        def request(self, _method, _path):
            return b"png"

    path = aries.save_image(Studio(), aries.new_id(), str(tmp_path / "out"), False)
    assert Path(path).parent == tmp_path / "out"
    assert Path(path).read_bytes() == b"png"


@pytest.mark.parametrize(
    "new, kept",
    [
        ("https://studio.example/b", True),
        ("https://evil.example/b", False),
        ("https://studio.example.evil.example/b", False),
        ("http://studio.example/b", False),
        ("https://studio.example:8443/b", False),
    ],
)
def test_a_redirect_keeps_the_token_only_on_the_same_site(new, kept):
    req = aries.urllib.request.Request("https://studio.example/a", headers={"Authorization": "Bearer aries_x"})
    out = aries.KeepTokenHome().redirect_request(req, None, 302, "Found", {}, new)
    assert out.has_header("Authorization") is kept


def test_a_redirect_to_https_on_the_same_host_keeps_the_token():
    req = aries.urllib.request.Request("http://studio.example:8080/a", headers={"Authorization": "Bearer aries_x"})
    out = aries.KeepTokenHome().redirect_request(req, None, 301, "Moved", {}, "https://studio.example/a")
    assert out.has_header("Authorization")
