"""Integration and unit tests for Aries Studio Architecture v2 Async Backend.
Tests database state machine, storage lifecycle, scheduler, socket wire protocol, and manager HTTP APIs.
"""
import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
import pytest
from httpx import AsyncClient, ASGITransport

from aries_manager.ulid import generate_ulid
from aries_manager.db import Database, get_db_connection, init_db
from aries_manager.storage import (
    init_job_dir,
    get_job_paths,
    compute_sha256,
    check_disk_watermark,
    delete_job_files_for_acknowledged,
)
from aries_manager.scheduler import CardScheduler
from aries_manager.server import ManagerSocketServer
from aries_manager.api import app


def _backdate(db: Database, job_id: str, seconds: float) -> None:
    conn = get_db_connection(db.db_path)
    with conn:
        conn.execute("UPDATE jobs SET created_at = ? WHERE id = ?;", (time.time() - seconds, job_id))
    conn.close()


@pytest.fixture
def tmp_env(monkeypatch):
    """Fixture providing an isolated temporary directory for DB, jobs, and socket."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="aries_test_"))
    db_path = tmp_dir / "manager.sqlite"
    socket_path = tmp_dir / "manager.sock"
    jobs_dir = tmp_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("aries_manager.config.DATA_DIR", tmp_dir)
    monkeypatch.setattr("aries_manager.config.DB_PATH", db_path)
    monkeypatch.setattr("aries_manager.config.SOCKET_PATH", socket_path)
    monkeypatch.setattr("aries_manager.config.JOBS_DIR", jobs_dir)

    init_db(db_path, num_cards=2)
    test_db = Database(db_path)

    yield {
        "dir": tmp_dir,
        "db_path": db_path,
        "socket_path": socket_path,
        "jobs_dir": jobs_dir,
        "db": test_db,
    }

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_ulid_generation():
    """Verify ULID uniqueness, length, and chronological ordering."""
    u1 = generate_ulid()
    time.sleep(0.002)
    u2 = generate_ulid()
    assert len(u1) == 26
    assert len(u2) == 26
    assert u1 != u2
    assert u1 < u2


def test_database_state_machine(tmp_env):
    """Verify job submissions, atomic transitions, retries, and failure semantics."""
    db: Database = tmp_env["db"]

    # 1. Submit jobs (one with enhance, one without)
    j1 = generate_ulid()
    j2 = generate_ulid()
    v1, ids = db.submit_jobs([
        {"id": j1, "prompt": "a beautiful forest", "width": 2048, "height": 2048, "enhance": True},
        {"id": j2, "prompt": "a neon city", "width": 2048, "height": 2048, "enhance": False},
    ])
    assert v1 > 0
    assert ids == [j1, j2]

    # Verify initial statuses
    job1 = db.get_job(j1)
    assert job1["status"] == "enhance_queued"
    assert job1["stage"] == "enhance"
    assert job1["attempts"] == 0

    job2 = db.get_job(j2)
    assert job2["status"] == "encode_queued"
    assert job2["stage"] == "encode"

    # 2. Claim job1 for enhance stage
    claimed = db.claim_oldest_queued_job("enhance", card=0, pid=1234)
    assert claimed is not None
    assert claimed["id"] == j1
    assert claimed["status"] == "enhancing"
    assert claimed["card"] == 0
    assert claimed["worker_pid"] == 1234

    # Record progress
    db.record_progress(j1, step=1, steps=1, s_per_step=0.5)
    j1_prog = db.get_job(j1)
    assert j1_prog["step"] == 1
    assert j1_prog["s_per_step"] == 0.5

    # 3. Advance enhance -> encode_queued
    db.advance_stage_success(j1, "enhance", {"enhance_s": 1.2}, peak_vram=1000)
    job1 = db.get_job(j1)
    assert job1["status"] == "encode_queued"
    assert job1["stage"] == "encode"
    assert job1["attempts"] == 0
    assert job1["timings"]["enhance_s"] == 1.2

    # 4. Advance encode -> denoise_queued
    db.claim_oldest_queued_job("encode", card=0, pid=1234)
    db.advance_stage_success(j1, "encode", {"encode_s": 0.8}, peak_vram=2000)
    job1 = db.get_job(j1)
    assert job1["status"] == "denoise_queued"
    assert job1["stage"] == "denoise"

    # 5. Advance denoise -> decode_queued
    db.claim_oldest_queued_job("denoise", card=0, pid=1234)
    db.advance_stage_success(j1, "denoise", {"denoise_s": 10.5, "s_per_step": 0.26}, peak_vram=3000)
    job1 = db.get_job(j1)
    assert job1["status"] == "decode_queued"
    assert job1["stage"] == "decode"

    # 6. Advance decode -> stored
    db.claim_oldest_queued_job("decode", card=1, pid=5678)
    dummy_sha256 = "abcd1234hash"
    db.advance_stage_success(j1, "decode", {"decode_s": 2.1}, peak_vram=4000, sha256=dummy_sha256)
    job1 = db.get_job(j1)
    assert job1["status"] == "stored"
    assert job1["stage"] == "stored"
    assert job1["sha256"] == dummy_sha256
    assert job1["completed_at"] is not None
    assert job1["timings"]["total_s"] >= 0
    assert job1["timings"]["decode_s"] == 2.1

    # 7. Test failure semantics on job2
    db.claim_oldest_queued_job("encode", card=0, pid=9999)
    # First failure -> requeues
    db.record_stage_failure(j2, "encode", "GPU memory fault", crash_class="gpu_fault")
    job2 = db.get_job(j2)
    assert job2["status"] == "encode_queued"
    assert job2["attempts"] == 1
    assert job2["error"] == "GPU memory fault"

    # Second failure -> permanently failed (no crash loops)
    db.claim_oldest_queued_job("encode", card=0, pid=9999)
    db.record_stage_failure(j2, "encode", "Second GPU failure", crash_class="gpu_fault")
    job2 = db.get_job(j2)
    assert job2["status"] == "failed"
    assert job2["attempts"] == 2

    # 8. Test polling cursor
    ver, changed_all = db.get_jobs_changed_since(0)
    assert len(changed_all) >= 2
    ver2, changed_recent = db.get_jobs_changed_since(ver)
    assert len(changed_recent) == 0

    # 9. Test deleting failed jobs
    deleted_ids = db.delete_failed_jobs(hard_delete=True)
    assert deleted_ids == [j2]
    assert db.get_job(j2) is None


def test_startup_reconciliation(tmp_env):
    """Verify ARCHITECTURE.md §8: on manager start, in-flight jobs revert to queued without incrementing attempts."""
    db: Database = tmp_env["db"]
    j = generate_ulid()
    db.submit_jobs([{"id": j, "prompt": "test in-flight", "width": 2048, "height": 2048}])

    # Claim into encoding
    db.claim_oldest_queued_job("encode", card=0, pid=7777)
    job = db.get_job(j)
    assert job["status"] == "encoding"
    assert job["attempts"] == 0

    # Simulate manager crash & restart
    result = db.startup_reconcile()
    assert j in result["reconciled_jobs"]

    reconciled_job = db.get_job(j)
    assert reconciled_job["status"] == "encode_queued"
    assert reconciled_job["attempts"] == 0
    assert reconciled_job["worker_pid"] is None


def test_storage_lifecycle(tmp_env):
    """Verify job directories, atomic file writes, SHA-256 calculation, and GC."""
    jid = generate_ulid()
    req_data = {"id": jid, "prompt": "aurora borealis", "width": 2048, "height": 2048}

    paths = init_job_dir(jid, req_data)
    req_file = Path(paths["request_json"])
    assert req_file.exists()
    assert json.loads(req_file.read_text())["prompt"] == "aurora borealis"

    # Create dummy image.png and test SHA-256
    img_file = Path(paths["image_png"])
    img_file.write_bytes(b"PNG_FAKE_IMAGE_DATA_12345")
    sha = compute_sha256(img_file)
    assert len(sha) == 64

    # Test disk watermark
    is_above, ratio, free_b = check_disk_watermark()
    assert isinstance(is_above, bool)
    assert 0.0 <= ratio <= 1.0

    # Test acknowledgment deletion
    delete_job_files_for_acknowledged(jid)
    assert not img_file.exists()


def test_scheduler_constraints_and_stickiness(tmp_env):
    """Verify stage candidate selection, RAM constraints, and stickiness yielding."""
    db: Database = tmp_env["db"]
    scheduler = CardScheduler(db, num_cards=2)

    # 1. No jobs queued -> pick_next_stage is None
    assert scheduler.pick_next_stage(0) is None

    # 2. Submit encode and denoise jobs
    j1 = generate_ulid()
    j2 = generate_ulid()
    db.submit_jobs([
        {"id": j1, "prompt": "prompt 1", "width": 2048, "height": 2048},
        {"id": j2, "prompt": "prompt 2", "width": 2048, "height": 2048},
    ])

    # Any card can run encode if no other card is running encode
    assert scheduler.can_stage_run_on_card("encode", 0) is True
    assert scheduler.can_stage_run_on_card("encode", 1) is True
    assert scheduler.pick_next_stage(0) == "encode"

    # When Card 0 is running encode, Card 1 cannot run encode (single instance limit)
    scheduler.card_stages[0] = "encode"
    assert scheduler.can_stage_run_on_card("encode", 1) is False
    assert scheduler.pick_next_stage(1) is None
    scheduler.card_stages[0] = None

    # 3. Test card unavailability
    scheduler.mark_card_unavailable(1, "simulated hardware fault")
    assert scheduler.card_states[1] == "unavailable"
    assert scheduler.can_stage_run_on_card("encode", 1) is False
    assert scheduler.can_stage_run_on_card("encode", 0) is True
    assert scheduler.pick_next_stage(0) == "encode"


@pytest.mark.asyncio
async def test_socket_protocol_handshake_and_flow(tmp_env):
    """Test Unix socket protocol: worker connects, handshakes, claims job, and completes."""
    db: Database = tmp_env["db"]
    scheduler = CardScheduler(db, num_cards=2)
    socket_path = tmp_env["socket_path"]
    server = ManagerSocketServer(db, scheduler, socket_path=socket_path)

    await server.start()

    # Submit a job
    jid = generate_ulid()
    init_job_dir(jid, {"id": jid, "prompt": "socket test", "width": 2048, "height": 2048})
    db.submit_jobs([{"id": jid, "prompt": "socket test", "width": 2048, "height": 2048}])

    # Connect dummy worker client
    reader, writer = await asyncio.open_unix_connection(str(socket_path))

    # Send hello
    writer.write(json.dumps({
        "event": "hello",
        "stage": "encode",
        "card": 0,
        "pid": 4321,
        "versions": {"torch": "2.5.0+xpu"},
        "load_s": 1.25,
    }).encode("utf-8") + b"\n")
    await writer.drain()

    # Send claim
    writer.write(json.dumps({"event": "claim"}).encode("utf-8") + b"\n")
    await writer.drain()

    # Read manager response
    line = await reader.readline()
    resp = json.loads(line.decode("utf-8"))
    assert resp["action"] == "job"
    assert resp["id"] == jid

    # Send progress
    writer.write(json.dumps({
        "event": "progress",
        "id": jid,
        "step": 1,
        "steps": 1,
        "s_per_step": 0.1,
    }).encode("utf-8") + b"\n")
    await writer.drain()

    # Send done
    writer.write(json.dumps({
        "event": "done",
        "id": jid,
        "timings": {"encode_s": 0.45},
        "peak_vram": 5000000,
    }).encode("utf-8") + b"\n")
    await writer.drain()

    # Give manager moment to process
    await asyncio.sleep(0.05)

    job = db.get_job(jid)
    assert job["status"] == "denoise_queued"
    assert job["stage"] == "denoise"
    assert job["timings"]["encode_s"] == 0.45

    writer.close()
    await writer.wait_closed()
    await server.stop()


@pytest.mark.asyncio
async def test_manager_api_full_cycle(tmp_env):
    """Test full HTTP API lifecycle: POST /jobs, GET /jobs, /status, /cards, /metrics, cancel, and image ACK."""
    # Wire app dependencies to temporary test environment
    monkeypatch_db = tmp_env["db"]
    transport = ASGITransport(app=app)

    # Override db and socket server paths in app modules
    import aries_manager.api as api_mod
    api_mod.db = monkeypatch_db
    api_mod.scheduler = CardScheduler(monkeypatch_db, num_cards=2)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. GET /status
        resp = await client.get("/status")
        assert resp.status_code == 200
        status_data = resp.json()
        assert "cards" in status_data
        assert len(status_data["cards"]) == 2

        # 2. POST /jobs (submission)
        jid = generate_ulid()
        submit_resp = await client.post("/jobs", json={
            "id": jid,
            "prompt": "futuristic flying car",
            "aspect": "16:9",
            "steps": 40,
            "seed": 100,
        })
        assert submit_resp.status_code == 201
        submit_json = submit_resp.json()
        assert submit_json["version"] > 0
        assert jid in submit_json["job_ids"]

        # 3. GET /jobs (returns list of current jobs directly)
        poll_resp = await client.get("/jobs")
        assert poll_resp.status_code == 200
        jobs_list = poll_resp.json()
        assert isinstance(jobs_list, list)
        assert any(j["id"] == jid for j in jobs_list)

        # 4. GET /jobs/{id}
        detail_resp = await client.get(f"/jobs/{jid}")
        assert detail_resp.status_code == 200
        detail_json = detail_resp.json()
        assert detail_json["prompt"] == "futuristic flying car"
        assert detail_json["status"] == "encode_queued"

        # 5. GET /metrics
        metrics_resp = await client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "aries_manager_" in metrics_resp.text

        # 7. Create a fake finished image and test GET /jobs/{id}/image and DELETE /jobs/{id}/image
        paths = get_job_paths(jid)
        img_p = Path(paths["image_png"])
        img_p.parent.mkdir(parents=True, exist_ok=True)
        img_p.write_bytes(b"\x89PNG\r\n\x1a\nfakeimagebytes")
        fake_hash = compute_sha256(img_p)

        # Walk the job through its stages to stored
        monkeypatch_db.advance_stage_success(jid, "encode", {"encode_s": 1.0}, peak_vram=1000)
        monkeypatch_db.advance_stage_success(jid, "denoise", {"denoise_s": 1.0}, peak_vram=1000)
        monkeypatch_db.advance_stage_success(jid, "decode", {"decode_s": 1.0}, peak_vram=1000, sha256=fake_hash)

        # GET image
        get_img_resp = await client.get(f"/jobs/{jid}/image")
        assert get_img_resp.status_code == 200
        assert get_img_resp.headers["X-Content-SHA256"] == fake_hash
        assert get_img_resp.content == b"\x89PNG\r\n\x1a\nfakeimagebytes"

        # DELETE image (acknowledge)
        del_img_resp = await client.delete(f"/jobs/{jid}/image")
        assert del_img_resp.status_code == 200
        assert del_img_resp.json()["status"] == "acknowledged"

        # Image file must now be deleted by ACK GC
        assert not img_p.exists()
        updated_job = monkeypatch_db.get_job(jid)
        assert updated_job["status"] == "acknowledged"

        # 8. POST /jobs/{id}/cancel on another job
        jid2 = generate_ulid()
        await client.post("/jobs", json={"id": jid2, "prompt": "cancel me"})
        cancel_resp = await client.post(f"/jobs/{jid2}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

        # 9. Test active job deletion rejection (400)
        jid3 = generate_ulid()
        await client.post("/jobs", json={"id": jid3, "prompt": "active queued job"})
        del_active_resp = await client.delete(f"/jobs/{jid3}")
        assert del_active_resp.status_code == 400
        assert "Cannot delete job in active state" in del_active_resp.json()["detail"]

        # 10. Create failed job and test DELETE /jobs/failed
        jid_fail = generate_ulid()
        await client.post("/jobs", json={"id": jid_fail, "prompt": "fail me"})
        monkeypatch_db.record_stage_failure(jid_fail, "encode", "err1")
        monkeypatch_db.record_stage_failure(jid_fail, "encode", "err2")
        failed_job = monkeypatch_db.get_job(jid_fail)
        assert failed_job["status"] == "failed"

        # Verify failed job is currently listed in GET /jobs
        jobs_list = (await client.get("/jobs")).json()
        assert any(j["id"] == jid_fail for j in jobs_list)

        # Bulk delete failed jobs
        del_failed_resp = await client.delete("/jobs/failed")
        assert del_failed_resp.status_code == 200
        assert del_failed_resp.json()["status"] == "deleted"
        assert jid_fail in del_failed_resp.json()["deleted_jobs"]

        # Verify failed job is no longer listed in GET /jobs
        jobs_list_after = (await client.get("/jobs")).json()
        assert not any(j["id"] == jid_fail for j in jobs_list_after)

        # 11. Test DELETE /jobs/{id} on cancelled job
        del_cancelled_resp = await client.delete(f"/jobs/{jid2}")
        assert del_cancelled_resp.status_code == 200
        assert del_cancelled_resp.json()["status"] == "deleted"
        assert monkeypatch_db.get_job(jid2) is None


def test_scheduler_no_duplicate_denoise_worker(tmp_env):
    """Verify that if only 1 job is queued for denoise, it does NOT spin up a second card."""
    db: Database = tmp_env["db"]
    scheduler = CardScheduler(db, num_cards=2)

    # 1. Submit 1 job directly into denoise_queued
    j1 = generate_ulid()
    db.submit_jobs([{"id": j1, "prompt": "job 1", "width": 2048, "height": 2048}])
    # Advance to denoise_queued
    db.advance_stage_success(j1, "encode", {"encode_s": 1.0}, peak_vram=1000)
    job = db.get_job(j1)
    assert job["status"] == "denoise_queued"
    assert db.get_queue_depths()["denoise"] == 1

    # Card 0 should pick denoise
    stage_for_card0 = scheduler.pick_next_stage(card_idx=0)
    assert stage_for_card0 == "denoise"

    # Simulate Card 0 worker spawning (loading weights, job not yet claimed)
    scheduler.card_states[0] = "running"
    scheduler.card_stages[0] = "denoise"
    scheduler.card_jobs[0] = None

    # Card 1 should NOT pick denoise because 1 job is already covered by Card 0
    assert not scheduler.can_stage_run_on_card("denoise", card_idx=1)
    stage_for_card1 = scheduler.pick_next_stage(card_idx=1)
    assert stage_for_card1 is None

    # 2. Now submit a second job into denoise_queued (depth = 2)
    j2 = generate_ulid()
    db.submit_jobs([{"id": j2, "prompt": "job 2", "width": 2048, "height": 2048}])
    db.advance_stage_success(j2, "encode", {"encode_s": 1.0}, peak_vram=1000)
    assert db.get_queue_depths()["denoise"] == 2

    # Now Card 1 CAN pick denoise because 2 jobs > 1 spawning worker
    assert scheduler.can_stage_run_on_card("denoise", card_idx=1)
    stage_for_card1 = scheduler.pick_next_stage(card_idx=1)
    assert stage_for_card1 == "denoise"

    # 3. Simulate Card 0 claiming Job 1 and running
    scheduler.set_card_job(0, j1)
    # Card 1 has not claimed yet; submit a 3rd job (depth = 2 in denoise_queued)
    # Card 1 will claim Job 2, Card 0 is busy with Job 1.
    scheduler.card_states[1] = "running"
    scheduler.card_stages[1] = "denoise"
    scheduler.set_card_job(1, j2)

    # Both cards are busy running jobs
    assert not scheduler.can_stage_run_on_card("denoise", card_idx=0)
    assert not scheduler.can_stage_run_on_card("denoise", card_idx=1)


def test_scheduler_enhance_two_workers_when_needed(tmp_env):
    """Verify that enhance stage can run on 2 cards when needed, without choking or redundant spawns."""
    db: Database = tmp_env["db"]
    scheduler = CardScheduler(db, num_cards=2)

    # 1. Submit 1 job directly into enhance_queued
    j1 = generate_ulid()
    db.submit_jobs([{"id": j1, "prompt": "enhance prompt 1", "width": 2048, "height": 2048, "enhance": True}])
    job = db.get_job(j1)
    assert job["status"] == "enhance_queued"
    assert db.get_queue_depths()["enhance"] == 1

    # Card 0 should pick enhance
    stage_for_card0 = scheduler.pick_next_stage(card_idx=0)
    assert stage_for_card0 == "enhance"

    # Simulate Card 0 worker spawning (loading weights, job not yet claimed)
    scheduler.card_states[0] = "running"
    scheduler.card_stages[0] = "enhance"
    scheduler.card_jobs[0] = None

    # Card 1 should NOT pick enhance because 1 job is covered by spawning Card 0
    assert not scheduler.can_stage_run_on_card("enhance", card_idx=1)
    stage_for_card1 = scheduler.pick_next_stage(card_idx=1)
    assert stage_for_card1 is None

    # 2. Submit a second job into enhance_queued (depth = 2)
    j2 = generate_ulid()
    db.submit_jobs([{"id": j2, "prompt": "enhance prompt 2", "width": 2048, "height": 2048, "enhance": True}])
    assert db.get_queue_depths()["enhance"] == 2

    # Now Card 1 CAN pick enhance because 2 jobs are queued > 1 spawning worker
    assert scheduler.can_stage_run_on_card("enhance", card_idx=1)
    stage_for_card1 = scheduler.pick_next_stage(card_idx=1)
    assert stage_for_card1 == "enhance"

    # 3. Simulate Card 0 claiming Job 1 and Card 1 claiming Job 2
    scheduler.set_card_job(0, j1)
    scheduler.card_states[1] = "running"
    scheduler.card_stages[1] = "enhance"
    scheduler.set_card_job(1, j2)

    # Both cards are busy running enhance jobs concurrently
    assert not scheduler.can_stage_run_on_card("enhance", card_idx=0)
    assert not scheduler.can_stage_run_on_card("enhance", card_idx=1)


def test_max_wait_does_not_livelock_new_worker(tmp_env):
    """Verify that should_worker_exit_for_max_wait compares relative waiting times correctly."""
    db: Database = tmp_env["db"]
    scheduler = CardScheduler(db, num_cards=2)
    scheduler.decode_ram_floor = 1024

    # 1. Job 1 in enhance_queued submitted 15 minutes ago
    j1 = generate_ulid()
    db.submit_jobs([{"id": j1, "prompt": "enhance prompt", "width": 2048, "height": 2048, "enhance": True}])
    _backdate(db, j1, 900)

    # 2. Job 2 in decode_queued submitted 20 minutes ago (older than Job 1)
    j2 = generate_ulid()
    db.submit_jobs([{"id": j2, "prompt": "decode prompt", "width": 2048, "height": 2048, "enhance": False}])
    db.advance_stage_success(j2, "encode", {"encode_s": 1.0}, peak_vram=1000)
    db.advance_stage_success(j2, "denoise", {"denoise_s": 10.0}, peak_vram=1000)
    _backdate(db, j2, 1200)

    # Oldest waiting job is decode (1200s > 900s)
    assert scheduler.pick_next_stage(card_idx=0) == "decode"

    # Decode worker should NOT yield to enhance because decode is older (1200s > 900s)
    should_exit, reason = scheduler.should_worker_exit_for_max_wait("decode")
    assert not should_exit
    assert reason is None


def test_enhancer_stays_warm_when_card_not_needed(tmp_env):
    """Verify that enhance worker on Card 1 stays warm when Card 0 can take queued work."""
    db: Database = tmp_env["db"]
    scheduler = CardScheduler(db, num_cards=2)

    # 1. Card 1 is running enhance, Card 0 is idle
    scheduler.card_states[0] = "idle"
    scheduler.card_stages[0] = None
    scheduler.card_states[1] = "running"
    scheduler.card_stages[1] = "enhance"

    # Job is in encode_queued (e.g. just finished enhance)
    j1 = generate_ulid()
    db.submit_jobs([{"id": j1, "prompt": "prompt 1", "width": 2048, "height": 2048, "enhance": False}])
    assert db.get_queue_depths()["encode"] == 1
    assert db.get_queue_depths()["enhance"] == 0

    # Card 1 is NOT needed by encode because Card 0 is idle and can run encode
    needed, reason = scheduler.is_card_needed_by_waiting_stages(card_idx=1, current_stage="enhance")
    assert not needed
    assert reason is None

    # 2. Now simulate Card 0 busy running denoise
    scheduler.card_states[0] = "running"
    scheduler.card_stages[0] = "denoise"

    # If encode is waiting and Card 0 is busy denoising, Card 1 IS needed to run encode!
    needed, reason = scheduler.is_card_needed_by_waiting_stages(card_idx=1, current_stage="enhance")
    assert needed
    assert "encode" in reason


def test_eviction_metrics_and_stage_runs_audit(tmp_env):
    """Verify that stage runs audit table and eviction metrics record queue depths and reasons."""
    from fastapi.testclient import TestClient
    from aries_manager.api import app
    from aries_manager.metrics import (
        WORKER_EVICTIONS_TOTAL,
        EVICTION_QUEUE_DEPTH,
        EVICTION_JOBS_PROCESSED,
    )

    db: Database = tmp_env["db"]

    # 1. Insert a stage run with eviction reason and queue depths
    depths_snapshot = {"enhance": 1, "encode": 3, "denoise": 0, "decode": 0}
    db.record_stage_run(
        stage="enhance",
        card=1,
        pid=99999,
        started_at=time.time() - 60,
        ended_at=time.time(),
        exit_code=0,
        jobs_processed=5,
        exit_reason="card_needed: queued stage 'encode' has depth 3 > 0 active workers on other cards",
        queue_depths=depths_snapshot,
    )

    recent_runs = db.get_recent_stage_runs(limit=10)
    assert len(recent_runs) >= 1
    latest = recent_runs[0]
    assert latest["stage"] == "enhance"
    assert latest["card"] == 1
    assert latest["jobs_processed"] == 5
    assert "card_needed" in latest["exit_reason"]
    assert latest["queue_depths"] == depths_snapshot

    # 2. Test /stage_runs API endpoint
    import aries_manager.api as api_mod
    api_mod.db = db
    client = TestClient(app)
    resp = client.get("/stage_runs?limit=10")
    assert resp.status_code == 200
    runs_from_api = resp.json()
    assert isinstance(runs_from_api, list)
    assert len(runs_from_api) >= 1
    assert runs_from_api[0]["stage"] == "enhance"
    assert runs_from_api[0]["queue_depths"] == depths_snapshot

    # 3. Test metric observations
    WORKER_EVICTIONS_TOTAL.labels(stage="enhance", card="1", reason="card_needed").inc()
    EVICTION_JOBS_PROCESSED.labels(stage="enhance", reason="card_needed").observe(5)
    for s, d in depths_snapshot.items():
        EVICTION_QUEUE_DEPTH.labels(evicted_stage="enhance", target_stage=s, reason="card_needed").observe(d)


def test_enhance_progress_telemetry_db(tmp_env):
    """Enhance progress stores sub_stage and enhance_stats; completion clears the sub-stage."""
    db: Database = tmp_env["db"]
    jid = generate_ulid()
    db.submit_jobs([{"id": jid, "prompt": "a futuristic vehicle", "width": 2048, "height": 2048, "enhance": True}])

    # Initially in enhance_queued
    job = db.get_job(jid)
    assert job["status"] == "enhance_queued"
    assert job["sub_stage"] is None

    # Claim job: it starts in prefill
    db.claim_oldest_queued_job("enhance", card=1, pid=5001)
    assert db.get_job(jid)["sub_stage"] == "prefill"

    # 1. Record live progress: enhance reports sub_stage and stats, not step counts
    stats = {
        "prompt_tokens": 64,
        "generated_tokens": 120,
        "thought_tokens": 80,
        "reply_tokens": 40,
        "prefill_s": 0.45,
        "prefill_tps": 142.2,
        "generation_tps": 18.5,
    }
    db.record_progress(jid, sub_stage="thinking", enhance_stats=stats)

    job_progress = db.get_job(jid)
    assert job_progress["sub_stage"] == "thinking"
    assert job_progress["enhance_stats"] == stats
    assert job_progress["step"] == 0
    assert job_progress["s_per_step"] is None

    # 2. Advance stage success -> transition from enhance to encode
    timings = {
        "enhance_s": 6.8,
        "prefill_tps": 142.2,
        "generation_tps": 18.5,
        "prompt_tokens": 64,
        "thought_tokens": 80,
        "reply_tokens": 40,
    }
    db.advance_stage_success(
        jid,
        stage="enhance",
        timings=timings,
        peak_vram=1000000000,
        enhanced_prompt="A sleek hovercraft gliding through neon-lit streets",
        thinking="Focus on aerodynamics and reflective materials",
    )

    job_completed = db.get_job(jid)
    assert job_completed["status"] == "encode_queued"
    assert job_completed["sub_stage"] is None
    assert job_completed["enhanced_prompt"] == "A sleek hovercraft gliding through neon-lit streets"
    assert job_completed["thinking"] == "Focus on aerodynamics and reflective materials"
    assert job_completed["timings"]["generation_tps"] == 18.5
    assert job_completed["timings"]["prefill_tps"] == 142.2





def test_cancelled_job_is_not_resurrected_by_worker_events(tmp_env):
    """A worker's error/done arriving after a cancel must not requeue or advance the job."""
    db: Database = tmp_env["db"]
    jid = generate_ulid()
    db.submit_jobs([{"id": jid, "prompt": "cancel race", "width": 2048, "height": 2048, "enhance": True}])
    db.claim_oldest_queued_job("enhance", card=1, pid=4242)
    assert db.cancel_job(jid) is not None

    # The worker notices the cancel and reports an error...
    assert db.record_stage_failure(jid, "enhance", "cancelled") is None
    # ...or finishes the stage anyway.
    assert db.advance_stage_success(jid, "enhance", {"enhance_s": 1.0}, peak_vram=0) is None

    job = db.get_job(jid)
    assert job["status"] == "cancelled"
    assert job["attempts"] == 0


def test_progress_fields_reset_between_stages(tmp_env):
    """s/step and ETA from one stage must not leak into the next one."""
    db: Database = tmp_env["db"]
    jid = generate_ulid()
    db.submit_jobs([{"id": jid, "prompt": "progress reset", "width": 2048, "height": 2048}])
    db.advance_stage_success(jid, "encode", {"encode_s": 1.0}, peak_vram=0)
    db.claim_oldest_queued_job("denoise", card=0, pid=4343)
    db.record_progress(jid, step=10, steps=40, s_per_step=6.4)
    assert db.get_job(jid)["eta_s"] == 192.0

    db.advance_stage_success(jid, "denoise", {"denoise_s": 256.0}, peak_vram=0)
    job = db.get_job(jid)
    assert job["status"] == "decode_queued"
    assert (job["step"], job["s_per_step"], job["eta_s"]) == (0, None, None)

    # A failed retry also starts clean.
    db.claim_oldest_queued_job("decode", card=1, pid=4444)
    db.record_progress(jid, step=1, steps=1, s_per_step=5.0)
    db.record_stage_failure(jid, "decode", "boom")
    job = db.get_job(jid)
    assert job["status"] == "decode_queued"
    assert (job["step"], job["s_per_step"], job["eta_s"]) == (0, None, None)


class _DeadProcess:
    """A worker process that exits immediately with `code`, without any output."""

    def __init__(self, code: int, pid: int = 7777):
        self.code, self.pid = code, pid
        self.stdout = self.stderr = None

    async def wait(self) -> int:
        return self.code


@pytest.mark.asyncio
async def test_worker_startup_failure_backs_off(tmp_env):
    """A worker dying before `hello` must not be respawned every second."""
    db: Database = tmp_env["db"]
    scheduler = CardScheduler(db, num_cards=2)
    scheduler.decode_ram_floor = 0
    db.submit_jobs([{"id": generate_ulid(), "prompt": "p", "width": 2048, "height": 2048}])
    assert scheduler.pick_next_stage(0) == "encode"

    scheduler.card_states[0] = "running"
    await scheduler._monitor_worker_process("encode", 0, _DeadProcess(1))
    assert scheduler.card_states[0] == "idle"
    assert scheduler.card_startup_failures[0] == 1
    assert scheduler.pick_next_stage(0) is None       # backing off
    assert scheduler.pick_next_stage(1) == "encode"   # other card unaffected

    first_retry = scheduler.card_retry_at[0]
    scheduler.card_states[0] = "running"
    await scheduler._monitor_worker_process("encode", 0, _DeadProcess(1))
    assert scheduler.card_retry_at[0] > first_retry   # delay doubles

    # A worker that connected (sent hello) and then exited cleanly resets the backoff.
    scheduler.card_states[0] = "running"
    scheduler.card_load_s[0] = 3.2
    await scheduler._monitor_worker_process("encode", 0, _DeadProcess(0))
    assert scheduler.card_startup_failures[0] == 0
    assert scheduler.pick_next_stage(0) == "encode"
