"""Field-quality audit tests for GET /api/v1/devices/{device_id}/field-quality."""
from __future__ import annotations

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "test.db"),
        window_size=50,
        gap_timeout_seconds=0.25,
        sweep_enabled=False,  # deterministic: gaps are declared lazily on access
    )
    with TestClient(create_app(settings)) as c:
        yield c


def submit(client, device, records):
    """records: list of (seq, payload) pairs."""
    resp = client.post(
        "/api/v1/batches",
        json={"device_id": device,
              "records": [{"seq": s, "payload": p} for s, p in records]},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def quality(client, device, **params):
    resp = client.get(f"/api/v1/devices/{device}/field-quality", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


# --------------------------------------------------------------------- #
# counts, validity rules and statistics
# --------------------------------------------------------------------- #

def test_basic_counts_and_stats(client):
    submit(client, "dev-a", [(s, {"temp": float(s)}) for s in range(1, 6)])
    body = quality(client, "dev-a", seq_from=1, seq_to=5, field="temp")
    assert body == {
        "device_id": "dev-a",
        "field": "temp",
        "seq_from": 1,
        "seq_to": 5,
        "committed_count": 5,
        "valid_count": 5,
        "invalid_count": 0,
        "uncommitted_ranges": [],
        "min": 1.0,
        "max": 5.0,
        "avg": 3.0,
    }


def test_missing_field_and_non_numeric_values_are_invalid(client):
    submit(client, "dev-a", [
        (1, {"temp": 21.5}),          # valid
        (2, {}),                      # field missing
        (3, {"temp": True}),          # bool is not a measurement
        (4, {"temp": "22.0"}),        # string is not a measurement
        (5, {"temp": None}),          # null
        (6, {"temp": [1, 2]}),        # array
        (7, {"temp": 30}),            # valid int
    ])
    body = quality(client, "dev-a", seq_from=1, seq_to=7, field="temp")
    assert body["committed_count"] == 7
    assert body["valid_count"] == 2
    assert body["invalid_count"] == 5
    assert body["min"] == 21.5
    assert body["max"] == 30.0
    assert body["avg"] == pytest.approx((21.5 + 30.0) / 2)


def test_non_finite_values_are_invalid(client):
    # NaN/Infinity are not strict JSON, but Python's json round-trips them,
    # so a non-strict client can get them into a payload; post the raw body
    # because httpx's json= serializer is strict.
    import json as jsonlib

    body = jsonlib.dumps({
        "device_id": "dev-a",
        "records": [
            {"seq": 1, "payload": {"temp": float("nan")}},
            {"seq": 2, "payload": {"temp": float("inf")}},
            {"seq": 3, "payload": {"temp": float("-inf")}},
            {"seq": 4, "payload": {"temp": 2.0}},
        ],
    })
    resp = client.post("/api/v1/batches", content=body,
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 200, resp.text
    body = quality(client, "dev-a", seq_from=1, seq_to=4, field="temp")
    assert body["valid_count"] == 1
    assert body["invalid_count"] == 3
    assert (body["min"], body["max"], body["avg"]) == (2.0, 2.0, 2.0)


def test_only_top_level_field_is_read(client):
    submit(client, "dev-a", [
        (1, {"outer": {"temp": 99}}),  # nested value is invisible
        (2, {"temp": {"nested": 1}}),  # object value is not numeric
        (3, {"temp": 5}),
    ])
    body = quality(client, "dev-a", seq_from=1, seq_to=3, field="temp")
    assert body["valid_count"] == 1
    assert body["invalid_count"] == 2
    assert body["min"] == body["max"] == body["avg"] == 5.0


def test_no_valid_values_yields_null_stats(client):
    submit(client, "dev-a", [(1, {"other": 1}), (2, {"temp": "hot"})])
    body = quality(client, "dev-a", seq_from=1, seq_to=2, field="temp")
    assert body["committed_count"] == 2
    assert body["valid_count"] == 0
    assert body["invalid_count"] == 2
    assert body["min"] is None and body["max"] is None and body["avg"] is None


def test_empty_interval_has_zero_counts_and_null_stats(client):
    submit(client, "dev-a", [(10, {"temp": 1})])
    body = quality(client, "dev-a", seq_from=1, seq_to=5, field="temp")
    assert body["committed_count"] == 0
    assert body["valid_count"] == 0
    assert body["invalid_count"] == 0
    assert body["uncommitted_ranges"] == [[1, 5]]
    assert body["min"] is None and body["max"] is None and body["avg"] is None


# --------------------------------------------------------------------- #
# uncommitted ranges: buffered / gap_filled / never-received
# --------------------------------------------------------------------- #

def test_uncommitted_ranges_merge_adjacent_gaps(client):
    # committed after the sweeps: 1,2,4,7,8,9 -> holes at 3, 5-6, 10
    submit(client, "dev-a", [(s, {"temp": 1}) for s in (1, 2, 4, 7, 8, 9)])
    time.sleep(0.3)  # let both pending gaps expire; the audit sweeps lazily
    body = quality(client, "dev-a", seq_from=1, seq_to=10, field="temp")
    assert body["committed_count"] == 6
    assert body["uncommitted_ranges"] == [[3, 3], [5, 6], [10, 10]]


def test_buffered_records_are_uncommitted(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "test.db"), window_size=50,
        gap_timeout_seconds=600, sweep_enabled=False,
    )
    with TestClient(create_app(settings)) as client:
        submit(client, "dev-a", [(s, {"temp": s}) for s in (1, 2, 3, 5, 6)])
        body = quality(client, "dev-a", seq_from=1, seq_to=6, field="temp")
        # 5 and 6 are buffered behind the pending gap at 4: not committed.
        assert body["committed_count"] == 3
        assert body["valid_count"] == 3
        assert body["uncommitted_ranges"] == [[4, 6]]


def test_gap_filled_records_are_uncommitted(client):
    submit(client, "dev-a", [(s, {"temp": s}) for s in (1, 2, 3, 5, 6)])
    time.sleep(0.3)  # let the pending gap at 4 expire
    gaps = client.get("/api/v1/devices/dev-a/gaps").json()["gaps"]  # lazy 断档
    assert [(g["start_seq"], g["end_seq"]) for g in gaps] == [(4, 4)]

    resp = client.post("/api/v1/devices/dev-a/gaps/1/replay")
    assert resp.status_code == 200
    body = submit(client, "dev-a", [(4, {"temp": 4})])  # 断档补录
    assert body["results"][0]["status"] == "gap_filled"

    body = quality(client, "dev-a", seq_from=1, seq_to=6, field="temp")
    # The gap declaration flushed 5,6 to committed; the backfilled 4 is
    # gap_filled and stays uncommitted for this audit.
    assert body["committed_count"] == 5
    assert body["valid_count"] == 5
    assert body["uncommitted_ranges"] == [[4, 4]]


def test_duplicate_resubmission_does_not_change_counts(client):
    records = [(s, {"temp": s}) for s in (1, 2, 3)]
    submit(client, "dev-a", records)
    before = quality(client, "dev-a", seq_from=1, seq_to=3, field="temp")
    body = submit(client, "dev-a", records)  # full retry of the batch
    assert all(r["status"] == "duplicate" for r in body["results"])
    after = quality(client, "dev-a", seq_from=1, seq_to=3, field="temp")
    assert after == before
    assert after["committed_count"] == 3


def test_lazy_gap_declaration_is_visible_to_the_audit(client):
    submit(client, "dev-a", [(s, {"temp": s}) for s in (1, 2, 4, 5)])
    body = quality(client, "dev-a", seq_from=1, seq_to=5, field="temp")
    assert body["committed_count"] == 2  # 4,5 still buffered
    assert body["uncommitted_ranges"] == [[3, 5]]

    time.sleep(0.3)  # gap at 3 expires; the audit itself triggers the sweep
    body = quality(client, "dev-a", seq_from=1, seq_to=5, field="temp")
    assert body["committed_count"] == 4  # 4,5 flushed to committed
    assert body["uncommitted_ranges"] == [[3, 3]]


# --------------------------------------------------------------------- #
# request validation
# --------------------------------------------------------------------- #

def test_unknown_device_404(client):
    resp = client.get("/api/v1/devices/nope/field-quality",
                      params={"seq_from": 1, "seq_to": 5, "field": "temp"})
    assert resp.status_code == 404
    assert "nope" in resp.json()["detail"]


def test_inverted_range_400(client):
    submit(client, "dev-a", [(1, {"temp": 1})])
    resp = client.get("/api/v1/devices/dev-a/field-quality",
                      params={"seq_from": 5, "seq_to": 2, "field": "temp"})
    assert resp.status_code == 400


def test_negative_seq_422(client):
    submit(client, "dev-a", [(1, {"temp": 1})])
    resp = client.get("/api/v1/devices/dev-a/field-quality",
                      params={"seq_from": -1, "seq_to": 2, "field": "temp"})
    assert resp.status_code == 422


def test_span_limit(client):
    submit(client, "dev-a", [(1, {"temp": 1})])
    # exactly 10_000 sequences is allowed
    resp = client.get("/api/v1/devices/dev-a/field-quality",
                      params={"seq_from": 0, "seq_to": 9999, "field": "temp"})
    assert resp.status_code == 200
    # 10_001 is rejected
    resp = client.get("/api/v1/devices/dev-a/field-quality",
                      params={"seq_from": 0, "seq_to": 10_000, "field": "temp"})
    assert resp.status_code == 400


def test_empty_or_blank_field_rejected(client):
    submit(client, "dev-a", [(1, {"temp": 1})])
    for params in (
        {"seq_from": 1, "seq_to": 2},                    # field missing
        {"seq_from": 1, "seq_to": 2, "field": ""},       # empty
        {"seq_from": 1, "seq_to": 2, "field": "   "},    # blank
    ):
        resp = client.get("/api/v1/devices/dev-a/field-quality", params=params)
        assert resp.status_code in (400, 422), params


# --------------------------------------------------------------------- #
# snapshot consistency under concurrent ingestion (real uvicorn server)
# --------------------------------------------------------------------- #

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_audit_is_consistent_while_batches_arrive(tmp_path):
    """Concurrent writers commit 1..800 while a reader audits [1, 800]:
    every response must be internally consistent (one snapshot)."""
    import uvicorn

    settings = Settings(
        db_path=str(tmp_path / "conc_quality.db"),
        window_size=10_000, gap_timeout_seconds=600, sweep_enabled=False,
    )
    port = _free_port()
    config = uvicorn.Config(create_app(settings), host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                httpx.get(f"{base}/health", timeout=0.2)
                break
            except Exception:
                time.sleep(0.05)
        else:
            pytest.fail("server did not start")

        span = 800
        stop = threading.Event()
        inconsistencies = []

        with httpx.Client(base_url=base, timeout=10) as c:
            # anchor the device (seq 0 is outside the audited range)
            r = c.post("/api/v1/batches", json={
                "device_id": "dev-a", "records": [{"seq": 0, "payload": {}}],
            })
            assert r.status_code == 200

        def writer(shard):
            with httpx.Client(base_url=base, timeout=10) as c:
                seqs = range(shard, span + 1, 8)
                records = [
                    {"seq": s, "payload": {} if s % 10 == 0 else {"temp": s}}
                    for s in seqs
                ]
                r = c.post("/api/v1/batches",
                           json={"device_id": "dev-a", "records": records})
                assert r.status_code == 200

        def reader():
            with httpx.Client(base_url=base, timeout=10) as c:
                while not stop.is_set():
                    r = c.get("/api/v1/devices/dev-a/field-quality",
                              params={"seq_from": 1, "seq_to": span, "field": "temp"})
                    assert r.status_code == 200
                    b = r.json()
                    holes = sum(e - s + 1 for s, e in b["uncommitted_ranges"])
                    ok = (
                        b["committed_count"] == b["valid_count"] + b["invalid_count"]
                        and b["committed_count"] + holes == span
                        and (b["valid_count"] == 0
                             or b["min"] <= b["avg"] <= b["max"])
                    )
                    if not ok:
                        inconsistencies.append(b)

        readers = [threading.Thread(target=reader) for _ in range(2)]
        for t in readers:
            t.start()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(writer, range(1, 9)))
        stop.set()
        for t in readers:
            t.join()
        assert inconsistencies == []

        with httpx.Client(base_url=base, timeout=10) as c:
            b = c.get("/api/v1/devices/dev-a/field-quality",
                      params={"seq_from": 1, "seq_to": span, "field": "temp"}).json()
        assert b["committed_count"] == span
        assert b["uncommitted_ranges"] == []
        assert b["invalid_count"] == 80          # every 10th seq has no field
        assert b["valid_count"] == span - 80
        assert b["min"] == 1.0 and b["max"] == 799.0
        assert b["avg"] == pytest.approx(400.0)
    finally:
        server.should_exit = True
        thread.join(timeout=5)
