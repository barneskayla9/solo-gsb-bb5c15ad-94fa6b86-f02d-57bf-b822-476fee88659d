"""Field data-quality audit tests: GET /api/v1/devices/{id}/field-quality."""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "test.db"),
        window_size=100,
        gap_timeout_seconds=0.25,
        sweep_enabled=False,  # deterministic: gaps are declared lazily
    )
    with TestClient(create_app(settings)) as c:
        yield c


QPATH = "/api/v1/devices/dev-a/field-quality"


def audit(client, **params):
    resp = client.get(QPATH, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def submit(client, records, device="dev-a"):
    resp = client.post(
        "/api/v1/batches",
        json={"device_id": device, "records": records},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def rec(seq, **payload):
    return {"seq": seq, "payload": payload}


# --------------------------------------------------------------------- #
# counts & statistics
# --------------------------------------------------------------------- #

def test_counts_and_stats_over_valid_values(client):
    submit(client, [
        rec(1, temp=10.0), rec(2, temp=20), rec(3, temp=30.0),
    ])
    body = audit(client, field="temp", seq_from=1, seq_to=3)
    assert body["committed_count"] == 3
    assert body["valid_count"] == 3
    assert body["invalid_count"] == 0
    assert body["missing_ranges"] == []
    assert body["min_value"] == 10.0
    assert body["max_value"] == 30.0
    assert body["avg_value"] == 20.0
    # request echo
    assert body["device_id"] == "dev-a"
    assert body["field"] == "temp"
    assert body["seq_from"] == 1
    assert body["seq_to"] == 3


def test_invalid_values_are_excluded_from_stats(client):
    # bool / str / NaN / Infinity / missing key are all invalid.
    # NaN/Infinity are non-standard JSON tokens, so post the raw body
    # (Python's json emits them and pydantic accepts them on parse).
    raw = {
        "device_id": "dev-a",
        "records": [
            {"seq": 1, "payload": {"temp": True}},       # bool
            {"seq": 2, "payload": {"temp": "hot"}},      # string
            {"seq": 3, "payload": {"temp": float("nan")}},
            {"seq": 4, "payload": {"temp": float("inf")}},
            {"seq": 5, "payload": {"other": 1.0}},       # field missing
            {"seq": 6, "payload": {"temp": None}},       # null
            {"seq": 7, "payload": {"temp": [1, 2]}},     # list
            {"seq": 8, "payload": {"temp": {"x": 1}}},   # nested object
            {"seq": 9, "payload": {"temp": 5.0}},        # the only finite valid value
            {"seq": 10, "payload": {"temp": 0, "flag": False}},  # integer is valid
        ],
    }
    resp = client.post(
        "/api/v1/batches",
        content=json.dumps(raw),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200, resp.text
    body = audit(client, field="temp", seq_from=1, seq_to=10)
    assert body["committed_count"] == 10
    assert body["valid_count"] == 2
    assert body["invalid_count"] == 8
    assert body["min_value"] == 0.0
    assert body["max_value"] == 5.0
    assert body["avg_value"] == 2.5


def test_nested_field_is_not_considered(client):
    # Only top-level keys count; "temp" inside a nested object is missing.
    submit(client, [rec(1, data={"temp": 1.0}), rec(2, temp=2.0)])
    body = audit(client, field="temp", seq_from=1, seq_to=2)
    assert body["valid_count"] == 1
    assert body["invalid_count"] == 1
    assert body["min_value"] == body["max_value"] == 2.0


def test_no_valid_values_means_null_stats(client):
    submit(client, [rec(1, temp="x"), rec(2, temp=True)])
    body = audit(client, field="temp", seq_from=1, seq_to=2)
    assert body["valid_count"] == 0
    assert body["invalid_count"] == 2
    assert body["min_value"] is None
    assert body["max_value"] is None
    assert body["avg_value"] is None


def test_empty_interval_returns_zero_counts_null_stats(client):
    submit(client, [rec(10, temp=1.0)])
    body = audit(client, field="temp", seq_from=1, seq_to=9)
    assert body["committed_count"] == 0
    assert body["valid_count"] == 0
    assert body["invalid_count"] == 0
    assert body["missing_ranges"] == [[1, 9]]
    assert body["min_value"] is None
    assert body["max_value"] is None
    assert body["avg_value"] is None


def test_interval_is_clipped_to_requested_range(client):
    submit(client, [rec(s, temp=float(s)) for s in range(0, 8)])
    body = audit(client, field="temp", seq_from=2, seq_to=5)
    assert body["committed_count"] == 4
    assert body["missing_ranges"] == []
    assert body["min_value"] == 2.0
    assert body["max_value"] == 5.0
    assert body["avg_value"] == 3.5


# --------------------------------------------------------------------- #
# uncommitted ranges: buffered / gap_filled / declared gaps merge
# --------------------------------------------------------------------- #

def test_missing_ranges_merge_hole_and_buffered(client):
    # 1 committed; 3,4,6 buffered behind the never-filled holes 2 and 5.
    # Only committed points break a missing range, so 2..6 is one merged gap.
    submit(client, [rec(1, temp=1.0), rec(3, temp=3.0), rec(4, temp=4.0),
                    rec(6, temp=6.0)])
    body = audit(client, field="temp", seq_from=1, seq_to=6)
    assert body["committed_count"] == 1
    assert body["valid_count"] == 1
    assert body["missing_ranges"] == [[2, 6]]


def test_adjacent_uncommitted_sequences_merge_into_one_range(client):
    # anchor at 6: 6 commits, 8/9 buffer behind the hole at 7; 1..5 were
    # never seen (below the anchor, no declared 断档) — holes on both sides.
    submit(client, [rec(6, temp=6.0), rec(8, temp=8.0), rec(9, temp=9.0)])
    body = audit(client, field="temp", seq_from=1, seq_to=9)
    assert body["committed_count"] == 1
    assert body["missing_ranges"] == [[1, 5], [7, 9]]


def test_declared_gap_and_gap_filled_count_as_uncommitted(client):
    submit(client, [rec(1, temp=1.0), rec(2, temp=2.0), rec(4, temp=4.0),
                    rec(5, temp=5.0)])
    time.sleep(0.3)
    # lazy gap declaration on read: 断档 at seq 3
    gaps = client.get("/api/v1/devices/dev-a/gaps").json()["gaps"]
    assert [(g["start_seq"], g["end_seq"], g["status"]) for g in gaps] == [(3, 3, "open")]
    # seq 4,5 drained to committed once the gap was declared
    body = audit(client, field="temp", seq_from=1, seq_to=6)
    assert body["committed_count"] == 4
    assert body["missing_ranges"] == [[3, 3], [6, 6]]

    # backfill the gap: the replay lands as gap_filled and must still count
    # as an uncommitted hole for the audit (only 'committed' counts).
    body = submit(client, [rec(3, temp=3.0)])
    assert body["results"][0]["status"] == "gap_filled"
    body = audit(client, field="temp", seq_from=1, seq_to=6)
    assert body["committed_count"] == 4  # gap_filled not counted
    assert body["valid_count"] == 4
    assert body["missing_ranges"] == [[3, 3], [6, 6]]


def test_missing_ranges_when_no_records_at_all(client):
    submit(client, [rec(100, temp=1.0)])
    body = audit(client, field="temp", seq_from=0, seq_to=0)
    assert body["missing_ranges"] == [[0, 0]]
    body = audit(client, field="temp", seq_from=200, seq_to=202)
    assert body["missing_ranges"] == [[200, 202]]


def test_range_starts_and_ends_inside_holes(client):
    submit(client, [rec(5, temp=5.0)])
    # partial ranges before and after the committed point
    assert audit(client, field="temp", seq_from=1, seq_to=4)["missing_ranges"] == [[1, 4]]
    assert audit(client, field="temp", seq_from=6, seq_to=9)["missing_ranges"] == [[6, 9]]
    assert audit(client, field="temp", seq_from=1, seq_to=9)["missing_ranges"] == [
        [1, 4], [6, 9],
    ]


# --------------------------------------------------------------------- #
# duplicates never count
# --------------------------------------------------------------------- #

def test_duplicate_delivery_does_not_change_audit(client):
    submit(client, [rec(1, temp=1.0), rec(2, temp=2.0)])
    first = audit(client, field="temp", seq_from=1, seq_to=2)
    # redeliver the same seqs (batch retry): all duplicates, same counts
    resubmit = submit(client, [rec(1, temp=999.0), rec(2, temp="junk")])
    assert {r["status"] for r in resubmit["results"]} == {"duplicate"}
    second = audit(client, field="temp", seq_from=1, seq_to=2)
    assert first == second
    assert second["committed_count"] == 2
    assert second["valid_count"] == 2
    assert second["avg_value"] == 1.5


# --------------------------------------------------------------------- #
# validation & error handling
# --------------------------------------------------------------------- #

def test_unknown_device_404(client):
    resp = client.get("/api/v1/devices/nope/field-quality",
                      params={"field": "temp", "seq_from": 0, "seq_to": 10})
    assert resp.status_code == 404
    assert "nope" in resp.json()["detail"]


def test_blank_field_rejected(client):
    submit(client, [rec(1, temp=1.0)])
    for blank in ("", "   ", "\t"):
        resp = client.get(QPATH, params={"field": blank, "seq_from": 1, "seq_to": 1})
        assert resp.status_code in (400, 422), blank
    # surrounding whitespace on a real name is trimmed and works
    body = audit(client, field="  temp  ", seq_from=1, seq_to=1)
    assert body["field"] == "temp"
    assert body["valid_count"] == 1


def test_negative_seq_rejected(client):
    submit(client, [rec(0, temp=1.0)])
    assert client.get(QPATH, params={"field": "temp", "seq_from": -1, "seq_to": 1}).status_code == 422
    assert client.get(QPATH, params={"field": "temp", "seq_from": 0, "seq_to": -1}).status_code == 422


def test_inverted_range_400(client):
    submit(client, [rec(1, temp=1.0)])
    resp = client.get(QPATH, params={"field": "temp", "seq_from": 5, "seq_to": 2})
    assert resp.status_code == 400


def test_range_length_limit_400(client):
    submit(client, [rec(0, temp=1.0)])
    # exactly 10001 seqs -> rejected
    resp = client.get(QPATH, params={"field": "temp", "seq_from": 0, "seq_to": 10_000})
    assert resp.status_code == 400
    # exactly 10000 seqs -> accepted
    ok = client.get(QPATH, params={"field": "temp", "seq_from": 0, "seq_to": 9_999})
    assert ok.status_code == 200


def test_missing_required_params_422(client):
    submit(client, [rec(1, temp=1.0)])
    assert client.get(QPATH, params={"seq_from": 1, "seq_to": 1}).status_code == 422
    assert client.get(QPATH, params={"field": "temp", "seq_to": 1}).status_code == 422


# --------------------------------------------------------------------- #
# snapshot consistency
# --------------------------------------------------------------------- #

def test_audit_reads_one_snapshot_under_concurrent_ingest(tmp_path):
    """Audit rows must come from a single snapshot even while writers commit.

    Spawn a real uvicorn server: writer threads keep committing new seqs while
    reader threads hammer the audit over a moving window. Every response must
    be internally self-consistent: committed = valid + invalid and the missing
    ranges must exactly cover the non-committed seqs of the reported window.
    """
    import socket
    import uvicorn
    import httpx
    from concurrent.futures import ThreadPoolExecutor

    settings = Settings(
        db_path=str(tmp_path / "test.db"),
        window_size=1_000_000,
        gap_timeout_seconds=3600,
        sweep_enabled=False,
    )

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(create_app(settings), host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)
    thread = __import__("threading").Thread(target=server.run, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}/api/v1"
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                httpx.get(f"{base.replace('/api/v1', '')}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.05)

        stop = False

        def writer(worker_id):
            n = 0
            with httpx.Client(timeout=10) as http:
                while not stop:
                    # each writer owns a residue class mod 8 for unique seqs
                    seq = worker_id + 8 * n
                    n += 1
                    http.post(f"{base}/batches", json={
                        "device_id": "dev-a",
                        "records": [{"seq": seq,
                                     "payload": {"v": float(seq) if seq % 7 else "bad"}}],
                    })
                    time.sleep(0.002)

        def reader():
            seen = 0
            with httpx.Client(timeout=10) as http:
                while not stop and seen < 40:
                    resp = http.get(f"{base}/devices/dev-a/field-quality",
                                    params={"field": "v", "seq_from": 0, "seq_to": 500})
                    assert resp.status_code == 200, resp.text
                    body = resp.json()
                    assert body["committed_count"] == body["valid_count"] + body["invalid_count"]
                    # missing ranges partition exactly the seqs without commit
                    covered = sum(b - a + 1 for a, b in body["missing_ranges"])
                    assert covered + body["committed_count"] == 501
                    # ranges are sorted, disjoint, closed and non-adjacent
                    prev_end = -2
                    for a, b in body["missing_ranges"]:
                        assert 0 <= a <= b <= 500
                        assert a > prev_end + 1
                        prev_end = b
                    seen += 1

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(writer, i) for i in range(8)] + [pool.submit(reader) for _ in range(2)]
            time.sleep(6)
            stop = True
            for f in futures:
                f.result(timeout=30)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
