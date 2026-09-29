"""Per-field data-quality audit over a closed seq interval.

Operators pick a device, a closed seq range and a top-level payload field;
the report counts the committed records in the interval, how many of them
carry a valid numeric value for the field versus a missing/invalid one,
which contiguous seq ranges are still uncommitted, and the min/max/avg of
the valid values (all three null when there are none).

Only `committed` rows count: `buffered` records (still waiting on a gap
ahead), `gap_filled` backfills (断档补录) and idempotent duplicate
resubmissions are excluded — duplicates never create a second row at all.
A value is valid only when it is a real, finite number: booleans, strings
and NaN/Infinity are invalid (the same rule the alert engine applies).

The caller runs this inside its BEGIN IMMEDIATE transaction, right after
the lazy 断档 sweep, so the counts, the statistics and the uncommitted
ranges all come from one consistent database snapshot even while batches
arrive concurrently.
"""
from __future__ import annotations

import json
import math

from .alerts import _as_number
from .window import _to_ranges


def field_quality(conn, device_id: str, seq_from: int, seq_to: int, field: str) -> dict:
    """Audit the top-level `field` over committed records in [seq_from, seq_to]."""
    rows = conn.execute(
        "SELECT seq, payload FROM records "
        "WHERE device_id = ? AND status = 'committed' AND seq BETWEEN ? AND ? "
        "ORDER BY seq",
        (device_id, seq_from, seq_to),
    ).fetchall()

    values: list[float] = []
    invalid = 0
    committed_seqs: list[int] = []
    for row in rows:
        committed_seqs.append(row["seq"])
        try:
            payload = json.loads(row["payload"])
        except ValueError:  # defensive: payloads are JSON objects by contract
            payload = None
        raw = payload.get(field) if isinstance(payload, dict) else None
        number = _as_number(raw)
        if number is None:
            invalid += 1
        else:
            values.append(number)

    committed = set(committed_seqs)
    uncommitted = [s for s in range(seq_from, seq_to + 1) if s not in committed]

    return {
        "device_id": device_id,
        "field": field,
        "seq_from": seq_from,
        "seq_to": seq_to,
        "committed_count": len(committed_seqs),
        "valid_count": len(values),
        "invalid_count": invalid,
        "uncommitted_ranges": _to_ranges(uncommitted),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "avg": (math.fsum(values) / len(values)) if values else None,
    }
