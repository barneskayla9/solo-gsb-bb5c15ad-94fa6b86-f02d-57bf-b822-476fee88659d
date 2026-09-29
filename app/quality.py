"""Per-device telemetry field data-quality audit (字段数据质量核查).

Given a device, a closed seq interval [seq_from, seq_to] and a top-level
payload field name, the audit reports how the field's data looks *among
committed records*:

* committed_count  - committed records in the interval (the audit universe);
* valid_count      - committed records whose top-level field is a finite
                     number (booleans/strings/NaN/Infinity are invalid);
* invalid_count    - committed records with the field missing or carrying an
                     invalid value (committed_count - valid_count);
* missing_ranges   - contiguous seq intervals inside [seq_from, seq_to] that
                     have no committed record, merging across buffered
                     records, gap_filled backfills and declared gaps;
* min/max/avg      - statistics over the valid values, each null when there
                     is no valid value.

Only rows with status 'committed' participate: buffered records, 断档补录
(gap_filled) rows and duplicate deliveries are not counted (duplicates never
create a second row anyway). ``buffered`` and ``gap_filled`` sequences count
as uncommitted holes for missing_ranges.

All SELECTs for one audit run inside a single deferred read transaction
(app.db.read_txn), so under concurrent ingest the counts, statistics and gap
ranges always come from one database snapshot.
"""
from __future__ import annotations

import json
from typing import Optional

from .alerts import as_number


class DeviceNotFound(Exception):
    """The audited device does not exist."""


class QualityAuditor:
    def __init__(self, conn):
        self.conn = conn

    def audit(self, device_id: str, field: str, seq_from: int, seq_to: int) -> dict:
        """Run the field-quality audit; call inside a read_txn snapshot."""
        dev = self.conn.execute(
            "SELECT 1 FROM devices WHERE device_id = ?", (device_id,)
        ).fetchone()
        if dev is None:
            raise DeviceNotFound(device_id)

        committed = self.conn.execute(
            "SELECT seq, payload FROM records "
            "WHERE device_id = ? AND status = 'committed' AND seq BETWEEN ? AND ? "
            "ORDER BY seq",
            (device_id, seq_from, seq_to),
        ).fetchall()

        committed_count = len(committed)
        valid_count = 0
        vmin: Optional[float] = None
        vmax: Optional[float] = None
        # Running mean update: stays bounded by the value magnitudes, so it
        # cannot overflow to Infinity the way a raw sum/divide could when
        # payloads carry extreme (still finite) doubles.
        vavg: float = 0.0

        for row in committed:
            try:
                payload = json.loads(row["payload"])
            except (ValueError, TypeError):
                payload = None
            if not isinstance(payload, dict):
                continue
            number = as_number(payload.get(field))
            if number is None:  # missing key, bool, str or non-finite number
                continue
            valid_count += 1
            vavg += (number - vavg) / valid_count
            if vmin is None or number < vmin:
                vmin = number
            if vmax is None or number > vmax:
                vmax = number

        # Committed seqs (sorted, unique via PK) describe the filled points;
        # every other seq in the closed interval is an uncommitted hole.
        # Walk the committed seqs once, emitting the holes between them and
        # the interval edges; adjacent holes merge into a single range.
        missing_ranges: list[list[int]] = []
        cursor = seq_from

        def add_hole(start: int, end: int) -> None:
            if start <= end:
                missing_ranges.append([start, end])

        for row in committed:
            s = row["seq"]
            if s > cursor:
                add_hole(cursor, s - 1)
            cursor = s + 1
        add_hole(cursor, seq_to)

        return {
            "device_id": device_id,
            "field": field,
            "seq_from": seq_from,
            "seq_to": seq_to,
            "committed_count": committed_count,
            "valid_count": valid_count,
            "invalid_count": committed_count - valid_count,
            "missing_ranges": missing_ranges,
            "min_value": vmin,
            "max_value": vmax,
            "avg_value": vavg if valid_count else None,
        }
