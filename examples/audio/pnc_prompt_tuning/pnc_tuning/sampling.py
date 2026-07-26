# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic, group-aware subset selection."""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pnc_tuning.config import InputFields, SamplingConfig

_DURATION_BUCKETS = ("lt300", "300_599", "600_899", "900_1799", "ge1800", "unknown")
_FIVE_MINUTES = 300
_TEN_MINUTES = 600
_FIFTEEN_MINUTES = 900
_THIRTY_MINUTES = 1800


def duration_bucket(value: object) -> str:
    """Map a duration to the pilot buckets in the design document."""

    try:
        duration = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if duration < _FIVE_MINUTES:
        return "lt300"
    if duration < _TEN_MINUTES:
        return "300_599"
    if duration < _FIFTEEN_MINUTES:
        return "600_899"
    if duration < _THIRTY_MINUTES:
        return "900_1799"
    return "ge1800"


def _hash_int(*parts: str) -> int:
    payload = "\x1f".join(parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _split_for_group(group_id: str, language: str, config: SamplingConfig) -> str:
    quota_map = asdict(config.quotas)
    names = [name for name, quota in quota_map.items() if quota > 0]
    weights = [quota_map[name] for name in names]
    total = sum(weights)
    point = _hash_int(config.seed, language, group_id, "split") % total
    cumulative = 0
    for name, weight in zip(names, weights):  # noqa: B905 - Python 3.8 smoke-test compatibility
        cumulative += weight
        if point < cumulative:
            return name
    return names[-1]


def _normalized_row(row: dict[str, Any], fields: InputFields, split: str) -> dict[str, Any]:
    row_id = str(row.get(fields.id, ""))
    language = str(row.get(fields.language, "")).lower()
    text = str(row.get(fields.text, "") or "")
    group_id = str(row.get(fields.group, "") or row_id)
    normalized = {
        "id": row_id,
        "group_id": group_id,
        "language": language,
        "text": text,
        "split": split,
        "duration": row.get(fields.duration),
        "duration_bucket": duration_bucket(row.get(fields.duration)),
    }
    reference = row.get(fields.reference)
    if reference not in (None, ""):
        normalized["reference"] = str(reference)
    complete = row.get(fields.complete)
    if isinstance(complete, bool):
        normalized["complete"] = complete
    return normalized


def select_subset(  # noqa: C901, PLR0912, PLR0915
    rows: Iterable[dict[str, Any]],
    *,
    fields: InputFields,
    config: SamplingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select deterministic per-language/split rows while avoiding group leakage."""

    quotas = asdict(config.quotas)
    max_quota = max(quotas.values(), default=0)
    heap_limit = max(10, max_quota * 3)
    heaps: dict[tuple[str, str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    counters: Counter[str] = Counter()

    for row in rows:
        counters["rows_seen"] += 1
        language = str(row.get(fields.language, "")).strip().lower()
        if language not in config.languages:
            counters["skipped_language"] += 1
            continue
        text = str(row.get(fields.text, "") or "")
        if config.require_nonempty_text and not text.strip():
            counters["skipped_empty_text"] += 1
            continue
        row_id = str(row.get(fields.id, "") or "")
        if not row_id:
            counters["skipped_missing_id"] += 1
            continue
        group_id = str(row.get(fields.group, "") or row_id)
        split = _split_for_group(group_id, language, config)
        normalized = _normalized_row(row, fields, split)
        bucket = normalized["duration_bucket"]
        priority = _hash_int(config.seed, language, split, group_id, row_id)
        key = (language, split, bucket)
        item = (-priority, row_id, normalized)
        heap = heaps[key]
        if len(heap) < heap_limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
        counters["eligible_rows"] += 1

    by_split: dict[tuple[str, str], dict[str, list[tuple[int, dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (language, split, bucket), heap in heaps.items():
        by_split[(language, split)][bucket] = sorted(
            [(-negative_priority, row) for negative_priority, _, row in heap],
            key=lambda item: item[0],
        )

    selected: list[dict[str, Any]] = []
    selected_groups: set[str] = set()
    selected_counts: Counter[str] = Counter()
    for language in config.languages:
        for split, quota in quotas.items():
            if quota <= 0:
                continue
            bucket_rows = by_split.get((language, split), {})
            offsets = dict.fromkeys(_DURATION_BUCKETS, 0)
            while selected_counts[f"{language}:{split}"] < quota:
                progressed = False
                for bucket in _DURATION_BUCKETS:
                    values = bucket_rows.get(bucket, [])
                    while offsets[bucket] < len(values):
                        _, row = values[offsets[bucket]]
                        offsets[bucket] += 1
                        if row["group_id"] in selected_groups:
                            continue
                        selected.append(row)
                        selected_groups.add(row["group_id"])
                        selected_counts[f"{language}:{split}"] += 1
                        progressed = True
                        break
                    if selected_counts[f"{language}:{split}"] >= quota:
                        break
                if not progressed:
                    break

    selected.sort(key=lambda row: (row["language"], row["split"], row["id"]))
    report = {
        **dict(counters),
        "selected_rows": len(selected),
        "selected_counts": dict(sorted(selected_counts.items())),
        "underfilled": {
            f"{language}:{split}": quota - selected_counts[f"{language}:{split}"]
            for language in config.languages
            for split, quota in quotas.items()
            if quota > selected_counts[f"{language}:{split}"]
        },
    }
    return selected, report
