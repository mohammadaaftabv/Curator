# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Finite, cost-aware local batching for GPU inference stages.

The backend still decides which parent rows reach ``process_batch``. This
policy only partitions that finite worker-local window into duration-coherent
adapter calls and realigns their results to the original row order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class BatchPolicy:
    """Duration-bucketed batching policy for one local candidate window."""

    enabled: bool = True
    strategy: str = "duration_bucketed"
    buckets_sec: list[float] = field(default_factory=lambda: [0.0, 600.0, 1200.0, 2400.0])
    max_items_per_batch_by_bucket: list[int] = field(default_factory=lambda: [32, 16, 8, 4])
    max_audio_sec_per_batch: float | None = 2400.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            msg = f"BatchPolicy: enabled must be a bool, got {type(self.enabled).__name__}"
            raise TypeError(msg)
        if not self.enabled:
            return
        if self.strategy != "duration_bucketed":
            msg = (
                f"BatchPolicy: strategy={self.strategy!r} not yet implemented; only 'duration_bucketed' is supported."
            )
            raise ValueError(msg)
        self._validate_bucket_edges()
        self._validate_batch_caps()

    def _validate_bucket_edges(self) -> None:
        if not self.buckets_sec:
            msg = "BatchPolicy: buckets_sec must contain at least one edge"
            raise ValueError(msg)
        for edge in self.buckets_sec:
            if isinstance(edge, bool) or not isinstance(edge, Real):
                msg = f"BatchPolicy: every buckets_sec entry must be numeric, got {type(edge).__name__}"
                raise TypeError(msg)
        if self.buckets_sec[0] != 0.0:
            msg = f"BatchPolicy: buckets_sec must start at 0.0, got {self.buckets_sec[0]}"
            raise ValueError(msg)
        for current, following in zip(self.buckets_sec, self.buckets_sec[1:], strict=False):
            if following <= current:
                msg = f"BatchPolicy: buckets_sec must be strictly increasing; got {current} -> {following}"
                raise ValueError(msg)

    def _validate_batch_caps(self) -> None:
        if len(self.max_items_per_batch_by_bucket) != len(self.buckets_sec):
            msg = (
                "BatchPolicy: max_items_per_batch_by_bucket has "
                f"{len(self.max_items_per_batch_by_bucket)} entries but buckets_sec has "
                f"{len(self.buckets_sec)}; lengths must match"
            )
            raise ValueError(msg)
        for cap in self.max_items_per_batch_by_bucket:
            if isinstance(cap, bool) or not isinstance(cap, int):
                msg = (
                    f"BatchPolicy: every max_items_per_batch_by_bucket entry must be an int, got {type(cap).__name__}"
                )
                raise TypeError(msg)
            if cap <= 0:
                msg = f"BatchPolicy: every max_items_per_batch_by_bucket entry must be > 0, got {cap}"
                raise ValueError(msg)
        if self.max_audio_sec_per_batch is not None:
            if isinstance(self.max_audio_sec_per_batch, bool) or not isinstance(self.max_audio_sec_per_batch, Real):
                msg = (
                    "BatchPolicy: max_audio_sec_per_batch must be numeric or None, "
                    f"got {type(self.max_audio_sec_per_batch).__name__}"
                )
                raise TypeError(msg)
            if self.max_audio_sec_per_batch <= 0:
                msg = f"BatchPolicy: max_audio_sec_per_batch must be > 0 (or None), got {self.max_audio_sec_per_batch}"
                raise ValueError(msg)

    @property
    def num_buckets(self) -> int:
        return len(self.buckets_sec)

    def bucket_for(self, cost: float) -> int:
        """Return the left-edge bucket for a non-negative item cost."""
        for index in range(self.num_buckets - 1, -1, -1):
            if cost >= self.buckets_sec[index]:
                return index
        return 0

    def bucketize(
        self,
        items: list[Any],
        cost_fn: Callable[[Any], float],
    ) -> list[tuple[list[int], list[Any]]]:
        """Partition a finite local window into bounded same-bucket calls."""
        if not items:
            return []
        if not self.enabled:
            return [(list(range(len(items))), list(items))]

        bucket_queues: list[list[tuple[int, Any, float]]] = [[] for _ in range(self.num_buckets)]
        for index, item in enumerate(items):
            cost = max(0.0, float(cost_fn(item)))
            bucket_queues[self.bucket_for(cost)].append((index, item, cost))

        planned: list[tuple[list[int], list[Any], float]] = []
        for bucket_index, queued in enumerate(bucket_queues):
            item_cap = self.max_items_per_batch_by_bucket[bucket_index]
            batch: list[tuple[int, Any, float]] = []
            total_cost = 0.0
            for queued_item in queued:
                cost = queued_item[2]
                exceeds_item_cap = len(batch) >= item_cap
                exceeds_cost_cap = (
                    bool(batch)
                    and self.max_audio_sec_per_batch is not None
                    and total_cost + cost > self.max_audio_sec_per_batch
                )
                if exceeds_item_cap or exceeds_cost_cap:
                    planned.append(self._finalize_batch(batch, total_cost))
                    batch = []
                    total_cost = 0.0
                batch.append(queued_item)
                total_cost += cost
                if len(batch) >= item_cap or (
                    self.max_audio_sec_per_batch is not None and total_cost >= self.max_audio_sec_per_batch
                ):
                    planned.append(self._finalize_batch(batch, total_cost))
                    batch = []
                    total_cost = 0.0
            if batch:
                planned.append(self._finalize_batch(batch, total_cost))

        planned.sort(key=lambda entry: entry[2], reverse=True)
        return [(indices, batch_items) for indices, batch_items, _cost in planned]

    @staticmethod
    def _finalize_batch(
        batch: list[tuple[int, Any, float]],
        total_cost: float,
    ) -> tuple[list[int], list[Any], float]:
        return (
            [index for index, _item, _cost in batch],
            [item for _index, item, _cost in batch],
            total_cost,
        )


def run_bucketed(
    items: list[Any],
    run_fn: Callable[[list[Any]], list[Any]],
    *,
    cost_fn: Callable[[Any], float],
    policy: BatchPolicy | None = None,
) -> list[Any]:
    """Run local sub-batches and restore results to input order."""
    if not items:
        return []
    sub_batches = (
        policy.bucketize(items, cost_fn=cost_fn)
        if policy is not None and policy.enabled
        else [(list(range(len(items))), list(items))]
    )

    results: list[Any] = [None] * len(items)
    for indices, sub_items in sub_batches:
        sub_results = run_fn(sub_items)
        if len(sub_results) != len(sub_items):
            msg = f"run_fn returned {len(sub_results)} results for {len(sub_items)} items (must match 1:1)"
            raise RuntimeError(msg)
        for index, result in zip(indices, sub_results, strict=True):
            results[index] = result
    return results
