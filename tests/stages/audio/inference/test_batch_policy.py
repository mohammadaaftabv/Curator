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

"""Tests for finite worker-local duration bucketing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio.inference.asr import ASRStage
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy, run_bucketed
from nemo_curator.tasks import AudioTask


def test_batch_policy_validates_edges_and_caps() -> None:
    with pytest.raises(ValueError, match="start at 0"):
        BatchPolicy(buckets_sec=[1, 10], max_items_per_batch_by_bucket=[2, 1])
    with pytest.raises(ValueError, match="strictly increasing"):
        BatchPolicy(buckets_sec=[0, 10, 10], max_items_per_batch_by_bucket=[2, 1, 1])
    with pytest.raises(ValueError, match="lengths must match"):
        BatchPolicy(buckets_sec=[0, 10], max_items_per_batch_by_bucket=[2])


def test_bucket_for_uses_left_edges_and_clamps_to_last_bucket() -> None:
    policy = BatchPolicy(
        buckets_sec=[0, 60, 600],
        max_items_per_batch_by_bucket=[8, 4, 1],
    )

    assert [policy.bucket_for(cost) for cost in (0, 59.9, 60, 599.9, 600, 10_000)] == [0, 0, 1, 1, 2, 2]


def test_bucketize_respects_item_and_total_cost_caps() -> None:
    policy = BatchPolicy(
        buckets_sec=[0],
        max_items_per_batch_by_bucket=[3],
        max_audio_sec_per_batch=100,
    )

    batches = policy.bucketize([40, 40, 40, 120], cost_fn=float)

    assert [[float(item) for item in items] for _indices, items in batches] == [[120.0], [40.0, 40.0], [40.0]]


def test_run_bucketed_restores_original_order() -> None:
    policy = BatchPolicy(
        buckets_sec=[0, 60],
        max_items_per_batch_by_bucket=[2, 2],
        max_audio_sec_per_batch=200,
    )
    calls: list[list[int]] = []

    def run_fn(items: list[int]) -> list[str]:
        calls.append(items)
        return [f"result-{item}" for item in items]

    results = run_bucketed([10, 70, 20, 80, 90], run_fn, cost_fn=float, policy=policy)

    assert calls == [[70, 80], [90], [10, 20]]
    assert results == ["result-10", "result-70", "result-20", "result-80", "result-90"]


class _RecordingAdapter:
    model_id = "nvidia/stt_en_fastconformer_ctc_large"

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def estimate_item_cost(self, item: dict[str, Any]) -> float:
        return float(item["audio_seconds"])

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        self.calls.append(items)
        return [ASRResult(text=str(item["task_id"])) for item in items]


def test_asr_stage_buckets_only_within_its_local_process_batch_window() -> None:
    policy = BatchPolicy(
        buckets_sec=[0, 3],
        max_items_per_batch_by_bucket=[2, 1],
        max_audio_sec_per_batch=10,
    )
    stage = ASRStage(
        adapter_target="nemo_curator.models.asr.NeMoASRAdapter",
        model_id="nvidia/stt_en_fastconformer_ctc_large",
        batch_size=8,
        batch_policy=policy,
    )
    adapter = _RecordingAdapter()
    stage._adapter = adapter
    tasks = [
        AudioTask(task_id=str(index), data={"waveform": np.zeros(seconds * 10), "sampling_rate": 10})
        for index, seconds in enumerate((1, 4, 2, 5))
    ]

    results = stage.process_batch(tasks)

    assert [[item["task_id"] for item in call] for call in adapter.calls] == [["3"], ["1"], ["0", "2"]]
    assert [task.data["pred_text"] for task in results] == ["0", "1", "2", "3"]
    assert all("waveform" not in task.data for task in results)
