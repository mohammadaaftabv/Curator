# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import pytest

from nemo_curator.core.serve.dynamo.admission_proxy import AIMDState, QueueMetricSampler, _parse_retry_after
from nemo_curator.core.serve.dynamo.config import DynamoAdmissionConfig


def test_admission_config_matches_optimized_workflow_defaults() -> None:
    config = DynamoAdmissionConfig(max_waiting_requests=2048)

    assert config.max_concurrent_requests == 8192
    assert config.reduce_factor == 0.75
    assert config.additive_increase == 1
    assert config.success_window == 25
    assert config.cooldown_seconds == 2.0
    assert config.ceiling_overshoot == 0.10


@pytest.mark.asyncio
async def test_aimd_reduces_once_per_429_cascade_and_recovers_additively() -> None:
    aimd = AIMDState(maximum=100, success_window=2)

    await aimd.rate_limited(2.0, release_permit=False)
    assert aimd.current_limit == 75
    assert aimd.rate_limit_ceiling == 100

    await aimd.rate_limited(2.0, release_permit=False)
    assert aimd.current_limit == 75

    await aimd.release_success()
    await aimd.release_success()
    assert aimd.current_limit == 76


@pytest.mark.asyncio
async def test_startup_ramp_is_aborted_by_first_rate_limit() -> None:
    aimd = AIMDState(maximum=64, rampup_seconds=60)
    assert aimd.current_limit == 1

    await aimd.rate_limited(None, release_permit=False)

    assert not aimd.rampup_active
    assert aimd.current_limit == 1


def test_metric_sampler_marks_stale_or_missing_sample_unavailable() -> None:
    sampler = QueueMetricSampler(
        urls=["http://worker/metrics"],
        metric_name="vllm:num_requests_waiting",
        poll_interval=0.25,
        stale_after=2.0,
        fail_open=True,
        aggregation="min",
    )
    assert sampler.snapshot() == (None, False)


@pytest.mark.parametrize(("value", "expected"), [("2", 2.0), ("0.25", 0.25), (None, None), ("date", None)])
def test_parse_retry_after(value: str | None, expected: float | None) -> None:
    assert _parse_retry_after(value) == expected
