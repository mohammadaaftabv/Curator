# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from nemo_curator.core.serve.dynamo.admission_proxy import (
    AdmissionProxy,
    AIMDState,
    QueueMetricSampler,
    _is_exempt_path,
    _parse_retry_after,
    _parser,
)
from nemo_curator.core.serve.dynamo.config import DynamoAdmissionConfig


def test_admission_config_matches_optimized_workflow_defaults() -> None:
    config = DynamoAdmissionConfig(max_waiting_requests=2048)

    assert config.max_concurrent_requests == 8192
    assert config.reduce_factor == 0.75
    assert config.additive_increase == 1
    assert config.success_window == 25
    assert config.cooldown_seconds == 2.0
    assert config.ceiling_overshoot == 0.10


def test_admission_config_allows_zero_queue_threshold() -> None:
    assert DynamoAdmissionConfig(max_waiting_requests=0).max_waiting_requests == 0


@pytest.mark.asyncio
async def test_aimd_reduces_once_per_429_cascade_and_recovers_additively() -> None:
    aimd = AIMDState(maximum=100, success_window=2)

    await aimd.rate_limited(2.0, release_permit=False)
    assert aimd.current_limit == 75
    assert aimd.rate_limit_ceiling == 100
    assert aimd.rate_limit_events_total == 1
    assert aimd.multiplicative_decreases_total == 1

    await aimd.rate_limited(2.0, release_permit=False)
    assert aimd.current_limit == 75
    assert aimd.rate_limit_events_total == 2
    assert aimd.multiplicative_decreases_total == 1

    await aimd.release_success()
    await aimd.release_success()
    assert aimd.current_limit == 76
    assert aimd.additive_increases_total == 1


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


@pytest.mark.parametrize(
    "path",
    ["/health", "/health/ready", "/v1/models", "/v1/models/gemma", "/is_scaling_elastic_ep/status"],
)
def test_control_path_prefixes_are_exempt(path: str) -> None:
    assert _is_exempt_path(path)


def test_chat_completions_is_not_exempt() -> None:
    assert not _is_exempt_path("/v1/chat/completions")


@pytest.mark.asyncio
async def test_overloaded_request_returns_openai_429_and_updates_telemetry() -> None:
    args = _parser().parse_args(
        [
            "--port",
            "8000",
            "--upstream",
            "http://dynamo:8001",
            "--metrics-url",
            "http://worker:18081/metrics",
            "--max-waiting-requests",
            "0",
            "--max-concurrent-requests",
            "100",
        ]
    )
    proxy = AdmissionProxy(args)
    proxy.sampler.value = 1
    proxy.sampler.sampled_at = time.monotonic()

    response = await proxy.handle(SimpleNamespace(path="/v1/chat/completions"))

    assert response.status == 429
    assert response.headers["Retry-After"] == "2.0"
    assert response.headers["X-Curator-Queue-Depth"] == "1"
    assert response.headers["X-Curator-AIMD-Limit"] == "75"
    body = json.loads(response.text)
    assert body["error"]["queue_depth"] == 1
    assert body["error"]["max_waiting_requests"] == 0
    assert proxy.queue_rejections_total == 1
    assert proxy.aimd.rate_limit_events_total == 1
    assert proxy.aimd.multiplicative_decreases_total == 1

    metrics = proxy._proxy_metrics().text
    assert "curator_admission_queue_rejections_total 1" in metrics
    assert "curator_admission_multiplicative_decreases_total 1" in metrics


@pytest.mark.asyncio
async def test_gateway_forwards_openai_request_and_response() -> None:
    async def complete(request: web.Request) -> web.Response:
        return web.json_response({"received": await request.json()})

    async def metrics(_request: web.Request) -> web.Response:
        return web.Response(text="vllm:num_requests_waiting 0\n")

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", complete)
    upstream_app.router.add_get("/metrics", metrics)
    upstream = TestServer(upstream_app)
    await upstream.start_server()

    args = _parser().parse_args(
        [
            "--port",
            "8000",
            "--upstream",
            str(upstream.make_url("/")).rstrip("/"),
            "--metrics-url",
            str(upstream.make_url("/metrics")),
            "--max-waiting-requests",
            "0",
        ]
    )
    proxy = AdmissionProxy(args)
    gateway_app = web.Application()
    gateway_app.router.add_route("*", "/{path_info:.*}", proxy.handle)
    gateway_app.on_startup.append(proxy.start)
    gateway_app.on_cleanup.append(proxy.close)
    client = TestClient(TestServer(gateway_app))
    await client.start_server()

    try:
        response = await client.post("/v1/chat/completions", json={"model": "gemma", "messages": []})
        assert response.status == 200
        assert await response.json() == {"received": {"model": "gemma", "messages": []}}
        assert proxy.forwarded_requests_total == 1
        assert proxy.aimd.in_flight == 0
    finally:
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_upstream_429_is_forwarded_and_reduces_shared_aimd_window() -> None:
    async def rate_limited(_request: web.Request) -> web.Response:
        return web.json_response(
            {"error": {"type": "rate_limit_error"}},
            status=429,
            headers={"Retry-After": "0.25"},
        )

    async def metrics(_request: web.Request) -> web.Response:
        return web.Response(text="vllm:num_requests_waiting 0\n")

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", rate_limited)
    upstream_app.router.add_get("/metrics", metrics)
    upstream = TestServer(upstream_app)
    await upstream.start_server()

    args = _parser().parse_args(
        [
            "--port",
            "8000",
            "--upstream",
            str(upstream.make_url("/")).rstrip("/"),
            "--metrics-url",
            str(upstream.make_url("/metrics")),
            "--max-waiting-requests",
            "0",
            "--max-concurrent-requests",
            "100",
        ]
    )
    proxy = AdmissionProxy(args)
    gateway_app = web.Application()
    gateway_app.router.add_route("*", "/{path_info:.*}", proxy.handle)
    gateway_app.on_startup.append(proxy.start)
    gateway_app.on_cleanup.append(proxy.close)
    client = TestClient(TestServer(gateway_app))
    await client.start_server()

    try:
        response = await client.post("/v1/chat/completions", json={"model": "gemma", "messages": []})
        assert response.status == 429
        assert response.headers["Retry-After"] == "0.25"
        assert await response.json() == {"error": {"type": "rate_limit_error"}}
        assert proxy.forwarded_requests_total == 1
        assert proxy.upstream_429_total == 1
        assert proxy.aimd.rate_limit_events_total == 1
        assert proxy.aimd.multiplicative_decreases_total == 1
        assert proxy.aimd.current_limit == 75
        assert proxy.aimd.in_flight == 0
    finally:
        await client.close()
        await upstream.close()


@pytest.mark.parametrize(("value", "expected"), [("2", 2.0), ("0.25", 0.25), (None, None), ("date", None)])
def test_parse_retry_after(value: str | None, expected: float | None) -> None:
    assert _parse_retry_after(value) == expected
