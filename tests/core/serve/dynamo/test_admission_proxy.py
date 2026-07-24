# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import asyncio
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
    _parser,
    _prometheus_metric_total,
)
from nemo_curator.core.serve.dynamo.config import DynamoAdmissionConfig


def test_admission_config_has_only_workflow_inputs() -> None:
    config = DynamoAdmissionConfig(max_waiting_requests=2048)

    assert config.max_waiting_requests == 2048
    assert config.max_concurrent_requests == 8192


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_waiting_requests": -1}, "max_waiting_requests"),
        ({"max_waiting_requests": 0, "max_concurrent_requests": 0}, "max_concurrent_requests"),
    ],
)
def test_admission_config_rejects_invalid_limits(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DynamoAdmissionConfig(**kwargs)


def test_prometheus_metric_total_sums_labelled_series() -> None:
    payload = """
    vllm:num_requests_waiting{replica="0"} 2
    vllm:num_requests_waiting{replica="1"} 3
    vllm:num_requests_running 9
    """
    assert _prometheus_metric_total(payload) == 5


@pytest.mark.asyncio
async def test_aimd_reduces_once_per_429_cascade_and_recovers_additively() -> None:
    aimd = AIMDState(maximum=100, success_window=2, rampup_seconds=0)

    await aimd.rate_limited(release_permit=False)
    await aimd.rate_limited(release_permit=False)

    assert aimd.current_limit == 75
    assert aimd.rate_limit_events_total == 2
    assert aimd.multiplicative_decreases_total == 1

    await aimd.release_success()
    await aimd.release_success()
    assert aimd.current_limit == 76
    assert aimd.additive_increases_total == 1


def test_aimd_uses_bounded_startup_ramp() -> None:
    aimd = AIMDState(maximum=8192)

    assert aimd.rampup_seconds == 10.0
    assert aimd.current_limit == 1
    assert aimd.rampup_active


def test_metric_sampler_starts_fail_open() -> None:
    assert QueueMetricSampler(["http://worker/metrics"]).snapshot() == (None, False)


@pytest.mark.asyncio
async def test_metric_sampler_uses_least_loaded_available_worker() -> None:
    sampler = QueueMetricSampler(["http://worker-1/metrics", "http://worker-2/metrics"])
    reads_complete = asyncio.Event()
    read_count = 0

    async def read(_session: object, url: str) -> float:
        nonlocal read_count
        read_count += 1
        if read_count == len(sampler.urls):
            reads_complete.set()
        return {"http://worker-1/metrics": 9.0, "http://worker-2/metrics": 1.0}[url]

    sampler._read = read
    task = asyncio.create_task(sampler._run(SimpleNamespace()))
    try:
        await asyncio.wait_for(reads_complete.wait(), timeout=1)
        for _ in range(10):
            if sampler.snapshot()[1]:
                break
            await asyncio.sleep(0)
        assert sampler.snapshot() == (1.0, True)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_overloaded_request_returns_retryable_openai_429() -> None:
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
    proxy.aimd.rampup_active = False
    proxy.aimd.current_limit = 100
    proxy.sampler.value = 1
    proxy.sampler.sampled_at = time.monotonic()

    response = await proxy.handle(SimpleNamespace(path="/v1/chat/completions"))

    assert response.status == 429
    assert response.headers["Retry-After"] == "2"
    assert response.headers["X-Should-Retry"] == "true"
    assert response.headers["X-Curator-AIMD-Limit"] == "75"
    body = json.loads(response.text)
    assert body["error"]["queue_depth"] == 1
    assert proxy.queue_rejections_total == 1


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
            "--max-concurrent-requests",
            "1",
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
async def test_upstream_429_is_forwarded_and_reduces_shared_window() -> None:
    async def rate_limited(_request: web.Request) -> web.Response:
        return web.json_response({"error": {"type": "rate_limit_error"}}, status=429)

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
    proxy.aimd.rampup_active = False
    proxy.aimd.current_limit = 100
    gateway_app = web.Application()
    gateway_app.router.add_route("*", "/{path_info:.*}", proxy.handle)
    gateway_app.on_startup.append(proxy.start)
    gateway_app.on_cleanup.append(proxy.close)
    client = TestClient(TestServer(gateway_app))
    await client.start_server()

    try:
        response = await client.post("/v1/chat/completions", json={"model": "gemma", "messages": []})
        assert response.status == 429
        assert proxy.upstream_429_total == 1
        assert proxy.aimd.current_limit == 75
        assert proxy.aimd.in_flight == 0
    finally:
        await client.close()
        await upstream.close()
