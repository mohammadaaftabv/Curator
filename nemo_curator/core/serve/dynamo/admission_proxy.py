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

"""Queue-aware HTTP admission proxy for a Dynamo OpenAI frontend.

This is the Dynamo-compatible equivalent of Big Iron's vLLM ASGI queue
middleware plus Data Designer's AIMD throttle. Dynamo's frontend is a separate
Rust-backed service and does not expose vLLM's ASGI middleware hook, so Curator
runs this small gateway directly in front of it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http
import math
import time
from dataclasses import dataclass

from aiohttp import ClientSession, ClientTimeout, web

from nemo_curator.models.client.openai_client import _prometheus_metric_total

_CAPACITY_POLL_INTERVAL = 0.05
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
_EXEMPT_PATH_PREFIXES = (
    "/health",
    "/ready",
    "/metrics",
    "/version",
    "/v1/models",
    "/ping",
    "/is_scaling_elastic_ep",
)


@dataclass
class AIMDState:
    """One model-wide AIMD window, matching Data Designer's chat domain."""

    maximum: int
    reduce_factor: float = 0.75
    additive_increase: int = 1
    success_window: int = 25
    cooldown_seconds: float = 2.0
    ceiling_overshoot: float = 0.10
    rampup_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.current_limit = 1 if self.rampup_seconds > 0 and self.maximum > 1 else self.maximum
        self.in_flight = 0
        self.blocked_until = 0.0
        self.success_streak = 0
        self.rate_limit_ceiling = 0
        self.consecutive_429s = 0
        self.rate_limit_events_total = 0
        self.multiplicative_decreases_total = 0
        self.additive_increases_total = 0
        self.rampup_started_at = time.monotonic()
        self.rampup_active = self.rampup_seconds > 0 and self.maximum > 1
        self._condition = asyncio.Condition()

    def _apply_ramp(self, now: float) -> None:
        if not self.rampup_active:
            return
        elapsed = max(0.0, now - self.rampup_started_at)
        if elapsed >= self.rampup_seconds:
            self.current_limit = self.maximum
            self.rampup_active = False
            return
        slots = math.floor((self.maximum - 1) * elapsed / self.rampup_seconds)
        self.current_limit = min(self.maximum, 1 + slots)

    def _soft_ceiling(self) -> int:
        if self.rate_limit_ceiling <= 0:
            return self.maximum
        overshoot = max(1, math.floor(self.rate_limit_ceiling * self.ceiling_overshoot))
        return min(self.maximum, self.rate_limit_ceiling + overshoot)

    async def acquire(self) -> None:
        async with self._condition:
            while True:
                now = time.monotonic()
                self._apply_ramp(now)
                cooldown = self.blocked_until - now
                if cooldown > 0:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._condition.wait(), timeout=cooldown)
                    continue
                if self.in_flight < self.current_limit:
                    self.in_flight += 1
                    return
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._condition.wait(), timeout=_CAPACITY_POLL_INTERVAL)

    async def release_success(self) -> None:
        async with self._condition:
            self.in_flight = max(0, self.in_flight - 1)
            self.consecutive_429s = 0
            self._apply_ramp(time.monotonic())
            if self.rampup_active:
                self.success_streak = 0
            else:
                self.success_streak += 1
                if self.success_streak >= self.success_window:
                    previous = self.current_limit
                    self.current_limit = min(
                        self.current_limit + self.additive_increase,
                        self._soft_ceiling(),
                    )
                    if self.current_limit > previous:
                        self.additive_increases_total += 1
                    self.success_streak = 0
            self._condition.notify_all()

    async def release_failure(self) -> None:
        async with self._condition:
            self.in_flight = max(0, self.in_flight - 1)
            if self.in_flight == 0:
                self.consecutive_429s = 0
            self._condition.notify_all()

    async def rate_limited(self, retry_after: float | None, *, release_permit: bool) -> None:
        async with self._condition:
            if release_permit:
                self.in_flight = max(0, self.in_flight - 1)
            self.rampup_active = False
            previous = self.current_limit
            first_in_cascade = self.consecutive_429s == 0
            self.consecutive_429s += 1
            self.rate_limit_events_total += 1
            self.blocked_until = time.monotonic() + (retry_after or self.cooldown_seconds)
            self.success_streak = 0
            if first_in_cascade:
                self.current_limit = max(1, math.floor(previous * self.reduce_factor))
                if self.current_limit < previous:
                    self.multiplicative_decreases_total += 1
                if self.rate_limit_ceiling == 0:
                    self.rate_limit_ceiling = previous
                else:
                    self.rate_limit_ceiling = min(self.rate_limit_ceiling, previous)
            self._condition.notify_all()


class QueueMetricSampler:
    def __init__(  # noqa: PLR0913
        self,
        *,
        urls: list[str],
        metric_name: str,
        poll_interval: float,
        stale_after: float,
        fail_open: bool,
        aggregation: str = "min",
    ) -> None:
        self.urls = urls
        self.metric_name = metric_name
        self.poll_interval = poll_interval
        self.stale_after = stale_after
        self.fail_open = fail_open
        self.aggregation = aggregation
        self.value: float | None = None
        self.sampled_at = 0.0
        self.last_error: str | None = None
        self._task: asyncio.Task | None = None

    def start(self, session: ClientSession) -> None:
        self._task = asyncio.create_task(self._run(session))

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self, session: ClientSession) -> None:
        while True:
            try:
                values: list[float] = []
                for url in self.urls:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        values.append(_prometheus_metric_total(await response.text(), self.metric_name))
                # A load balancer can retry another vLLM backend, so its client
                # sees 429 only once all backends are saturated. Dynamo routes
                # internally; min(queue) preserves that same semantics.
                # max/sum remain available for deliberately stricter policies.
                if self.aggregation == "sum":
                    self.value = sum(values) if values else None
                elif self.aggregation == "max":
                    self.value = max(values) if values else None
                else:
                    self.value = min(values) if values else None
                self.sampled_at = time.monotonic()
                self.last_error = None
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
            await asyncio.sleep(self.poll_interval)

    def snapshot(self) -> tuple[float | None, bool]:
        fresh = self.value is not None and time.monotonic() - self.sampled_at <= self.stale_after
        return (self.value if fresh else None), fresh


class AdmissionProxy:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session: ClientSession | None = None
        self.sampler = QueueMetricSampler(
            urls=args.metrics_url,
            metric_name=args.metric_name,
            poll_interval=args.poll_interval_seconds,
            stale_after=args.stale_after_seconds,
            fail_open=args.fail_open,
            aggregation=args.queue_aggregation,
        )
        self.aimd = AIMDState(
            maximum=args.max_concurrent_requests,
            reduce_factor=args.aimd_reduce_factor,
            additive_increase=args.aimd_additive_increase,
            success_window=args.aimd_success_window,
            cooldown_seconds=args.aimd_cooldown_seconds,
            ceiling_overshoot=args.aimd_ceiling_overshoot,
            rampup_seconds=args.aimd_rampup_seconds,
        )
        self.forwarded_requests_total = 0
        self.queue_rejections_total = 0
        self.upstream_429_total = 0
        self.upstream_failures_total = 0

    async def start(self, _app: web.Application) -> None:
        # Preserve upstream bytes and Content-Encoding/Content-Length headers
        # exactly while streaming through the gateway.
        self.session = ClientSession(timeout=ClientTimeout(total=None), auto_decompress=False)
        self.sampler.start(self.session)

    async def close(self, _app: web.Application) -> None:
        await self.sampler.close()
        if self.session is not None:
            await self.session.close()

    def _retry_after(self) -> float:
        return self.args.retry_after_seconds or self.args.aimd_cooldown_seconds

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.path == "/metrics":
            return self._proxy_metrics()

        is_exempt = _is_exempt_path(request.path)
        queue_depth, fresh = self.sampler.snapshot()
        if not is_exempt and (
            (fresh and queue_depth > self.args.max_waiting_requests) or (not fresh and not self.args.fail_open)
        ):
            retry_after = self._retry_after()
            await self.aimd.rate_limited(retry_after, release_permit=False)
            self.queue_rejections_total += 1
            return self._overloaded(queue_depth, retry_after, metrics_fresh=fresh)

        permit_acquired = False
        if not is_exempt:
            await self.aimd.acquire()
            permit_acquired = True

        assert self.session is not None  # noqa: S101
        upstream_url = f"{self.args.upstream.rstrip('/')}{request.rel_url}"
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS and k.lower() != "host"
        }
        try:
            self.forwarded_requests_total += 1
            async with self.session.request(
                request.method,
                upstream_url,
                headers=headers,
                data=request.content.iter_chunked(64 * 1024),
                allow_redirects=False,
            ) as upstream:
                response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}
                response = web.StreamResponse(status=upstream.status, headers=response_headers)
                await response.prepare(request)
                async for chunk in upstream.content.iter_chunked(64 * 1024):
                    await response.write(chunk)
                await response.write_eof()

                if permit_acquired:
                    if upstream.status == http.HTTPStatus.TOO_MANY_REQUESTS:
                        self.upstream_429_total += 1
                        retry = _parse_retry_after(upstream.headers.get("Retry-After"))
                        await self.aimd.rate_limited(retry, release_permit=True)
                    elif upstream.status < http.HTTPStatus.BAD_REQUEST:
                        await self.aimd.release_success()
                    else:
                        self.upstream_failures_total += 1
                        await self.aimd.release_failure()
                return response
        except Exception:
            self.upstream_failures_total += 1
            if permit_acquired:
                await self.aimd.release_failure()
            raise

    def _overloaded(self, queue_depth: float | None, retry_after: float, *, metrics_fresh: bool) -> web.Response:
        body = {
            "error": {
                "message": "Dynamo admission queue is overloaded",
                "type": "rate_limit_error",
                "code": "queue_overloaded",
                "queue_depth": queue_depth,
                "max_waiting_requests": self.args.max_waiting_requests,
            }
        }
        return web.json_response(
            body,
            status=http.HTTPStatus.TOO_MANY_REQUESTS,
            headers={
                "Retry-After": str(retry_after),
                "X-Curator-Queue-Depth": "unknown" if queue_depth is None else str(queue_depth),
                "X-Curator-Queue-Metrics-Fresh": str(metrics_fresh).lower(),
                "X-Curator-AIMD-Limit": str(self.aimd.current_limit),
            },
        )

    def _proxy_metrics(self) -> web.Response:
        queue_depth, fresh = self.sampler.snapshot()
        lines = [
            "# TYPE curator_admission_current_limit gauge",
            f"curator_admission_current_limit {self.aimd.current_limit}",
            "# TYPE curator_admission_in_flight gauge",
            f"curator_admission_in_flight {self.aimd.in_flight}",
            "# TYPE curator_admission_queue_metric_fresh gauge",
            f"curator_admission_queue_metric_fresh {int(fresh)}",
            "# TYPE curator_admission_forwarded_requests_total counter",
            f"curator_admission_forwarded_requests_total {self.forwarded_requests_total}",
            "# TYPE curator_admission_queue_rejections_total counter",
            f"curator_admission_queue_rejections_total {self.queue_rejections_total}",
            "# TYPE curator_admission_upstream_429_total counter",
            f"curator_admission_upstream_429_total {self.upstream_429_total}",
            "# TYPE curator_admission_upstream_failures_total counter",
            f"curator_admission_upstream_failures_total {self.upstream_failures_total}",
            "# TYPE curator_admission_rate_limit_events_total counter",
            f"curator_admission_rate_limit_events_total {self.aimd.rate_limit_events_total}",
            "# TYPE curator_admission_multiplicative_decreases_total counter",
            f"curator_admission_multiplicative_decreases_total {self.aimd.multiplicative_decreases_total}",
            "# TYPE curator_admission_additive_increases_total counter",
            f"curator_admission_additive_increases_total {self.aimd.additive_increases_total}",
        ]
        if queue_depth is not None:
            lines += [
                "# TYPE curator_admission_worker_queue_depth gauge",
                f"curator_admission_worker_queue_depth {queue_depth}",
            ]
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    with contextlib.suppress(ValueError):
        parsed = float(value)
        return parsed if parsed > 0 else None
    return None


def _is_exempt_path(path: str) -> bool:
    """Match the health/control path prefixes exempted by vLLM middleware."""
    return any(path.startswith(prefix) for prefix in _EXEMPT_PATH_PREFIXES)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - cluster-facing service
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--metrics-url", action="append", required=True)
    parser.add_argument("--metric-name", default="vllm:num_requests_waiting")
    parser.add_argument("--max-waiting-requests", type=int, required=True)
    parser.add_argument("--max-concurrent-requests", type=int, default=8192)
    parser.add_argument("--queue-aggregation", choices=["min", "max", "sum"], default="min")
    parser.add_argument("--poll-interval-seconds", type=float, default=0.25)
    parser.add_argument("--stale-after-seconds", type=float, default=2.0)
    parser.add_argument("--retry-after-seconds", type=float, default=None)
    parser.add_argument("--fail-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aimd-reduce-factor", type=float, default=0.75)
    parser.add_argument("--aimd-additive-increase", type=int, default=1)
    parser.add_argument("--aimd-success-window", type=int, default=25)
    parser.add_argument("--aimd-cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--aimd-ceiling-overshoot", type=float, default=0.10)
    parser.add_argument("--aimd-rampup-seconds", type=float, default=0.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    proxy = AdmissionProxy(args)
    app = web.Application(client_max_size=16 * 1024**2)
    app.router.add_route("*", "/{path_info:.*}", proxy.handle)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.close)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
