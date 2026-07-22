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

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from nemo_curator.core.serve.base import BaseModelConfig, BaseServerConfig
from nemo_curator.core.serve.dynamo.constants import (
    DEFAULT_DYNAMO_EVENT_PLANE,
    DEFAULT_DYNAMO_NAMESPACE,
    DEFAULT_DYNAMO_REQUEST_PLANE,
)


@dataclass
class DynamoRoleConfig:
    """Per-role config for disaggregated Dynamo serving."""

    num_replicas: int = 1
    engine_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_replicas < 0:
            msg = f"num_replicas must be >= 0, got {self.num_replicas}"
            raise ValueError(msg)


@dataclass
class DynamoRouterConfig:
    """Frontend router config for Dynamo.

    ``mode=None`` means "auto": Curator picks ``"kv"`` if any model uses
    ``mode="disagg"``, else leaves ``--router-mode`` unset so the Dynamo
    frontend falls back to its own ``round_robin`` default. ``kv_events``
    only applies when ``mode == "kv"``: pass ``kv_events=True`` to opt into
    exact ZMQ KV-cache event publishing; the default uses the router's
    approximate tree-based tracking. Anything else is forwarded to the
    Dynamo frontend as CLI args via ``router_kwargs``.
    """

    mode: Literal["round_robin", "random", "kv", "direct"] | None = None
    kv_events: bool = False
    router_kwargs: dict[str, Any] = field(default_factory=dict)

    _RESERVED_ROUTER_KWARGS: ClassVar[frozenset[str]] = frozenset({"router_mode", "router_kv_events"})

    def __post_init__(self) -> None:
        if self.mode is not None and self.mode != "kv" and self.kv_events:
            msg = f"kv_events=True is only meaningful when mode='kv'; got mode={self.mode!r}."
            raise ValueError(msg)
        reserved = self._RESERVED_ROUTER_KWARGS & set(self.router_kwargs)
        if reserved:
            reserved_str = ", ".join(sorted(reserved))
            typed_fields = ", ".join(sorted(k.removeprefix("router_") for k in reserved))
            msg = (
                f"router_kwargs conflicts with typed field(s): {reserved_str}. "
                f"Set these directly on DynamoRouterConfig (.{typed_fields}) instead."
            )
            raise ValueError(msg)


@dataclass
class DynamoAdmissionConfig:
    """Model-level queue admission and AIMD control for the Dynamo frontend.

    The admission proxy reads the vLLM workers' Prometheus queue metric,
    returns HTTP 429 when the queue is overloaded, and applies one shared AIMD
    concurrency window in front of all Curator stage actors using the model.
    Defaults match the optimized Data Designer workflow.
    """

    max_waiting_requests: int
    max_concurrent_requests: int = 8192
    poll_interval_seconds: float = 0.25
    stale_after_seconds: float = 2.0
    retry_after_seconds: float | None = None
    metric_name: str = "vllm:num_requests_waiting"
    fail_open: bool = True
    metrics_urls: list[str] = field(default_factory=list)
    queue_aggregation: Literal["min", "max", "sum"] = "min"
    reduce_factor: float = 0.75
    additive_increase: int = 1
    success_window: int = 25
    cooldown_seconds: float = 2.0
    ceiling_overshoot: float = 0.10
    rampup_seconds: float = 0.0

    def __post_init__(self) -> None:
        checks = [
            (self.max_waiting_requests < 1, "max_waiting_requests must be >= 1"),
            (self.max_concurrent_requests < 1, "max_concurrent_requests must be >= 1"),
            (self.poll_interval_seconds <= 0, "poll_interval_seconds must be > 0"),
            (self.stale_after_seconds < 0, "stale_after_seconds must be >= 0"),
            (
                self.retry_after_seconds is not None and self.retry_after_seconds <= 0,
                "retry_after_seconds must be > 0 when set",
            ),
            (not self.metric_name.strip(), "metric_name must not be empty"),
            (self.queue_aggregation not in {"min", "max", "sum"}, "queue_aggregation must be min, max, or sum"),
            (not 0 < self.reduce_factor < 1, "reduce_factor must be in (0, 1)"),
            (self.additive_increase < 1, "additive_increase must be >= 1"),
            (self.success_window < 1, "success_window must be >= 1"),
            (self.cooldown_seconds <= 0, "cooldown_seconds must be > 0"),
            (self.ceiling_overshoot < 0, "ceiling_overshoot must be >= 0"),
            (self.rampup_seconds < 0, "rampup_seconds must be >= 0"),
        ]
        for invalid, message in checks:
            if invalid:
                raise ValueError(message)


@dataclass
class DynamoVLLMModelConfig(BaseModelConfig):
    """Dynamo vLLM model config.

    Typed fields cover deployment/placement knobs Curator branches on; anything
    else is forwarded to ``python -m dynamo.vllm`` via ``dynamo_kwargs``.
    ``kv_events_config`` and ``kv_transfer_config`` are Curator-managed
    (``init=False``): events are derived from router state + port allocation,
    transfer defaults to NixlConnector for disagg.
    """

    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    install_runtime_dependencies: bool = True
    num_replicas: int = 1
    mode: Literal["aggregated", "disagg"] = "aggregated"
    prefill: DynamoRoleConfig | None = None
    decode: DynamoRoleConfig | None = None
    dynamo_kwargs: dict[str, Any] = field(default_factory=dict)
    kv_events_config: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    kv_transfer_config: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_replicas < 1:
            msg = f"num_replicas must be >= 1, got {self.num_replicas}"
            raise ValueError(msg)
        if self.mode == "aggregated" and (self.prefill is not None or self.decode is not None):
            msg = "prefill/decode are only valid with mode='disagg'"
            raise ValueError(msg)
        if self.mode == "disagg":
            if (self.prefill is None) != (self.decode is None):
                msg = "mode='disagg' requires both prefill and decode to be specified, or neither"
                raise ValueError(msg)
            if self.prefill is not None and (self.prefill.num_replicas < 1 or self.decode.num_replicas < 1):
                msg = "mode='disagg' requires prefill.num_replicas >= 1 and decode.num_replicas >= 1"
                raise ValueError(msg)


@dataclass
class DynamoServerConfig(BaseServerConfig):
    """Server-level Dynamo config."""

    model_configs: ClassVar[tuple[type[BaseModelConfig], ...]] = (DynamoVLLMModelConfig,)

    etcd_endpoint: str | None = None
    nats_url: str | None = None
    namespace: str = DEFAULT_DYNAMO_NAMESPACE
    request_plane: str = DEFAULT_DYNAMO_REQUEST_PLANE
    event_plane: str = DEFAULT_DYNAMO_EVENT_PLANE
    router: DynamoRouterConfig = field(default_factory=DynamoRouterConfig)
    admission: DynamoAdmissionConfig | None = None
