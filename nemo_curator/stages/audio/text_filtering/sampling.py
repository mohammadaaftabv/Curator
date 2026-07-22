# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Sampling-configuration helpers shared by text-pipeline launchers."""

from __future__ import annotations

import argparse
import json

SAMPLING_STAGE_KEYS = frozenset(
    {
        "recover_entities",
        "pnc",
        "language_id",
        "tn",
        "itn",
        "itn_no_disfluencies",
        "captioning",
        "context_asr",
        "code_switching",
        "speech_qa",
    }
)


def parse_stage_sampling_config(value: str) -> dict[str, dict[str, float]]:
    """Parse and validate per-stage temperature/top-p overrides."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        msg = f"Expected a JSON object, got invalid JSON: {exc}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"Expected a JSON object, got {type(parsed).__name__}"
        raise argparse.ArgumentTypeError(msg)

    unknown_stages = set(parsed) - SAMPLING_STAGE_KEYS
    if unknown_stages:
        msg = f"Unknown sampling stage(s): {', '.join(sorted(unknown_stages))}"
        raise argparse.ArgumentTypeError(msg)

    normalized: dict[str, dict[str, float]] = {}
    for stage, config in parsed.items():
        normalized[stage] = _normalize_stage(stage, config)
    return normalized


def _normalize_stage(stage: str, config: object) -> dict[str, float]:
    if not isinstance(config, dict):
        msg = f"Sampling config for {stage!r} must be a JSON object"
        raise argparse.ArgumentTypeError(msg)
    unknown_fields = set(config) - {"temperature", "top_p"}
    if unknown_fields:
        msg = f"Unknown sampling field(s) for {stage!r}: {', '.join(sorted(unknown_fields))}"
        raise argparse.ArgumentTypeError(msg)

    normalized: dict[str, float] = {}
    for field_name, field_value in config.items():
        if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
            msg = f"{stage}.{field_name} must be a number"
            raise argparse.ArgumentTypeError(msg)
        numeric_value = float(field_value)
        if field_name == "temperature" and numeric_value < 0:
            msg = f"{stage}.temperature must be >= 0"
            raise argparse.ArgumentTypeError(msg)
        if field_name == "top_p" and not 0 < numeric_value <= 1:
            msg = f"{stage}.top_p must be in (0, 1]"
            raise argparse.ArgumentTypeError(msg)
        normalized[field_name] = numeric_value
    return normalized


def sampling_for_stage(
    *,
    stage_sampling_config: dict[str, dict[str, float]],
    stage: str,
    default_temperature: float,
    default_top_p: float,
) -> dict[str, float]:
    """Resolve a stage override over its backwards-compatible defaults."""
    return {
        "temperature": default_temperature,
        "top_p": default_top_p,
        **stage_sampling_config.get(stage, {}),
    }
