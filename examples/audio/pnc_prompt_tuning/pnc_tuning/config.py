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

"""Typed configuration loader for PNC experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pnc_tuning.io_utils import DEFAULT_DRACO_WORK_ROOT, lexical_absolute_path, load_json
from pnc_tuning.languages import LANGUAGE_SPECS

CURATOR_ROOT = Path(__file__).resolve().parents[4]
COMMON_YAML_RELATIVE_PATH = Path("tutorials/audio/granary_v2_postprocessing/common.yaml")
DEFAULT_COMMON_YAML_PATH = (CURATOR_ROOT / COMMON_YAML_RELATIVE_PATH).resolve()
CANONICAL_PNC_INSERTIONS = ".,?!"


@dataclass(frozen=True)
class InputFields:
    """Field mapping for source JSONL rows."""

    id: str = "id"
    text: str = "text"
    language: str = "language"
    reference: str = "reference"
    complete: str = "complete"
    group: str = "audio_item_id"
    duration: str = "actual_duration"


@dataclass(frozen=True)
class SplitQuota:
    """Per-language row limits for each experiment split."""

    smoke: int = 5
    development: int = 50
    calibration: int = 25
    challenge: int = 10
    test: int = 50


@dataclass(frozen=True)
class SamplingConfig:
    """Deterministic subset configuration."""

    seed: str = "indic-pnc-v1"
    languages: tuple[str, ...] = tuple(LANGUAGE_SPECS)
    quotas: SplitQuota = field(default_factory=SplitQuota)
    require_nonempty_text: bool = True


@dataclass(frozen=True)
class NvidiaConfig:
    """OpenAI-compatible NVIDIA endpoint configuration."""

    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    timeout_seconds: float = 120.0
    max_retries: int = 3


@dataclass(frozen=True)
class ModelRoleConfig:
    """Ordered model candidates for one panel role."""

    name: str
    candidate_models: tuple[str, ...]
    supported_languages: tuple[str, ...]
    required: bool = False
    evaluate_language_quality: bool = True


@dataclass(frozen=True)
class CalibrationConfig:
    """Per-language judge-route enablement thresholds."""

    min_rows: int = 25
    min_agreement_rate: float = 0.80
    max_false_accept_rate: float = 0.05
    max_false_reject_rate: float = 0.10


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    work_root: Path
    input_paths: tuple[str, ...]
    fields: InputFields
    sampling: SamplingConfig
    nvidia: NvidiaConfig
    calibration: CalibrationConfig
    judge_roles: tuple[ModelRoleConfig, ...]
    common_yaml_path: Path = DEFAULT_COMMON_YAML_PATH
    allowed_punctuation: str = CANONICAL_PNC_INSERTIONS


def _dict(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = f"{label} must be a JSON object"
        raise TypeError(msg)
    return value


def _tuple(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = "Expected a JSON array of strings"
        raise TypeError(msg)
    return tuple(value)


def load_config(path: str | Path) -> ExperimentConfig:  # noqa: C901
    """Load and validate a JSON experiment configuration."""

    raw = load_json(path)
    fields_raw = _dict(raw.get("fields"), "fields")
    sampling_raw = _dict(raw.get("sampling"), "sampling")
    quotas_raw = _dict(sampling_raw.get("quotas"), "sampling.quotas")
    nvidia_raw = _dict(raw.get("nvidia"), "nvidia")
    calibration_raw = _dict(raw.get("calibration"), "calibration")

    fields = InputFields(
        **{key: value for key, value in fields_raw.items() if key in InputFields.__dataclass_fields__}
    )
    quotas = SplitQuota(
        **{key: int(value) for key, value in quotas_raw.items() if key in SplitQuota.__dataclass_fields__}
    )
    if any(value < 0 for value in quotas.__dict__.values()):
        msg = "Sampling quotas must be non-negative"
        raise ValueError(msg)
    if not any(quotas.__dict__.values()):
        msg = "At least one sampling quota must be positive"
        raise ValueError(msg)
    sampling = SamplingConfig(
        seed=str(sampling_raw.get("seed", "indic-pnc-v1")),
        languages=_tuple(sampling_raw.get("languages"), tuple(LANGUAGE_SPECS)),
        quotas=quotas,
        require_nonempty_text=bool(sampling_raw.get("require_nonempty_text", True)),
    )
    unsupported = sorted(set(sampling.languages) - set(LANGUAGE_SPECS))
    if unsupported:
        msg = f"Unsupported sampling languages: {', '.join(unsupported)}"
        raise ValueError(msg)

    nvidia = NvidiaConfig(
        base_url=str(nvidia_raw.get("base_url", NvidiaConfig.base_url)),
        api_key_env=str(nvidia_raw.get("api_key_env", NvidiaConfig.api_key_env)),
        timeout_seconds=float(nvidia_raw.get("timeout_seconds", NvidiaConfig.timeout_seconds)),
        max_retries=int(nvidia_raw.get("max_retries", NvidiaConfig.max_retries)),
    )
    calibration = CalibrationConfig(
        min_rows=int(calibration_raw.get("min_rows", CalibrationConfig.min_rows)),
        min_agreement_rate=float(calibration_raw.get("min_agreement_rate", CalibrationConfig.min_agreement_rate)),
        max_false_accept_rate=float(
            calibration_raw.get("max_false_accept_rate", CalibrationConfig.max_false_accept_rate)
        ),
        max_false_reject_rate=float(
            calibration_raw.get("max_false_reject_rate", CalibrationConfig.max_false_reject_rate)
        ),
    )

    roles_raw = raw.get("judge_roles", [])
    if not isinstance(roles_raw, list):
        msg = "judge_roles must be a JSON array"
        raise TypeError(msg)
    roles = []
    for role_raw in roles_raw:
        role = _dict(role_raw, "judge_roles[]")
        roles.append(
            ModelRoleConfig(
                name=str(role["name"]),
                candidate_models=_tuple(role.get("candidate_models"), ()),
                supported_languages=_tuple(role.get("supported_languages"), ("*",)),
                required=bool(role.get("required", False)),
                evaluate_language_quality=bool(role.get("evaluate_language_quality", True)),
            )
        )
    if not roles:
        msg = "At least one judge role must be configured"
        raise ValueError(msg)

    input_paths = raw.get("input_paths", [])
    if not isinstance(input_paths, list) or not all(isinstance(item, str) for item in input_paths):
        msg = "input_paths must be a JSON array of strings"
        raise TypeError(msg)

    # Keep Draco's approved /lustre/fsw/.../users path exactly as configured.
    # Path.resolve() rewrites that alias to /lustre/fs11/.../projects on Draco.
    work_root = lexical_absolute_path(raw.get("work_root", DEFAULT_DRACO_WORK_ROOT))
    if work_root.as_posix().startswith("/lustre/") and work_root != DEFAULT_DRACO_WORK_ROOT:
        msg = f"Draco work_root must be exactly {DEFAULT_DRACO_WORK_ROOT}, got {work_root}"
        raise ValueError(msg)

    common_yaml_value = Path(raw.get("common_yaml_path", COMMON_YAML_RELATIVE_PATH)).expanduser()
    common_yaml_path = (
        common_yaml_value.resolve()
        if common_yaml_value.is_absolute()
        else (CURATOR_ROOT / common_yaml_value).resolve()
    )
    if common_yaml_path != DEFAULT_COMMON_YAML_PATH:
        msg = (
            "common_yaml_path is fixed by the Granary v2 contract; expected "
            f"{DEFAULT_COMMON_YAML_PATH}, got {common_yaml_path}"
        )
        raise ValueError(msg)

    allowed_punctuation = str(raw.get("allowed_punctuation", CANONICAL_PNC_INSERTIONS))
    if allowed_punctuation != CANONICAL_PNC_INSERTIONS:
        msg = (
            "The generator insertion set is fixed by the common.yaml PNC contract; "
            f"expected {CANONICAL_PNC_INSERTIONS!r}, got {allowed_punctuation!r}"
        )
        raise ValueError(msg)

    return ExperimentConfig(
        work_root=work_root,
        input_paths=tuple(input_paths),
        fields=fields,
        sampling=sampling,
        nvidia=nvidia,
        calibration=calibration,
        judge_roles=tuple(roles),
        common_yaml_path=common_yaml_path,
        allowed_punctuation=allowed_punctuation,
    )
