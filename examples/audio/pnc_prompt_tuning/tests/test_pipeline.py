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

# ruff: noqa: PLR2004, S101, SIM108

"""Offline end-to-end tests using a deterministic fake chat client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from pnc_tuning.config import (
    CalibrationConfig,
    ExperimentConfig,
    InputFields,
    ModelRoleConfig,
    NvidiaConfig,
    SamplingConfig,
)
from pnc_tuning.normalization import CommonYamlNormalizer
from pnc_tuning.pipeline import (
    aggregate_candidates,
    attach_transcripts,
    calibrate_judges,
    generate_candidates,
    import_candidates,
    judge_candidates,
    pairwise_compare,
    promotion_report,
    validate_candidates,
)

if TYPE_CHECKING:
    from pathlib import Path

_JUDGE_JSON = json.dumps(
    {
        "content_preservation": "pass",
        "language_script_preservation": "pass",
        "sentence_termination": "correct",
        "intra_sentence_punctuation": "correct",
        "capitalization": "not_applicable",
        "completeness_handling": "correct",
        "overall": "pass",
        "error_spans": [],
        "confidence": "high",
        "reason": "Candidate is correct.",
    }
)


@dataclass(frozen=True)
class _Result:
    model: str
    content: str
    usage: dict[str, int]
    cached: bool
    cache_key: str


class _FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def list_models(self) -> set[str]:
        return {"broad-model", "arbiter-model"}

    def chat(self, **kwargs) -> _Result:
        self.calls.append(kwargs)
        if kwargs.get("json_output"):
            content = _JUDGE_JSON
        else:
            content = "नमस्ते."
        return _Result(
            model=str(kwargs["model"]),
            content=content,
            usage={"total_tokens": 10},
            cached=False,
            cache_key=f"key-{len(self.calls)}",
        )


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        work_root=tmp_path,
        input_paths=(),
        fields=InputFields(),
        sampling=SamplingConfig(languages=("hi",)),
        nvidia=NvidiaConfig(),
        calibration=CalibrationConfig(
            min_rows=1,
            min_agreement_rate=1.0,
            max_false_accept_rate=0.0,
            max_false_reject_rate=0.0,
        ),
        judge_roles=(
            ModelRoleConfig(
                name="broad_multilingual",
                candidate_models=("broad-model",),
                supported_languages=("*",),
                required=True,
            ),
            ModelRoleConfig(
                name="policy_arbiter",
                candidate_models=("arbiter-model",),
                supported_languages=("*",),
                required=True,
                evaluate_language_quality=False,
            ),
        ),
    )


def _normalizer(config: ExperimentConfig) -> CommonYamlNormalizer:
    return CommonYamlNormalizer.load(config.common_yaml_path)


def test_transcript_overlay_preserves_source_and_rejects_duplicates(config: ExperimentConfig) -> None:
    metadata = [{"id": "1", "text": "", "language": "hi", "audio_item_id": "video"}]
    rows, report = attach_transcripts(
        metadata,
        [{"id": "1", "text": "नमस्ते", "language": "hi", "complete": True}],
        config=config,
    )
    assert metadata[0]["text"] == ""
    assert rows[0]["text"] == "नमस्ते"
    assert report["matched_rows"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        attach_transcripts(metadata, [{"id": "1", "text": "a"}, {"id": "1", "text": "b"}], config=config)


def test_generation_validation_judging_aggregation_and_calibration(config: ExperimentConfig) -> None:
    source = [{"id": "1", "language": "hi", "text": "नमस्ते", "split": "calibration"}]
    client = _FakeClient()
    candidates = generate_candidates(
        source,
        prompt_templates={"p1": {"*": "Restore {language_name}: {text}"}},
        generator_models=["generator"],
        client=client,
    )
    validated = validate_candidates(
        candidates,
        allowed_punctuation=".,?!",
        normalizer=_normalizer(config),
    )
    assert validated[0]["validation"]["passed"]
    assert validated[0]["candidate_raw"] == "नमस्ते."
    assert validated[0]["candidate_common"] == "नमस्ते."

    judgments = judge_candidates(
        validated,
        config=config,
        judge_template=(
            "{language_name} {language_code} {script_policy} {capitalization_policy} "
            "{complete_or_incomplete} {raw_text} {candidate_text} {gate_results} {judge_scope}"
        ),
        client=client,
        available_models=client.list_models(),
    )
    assert len(judgments) == 2
    aggregated = aggregate_candidates(validated, judgments)
    assert aggregated[0]["panel"]["overall"] == "pass"
    assert not aggregated[0]["panel"]["needs_human_review"]

    labels = [
        {
            "candidate_key": aggregated[0]["candidate_key"],
            "human_overall": "pass",
            "human_rubric": {"overall": "pass"},
        }
    ]
    calibration = calibrate_judges(labels, judgments, config=config)
    assert all(value["route_enabled"] for value in calibration.values())


def test_uncalibrated_routes_are_not_called(config: ExperimentConfig) -> None:
    client = _FakeClient()
    row = {
        "id": "1",
        "language": "hi",
        "text": "नमस्ते",
        "candidate": "नमस्ते.",
        "text_common": "नमस्ते",
        "candidate_common": "नमस्ते.",
        "normalization": {"profile": "common.yaml"},
        "prompt_id": "p1",
        "generator_model": "generator",
        "validation": {"passed": True},
    }
    judgments = judge_candidates(
        [row],
        config=config,
        judge_template=(
            "{language_name} {language_code} {script_policy} {capitalization_policy} "
            "{complete_or_incomplete} {raw_text} {candidate_text} {gate_results} {judge_scope}"
        ),
        client=client,
        available_models=client.list_models(),
        calibrated_routes={"hi:broad-model"},
    )
    assert [item["judge_model"] for item in judgments] == ["broad-model"]


def test_unparseable_judge_response_forces_human_review() -> None:
    candidate = {
        "id": "1",
        "language": "hi",
        "text": "नमस्ते",
        "candidate": "नमस्ते.",
        "text_common": "नमस्ते",
        "candidate_common": "नमस्ते.",
        "normalization": {"profile": "common.yaml"},
        "candidate_key": "key",
        "prompt_id": "p1",
        "generator_model": "generator",
        "validation": {"passed": True},
    }
    judgment = {
        "candidate_key": "key",
        "judge_role": "broad",
        "judge_model": "judge",
        "judge": {},
        "judge_error": "invalid JSON",
    }
    aggregated = aggregate_candidates([candidate], [judgment])
    assert aggregated[0]["panel"]["overall"] == "review"
    assert aggregated[0]["panel"]["needs_human_review"]


def test_p2_requires_two_gold_demonstrations() -> None:
    client = _FakeClient()
    with pytest.raises(ValueError, match="requires two"):
        generate_candidates(
            [{"id": "1", "language": "hi", "text": "नमस्ते"}],
            prompt_templates={"p2": {"*": "{demonstrations}\n{text}"}},
            generator_models=["generator"],
            client=client,
        )


def test_imported_baseline_and_paired_promotion_report() -> None:
    subset = [
        {
            "id": str(index),
            "language": "hi",
            "text": "नमस्ते",
            "reference": "नमस्ते.",
        }
        for index in range(4)
    ]
    imported, report = import_candidates(
        subset,
        [{"id": str(index), "pnc_text": "नमस्ते"} for index in range(4)],
        result_id_field="id",
        candidate_field="pnc_text",
        prompt_id="p0",
        generator_model="target",
    )
    assert report["matched_rows"] == 4
    rows = [
        {
            **row,
            "text_common": row["text"],
            "candidate_common": row["candidate"],
            "reference_common": row["reference"],
            "normalization": {"profile": "common.yaml"},
            "validation": {"passed": True},
        }
        for row in imported
    ]
    rows.extend(
        {
            **row,
            "prompt_id": "p1",
            "candidate": "नमस्ते.",
            "text_common": row["text"],
            "candidate_common": "नमस्ते.",
            "reference_common": row["reference"],
            "normalization": {"profile": "common.yaml"},
            "validation": {"passed": True},
        }
        for row in subset
    )
    decision = promotion_report(
        rows,
        baseline_prompt="p0",
        candidate_prompt="p1",
        allowed_punctuation=".,?!",
        bootstrap_samples=100,
    )
    assert decision["promotion_eligible"]
    assert decision["macro"]["mean_f1_difference"] == 1.0


def test_pairwise_rejects_position_inconsistency() -> None:
    class _BiasedClient(_FakeClient):
        def chat(self, **kwargs) -> _Result:
            self.calls.append(kwargs)
            return _Result(
                model=str(kwargs["model"]),
                content='{"winner":"A","confidence":"high","reason":"first"}',
                usage={},
                cached=False,
                cache_key=str(len(self.calls)),
            )

    base = {
        "id": "1",
        "language": "hi",
        "text": "नमस्ते",
        "text_common": "नमस्ते",
        "normalization": {"profile": "common.yaml"},
        "generator_model": "generator",
        "validation": {"passed": True},
    }
    rows = [
        {**base, "prompt_id": "p0", "candidate": "नमस्ते", "candidate_common": "नमस्ते"},
        {**base, "prompt_id": "p1", "candidate": "नमस्ते.", "candidate_common": "नमस्ते."},
    ]
    client = _BiasedClient()
    output = pairwise_compare(
        rows,
        prompt_a="p0",
        prompt_b="p1",
        judge_model="judge",
        pairwise_template=(
            "{language_name} {language_code} {script_policy} {capitalization_policy} "
            "{complete_or_incomplete} {raw_text} {candidate_a} {candidate_b}"
        ),
        client=client,
    )
    assert output[0]["winner"] == "review"
    assert len(client.calls) == 2
