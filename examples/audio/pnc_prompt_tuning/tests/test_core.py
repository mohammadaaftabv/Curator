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

# ruff: noqa: PLR2004, S101

"""Offline tests for sampling, prompts, routing, judges, and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pnc_tuning.config import InputFields, ModelRoleConfig, SamplingConfig, SplitQuota
from pnc_tuning.io_utils import ensure_within_work_root
from pnc_tuning.judging import (
    JudgeDecision,
    aggregate_panel,
    consistent_pairwise_winner,
    parse_judge_decision,
)
from pnc_tuning.metrics import punctuation_scores
from pnc_tuning.prompts import render_generation_prompt, render_template
from pnc_tuning.routing import resolve_panel
from pnc_tuning.sampling import select_subset

if TYPE_CHECKING:
    from pathlib import Path


def _decision(overall: str, confidence: str = "high") -> JudgeDecision:
    return JudgeDecision(
        content_preservation="pass",
        language_script_preservation="pass",
        sentence_termination="correct",
        intra_sentence_punctuation="correct",
        capitalization="not_applicable",
        completeness_handling="correct",
        overall=overall,
        confidence=confidence,
    )


def test_sampler_is_deterministic_and_group_disjoint() -> None:
    rows = [
        {
            "id": f"id-{index}",
            "audio_item_id": f"group-{index // 2}",
            "language": "hi",
            "text": f"पाठ {index}",
            "actual_duration": index * 100,
        }
        for index in range(30)
    ]
    config = SamplingConfig(
        seed="fixed",
        languages=("hi",),
        quotas=SplitQuota(smoke=2, development=4, calibration=3, challenge=2, test=4),
    )
    first, first_report = select_subset(rows, fields=InputFields(), config=config)
    second, second_report = select_subset(reversed(rows), fields=InputFields(), config=config)
    assert first == second
    assert first_report == second_report
    assert len({row["group_id"] for row in first}) == len(first)


def test_prompt_rendering_rejects_unknown_or_missing_placeholders() -> None:
    rendered = render_generation_prompt(
        "Language {language_code}\n<data>{text}</data>",
        text="say {language_code} literally",
        language="hi",
    )
    assert "say {language_code} literally" in rendered
    with pytest.raises(ValueError, match="unsupported"):
        render_template("{text} {unknown}", {"text": "x"}, required=("text",))
    with pytest.raises(ValueError, match="missing"):
        render_template("no transcript", {"text": "x"}, required=("text",))


def test_router_uses_fallback_and_language_scope() -> None:
    roles = (
        ModelRoleConfig(
            name="broad",
            candidate_models=("retired", "fallback"),
            supported_languages=("*",),
            required=True,
        ),
        ModelRoleConfig(
            name="indic",
            candidate_models=("sarvam",),
            supported_languages=("hi",),
        ),
    )
    routes = resolve_panel(available_models={"fallback", "sarvam"}, roles=roles, language="hi")
    assert [route.model_id for route in routes] == ["fallback", "sarvam"]
    urdu_routes = resolve_panel(available_models={"fallback", "sarvam"}, roles=roles, language="ur")
    assert [route.model_id for route in urdu_routes] == ["fallback"]
    with pytest.raises(RuntimeError, match="Required"):
        resolve_panel(available_models={"sarvam"}, roles=roles, language="ur")


def test_judge_parser_and_panel_fail_closed() -> None:
    parsed = parse_judge_decision(
        """```json
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
          "reason": "All constraints pass."
        }
        ```"""
    )
    assert parsed.overall == "pass"
    hard_fail = aggregate_panel([("broad", parsed)], hard_gate_passed=False)
    assert hard_fail.overall == "fail"
    assert hard_fail.needs_human_review
    disagreement = aggregate_panel(
        [("broad", _decision("pass")), ("policy_arbiter", _decision("fail"))],
        hard_gate_passed=True,
    )
    assert disagreement.needs_human_review


def test_pairwise_swap_consistency() -> None:
    assert consistent_pairwise_winner("A", "B") == "A"
    assert consistent_pairwise_winner("B", "A") == "B"
    assert consistent_pairwise_winner("tie", "tie") == "tie"
    assert consistent_pairwise_winner("A", "A") == "review"


def test_punctuation_metrics_use_content_boundaries() -> None:
    score = punctuation_scores("नमस्ते, दुनिया.", "नमस्ते दुनिया.")
    assert score["true_positive"] == 1
    assert score["expected"] == 2
    assert score["predicted"] == 1
    assert score["precision"] == 1.0
    assert score["recall"] == 0.5


def test_output_guard_rejects_escape(tmp_path: Path) -> None:
    target = ensure_within_work_root("artifacts/result.json", tmp_path)
    assert target == tmp_path / "artifacts/result.json"
    with pytest.raises(ValueError, match="outside"):
        ensure_within_work_root("../escape.json", tmp_path)


def test_output_guard_preserves_approved_symlink_alias_and_blocks_physical_escape(tmp_path: Path) -> None:
    physical_root = tmp_path / "physical-root"
    physical_root.mkdir()
    approved_alias = tmp_path / "approved-alias"
    approved_alias.symlink_to(physical_root, target_is_directory=True)

    target = ensure_within_work_root("artifacts/result.json", approved_alias)
    assert target == approved_alias / "artifacts/result.json"

    outside = tmp_path / "outside"
    outside.mkdir()
    escaping_alias = approved_alias / "escape"
    escaping_alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink target escapes"):
        ensure_within_work_root(escaping_alias / "result.json", approved_alias)
