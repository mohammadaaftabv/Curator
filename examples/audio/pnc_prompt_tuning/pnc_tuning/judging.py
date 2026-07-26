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

"""Categorical judge parsing and panel aggregation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

_ALLOWED = {
    "content_preservation": {"pass", "fail"},
    "language_script_preservation": {"pass", "fail"},
    "sentence_termination": {"correct", "missing", "extraneous", "uncertain"},
    "intra_sentence_punctuation": {"correct", "under", "over", "incorrect", "uncertain"},
    "capitalization": {"correct", "incorrect", "not_applicable", "uncertain"},
    "completeness_handling": {"correct", "forced_terminal", "missed_terminal", "uncertain"},
    "overall": {"pass", "review", "fail"},
    "confidence": {"high", "medium", "low"},
}


@dataclass(frozen=True)
class ErrorSpan:
    """Judge-identified error span."""

    raw_span: str
    candidate_span: str
    category: str


@dataclass(frozen=True)
class JudgeDecision:
    """Validated categorical output from one judge."""

    content_preservation: str
    language_script_preservation: str
    sentence_termination: str
    intra_sentence_punctuation: str
    capitalization: str
    completeness_handling: str
    overall: str
    error_spans: tuple[ErrorSpan, ...] = field(default_factory=tuple)
    confidence: str = "low"
    reason: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> JudgeDecision:
        """Validate a parsed JSON object."""

        for name, allowed in _ALLOWED.items():
            actual = str(value.get(name, "")).lower()
            if actual not in allowed:
                msg = f"Invalid {name}={actual!r}; expected one of {sorted(allowed)}"
                raise ValueError(msg)
        spans = []
        for item in value.get("error_spans", []) or []:
            if not isinstance(item, dict):
                msg = "error_spans entries must be JSON objects"
                raise TypeError(msg)
            spans.append(
                ErrorSpan(
                    raw_span=str(item.get("raw_span", "")),
                    candidate_span=str(item.get("candidate_span", "")),
                    category=str(item.get("category", "")),
                )
            )
        return cls(
            content_preservation=str(value["content_preservation"]).lower(),
            language_script_preservation=str(value["language_script_preservation"]).lower(),
            sentence_termination=str(value["sentence_termination"]).lower(),
            intra_sentence_punctuation=str(value["intra_sentence_punctuation"]).lower(),
            capitalization=str(value["capitalization"]).lower(),
            completeness_handling=str(value["completeness_handling"]).lower(),
            overall=str(value["overall"]).lower(),
            error_spans=tuple(spans),
            confidence=str(value["confidence"]).lower(),
            reason=str(value.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision."""

        return asdict(self)


@dataclass(frozen=True)
class PanelDecision:
    """Aggregated result for one candidate."""

    overall: str
    needs_human_review: bool
    reason: str
    vote_counts: dict[str, int]


@dataclass(frozen=True)
class PairwiseDecision:
    """Validated pairwise judge output."""

    winner: str
    confidence: str
    reason: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> PairwiseDecision:
        """Validate a parsed pairwise JSON object."""

        winner = str(value.get("winner", ""))
        confidence = str(value.get("confidence", "")).lower()
        if winner not in {"A", "B", "tie"}:
            msg = "Pairwise winner must be A, B, or tie"
            raise ValueError(msg)
        if confidence not in _ALLOWED["confidence"]:
            msg = "Pairwise confidence must be high, medium, or low"
            raise ValueError(msg)
        return cls(winner=winner, confidence=confidence, reason=str(value.get("reason", "")))

    def to_dict(self) -> dict[str, str]:
        """Serialize the decision."""

        return asdict(self)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse JSON even when a model wraps it in a Markdown fence."""

    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        msg = "Judge output must be a JSON object"
        raise TypeError(msg)
    return value


def parse_judge_decision(text: str) -> JudgeDecision:
    """Parse and validate one judge response."""

    return JudgeDecision.from_mapping(parse_json_object(text))


def parse_pairwise_decision(text: str) -> PairwiseDecision:
    """Parse and validate one pairwise response."""

    return PairwiseDecision.from_mapping(parse_json_object(text))


def aggregate_panel(
    decisions: list[tuple[str, JudgeDecision]],
    *,
    hard_gate_passed: bool,
) -> PanelDecision:
    """Aggregate calibrated panel votes, with hard-gate and arbiter precedence."""

    if not hard_gate_passed:
        return PanelDecision(
            overall="fail",
            needs_human_review=True,
            reason="deterministic_hard_gate_failed",
            vote_counts={},
        )
    if not decisions:
        return PanelDecision(
            overall="review",
            needs_human_review=True,
            reason="no_calibrated_judge_votes",
            vote_counts={},
        )
    if any(
        decision.content_preservation == "fail" or decision.language_script_preservation == "fail"
        for _, decision in decisions
    ):
        return PanelDecision(
            overall="fail",
            needs_human_review=True,
            reason="judge_reported_content_or_script_mutation",
            vote_counts=dict(Counter(decision.overall for _, decision in decisions)),
        )

    counts = Counter(decision.overall for _, decision in decisions)
    pass_votes = counts["pass"]
    fail_votes = counts["fail"]
    low_confidence = any(decision.confidence == "low" for _, decision in decisions)
    disagreement = len({decision.overall for _, decision in decisions}) > 1
    if pass_votes > fail_votes and pass_votes > counts["review"]:
        overall = "pass"
        reason = "panel_majority_pass"
    elif fail_votes > pass_votes and fail_votes >= counts["review"]:
        overall = "fail"
        reason = "panel_majority_fail"
    else:
        arbiter = next((decision for role, decision in decisions if "arbiter" in role.lower()), None)
        if arbiter is not None and arbiter.overall != "review":
            overall = arbiter.overall
            reason = "arbiter_tiebreak"
        else:
            overall = "review"
            reason = "panel_tie_or_review_plurality"
    return PanelDecision(
        overall=overall,
        needs_human_review=low_confidence or disagreement or overall == "review",
        reason=reason,
        vote_counts=dict(counts),
    )


def consistent_pairwise_winner(first_order: str, swapped_order: str) -> str:
    """Return a stable A/B winner after position swapping."""

    allowed = {"A", "B", "tie"}
    if first_order not in allowed or swapped_order not in allowed:
        msg = "Pairwise decisions must be A, B, or tie"
        raise ValueError(msg)
    mapped_swapped = {"A": "B", "B": "A", "tie": "tie"}[swapped_order]
    return first_order if first_order == mapped_swapped else "review"
