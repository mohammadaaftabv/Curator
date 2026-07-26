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

"""Reference metrics and judge-calibration summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pnc_tuning.config import CalibrationConfig

_RUBRIC_FIELDS = (
    "content_preservation",
    "language_script_preservation",
    "sentence_termination",
    "intra_sentence_punctuation",
    "capitalization",
    "completeness_handling",
    "overall",
)


def punctuation_events(text: str, *, allowed_punctuation: str = ".,?!") -> Counter[tuple[int, str]]:
    """Return punctuation events keyed by non-punctuation code-point boundary."""

    allowed = set(allowed_punctuation)
    boundary = 0
    events: Counter[tuple[int, str]] = Counter()
    for character in text:
        if character in allowed:
            events[(boundary, character)] += 1
        else:
            boundary += 1
    return events


def punctuation_scores(
    reference: str,
    candidate: str,
    *,
    allowed_punctuation: str = ".,?!",
) -> dict[str, Any]:
    """Compute exact punctuation-event precision, recall, F1, and per-class counts."""

    reference_events = punctuation_events(reference, allowed_punctuation=allowed_punctuation)
    candidate_events = punctuation_events(candidate, allowed_punctuation=allowed_punctuation)
    true_positive = sum((reference_events & candidate_events).values())
    predicted = sum(candidate_events.values())
    expected = sum(reference_events.values())
    precision = true_positive / predicted if predicted else (1.0 if expected == 0 else 0.0)
    recall = true_positive / expected if expected else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_class = {}
    for mark in allowed_punctuation:
        ref = Counter({key: count for key, count in reference_events.items() if key[1] == mark})
        cand = Counter({key: count for key, count in candidate_events.items() if key[1] == mark})
        tp = sum((ref & cand).values())
        per_class[mark] = {
            "true_positive": tp,
            "predicted": sum(cand.values()),
            "expected": sum(ref.values()),
        }
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "predicted": predicted,
        "expected": expected,
        "per_class": per_class,
        "exact_match": reference == candidate,
    }


def calibration_report(
    rows: list[dict[str, Any]],
    *,
    thresholds: CalibrationConfig,
) -> dict[str, Any]:
    """Summarize judge-vs-human outcomes by language and model."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["language"]), str(row["judge_model"]))].append(row)

    report = {}
    for (language, model), values in sorted(groups.items()):
        counts = Counter()
        confusion: dict[str, Counter[str]] = defaultdict(Counter)
        for value in values:
            human = str(value["human_overall"])
            judge = str(value["judge_overall"])
            counts["rows"] += 1
            counts["agreement"] += human == judge
            counts["false_accept"] += human == "fail" and judge == "pass"
            counts["false_reject"] += human == "pass" and judge == "fail"
            counts["judge_review"] += judge == "review"
            human_rubric = value.get("human_rubric", {})
            judge_rubric = value.get("judge_rubric", {})
            if isinstance(human_rubric, Mapping) and isinstance(judge_rubric, Mapping):
                for field in _RUBRIC_FIELDS:
                    if (
                        field in human_rubric
                        and field in judge_rubric
                        and str(human_rubric[field]).strip()
                        and str(judge_rubric[field]).strip()
                    ):
                        confusion[field][f"{human_rubric[field]}->{judge_rubric[field]}"] += 1
        total = counts["rows"]
        agreement_rate = counts["agreement"] / total if total else 0.0
        false_accept_rate = counts["false_accept"] / total if total else 0.0
        false_reject_rate = counts["false_reject"] / total if total else 0.0
        report[f"{language}:{model}"] = {
            **dict(counts),
            "agreement_rate": agreement_rate,
            "false_accept_rate": false_accept_rate,
            "false_reject_rate": false_reject_rate,
            "confusion": {
                field: dict(sorted(field_counts.items())) for field, field_counts in sorted(confusion.items())
            },
            "route_enabled": (
                total >= thresholds.min_rows
                and agreement_rate >= thresholds.min_agreement_rate
                and false_accept_rate <= thresholds.max_false_accept_rate
                and false_reject_rate <= thresholds.max_false_reject_rate
            ),
            "thresholds": {
                "min_rows": thresholds.min_rows,
                "min_agreement_rate": thresholds.min_agreement_rate,
                "max_false_accept_rate": thresholds.max_false_accept_rate,
                "max_false_reject_rate": thresholds.max_false_reject_rate,
            },
        }
    return report
