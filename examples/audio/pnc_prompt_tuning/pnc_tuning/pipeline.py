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

"""Composable phases for the Indic PNC prompt-tuning experiment."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict
from statistics import fmean
from typing import TYPE_CHECKING, Any, Protocol

from pnc_tuning.judging import (
    JudgeDecision,
    aggregate_panel,
    consistent_pairwise_winner,
    parse_judge_decision,
    parse_pairwise_decision,
)
from pnc_tuning.metrics import calibration_report, punctuation_scores
from pnc_tuning.prompts import render_generation_prompt, render_judge_prompt, render_pairwise_prompt
from pnc_tuning.routing import resolve_panel
from pnc_tuning.sampling import select_subset
from pnc_tuning.validation import validate_preservation

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pnc_tuning.config import ExperimentConfig
    from pnc_tuning.normalization import CommonYamlNormalizer
    from pnc_tuning.nvidia_client import ChatResult

_MIN_BOOTSTRAP_SAMPLES = 100


class ChatClient(Protocol):
    """Minimal client contract used by online phases and offline tests."""

    def list_models(self) -> set[str]:
        """Return endpoint model IDs."""

    def chat(  # noqa: PLR0913
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        json_output: bool = False,
        disable_thinking: bool = True,
    ) -> ChatResult:
        """Return an object with model, content, usage, cached, and cache_key."""


def candidate_key(row: Mapping[str, Any]) -> str:
    """Return a stable key for a generated candidate."""

    parts = (row.get("id", ""), row.get("prompt_id", ""), row.get("generator_model", ""))
    return "\x1f".join(str(part) for part in parts)


def _require_common_fields(row: Mapping[str, Any], *, require_reference: bool = False) -> None:
    required = {"text_common", "candidate_common", "normalization"}
    if require_reference:
        required.add("reference_common")
    missing = sorted(required - set(row))
    if missing:
        msg = (
            "Candidate has not passed the authoritative common.yaml normalization phase; "
            f"missing fields: {', '.join(missing)}"
        )
        raise ValueError(msg)


def build_subset(
    rows: Iterable[dict[str, Any]],
    *,
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the deterministic, group-disjoint experiment subset."""

    return select_subset(rows, fields=config.fields, config=config.sampling)


def attach_transcripts(  # noqa: C901, PLR0913
    metadata_rows: Iterable[dict[str, Any]],
    transcript_rows: Iterable[dict[str, Any]],
    *,
    config: ExperimentConfig,
    transcript_id_field: str = "id",
    transcript_text_field: str = "text",
    transcript_language_field: str = "language",
    transcript_reference_field: str = "reference",
    transcript_complete_field: str = "complete",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join a small transcript overlay to read-only metadata without altering the source."""

    overlays = {}
    duplicate_ids = set()
    for row in transcript_rows:
        row_id = str(row.get(transcript_id_field, ""))
        if not row_id:
            continue
        if row_id in overlays:
            duplicate_ids.add(row_id)
        overlays[row_id] = row
    if duplicate_ids:
        preview = ", ".join(sorted(duplicate_ids)[:5])
        msg = f"Transcript input contains duplicate IDs: {preview}"
        raise ValueError(msg)

    output = []
    matched_ids = set()
    for metadata in metadata_rows:
        row_id = str(metadata.get(config.fields.id, ""))
        overlay = overlays.get(row_id)
        if overlay is None:
            continue
        text = str(overlay.get(transcript_text_field, "") or "")
        if not text.strip():
            continue
        merged = {
            config.fields.id: metadata.get(config.fields.id),
            config.fields.text: text,
            config.fields.language: metadata.get(config.fields.language),
            config.fields.group: metadata.get(config.fields.group),
            config.fields.duration: metadata.get(config.fields.duration),
        }
        language = overlay.get(transcript_language_field)
        if language not in (None, ""):
            merged[config.fields.language] = str(language).lower()
        reference = overlay.get(transcript_reference_field)
        if reference not in (None, ""):
            merged[config.fields.reference] = str(reference)
        complete = overlay.get(transcript_complete_field)
        if isinstance(complete, bool):
            merged[config.fields.complete] = complete
        output.append(merged)
        matched_ids.add(row_id)

    return output, {
        "transcript_rows": len(overlays),
        "matched_rows": len(output),
        "unmatched_transcript_ids": sorted(set(overlays) - matched_ids),
    }


def import_candidates(  # noqa: PLR0913
    subset_rows: Iterable[dict[str, Any]],
    result_rows: Iterable[dict[str, Any]],
    *,
    result_id_field: str,
    candidate_field: str,
    prompt_id: str,
    generator_model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join Curator or baseline output to the frozen subset without changing text."""

    results = {}
    duplicate_ids = set()
    for result in result_rows:
        row_id = str(result.get(result_id_field, ""))
        if not row_id:
            continue
        if row_id in results:
            duplicate_ids.add(row_id)
        results[row_id] = result
    if duplicate_ids:
        preview = ", ".join(sorted(duplicate_ids)[:5])
        msg = f"Candidate results contain duplicate IDs: {preview}"
        raise ValueError(msg)

    output = []
    matched_ids = set()
    for row in subset_rows:
        row_id = str(row["id"])
        result = results.get(row_id)
        if result is None:
            continue
        output.append(
            {
                **row,
                "prompt_id": prompt_id,
                "generator_model": generator_model,
                "candidate": str(result.get(candidate_field, "") or "").strip(),
                "generation": {"source": "imported"},
            }
        )
        matched_ids.add(row_id)
    return output, {
        "result_rows": len(results),
        "matched_rows": len(output),
        "unmatched_result_ids": sorted(set(results) - matched_ids),
    }


def generate_candidates(  # noqa: PLR0913
    rows: Iterable[dict[str, Any]],
    *,
    prompt_templates: Mapping[str, Mapping[str, str]],
    generator_models: Iterable[str],
    client: ChatClient,
    demonstrations: Mapping[str, str] | None = None,
    max_tokens: int = 512,
) -> list[dict[str, Any]]:
    """Generate one candidate for every row, prompt, and generator model."""

    output = []
    for row in rows:
        language = str(row["language"])
        text = str(row["text"])
        for prompt_id, language_templates in prompt_templates.items():
            template = language_templates.get(language, language_templates.get("*"))
            if template is None:
                msg = f"Prompt {prompt_id!r} has no template for language {language!r} and no '*' default"
                raise ValueError(msg)
            demonstration_text = "" if demonstrations is None else demonstrations.get(language, "")
            if "{demonstrations}" in template and not demonstration_text:
                msg = f"Prompt {prompt_id!r} requires two gold demonstrations for language {language!r}"
                raise ValueError(msg)
            prompt = render_generation_prompt(
                template,
                text=text,
                language=language,
                demonstrations=demonstration_text,
            )
            for generator_model in generator_models:
                result = client.chat(
                    model=generator_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                output.append(
                    {
                        **row,
                        "prompt_id": prompt_id,
                        "generator_model": generator_model,
                        "candidate": result.content.strip(),
                        "generation": {
                            "served_model": result.model,
                            "usage": result.usage,
                            "metadata": getattr(result, "metadata", {}),
                            "cached": result.cached,
                            "cache_key": result.cache_key,
                        },
                    }
                )
    return output


def validate_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    allowed_punctuation: str,
    normalizer: CommonYamlNormalizer,
) -> list[dict[str, Any]]:
    """Gate raw generator output, then apply the exact ``common.yaml`` profile."""

    output = []
    for row in rows:
        complete = row.get("complete")
        raw_text = str(row["text"])
        raw_candidate = str(row["candidate"])
        result = validate_preservation(
            raw_text,
            raw_candidate,
            allowed_punctuation=allowed_punctuation,
            complete=complete if isinstance(complete, bool) else None,
        )
        text_common = normalizer.normalize(raw_text)
        candidate_common = normalizer.normalize(raw_candidate)
        normalized = {
            **row,
            "candidate_key": candidate_key(row),
            "candidate_raw": raw_candidate,
            "text_common": text_common,
            "candidate_common": candidate_common,
            "candidate": candidate_common,
            "normalization": {
                "profile": "tutorials/audio/granary_v2_postprocessing/common.yaml",
                "sha256": normalizer.sha256,
                "raw_text_changed": text_common != raw_text,
                "raw_candidate_changed": candidate_common != raw_candidate,
            },
            "validation": result.to_dict(),
        }
        if "reference" in row:
            normalized["reference_common"] = normalizer.normalize(str(row["reference"]))
        output.append(normalized)
    return output


def judge_candidates(  # noqa: PLR0913
    rows: Iterable[dict[str, Any]],
    *,
    config: ExperimentConfig,
    judge_template: str,
    client: ChatClient,
    available_models: set[str],
    calibrated_routes: set[str] | None = None,
    allow_partial_panel: bool = False,
    max_tokens: int = 800,
) -> list[dict[str, Any]]:
    """Obtain one structured decision from every resolved judge role."""

    output = []
    routes_by_language = {}
    for row in rows:
        validation = dict(row.get("validation", {}))
        if not validation.get("passed", False):
            continue
        _require_common_fields(row)
        language = str(row["language"])
        if language not in routes_by_language:
            routes_by_language[language] = resolve_panel(
                available_models=available_models,
                roles=config.judge_roles,
                language=language,
                allow_partial=allow_partial_panel,
            )
        for route in routes_by_language[language]:
            route_key = f"{language}:{route.model_id}"
            if calibrated_routes is not None and route_key not in calibrated_routes:
                continue
            rendered = render_judge_prompt(
                judge_template,
                raw_text=str(row.get("text_common", row["text"])),
                candidate_text=str(row.get("candidate_common", row["candidate"])),
                language=language,
                complete=row.get("complete") if isinstance(row.get("complete"), bool) else None,
                gate_results=json.dumps(validation, ensure_ascii=False, sort_keys=True),
                evaluate_language_quality=route.evaluate_language_quality,
            )
            result = client.chat(
                model=route.model_id,
                messages=[{"role": "user", "content": rendered}],
                max_tokens=max_tokens,
                json_output=True,
            )
            judgment: dict[str, Any]
            error = ""
            try:
                judgment = parse_judge_decision(result.content).to_dict()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                judgment = {}
                error = f"{type(exc).__name__}: {exc}"
            output.append(
                {
                    "candidate_key": row.get("candidate_key", candidate_key(row)),
                    "id": row["id"],
                    "language": language,
                    "prompt_id": row["prompt_id"],
                    "generator_model": row["generator_model"],
                    "judge_role": route.role,
                    "judge_model": route.model_id,
                    "judge": judgment,
                    "judge_error": error,
                    "request": {
                        "served_model": result.model,
                        "usage": result.usage,
                        "metadata": getattr(result, "metadata", {}),
                        "cached": result.cached,
                        "cache_key": result.cache_key,
                    },
                }
            )
    return output


def aggregate_candidates(
    candidates: Iterable[dict[str, Any]],
    judgments: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine hard-gate results and model decisions into final panel outcomes."""

    decisions_by_candidate: dict[str, list[tuple[str, JudgeDecision]]] = defaultdict(list)
    errors_by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in judgments:
        key = str(row["candidate_key"])
        judge_value = row.get("judge")
        if isinstance(judge_value, dict) and judge_value:
            decisions_by_candidate[key].append((str(row["judge_role"]), JudgeDecision.from_mapping(judge_value)))
        elif row.get("judge_error"):
            errors_by_candidate[key].append(
                {
                    "judge_role": str(row.get("judge_role", "")),
                    "judge_model": str(row.get("judge_model", "")),
                    "error": str(row["judge_error"]),
                }
            )

    output = []
    for row in candidates:
        key = str(row.get("candidate_key", candidate_key(row)))
        validation = dict(row.get("validation", {}))
        panel = aggregate_panel(
            decisions_by_candidate.get(key, []),
            hard_gate_passed=bool(validation.get("passed", False)),
        )
        judge_errors = errors_by_candidate.get(key, [])
        panel_value = asdict(panel)
        if judge_errors and validation.get("passed", False):
            panel_value.update(
                {
                    "overall": "review",
                    "needs_human_review": True,
                    "reason": "one_or_more_judge_responses_failed_validation",
                }
            )
        output.append(
            {
                **row,
                "candidate_key": key,
                "panel": panel_value,
                "judge_errors": judge_errors,
            }
        )
    return output


def pairwise_compare(  # noqa: PLR0913
    rows: Iterable[dict[str, Any]],
    *,
    prompt_a: str,
    prompt_b: str,
    judge_model: str,
    pairwise_template: str,
    client: ChatClient,
    max_tokens: int = 300,
) -> list[dict[str, Any]]:
    """Compare two prompt variants twice with their positions swapped."""

    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["id"]),
            str(row["language"]),
            str(row["generator_model"]),
            str(row["prompt_id"]),
        )
        index[key] = row

    pair_keys = sorted(
        {
            (row_id, language, generator)
            for row_id, language, generator, prompt_id in index
            if prompt_id == prompt_a and (row_id, language, generator, prompt_b) in index
        }
    )
    output = []
    for row_id, language, generator_model in pair_keys:
        row_a = index[(row_id, language, generator_model, prompt_a)]
        row_b = index[(row_id, language, generator_model, prompt_b)]
        gates_passed = all(bool(dict(row.get("validation", {})).get("passed", False)) for row in (row_a, row_b))
        if not gates_passed:
            output.append(
                {
                    "id": row_id,
                    "language": language,
                    "generator_model": generator_model,
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "winner": "review",
                    "reason": "one_or_both_candidates_failed_hard_gate",
                    "orders": [],
                }
            )
            continue
        _require_common_fields(row_a)
        _require_common_fields(row_b)

        orders = []
        for order_name, first, second in (
            ("original", row_a, row_b),
            ("swapped", row_b, row_a),
        ):
            rendered = render_pairwise_prompt(
                pairwise_template,
                raw_text=str(first.get("text_common", first["text"])),
                candidate_a=str(first.get("candidate_common", first["candidate"])),
                candidate_b=str(second.get("candidate_common", second["candidate"])),
                language=language,
                complete=first.get("complete") if isinstance(first.get("complete"), bool) else None,
            )
            result = client.chat(
                model=judge_model,
                messages=[{"role": "user", "content": rendered}],
                max_tokens=max_tokens,
                json_output=True,
            )
            try:
                decision = parse_pairwise_decision(result.content)
                decision_value = decision.to_dict()
                error = ""
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                decision_value = {}
                error = f"{type(exc).__name__}: {exc}"
            orders.append(
                {
                    "order": order_name,
                    "decision": decision_value,
                    "error": error,
                    "request": {
                        "served_model": result.model,
                        "usage": result.usage,
                        "metadata": getattr(result, "metadata", {}),
                        "cached": result.cached,
                        "cache_key": result.cache_key,
                    },
                }
            )

        if any(item["error"] for item in orders):
            stable_winner = "review"
            reason = "unparseable_pairwise_response"
        else:
            winner = consistent_pairwise_winner(
                str(orders[0]["decision"]["winner"]),
                str(orders[1]["decision"]["winner"]),
            )
            stable_winner = {"A": prompt_a, "B": prompt_b}.get(winner, winner)
            reason = "position_consistent" if winner != "review" else "position_inconsistent"
        output.append(
            {
                "id": row_id,
                "language": language,
                "generator_model": generator_model,
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "winner": stable_winner,
                "reason": reason,
                "orders": orders,
            }
        )
    return output


def score_reference_rows(
    rows: Iterable[dict[str, Any]],
    *,
    allowed_punctuation: str,
) -> dict[str, Any]:
    """Score rows that contain human punctuation references."""

    scored = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "reference" not in row:
            continue
        _require_common_fields(row, require_reference=True)
        score = punctuation_scores(
            str(row.get("reference_common", row["reference"])),
            str(row.get("candidate_common", row["candidate"])),
            allowed_punctuation=allowed_punctuation,
        )
        value = {
            "id": row["id"],
            "language": row["language"],
            "prompt_id": row["prompt_id"],
            "generator_model": row["generator_model"],
            **score,
        }
        scored.append(value)
        grouped[(str(row["language"]), str(row["prompt_id"]))].append(score)

    summary = {}
    for (language, prompt_id), values in sorted(grouped.items()):
        summary[f"{language}:{prompt_id}"] = {
            "rows": len(values),
            "mean_precision": fmean(value["precision"] for value in values),
            "mean_recall": fmean(value["recall"] for value in values),
            "mean_f1": fmean(value["f1"] for value in values),
            "exact_match_rate": fmean(float(value["exact_match"]) for value in values),
        }
    return {"rows": scored, "summary": summary}


def _confidence_interval(values: list[float], *, samples: int, rng: random.Random) -> tuple[float, float]:
    bootstrapped = sorted(fmean(rng.choice(values) for _ in values) for _ in range(samples))
    lower_index = int(0.025 * (samples - 1))
    upper_index = int(0.975 * (samples - 1))
    return bootstrapped[lower_index], bootstrapped[upper_index]


def promotion_report(  # noqa: C901, PLR0912, PLR0913, PLR0915
    rows: Iterable[dict[str, Any]],
    *,
    baseline_prompt: str,
    candidate_prompt: str,
    allowed_punctuation: str,
    bootstrap_samples: int = 2000,
    seed: int = 1234,
    noninferiority_margin: float = 0.01,
) -> dict[str, Any]:
    """Build a paired, per-language bootstrap decision record."""

    if bootstrap_samples < _MIN_BOOTSTRAP_SAMPLES:
        msg = "bootstrap_samples must be at least 100"
        raise ValueError(msg)
    index = {}
    duplicates = set()
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        if prompt_id not in {baseline_prompt, candidate_prompt}:
            continue
        key = (str(row["id"]), str(row["language"]), prompt_id)
        if key in index:
            duplicates.add(key)
        index[key] = row
    if duplicates:
        preview = ", ".join(":".join(item) for item in sorted(duplicates)[:5])
        msg = f"Promotion input has duplicate id/language/prompt rows: {preview}"
        raise ValueError(msg)

    paired_by_language: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row_id, language, prompt_id in sorted(index):
        if prompt_id != baseline_prompt:
            continue
        candidate_key_value = (row_id, language, candidate_prompt)
        if candidate_key_value in index:
            paired_by_language[language].append(
                (index[(row_id, language, baseline_prompt)], index[candidate_key_value])
            )

    rng = random.Random(seed)  # noqa: S311 - deterministic statistical resampling
    per_language = {}
    macro_bootstrap: list[list[float]] = []
    all_candidate_gates_pass = True
    reference_pairs = 0
    for language, pairs in sorted(paired_by_language.items()):
        differences = []
        baseline_gate_passes = 0
        candidate_gate_passes = 0
        for baseline, candidate in pairs:
            baseline_passed = bool(dict(baseline.get("validation", {})).get("passed", False))
            candidate_passed = bool(dict(candidate.get("validation", {})).get("passed", False))
            baseline_gate_passes += baseline_passed
            candidate_gate_passes += candidate_passed
            all_candidate_gates_pass = all_candidate_gates_pass and candidate_passed
            if "reference" not in baseline or "reference" not in candidate:
                continue
            _require_common_fields(baseline, require_reference=True)
            _require_common_fields(candidate, require_reference=True)
            baseline_score = punctuation_scores(
                str(baseline.get("reference_common", baseline["reference"])),
                str(baseline.get("candidate_common", baseline["candidate"])),
                allowed_punctuation=allowed_punctuation,
            )
            candidate_score = punctuation_scores(
                str(candidate.get("reference_common", candidate["reference"])),
                str(candidate.get("candidate_common", candidate["candidate"])),
                allowed_punctuation=allowed_punctuation,
            )
            differences.append(float(candidate_score["f1"]) - float(baseline_score["f1"]))
            reference_pairs += 1
        if differences:
            lower, upper = _confidence_interval(differences, samples=bootstrap_samples, rng=rng)
            language_bootstrap = [
                fmean(rng.choice(differences) for _ in differences) for _ in range(bootstrap_samples)
            ]
            macro_bootstrap.append(language_bootstrap)
            mean_difference = fmean(differences)
        else:
            lower, upper = 0.0, 0.0
            mean_difference = 0.0
        per_language[language] = {
            "paired_rows": len(pairs),
            "reference_pairs": len(differences),
            "baseline_hard_gate_pass_rate": baseline_gate_passes / len(pairs) if pairs else 0.0,
            "candidate_hard_gate_pass_rate": candidate_gate_passes / len(pairs) if pairs else 0.0,
            "mean_f1_difference": mean_difference,
            "f1_difference_95pct_ci": [lower, upper],
            "noninferior": bool(differences) and lower >= -noninferiority_margin,
        }

    if macro_bootstrap:
        macro_values = sorted(
            fmean(language_values[index] for language_values in macro_bootstrap) for index in range(bootstrap_samples)
        )
        macro_lower = macro_values[int(0.025 * (bootstrap_samples - 1))]
        macro_upper = macro_values[int(0.975 * (bootstrap_samples - 1))]
        macro_mean = fmean(value["mean_f1_difference"] for value in per_language.values() if value["reference_pairs"])
    else:
        macro_lower, macro_upper, macro_mean = 0.0, 0.0, 0.0
    no_language_regression = bool(per_language) and all(value["noninferior"] for value in per_language.values())
    reasons = []
    if not all_candidate_gates_pass:
        reasons.append("candidate_has_hard_gate_failures")
    if not reference_pairs:
        reasons.append("no_paired_human_references")
    if not no_language_regression:
        reasons.append("one_or_more_languages_not_noninferior")
    if macro_lower <= 0:
        reasons.append("macro_gain_not_statistically_positive")
    return {
        "baseline_prompt": baseline_prompt,
        "candidate_prompt": candidate_prompt,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "noninferiority_margin": noninferiority_margin,
        "per_language": per_language,
        "macro": {
            "mean_f1_difference": macro_mean,
            "f1_difference_95pct_ci": [macro_lower, macro_upper],
        },
        "candidate_100pct_preservation": all_candidate_gates_pass,
        "promotion_eligible": not reasons,
        "blocking_reasons": reasons,
    }


def calibrate_judges(
    human_rows: Iterable[dict[str, Any]],
    judgment_rows: Iterable[dict[str, Any]],
    *,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Build route-level judge calibration statistics from human labels."""

    human_by_candidate = {
        str(row["candidate_key"]): row
        for row in human_rows
        if row.get("candidate_key") and str(row.get("human_overall", "")).strip()
    }
    normalized = []
    for row in judgment_rows:
        human = human_by_candidate.get(str(row.get("candidate_key", "")))
        if human is None:
            continue
        judgment = row.get("judge")
        judge_overall = judgment.get("overall") if isinstance(judgment, dict) else row.get("judge_overall")
        if not judge_overall:
            continue
        normalized.append(
            {
                "language": row["language"],
                "judge_model": row["judge_model"],
                "judge_overall": judge_overall,
                "human_overall": human["human_overall"],
                "judge_rubric": judgment if isinstance(judgment, dict) else {},
                "human_rubric": human.get("human_rubric", {}),
            }
        )
    return calibration_report(normalized, thresholds=config.calibration)


def experiment_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize hard gates, panel outcomes, and review load by language/prompt/model."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["language"]), str(row["prompt_id"]), str(row["generator_model"]))].append(row)

    summary = {}
    for (language, prompt_id, model), values in sorted(groups.items()):
        total = len(values)
        gate_passes = sum(bool(dict(value.get("validation", {})).get("passed", False)) for value in values)
        outcomes = defaultdict(int)
        human_review = 0
        judge_errors = 0
        for value in values:
            panel = dict(value.get("panel", {}))
            outcomes[str(panel.get("overall", "missing"))] += 1
            human_review += bool(panel.get("needs_human_review", True))
            judge_errors += len(value.get("judge_errors", []))
        summary[f"{language}:{prompt_id}:{model}"] = {
            "rows": total,
            "hard_gate_passes": gate_passes,
            "hard_gate_pass_rate": gate_passes / total if total else 0.0,
            "panel_outcomes": dict(sorted(outcomes.items())),
            "human_review_rows": human_review,
            "human_review_rate": human_review / total if total else 0.0,
            "judge_errors": judge_errors,
        }
    return summary
