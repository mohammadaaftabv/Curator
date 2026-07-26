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

"""Prompt loading and literal placeholder substitution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pnc_tuning.languages import get_language_spec

_PLACEHOLDER = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")


def load_prompt(path: str | Path) -> str:
    """Read a UTF-8 prompt template."""

    return Path(path).read_text(encoding="utf-8").strip()


def render_template(template: str, values: dict[str, Any], *, required: tuple[str, ...] = ()) -> str:
    """Replace known ``{name}`` placeholders without interpreting transcript braces."""

    placeholders = set(_PLACEHOLDER.findall(template))
    absent = sorted(set(required) - placeholders)
    if absent:
        msg = f"Prompt is missing required placeholders: {', '.join(absent)}"
        raise ValueError(msg)
    unknown = sorted(placeholders - set(values))
    if unknown:
        msg = f"Prompt contains unsupported placeholders: {', '.join(unknown)}"
        raise ValueError(msg)
    return _PLACEHOLDER.sub(lambda match: str(values[match.group(1)]), template)


def render_generation_prompt(
    template: str,
    *,
    text: str,
    language: str,
    demonstrations: str = "",
) -> str:
    """Render a PNC generation prompt for one transcript."""

    spec = get_language_spec(language)
    return render_template(
        template,
        {
            "text": text,
            "language": spec.name,
            "language_name": spec.name,
            "language_code": spec.code,
            "script_policy": spec.script_policy,
            "capitalization_policy": spec.capitalization_policy,
            "demonstrations": demonstrations,
        },
        required=("text",),
    )


def render_judge_prompt(  # noqa: PLR0913
    template: str,
    *,
    raw_text: str,
    candidate_text: str,
    language: str,
    complete: bool | None,
    gate_results: str,
    evaluate_language_quality: bool,
) -> str:
    """Render the absolute categorical judge prompt."""

    spec = get_language_spec(language)
    completeness = "unknown" if complete is None else ("complete" if complete else "incomplete")
    judge_scope = (
        "Evaluate native-language punctuation quality and all policy constraints."
        if evaluate_language_quality
        else (
            "Evaluate only policy compliance and the supplied deterministic evidence. "
            "Do not claim native-language expertise."
        )
    )
    return render_template(
        template,
        {
            "language_name": spec.name,
            "language_code": spec.code,
            "script_policy": spec.script_policy,
            "capitalization_policy": spec.capitalization_policy,
            "complete_or_incomplete": completeness,
            "raw_text": raw_text,
            "candidate_text": candidate_text,
            "gate_results": gate_results,
            "judge_scope": judge_scope,
        },
        required=("raw_text", "candidate_text"),
    )


def render_pairwise_prompt(  # noqa: PLR0913
    template: str,
    *,
    raw_text: str,
    candidate_a: str,
    candidate_b: str,
    language: str,
    complete: bool | None,
) -> str:
    """Render a pairwise prompt for position-swapped evaluation."""

    spec = get_language_spec(language)
    completeness = "unknown" if complete is None else ("complete" if complete else "incomplete")
    return render_template(
        template,
        {
            "language_name": spec.name,
            "language_code": spec.code,
            "script_policy": spec.script_policy,
            "capitalization_policy": spec.capitalization_policy,
            "complete_or_incomplete": completeness,
            "raw_text": raw_text,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
        },
        required=("raw_text", "candidate_a", "candidate_b"),
    )
