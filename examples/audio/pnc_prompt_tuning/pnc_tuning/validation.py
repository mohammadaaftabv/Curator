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

"""Deterministic Unicode-preservation gates for PNC outputs."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class InsertedPunctuation:
    """One punctuation code point inserted into the candidate."""

    output_index: int
    input_boundary: int
    mark: str


@dataclass(frozen=True)
class CaseChange:
    """One allowed Latin case-only change."""

    input_index: int
    output_index: int
    before: str
    after: str


@dataclass
class ValidationResult:
    """Hard-gate result for one raw/candidate pair."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    inserted_punctuation: list[InsertedPunctuation] = field(default_factory=list)
    case_changes: list[CaseChange] = field(default_factory=list)
    input_index: int = 0
    output_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for JSONL output."""

        return asdict(self)


def _is_latin_cased(character: str) -> bool:
    if len(character) != 1 or not character.isalpha():
        return False
    return "LATIN" in unicodedata.name(character, "")


def _equivalent(
    raw_character: str,
    candidate_character: str,
    *,
    allow_latin_case_changes: bool,
) -> tuple[bool, bool]:
    if raw_character == candidate_character:
        return True, False
    if (
        allow_latin_case_changes
        and _is_latin_cased(raw_character)
        and _is_latin_cased(candidate_character)
        and raw_character.casefold() == candidate_character.casefold()
    ):
        return True, True
    return False, False


def validate_preservation(  # noqa: C901
    raw_text: str,
    candidate_text: str,
    *,
    allowed_punctuation: str = ".,?!",
    allow_latin_case_changes: bool = True,
    complete: bool | None = None,
) -> ValidationResult:
    """Prove that a candidate only inserts allowed punctuation and optional Latin case changes.

    The scan consumes matching input characters before treating punctuation as
    an insertion. This preserves punctuation that was already present in the
    input and records only genuinely inserted marks.
    """

    result = ValidationResult(passed=False)
    if raw_text and not candidate_text:
        result.errors.append("nonempty_input_became_empty")
        return result
    if candidate_text.startswith("```") or candidate_text.endswith("```"):
        result.errors.append("markdown_fence")

    allowed = set(allowed_punctuation)
    raw_index = 0
    candidate_index = 0
    while raw_index < len(raw_text) and candidate_index < len(candidate_text):
        equivalent, case_changed = _equivalent(
            raw_text[raw_index],
            candidate_text[candidate_index],
            allow_latin_case_changes=allow_latin_case_changes,
        )
        if equivalent:
            if case_changed:
                result.case_changes.append(
                    CaseChange(
                        input_index=raw_index,
                        output_index=candidate_index,
                        before=raw_text[raw_index],
                        after=candidate_text[candidate_index],
                    )
                )
            raw_index += 1
            candidate_index += 1
            continue

        mark = candidate_text[candidate_index]
        if mark in allowed:
            result.inserted_punctuation.append(
                InsertedPunctuation(
                    output_index=candidate_index,
                    input_boundary=raw_index,
                    mark=mark,
                )
            )
            candidate_index += 1
            continue

        result.errors.append(
            "content_mismatch:"
            f"input[{raw_index}]={raw_text[raw_index]!r}:"
            f"output[{candidate_index}]={candidate_text[candidate_index]!r}"
        )
        result.input_index = raw_index
        result.output_index = candidate_index
        return result

    if raw_index < len(raw_text):
        result.errors.append(f"input_deleted_from_index:{raw_index}")
    while candidate_index < len(candidate_text):
        mark = candidate_text[candidate_index]
        if mark not in allowed:
            result.errors.append(f"extra_non_punctuation:output[{candidate_index}]={mark!r}")
            result.output_index = candidate_index
            break
        result.inserted_punctuation.append(
            InsertedPunctuation(
                output_index=candidate_index,
                input_boundary=raw_index,
                mark=mark,
            )
        )
        candidate_index += 1

    if complete is False and result.inserted_punctuation:
        final_content_index = len(candidate_text.rstrip()) - 1
        if any(
            item.output_index == final_content_index and item.mark in ".?!" for item in result.inserted_punctuation
        ):
            result.errors.append("forced_terminal_on_incomplete_input")

    result.input_index = raw_index
    result.output_index = candidate_index
    result.passed = not result.errors
    return result
