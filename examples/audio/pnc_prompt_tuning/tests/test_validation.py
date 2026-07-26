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

"""Unicode hard-gate tests across every target language."""

from __future__ import annotations

import pytest
from pnc_tuning.languages import LANGUAGE_SPECS, get_language_spec
from pnc_tuning.validation import validate_preservation

_SAMPLES = {
    "as": "অসমীয়া ভাষা",
    "bn": "বাংলা ভাষা",
    "gu": "ગુજરાતી ભાષા",
    "hi": "हिन्दी भाषा",
    "kn": "ಕನ್ನಡ ಭಾಷೆ",
    "ml": "മലയാളം ഭാഷ",
    "mr": "मराठी भाषा",
    "or": "ଓଡ଼ିଆ ଭାଷା",
    "pa": "ਪੰਜਾਬੀ ਭਾਸ਼ਾ",
    "ta": "தமிழ் மொழி",
    "te": "తెలుగు భాష",
    "ur": "اُردُو زبان",
}


@pytest.mark.parametrize(("language", "raw"), sorted(_SAMPLES.items()))
def test_all_target_scripts_allow_punctuation_only(language: str, raw: str) -> None:
    result = validate_preservation(raw, f"{raw}?")
    assert result.passed
    assert result.inserted_punctuation[0].mark == "?"
    assert get_language_spec(language).code == language


def test_language_registry_is_exactly_the_twelve_targets() -> None:
    assert set(LANGUAGE_SPECS) == set(_SAMPLES)
    assert not LANGUAGE_SPECS["as"].sarvam_supported
    assert not LANGUAGE_SPECS["ur"].sarvam_supported


@pytest.mark.parametrize(
    ("raw", "mutated"),
    [
        ("क्\u200dष", "क्ष"),
        ("क्\u200cष", "क्ष"),
        ("اُردُو", "اردُو"),
    ],
)
def test_combining_and_joining_controls_are_preserved(raw: str, mutated: str) -> None:
    assert validate_preservation(raw, f"{raw}.").passed
    assert not validate_preservation(raw, f"{mutated}.").passed


def test_latin_case_changes_are_allowed_but_spelling_changes_are_not() -> None:
    accepted = validate_preservation("नमस्ते openai", "नमस्ते OpenAI.")
    assert accepted.passed
    assert len(accepted.case_changes) == 3
    assert not validate_preservation("नमस्ते openai", "नमस्ते OpenAl.").passed


def test_existing_punctuation_is_consumed_before_insertions() -> None:
    result = validate_preservation("क्या? हाँ", "क्या? हाँ.")
    assert result.passed
    assert [item.mark for item in result.inserted_punctuation] == ["."]


def test_content_mutations_and_wrappers_fail_closed() -> None:
    assert not validate_preservation("नमस्ते", "नमस्ते मित्र.").passed
    assert not validate_preservation("नमस्ते", "```नमस्ते.```").passed
    assert not validate_preservation("اُردُو", "اردو.").passed
    assert not validate_preservation("हैलो", "हैलो।").passed


def test_incomplete_transcript_rejects_only_new_terminal_mark() -> None:
    assert not validate_preservation("यह आगे", "यह आगे.", complete=False).passed
    assert validate_preservation("यह आगे.", "यह आगे.", complete=False).passed
    assert validate_preservation("यह आगे", "यह, आगे", complete=False).passed
