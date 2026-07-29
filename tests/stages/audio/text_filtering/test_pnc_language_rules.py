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

from pathlib import Path

import pytest

from nemo_curator.stages.audio.text_filtering.pnc_language_rules import (
    PNC_LANGUAGE_CODES,
    load_pnc_language_rules,
)
from nemo_curator.stages.audio.text_filtering.remote_text_llm_stage import RemoteTextLLMStage
from nemo_curator.stages.audio.text_filtering.text_llm_stage import TextLLMStage

PROMPT_DIR = (
    Path(__file__).parents[4] / "nemo_curator" / "stages" / "audio" / "text_filtering" / "prompts"
)


def test_bundled_language_rules_have_exact_target_codes() -> None:
    rules = load_pnc_language_rules()

    assert tuple(rules) == PNC_LANGUAGE_CODES
    assert all(rule.strip() for rule in rules.values())


def test_pnc_prompt_uses_one_language_rules_placeholder() -> None:
    prompt = (PROMPT_DIR / "pnc_prompt.md").read_text(encoding="utf-8")

    assert prompt.count("{language_rules}") == 1
    assert "For Assamese, Bengali" not in prompt


def test_text_stage_renders_only_active_language_rule() -> None:
    stage = TextLLMStage(
        prompt_text="{language}\n{language_rules}\n{text}",
        language_rules={"hi": "Hindi-only rule.", "ur": "Urdu-only rule."},
    )
    stage._system_prompt = stage._resolve_prompt()

    rendered = stage._render_prompt_template("नमस्ते दुनिया", {"source_lang": "hi"})

    assert rendered == "hi\nHindi-only rule.\nनमस्ते दुनिया"
    assert "Urdu-only rule." not in rendered


def test_remote_stage_uses_same_row_scoped_rendering() -> None:
    stage = RemoteTextLLMStage(
        prompt_text="{language}\n{language_rules}\n{text}",
        language_rules={"hi": "Hindi-only rule.", "ur": "Urdu-only rule."},
    )
    stage._system_prompt = stage._resolve_prompt()

    messages = stage._build_messages("नमस्ते दुनिया", {"source_lang": "hi"})

    assert messages == [{"role": "user", "content": "hi\nHindi-only rule.\nनमस्ते दुनिया"}]


def test_bundled_hindi_render_contains_no_other_language_guidance() -> None:
    stage = TextLLMStage(
        prompt_file=str(PROMPT_DIR / "pnc_prompt.md"),
        language_rules=load_pnc_language_rules(),
    )
    stage._system_prompt = stage._resolve_prompt()

    rendered = stage._render_prompt_template("नमस्ते दुनिया", {"source_lang": "hi"})

    assert "Hindi" in rendered
    for other_language in (
        "Assamese",
        "Bengali",
        "Gujarati",
        "Kannada",
        "Malayalam",
        "Marathi",
        "Odia",
        "Punjabi",
        "Tamil",
        "Telugu",
        "Urdu",
        "Arabic",
        "Gurmukhi",
    ):
        assert other_language not in rendered


@pytest.mark.parametrize("task_data", [None, {}, {"source_lang": ""}, {"source_lang": "xx"}])
def test_language_rule_resolution_fails_closed(task_data: dict | None) -> None:
    stage = TextLLMStage(
        prompt_text="{language_rules}\n{text}",
        language_rules={"hi": "Hindi-only rule."},
    )
    stage._system_prompt = stage._resolve_prompt()

    with pytest.raises(ValueError, match=r"prompt requires|unsupported"):
        stage._render_prompt_template("नमस्ते दुनिया", task_data)


def test_prompt_without_language_rules_remains_backward_compatible() -> None:
    stage = TextLLMStage(prompt_text="{language}\n{text}")
    stage._system_prompt = stage._resolve_prompt()

    assert stage._render_prompt_template("hello", None) == "English\nhello"
