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

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nemo_curator.stages.audio.text_filtering.pnc_prompt_variants import (
    PNC_LANGUAGE_CODES,
    PNC_PROMPT_VERSIONS,
    get_pnc_prompt_configuration,
)
from nemo_curator.stages.audio.text_filtering.remote_text_llm_stage import RemoteTextLLMStage
from nemo_curator.stages.audio.text_filtering.text_llm_stage import TextLLMStage

_P0_SHA256 = "47e429e1d6ae5fb44aacae07164e6bdfc4d25cd1f3c9f7e965da93103b6ff39d"
_RECONSTRUCTION_CHECK = (
    "Before answering, silently remove only the punctuation you inserted and reverse only permitted "
    "Latin-case changes. The remaining text must reconstruct the input exactly. If it does not, "
    "return the input unchanged."
)


def _read_prompt(version: str) -> tuple[str, dict[str, str] | None]:
    prompt_file, language_blocks = get_pnc_prompt_configuration(version)
    return Path(prompt_file).read_text(encoding="utf-8").strip(), language_blocks


def test_p0_is_the_frozen_curator_reference() -> None:
    prompt_file, language_blocks = get_pnc_prompt_configuration("p0")

    assert language_blocks is None
    assert hashlib.sha256(Path(prompt_file).read_bytes()).hexdigest() == _P0_SHA256


@pytest.mark.parametrize("version", [pytest.param("p1"), pytest.param("p2"), pytest.param("p3")])
def test_variant_has_one_language_block_and_one_text_placeholder(version: str) -> None:
    prompt, language_blocks = _read_prompt(version)

    assert prompt.count("{language_block}") == 1
    assert prompt.count("{text}") == 1
    assert "{language}" not in prompt
    assert language_blocks is not None
    assert tuple(language_blocks) == PNC_LANGUAGE_CODES


def test_p2_is_p1_plus_only_the_reconstruction_check() -> None:
    p1, _ = _read_prompt("p1")
    p2, _ = _read_prompt("p2")
    expected = p1.replace(
        "\n\nReturn only the restored transcript.",
        f"\n\n{_RECONSTRUCTION_CHECK}\n\nReturn only the restored transcript.",
    )

    assert p2 == expected


def test_p3_retains_reconstruction_and_adds_boundary_rules() -> None:
    p1, _ = _read_prompt("p1")
    p2, _ = _read_prompt("p2")
    p3, _ = _read_prompt("p3")

    assert _RECONSTRUCTION_CHECK not in p1
    assert _RECONSTRUCTION_CHECK in p2
    assert _RECONSTRUCTION_CHECK in p3
    assert 'Use "?" only for a complete interrogative utterance.' not in p2
    assert 'Use "?" only for a complete interrogative utterance.' in p3
    assert 'Use "," only for a clear clause, list, or vocative boundary' in p3


def test_all_48_version_language_combinations_render() -> None:
    input_text = "unchanged transcript"

    for version in PNC_PROMPT_VERSIONS:
        prompt, language_blocks = _read_prompt(version)
        stage = TextLLMStage(prompt_text=prompt, language_blocks=language_blocks)
        stage._system_prompt = prompt

        for language_code in PNC_LANGUAGE_CODES:
            messages = stage._build_messages(input_text, {"source_lang": language_code})
            rendered = messages[0]["content"]

            assert messages[0]["role"] == "user"
            assert input_text in rendered
            assert "{text}" not in rendered
            assert "{language_block}" not in rendered
            if version == "p0":
                assert language_code in rendered
            else:
                assert f"({language_code})" in rendered


def test_p1_and_p2_exclude_language_boundary_cues() -> None:
    _, p1_blocks = _read_prompt("p1")
    _, p2_blocks = _read_prompt("p2")
    _, p3_blocks = _read_prompt("p3")

    assert p1_blocks is not None
    assert p2_blocks is not None
    assert p3_blocks is not None
    for code in PNC_LANGUAGE_CODES:
        assert p1_blocks[code] == p2_blocks[code]
        assert "Possible interrogative cues" not in p1_blocks[code]
        assert "interrogative" in p3_blocks[code].lower() or "question particles" in p3_blocks[code].lower()


def test_missing_or_unknown_language_fails_closed() -> None:
    prompt, language_blocks = _read_prompt("p1")
    stage = TextLLMStage(prompt_text=prompt, language_blocks=language_blocks)
    stage._system_prompt = prompt

    with pytest.raises(ValueError, match="No language block configured"):
        stage._build_messages("text", {"source_lang": "xx"})
    with pytest.raises(ValueError, match="No language block configured"):
        stage._build_messages("text", {})


def test_local_and_remote_paths_build_identical_messages() -> None:
    prompt, language_blocks = _read_prompt("p3")
    local = TextLLMStage(prompt_text=prompt, language_blocks=language_blocks)
    remote = RemoteTextLLMStage(prompt_text=prompt, language_blocks=language_blocks)
    local._system_prompt = prompt
    remote._system_prompt = prompt

    task_data = {"source_lang": "hi"}
    assert local._build_messages("यह एक परीक्षण है", task_data) == remote._build_messages(
        "यह एक परीक्षण है",
        task_data,
    )
