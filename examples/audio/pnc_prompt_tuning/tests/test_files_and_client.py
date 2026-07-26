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

# ruff: noqa: ANN202, PLR2004, S101

"""Repository-contract and request-cache tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pnc_tuning.config import NvidiaConfig, load_config
from pnc_tuning.nvidia_client import NvidiaClient

_TOOLKIT = Path(__file__).resolve().parents[1]
_REPO = _TOOLKIT.parents[2]


def test_example_config_and_prompt_registry_are_complete() -> None:
    config = load_config(_TOOLKIT / "config.example.json")
    assert len(config.sampling.languages) == 12
    assert len(config.judge_roles) == 3
    assert config.allowed_punctuation == ".,?!"
    assert config.common_yaml_path == _REPO / "tutorials/audio/granary_v2_postprocessing/common.yaml"
    registry = __import__("json").loads((_TOOLKIT / "prompt_registry.json").read_text(encoding="utf-8"))
    assert set(registry) == {"p0", "p1", "p2", "p3", "p4"}
    for value in registry.values():
        assert (_TOOLKIT / value["default"]).is_file()
        for override in value.get("language_overrides", {}).values():
            assert (_TOOLKIT / override).is_file()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("allowed_punctuation", ".,?!:;", "insertion set is fixed"),
        ("common_yaml_path", "other.yaml", "common_yaml_path is fixed"),
    ],
)
def test_config_rejects_alternate_pnc_profiles(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    raw = json.loads((_TOOLKIT / "config.example.json").read_text(encoding="utf-8"))
    raw["work_root"] = str(tmp_path)
    raw[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_p0_is_exact_repository_prompt() -> None:
    production = _REPO / "nemo_curator/stages/audio/text_filtering/prompts/pnc_prompt.md"
    baseline = (_TOOLKIT / "prompts/p0_current.md").read_text(encoding="utf-8")
    assert baseline.rstrip("\n") == production.read_text(encoding="utf-8").rstrip("\n")


def test_nvidia_chat_cache_avoids_duplicate_calls(tmp_path: Path) -> None:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="नमस्ते."),
                )
            ],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 3}),
            model="served-model",
        )

    client = NvidiaClient(NvidiaConfig(), work_root=tmp_path)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    arguments = {
        "model": "requested-model",
        "messages": [{"role": "user", "content": "punctuate"}],
        "max_tokens": 10,
    }
    first = client.chat(**arguments)
    second = client.chat(**arguments)
    assert first.content == second.content
    assert not first.cached
    assert second.cached
    assert len(calls) == 1
