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

"""CLI smoke tests for offline phases and output-boundary enforcement."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pnc_tuning.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def _write_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": str(index),
                    "audio_item_id": f"group-{index}",
                    "language": "hi",
                    "text": f"पाठ {index}",
                    "actual_duration": 20,
                },
                ensure_ascii=False,
            )
            for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "work_root": str(tmp_path / "work"),
                "input_paths": [str(source)],
                "sampling": {
                    "languages": ["hi"],
                    "quotas": {
                        "smoke": 2,
                        "development": 0,
                        "calibration": 0,
                        "challenge": 0,
                        "test": 0,
                    },
                },
                "judge_roles": [
                    {
                        "name": "broad",
                        "candidate_models": ["judge"],
                        "supported_languages": ["*"],
                        "required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config


def test_build_subset_cli_writes_only_below_work_root(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    assert (
        main(
            [
                "build-subset",
                "--config",
                str(config),
                "--output",
                "artifacts/subset.jsonl",
                "--report",
                "artifacts/report.json",
            ]
        )
        == 0
    )
    output = tmp_path / "work/artifacts/subset.jsonl"
    assert output.is_file()
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_cli_rejects_output_escape(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "build-subset",
                "--config",
                str(config),
                "--output",
                "../escape.jsonl",
            ]
        )
    assert exc.value.code == 2
    assert not (tmp_path / "escape.jsonl").exists()
