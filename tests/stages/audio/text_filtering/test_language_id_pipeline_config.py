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

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

RUN_TEXT_PIPELINE = Path(__file__).parents[4] / "examples" / "audio" / "text_processing" / "run_text_pipeline.py"

_RUNNER_SPEC = importlib.util.spec_from_file_location("curator_lid_run_text_pipeline", RUN_TEXT_PIPELINE)
if _RUNNER_SPEC is None or _RUNNER_SPEC.loader is None:
    message = f"Unable to load {RUN_TEXT_PIPELINE}"
    raise RuntimeError(message)
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)


def _parse(*extra_args: str):  # noqa: ANN202
    return _RUNNER._build_arg_parser().parse_args(
        ["--input_manifest", "input.jsonl", "--output_dir", "output", *extra_args]
    )


def _run_main_and_capture_stages(
    monkeypatch: pytest.MonkeyPatch,
    *extra_args: str,
) -> list[object]:
    captured: list[object] = []

    class _Stage:
        def __init__(self, **kwargs: object) -> None:
            self.name = str(kwargs.get("name", type(self).__name__))
            self.kwargs = kwargs

    class _Reader(_Stage):
        pass

    class _Writer(_Stage):
        pass

    class _FastText(_Stage):
        pass

    class _IndicLID(_Stage):
        pass

    class _Verifier(_Stage):
        pass

    class _TextLLM(_Stage):
        pass

    class _Pipeline:
        def __init__(self, *, name: str, stages: list[object]) -> None:
            del name
            captured[:] = stages

        def run(self, *, executor: object) -> None:
            del executor

    class _Executor:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    monkeypatch.setattr(_RUNNER, "ALMManifestReader", _Reader)
    monkeypatch.setattr(_RUNNER, "ShardedManifestWriterStage", _Writer)
    monkeypatch.setattr(_RUNNER, "FastTextLanguageIdentificationStage", _FastText)
    monkeypatch.setattr(_RUNNER, "IndicLIDLanguageIdentificationStage", _IndicLID)
    monkeypatch.setattr(_RUNNER, "LLMLanguageVerificationStage", _Verifier)
    monkeypatch.setattr(_RUNNER, "TextLLMStage", _TextLLM)
    monkeypatch.setattr(_RUNNER, "Pipeline", _Pipeline)
    monkeypatch.setattr(_RUNNER, "_resolve_shard_batches", lambda *_args: ["input.jsonl"])

    ray_data_module = types.ModuleType("nemo_curator.backends.ray_data")
    ray_data_module.RayDataExecutor = _Executor
    monkeypatch.setitem(sys.modules, "nemo_curator.backends.ray_data", ray_data_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUN_TEXT_PIPELINE),
            "--input_manifest",
            "input.jsonl",
            "--output_dir",
            "output",
            *extra_args,
        ],
    )

    _RUNNER.main()
    return captured


def test_llm_language_id_remains_the_default_contract() -> None:
    args = _parse("--enable_language_id")

    backend_config, output_key = _RUNNER._resolve_language_id_configuration(args)

    assert args.language_id_backend == "llm"
    assert backend_config is None
    assert output_key == "llm_language_prediction"


def test_direct_fasttext_backend_requires_its_model() -> None:
    args = _parse("--enable_language_id", "--language_id_backend", "fasttext")

    with pytest.raises(ValueError, match="requires --fasttext_lid_model_path"):
        _RUNNER._resolve_language_id_configuration(args)


def test_config_backend_loads_strict_per_language_routes(tmp_path: Path) -> None:
    config_path = tmp_path / "lid_backends.json"
    config_path.write_text(json.dumps({"HI": "FASTTEXT", "brx": "indiclid"}), encoding="utf-8")
    args = _parse(
        "--enable_language_id",
        "--language_id_backend",
        "config",
        "--language_id_backend_config_file",
        str(config_path),
        "--fasttext_lid_model_path",
        "lid.176.bin",
        "--indiclid_lid_model_path",
        "model_baseline_roman.bin",
    )

    backend_config, output_key = _RUNNER._resolve_language_id_configuration(args)

    assert backend_config == {"hi": "fasttext", "brx": "indiclid"}
    assert output_key == "llm_language_prediction"


def test_config_backend_rejects_llm_values(tmp_path: Path) -> None:
    config_path = tmp_path / "lid_backends.json"
    config_path.write_text('{"hi":"llm"}', encoding="utf-8")
    args = _parse(
        "--enable_language_id",
        "--language_id_backend",
        "config",
        "--language_id_backend_config_file",
        str(config_path),
    )

    with pytest.raises(ValueError, match="must be one of"):
        _RUNNER._resolve_language_id_configuration(args)


def test_language_id_options_require_enable_flag() -> None:
    args = _parse("--language_id_backend", "indiclid", "--indiclid_lid_model_path", "model.bin")

    with pytest.raises(ValueError, match="require --enable_language_id"):
        _RUNNER._resolve_language_id_configuration(args)


def test_main_validates_language_id_options_before_no_stages_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUN_TEXT_PIPELINE),
            "--input_manifest",
            "input.jsonl",
            "--output_dir",
            "output",
            "--language_id_backend",
            "indiclid",
            "--indiclid_lid_model_path",
            "model.bin",
        ],
    )

    with pytest.raises(ValueError, match="require --enable_language_id"):
        _RUNNER.main()


def test_default_llm_language_id_assembly_is_backward_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = _run_main_and_capture_stages(monkeypatch, "--enable_language_id", "--enable_pnc")

    assert [stage.name for stage in stages] == [
        "_Reader",
        "PnCRestoration",
        "LanguageID",
        "_Verifier",
        "_Writer",
    ]
    assert stages[2].kwargs["text_key"] == "pnc_text"
    assert stages[2].kwargs["output_text_key"] == "llm_language_prediction"


def test_llm_language_id_first_assembly_precedes_pnc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stages = _run_main_and_capture_stages(
        monkeypatch,
        "--enable_language_id",
        "--language_id_first",
        "--enable_pnc",
    )

    assert [stage.name for stage in stages] == [
        "_Reader",
        "LanguageID",
        "_Verifier",
        "PnCRestoration",
        "_Writer",
    ]
    assert stages[1].kwargs["text_key"] == "abbreviated_text"


@pytest.mark.parametrize(
    ("backend", "model_flag", "model_path", "expected_stage"),
    [
        ("fasttext", "--fasttext_lid_model_path", "lid.176.bin", "_FastText"),
        ("indiclid", "--indiclid_lid_model_path", "indiclid.bin", "_IndicLID"),
    ],
)
def test_direct_cpu_language_id_assembly_precedes_pnc(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    model_flag: str,
    model_path: str,
    expected_stage: str,
) -> None:
    stages = _run_main_and_capture_stages(
        monkeypatch,
        "--enable_language_id",
        "--enable_pnc",
        "--language_id_backend",
        backend,
        model_flag,
        model_path,
    )

    assert [type(stage).__name__ for stage in stages] == [
        "_Reader",
        expected_stage,
        "_Verifier",
        "_TextLLM",
        "_Writer",
    ]
    assert stages[1].name == "LanguageID"
    assert stages[3].name == "PnCRestoration"
    assert stages[1].kwargs["text_key"] == "abbreviated_text"
    assert stages[1].kwargs["output_text_key"] == "llm_language_prediction"


def test_config_assembly_routes_rows_through_both_cpu_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "lid_backends.json"
    config_path.write_text('{"hi":"fasttext","brx":"indiclid"}', encoding="utf-8")

    stages = _run_main_and_capture_stages(
        monkeypatch,
        "--enable_language_id",
        "--language_id_backend",
        "config",
        "--language_id_backend_config_file",
        str(config_path),
        "--fasttext_lid_model_path",
        "lid.176.bin",
        "--indiclid_lid_model_path",
        "indiclid.bin",
    )

    assert [type(stage).__name__ for stage in stages] == [
        "_Reader",
        "_FastText",
        "_IndicLID",
        "_Verifier",
        "_Writer",
    ]
    assert stages[1].kwargs["backend_by_language"] == {"hi": "fasttext", "brx": "indiclid"}
    assert stages[2].kwargs["backend_by_language"] == {"hi": "fasttext", "brx": "indiclid"}
