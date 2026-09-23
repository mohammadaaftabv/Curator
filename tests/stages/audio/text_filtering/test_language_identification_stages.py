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

from typing import TYPE_CHECKING

import pytest

from nemo_curator.stages.audio.text_filtering.fasttext_language_identification import (
    FastTextLanguageIdentificationStage,
)
from nemo_curator.stages.audio.text_filtering.indiclid_language_identification import (
    INDICLID_LABEL_TO_LANGUAGE,
    IndicLIDLanguageIdentificationStage,
)
from nemo_curator.stages.audio.text_filtering.llm_language_verification import LLMLanguageVerificationStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeFastTextModel:
    def __init__(self, labels: Sequence[str], probabilities: Sequence[float]) -> None:
        self._labels = list(labels)
        self._probabilities = list(probabilities)
        self.calls: list[tuple[list[str], int]] = []

    def predict(self, texts: list[str], k: int) -> tuple[list[list[str]], list[list[float]]]:
        self.calls.append((list(texts), k))
        return (
            [[self._labels[index]] for index in range(len(texts))],
            [[self._probabilities[index]] for index in range(len(texts))],
        )


def _attach_model(
    stage: FastTextLanguageIdentificationStage | IndicLIDLanguageIdentificationStage,
    model: _FakeFastTextModel,
    labels: set[str],
) -> None:
    stage._model = model
    stage._model_labels = frozenset(labels)


def test_fasttext_matches_eval_preprocessing_and_llm_row_contract() -> None:
    stage = FastTextLanguageIdentificationStage(model_path="unused.bin")
    model = _FakeFastTextModel(["__label__hi"], [1.0000499487])
    _attach_model(stage, model, {"hi"})
    task = AudioTask(
        data={
            "abbreviated_text": "  नमस्ते\n\t दुनिया  ",
            "source_lang": "hi",
            "_skipme": "",
            "additional_notes": {"Existing": "kept"},
        }
    )

    result = stage.process(task)

    assert model.calls == [(["नमस्ते दुनिया"], 1)]
    assert result.data == {
        "abbreviated_text": "  नमस्ते\n\t दुनिया  ",
        "source_lang": "hi",
        "_skipme": "",
        "llm_language_prediction": "hi",
        "additional_notes": {
            "Existing": "kept",
            "LanguageID": "applied (modified)",
        },
    }


def test_indiclid_matches_eval_preprocessing_and_maps_raw_label() -> None:
    stage = IndicLIDLanguageIdentificationStage(model_path="unused.bin")
    model = _FakeFastTextModel(["__label__kas_Arab"], [0.99])
    _attach_model(stage, model, {"kas_Arab"})
    task = AudioTask(data={"abbreviated_text": "  first\r\nsecond  ", "source_lang": "ks", "_skipme": ""})

    result = stage.process(task)

    assert model.calls == [(["  first  second  "], 1)]
    assert result.data["llm_language_prediction"] == "ks"
    assert result.data["additional_notes"] == {"LanguageID": "applied (modified)"}
    assert set(result.data) == {
        "abbreviated_text",
        "source_lang",
        "_skipme",
        "llm_language_prediction",
        "additional_notes",
    }


def test_cpu_prediction_uses_existing_llm_verifier_without_schema_changes() -> None:
    stage = IndicLIDLanguageIdentificationStage(model_path="unused.bin")
    model = _FakeFastTextModel(["__label__mai_Deva"], [0.9])
    _attach_model(stage, model, {"mai_Deva"})
    task = AudioTask(data={"abbreviated_text": "यह हिंदी वाक्य है", "source_lang": "hi", "_skipme": ""})

    stage.process(task)
    LLMLanguageVerificationStage().process(task)

    assert task.data["llm_language_prediction"] == "mai"
    assert task.data["_skipme"] == "Wrong language:LLMLanguageVerification"
    assert task.data["additional_notes"] == {
        "LanguageID": "applied (modified)",
        "LLMLanguageVerification": "wrong language (predicted=Maithili-mai, expected=Hindi-hi)",
    }
    assert "language_id_raw_label" not in task.data
    assert "language_id_probability" not in task.data
    assert "language_id_backend" not in task.data


def test_config_routing_runs_exactly_one_backend_per_row() -> None:
    backend_config = {"hi": "fasttext", "brx": "indiclid"}
    fasttext_stage = FastTextLanguageIdentificationStage(
        model_path="unused.bin",
        backend_by_language=backend_config,
    )
    indiclid_stage = IndicLIDLanguageIdentificationStage(
        model_path="unused.bin",
        backend_by_language=backend_config,
    )
    fasttext_model = _FakeFastTextModel(["__label__hi"], [0.9])
    indiclid_model = _FakeFastTextModel(["__label__brx_Deva"], [0.9])
    _attach_model(fasttext_stage, fasttext_model, {"hi"})
    _attach_model(indiclid_stage, indiclid_model, {"brx_Deva"})
    tasks = [
        AudioTask(data={"abbreviated_text": "हिंदी", "source_lang": "hi", "_skipme": ""}),
        AudioTask(data={"abbreviated_text": "बर' राव", "source_lang": "brx", "_skipme": ""}),
    ]

    fasttext_stage.process_batch(tasks)
    indiclid_stage.process_batch(tasks)

    assert fasttext_model.calls == [(["हिंदी"], 1)]
    assert indiclid_model.calls == [(["बर' राव"], 1)]
    assert [task.data["llm_language_prediction"] for task in tasks] == ["hi", "brx"]
    assert all(task.data["additional_notes"] == {"LanguageID": "applied (modified)"} for task in tasks)


def test_config_routing_fails_closed_for_unmapped_language() -> None:
    stage = IndicLIDLanguageIdentificationStage(
        model_path="unused.bin",
        backend_by_language={"brx": "indiclid"},
    )
    model = _FakeFastTextModel(["__label__brx_Deva"], [0.9])
    _attach_model(stage, model, {"brx_Deva"})
    task = AudioTask(data={"abbreviated_text": "हिंदी", "source_lang": "hi", "_skipme": ""})

    with pytest.raises(ValueError, match="No language-ID backend configured"):
        stage.process(task)


def test_fasttext_rejects_language_without_exact_lid176_label() -> None:
    stage = FastTextLanguageIdentificationStage(model_path="unused.bin")
    model = _FakeFastTextModel(["__label__hi"], [0.9])
    _attach_model(stage, model, {"hi"})
    task = AudioTask(data={"abbreviated_text": "बर' राव", "source_lang": "brx", "_skipme": ""})

    with pytest.raises(ValueError, match=r"no exact label.*route this language to IndicLID"):
        stage.process(task)


def test_indiclid_mapping_is_the_exact_26_label_inventory() -> None:
    assert len(INDICLID_LABEL_TO_LANGUAGE) == 26
    assert INDICLID_LABEL_TO_LANGUAGE["kas_Arab"] == "ks"
    assert INDICLID_LABEL_TO_LANGUAGE["kas_Deva"] == "ks"
    assert INDICLID_LABEL_TO_LANGUAGE["mni_Beng"] == "mni"
    assert INDICLID_LABEL_TO_LANGUAGE["mni_Meti"] == "mni"
    assert INDICLID_LABEL_TO_LANGUAGE["eng_Latn"] == "en"


def test_checkpoint_inventory_validation_matches_granary_evaluators() -> None:
    fasttext_stage = FastTextLanguageIdentificationStage(model_path="unused.bin")
    official_shape = set(fasttext_stage.granary_exact_languages) | {"gom"}
    official_shape.update(f"other-{index}" for index in range(176 - len(official_shape)))
    fasttext_stage._validate_model_labels(frozenset(official_shape))

    wrong_shape = (official_shape - {"hi"}) | {"brx"}
    with pytest.raises(ValueError, match="Granary language inventory mismatch"):
        fasttext_stage._validate_model_labels(frozenset(wrong_shape))

    indiclid_stage = IndicLIDLanguageIdentificationStage(model_path="unused.bin")
    indiclid_stage._validate_model_labels(frozenset(INDICLID_LABEL_TO_LANGUAGE))
    with pytest.raises(ValueError, match="IndicLID-FTN label inventory mismatch"):
        indiclid_stage._validate_model_labels(frozenset(INDICLID_LABEL_TO_LANGUAGE) - {"hin_Deva"})


def test_empty_batch_does_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = FastTextLanguageIdentificationStage(model_path="unused.bin")

    def _unexpected_setup() -> None:
        pytest.fail("setup must not run for an empty batch")

    monkeypatch.setattr(stage, "setup", _unexpected_setup)

    assert stage.process_batch([]) == []


def test_flagged_and_empty_rows_match_text_llm_stage_edits() -> None:
    stage = FastTextLanguageIdentificationStage(model_path="unused.bin")
    model = _FakeFastTextModel([], [])
    _attach_model(stage, model, {"hi"})
    flagged = AudioTask(data={"abbreviated_text": "हिंदी", "source_lang": "hi", "_skipme": "upstream reason"})
    empty = AudioTask(data={"abbreviated_text": "  ", "source_lang": "hi", "_skipme": ""})
    missing_text_and_language = AudioTask(data={"_skipme": ""})
    flagged_without_language = AudioTask(data={"abbreviated_text": "text", "_skipme": "upstream reason"})

    stage.process_batch([flagged, empty, missing_text_and_language, flagged_without_language])

    assert model.calls == []
    assert flagged.data["llm_language_prediction"] == ""
    assert flagged.data["additional_notes"] == {"LanguageID": "skipped (flagged)"}
    assert empty.data["llm_language_prediction"] == "  "
    assert empty.data["additional_notes"] == {"LanguageID": "skipped (empty)"}
    assert missing_text_and_language.data["llm_language_prediction"] == ""
    assert missing_text_and_language.data["additional_notes"] == {"LanguageID": "skipped (empty)"}
    assert flagged_without_language.data["llm_language_prediction"] == ""
    assert flagged_without_language.data["additional_notes"] == {"LanguageID": "skipped (flagged)"}
