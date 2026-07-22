# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import argparse

import pytest

from nemo_curator.stages.audio.text_filtering.sampling import parse_stage_sampling_config, sampling_for_stage


def test_stage_sampling_config_accepts_independent_stage_values() -> None:
    config = parse_stage_sampling_config(
        '{"pnc":{"temperature":0,"top_p":1},"speech_qa":{"temperature":0.7,"top_p":0.95}}'
    )
    assert sampling_for_stage(
        stage_sampling_config=config, stage="pnc", default_temperature=0.2, default_top_p=0.9
    ) == {"temperature": 0.0, "top_p": 1.0}
    assert sampling_for_stage(
        stage_sampling_config=config, stage="speech_qa", default_temperature=0.2, default_top_p=0.9
    ) == {"temperature": 0.7, "top_p": 0.95}
    assert sampling_for_stage(
        stage_sampling_config=config, stage="tn", default_temperature=0.2, default_top_p=0.9
    ) == {"temperature": 0.2, "top_p": 0.9}


def test_special_stage_defaults_are_preserved_without_override() -> None:
    assert sampling_for_stage(
        stage_sampling_config={}, stage="context_asr", default_temperature=0.1, default_top_p=0.95
    ) == {"temperature": 0.1, "top_p": 0.95}


@pytest.mark.parametrize(
    "value",
    [
        '{"unknown":{"temperature":0}}',
        '{"pnc":{"temperature":-1}}',
        '{"pnc":{"top_p":0}}',
        '{"pnc":{"seed":1}}',
    ],
)
def test_stage_sampling_config_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_stage_sampling_config(value)
