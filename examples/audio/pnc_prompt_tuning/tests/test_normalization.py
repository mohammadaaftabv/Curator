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

# ruff: noqa: S101

"""Tests for the exact Granary v2 ``common.yaml`` acceptance contract."""

from __future__ import annotations

import pytest
from pnc_tuning.config import DEFAULT_COMMON_YAML_PATH
from pnc_tuning.normalization import CommonYamlNormalizer


@pytest.fixture(scope="module")
def normalizer() -> CommonYamlNormalizer:
    return CommonYamlNormalizer.load(DEFAULT_COMMON_YAML_PATH)


def test_normalizer_matches_regex_stage_semantics(normalizer: CommonYamlNormalizer) -> None:
    value = "  हिन्दी।  اُردُو؟ [noise] (aside) foo؛bar!!!  "
    assert normalizer.normalize(value) == "हिन्दी. اُردُو? foo,bar!"


def test_current_common_yaml_fails_closed_for_odia(normalizer: CommonYamlNormalizer) -> None:
    report = normalizer.contract_report(["as", "or", "ur"])
    assert report["language_status"]["as"]["preserved"]
    assert report["language_status"]["ur"]["preserved"]
    assert not report["language_status"]["or"]["preserved"]
    assert report["incompatible_languages"] == ["or"]
    with pytest.raises(RuntimeError, match=r"or \(Odia\)"):
        normalizer.require_language_coverage(["as", "or", "ur"])


def test_no_unconfigured_punctuation_profile_is_needed(normalizer: CommonYamlNormalizer) -> None:
    assert normalizer.normalize("क्या؛ हाँ؟") == "क्या, हाँ?"
