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

"""Language policies for the twelve Granary v2 Indic targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    """Prompt and judge policy for one target language."""

    code: str
    name: str
    script: str
    script_policy: str
    capitalization_policy: str
    sarvam_supported: bool


_BRAHMIC_SCRIPT_POLICY = (
    "Native script has no uppercase/lowercase distinction. Preserve native-script letters, combining marks, virama, "
    'ZWJ, and ZWNJ exactly. Emit only canonical ASCII punctuation: ".", ",", "?", and "!".'
)
_URDU_SCRIPT_POLICY = (
    "Right-to-left Perso-Arabic script. Preserve every letter, joining control, and diacritic exactly; never "
    'transliterate. Emit only canonical ASCII punctuation: ".", ",", "?", and "!".'
)
_CAPITALIZATION_POLICY = (
    "Native-script capitalization is not applicable. In Latin spans, capitalize sentence starts and high-confidence "
    "proper nouns only; do not change spelling."
)


def _brahmic(code: str, name: str, script: str, *, sarvam_supported: bool) -> LanguageSpec:
    return LanguageSpec(
        code=code,
        name=name,
        script=script,
        script_policy=_BRAHMIC_SCRIPT_POLICY,
        capitalization_policy=_CAPITALIZATION_POLICY,
        sarvam_supported=sarvam_supported,
    )


LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "as": _brahmic("as", "Assamese", "Bengali-Assamese", sarvam_supported=False),
    "bn": _brahmic("bn", "Bengali", "Bengali-Assamese", sarvam_supported=True),
    "gu": _brahmic("gu", "Gujarati", "Gujarati", sarvam_supported=True),
    "hi": _brahmic("hi", "Hindi", "Devanagari", sarvam_supported=True),
    "kn": _brahmic("kn", "Kannada", "Kannada", sarvam_supported=True),
    "ml": _brahmic("ml", "Malayalam", "Malayalam", sarvam_supported=True),
    "mr": _brahmic("mr", "Marathi", "Devanagari", sarvam_supported=True),
    "or": _brahmic("or", "Odia", "Odia", sarvam_supported=True),
    "pa": _brahmic("pa", "Punjabi", "Gurmukhi", sarvam_supported=True),
    "ta": _brahmic("ta", "Tamil", "Tamil", sarvam_supported=True),
    "te": _brahmic("te", "Telugu", "Telugu", sarvam_supported=True),
    "ur": LanguageSpec(
        code="ur",
        name="Urdu",
        script="Perso-Arabic",
        script_policy=_URDU_SCRIPT_POLICY,
        capitalization_policy=_CAPITALIZATION_POLICY,
        sarvam_supported=False,
    ),
}


def get_language_spec(code: str) -> LanguageSpec:
    """Return the policy for a supported ISO language code."""

    normalized = code.strip().lower()
    try:
        return LANGUAGE_SPECS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(LANGUAGE_SPECS))
        msg = f"Unsupported language code {code!r}; expected one of: {supported}"
        raise ValueError(msg) from exc
