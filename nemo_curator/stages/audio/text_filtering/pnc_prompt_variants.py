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

"""Versioned PnC prompts with one per-row language substitution block."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

_PROMPT_DIR = Path(__file__).parent / "prompts"
_LANGUAGE_PROFILES = _PROMPT_DIR / "pnc_language_profiles.json"

PNC_PROMPT_VERSIONS: Final[tuple[str, ...]] = ("p0", "p1", "p2", "p3")
PNC_LANGUAGE_CODES: Final[tuple[str, ...]] = (
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
    "ur",
)

_PROMPT_FILES: Final[dict[str, Path]] = {
    "p0": _PROMPT_DIR / "pnc_prompt.md",
    "p1": _PROMPT_DIR / "pnc_prompt_p1.md",
    "p2": _PROMPT_DIR / "pnc_prompt_p2.md",
    "p3": _PROMPT_DIR / "pnc_prompt_p3.md",
}


def _load_language_profiles() -> dict[str, dict[str, str]]:
    profiles = json.loads(_LANGUAGE_PROFILES.read_text(encoding="utf-8"))
    if not isinstance(profiles, dict) or tuple(profiles) != PNC_LANGUAGE_CODES:
        msg = (
            f"Expected PnC language profiles in order {PNC_LANGUAGE_CODES}, "
            f"found {tuple(profiles) if isinstance(profiles, dict) else type(profiles).__name__}"
        )
        raise ValueError(msg)

    for code, profile in profiles.items():
        if not isinstance(profile, dict):
            msg = f"PnC language profile {code!r} must be an object"
            raise TypeError(msg)
        missing = {"name", "preservation", "boundary_cues"} - profile.keys()
        if missing:
            msg = f"PnC language profile {code!r} is missing fields: {sorted(missing)}"
            raise ValueError(msg)
        if not all(isinstance(profile[key], str) and profile[key].strip() for key in profile):
            msg = f"PnC language profile {code!r} contains an empty or non-string value"
            raise ValueError(msg)
    return profiles


def get_pnc_prompt_configuration(prompt_version: str) -> tuple[str, dict[str, str] | None]:
    """Return the prompt file and optional per-language block map.

    P0 deliberately returns no language blocks because it must remain the
    byte-for-byte Curator reference prompt. P1 and P2 use only each language's
    preservation block. P3 adds the boundary-cue suffix inside the same single
    ``{language_block}`` placeholder.
    """

    version = prompt_version.lower()
    if version not in PNC_PROMPT_VERSIONS:
        msg = f"Unknown PnC prompt version {prompt_version!r}; choose from {PNC_PROMPT_VERSIONS}"
        raise ValueError(msg)

    prompt_path = _PROMPT_FILES[version]
    if version == "p0":
        return str(prompt_path), None

    profiles = _load_language_profiles()
    language_blocks = {
        code: (
            profile["preservation"]
            if version in ("p1", "p2")
            else f"{profile['preservation']}\n{profile['boundary_cues']}"
        )
        for code, profile in profiles.items()
    }
    return str(prompt_path), language_blocks
