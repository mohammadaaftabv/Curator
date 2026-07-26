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

"""Exact Granary v2 ``common.yaml`` normalization and contract checks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pnc_tuning.languages import LANGUAGE_SPECS

_LANGUAGE_PROBES = {
    "as": "অসমীয়া ভাষা",
    "bn": "বাংলা ভাষা",
    "gu": "ગુજરાતી ભાષા",
    "hi": "हिन्दी भाषा",
    "kn": "ಕನ್ನಡ ಭಾಷೆ",
    "ml": "മലയാളം ഭാഷ",
    "mr": "मराठी भाषा",
    "or": "ଓଡ଼ିଆ ଭାଷା",
    "pa": "ਪੰਜਾਬੀ ਭਾਸ਼ਾ",
    "ta": "தமிழ் மொழி",
    "te": "తెలుగు భాష",
    "ur": "اُردُو زبان",
}


@dataclass(frozen=True)
class RegexRule:
    """One compiled substitution from the authoritative YAML file."""

    pattern: re.Pattern[str]
    replacement: str
    count: int


@dataclass(frozen=True)
class CommonYamlNormalizer:
    """Apply the same rule order and final whitespace cleanup as Curator."""

    path: Path
    sha256: str
    rules: tuple[RegexRule, ...]

    @classmethod
    def load(cls, path: str | Path) -> CommonYamlNormalizer:
        """Load and validate the authoritative ``common.yaml`` rule list."""

        resolved = Path(path).expanduser().resolve()
        payload = resolved.read_bytes()
        raw_rules = yaml.safe_load(payload.decode("utf-8"))
        if not isinstance(raw_rules, list) or not raw_rules:
            msg = f"common.yaml must contain a non-empty list of regex rules: {resolved}"
            raise ValueError(msg)

        rules = []
        for index, value in enumerate(raw_rules):
            if not isinstance(value, dict):
                msg = f"common.yaml rule {index} must be an object"
                raise TypeError(msg)
            if set(value) - {"pattern", "repl", "count"}:
                unexpected = ", ".join(sorted(set(value) - {"pattern", "repl", "count"}))
                msg = f"common.yaml rule {index} has unsupported keys: {unexpected}"
                raise ValueError(msg)
            pattern = value.get("pattern")
            replacement = value.get("repl")
            count = value.get("count", 0)
            if not isinstance(pattern, str) or not isinstance(replacement, str):
                msg = f"common.yaml rule {index} requires string pattern and repl values"
                raise TypeError(msg)
            if not isinstance(count, int) or count < 0:
                msg = f"common.yaml rule {index} count must be a non-negative integer"
                raise TypeError(msg)
            rules.append(RegexRule(pattern=re.compile(pattern), replacement=replacement, count=count))

        return cls(
            path=resolved,
            sha256=hashlib.sha256(payload).hexdigest(),
            rules=tuple(rules),
        )

    def normalize(self, text: str) -> str:
        """Mirror ``RegexSubstitutionStage`` exactly for one string."""

        value = f" {text} "
        for rule in self.rules:
            value = rule.pattern.sub(rule.replacement, value, count=rule.count)
        return re.sub(r"\s+", " ", value).strip()

    def contract_report(self, languages: tuple[str, ...] | list[str]) -> dict[str, Any]:
        """Report whether ``common.yaml`` preserves each configured native script."""

        statuses = {}
        for language in languages:
            if language not in LANGUAGE_SPECS:
                msg = f"Unsupported language in common.yaml contract check: {language}"
                raise ValueError(msg)
            probe = _LANGUAGE_PROBES[language]
            normalized = self.normalize(probe)
            statuses[language] = {
                "name": LANGUAGE_SPECS[language].name,
                "probe": probe,
                "normalized_probe": normalized,
                "preserved": normalized == probe,
            }
        incompatible = [language for language, value in statuses.items() if not value["preserved"]]
        return {
            "profile": "tutorials/audio/granary_v2_postprocessing/common.yaml",
            "path": str(self.path),
            "sha256": self.sha256,
            "rule_count": len(self.rules),
            "generator_insertions": ".,?!",
            "language_status": statuses,
            "incompatible_languages": incompatible,
            "compatible": not incompatible,
        }

    def require_language_coverage(self, languages: tuple[str, ...] | list[str]) -> None:
        """Fail closed when the authoritative profile would destroy a target script."""

        report = self.contract_report(languages)
        incompatible = report["incompatible_languages"]
        if incompatible:
            labels = ", ".join(
                f"{language} ({LANGUAGE_SPECS[language].name})" for language in incompatible
            )
            msg = (
                "The authoritative common.yaml does not preserve the configured target "
                f"language(s): {labels}. Update and repin common.yaml itself; do not add a "
                "second normalization policy."
            )
            raise RuntimeError(msg)
