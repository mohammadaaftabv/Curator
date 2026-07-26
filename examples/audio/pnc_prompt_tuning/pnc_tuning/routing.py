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

"""Runtime model discovery and heterogeneous judge-panel routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pnc_tuning.config import ModelRoleConfig


@dataclass(frozen=True)
class ResolvedRoute:
    """One role resolved to an endpoint model."""

    role: str
    model_id: str
    evaluate_language_quality: bool

    def to_dict(self) -> dict[str, str | bool]:
        """Serialize the route."""

        return asdict(self)


def role_supports_language(role: ModelRoleConfig, language: str) -> bool:
    """Return whether a role is configured for a language."""

    return "*" in role.supported_languages or language in role.supported_languages


def resolve_panel(
    *,
    available_models: set[str],
    roles: tuple[ModelRoleConfig, ...],
    language: str,
    allow_partial: bool = False,
) -> tuple[ResolvedRoute, ...]:
    """Resolve the first available candidate for each applicable role."""

    resolved = []
    missing_required = []
    for role in roles:
        if not role_supports_language(role, language):
            continue
        model = next((candidate for candidate in role.candidate_models if candidate in available_models), None)
        if model is None:
            if role.required:
                missing_required.append(role.name)
            continue
        resolved.append(
            ResolvedRoute(
                role=role.name,
                model_id=model,
                evaluate_language_quality=role.evaluate_language_quality,
            )
        )
    if missing_required and not allow_partial:
        missing = ", ".join(missing_required)
        msg = f"Required judge roles are unavailable for {language}: {missing}"
        raise RuntimeError(msg)
    if not resolved:
        msg = f"No judge route is available for language {language}"
        raise RuntimeError(msg)
    return tuple(resolved)
