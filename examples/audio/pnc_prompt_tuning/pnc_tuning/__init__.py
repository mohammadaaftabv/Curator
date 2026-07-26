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

"""Utilities for controlled Indic PNC prompt experiments."""

from pnc_tuning.languages import LANGUAGE_SPECS, LanguageSpec, get_language_spec
from pnc_tuning.validation import ValidationResult, validate_preservation

__all__ = [
    "LANGUAGE_SPECS",
    "LanguageSpec",
    "ValidationResult",
    "get_language_spec",
    "validate_preservation",
]
