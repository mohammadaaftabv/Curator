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

"""Safe JSON/JSONL I/O with a mandatory output-root boundary."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

DEFAULT_DRACO_WORK_ROOT = Path("/lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning")


def lexical_absolute_path(path: str | Path) -> Path:
    """Return an absolute normalized path without dereferencing symlinks."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def ensure_within_work_root(path: str | Path, work_root: str | Path) -> Path:
    """Return a lexical output path after lexical and physical containment checks."""

    root = lexical_absolute_path(work_root)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = lexical_absolute_path(candidate)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        msg = f"Refusing to write outside work root {root}: {candidate}"
        raise ValueError(msg) from exc
    if candidate == root:
        msg = f"Refusing to use the work-root directory itself as an output file: {root}"
        raise ValueError(msg)

    physical_root = root.resolve()
    physical_candidate = candidate.resolve()
    try:
        physical_candidate.relative_to(physical_root)
    except ValueError as exc:
        msg = (
            f"Refusing output path whose symlink target escapes work root {root}: "
            f"{candidate} -> {physical_candidate}"
        )
        raise ValueError(msg) from exc
    return candidate


def expand_input_paths(patterns: Iterable[str | Path]) -> list[Path]:
    """Expand read-only input globs without modifying the source tree."""

    paths: set[Path] = set()
    for pattern in patterns:
        value = str(Path(pattern).expanduser())
        matches = glob.glob(value)
        if not matches and Path(value).is_file():
            matches = [value]
        for match in matches:
            path = Path(match)
            if path.is_file():
                paths.add(path.resolve())
    if not paths:
        msg = "No readable input files matched the supplied paths"
        raise FileNotFoundError(msg)
    return sorted(paths)


def iter_jsonl(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from one or more JSONL files."""

    for path in paths:
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = f"Invalid JSON in {source}:{line_number}: {exc.msg}"
                    raise ValueError(msg) from exc
                if not isinstance(value, dict):
                    msg = f"Expected a JSON object in {source}:{line_number}"
                    raise TypeError(msg)
                yield value


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        msg = f"Expected a JSON object in {path}"
        raise TypeError(msg)
    return value


def _atomic_target(path: str | Path, work_root: str | Path) -> tuple[Path, Path]:
    target = ensure_within_work_root(path, work_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    return target, temporary


def atomic_write_json(path: str | Path, value: object, work_root: str | Path) -> Path:
    """Atomically write JSON next to the destination, never through ``/tmp``."""

    target, temporary = _atomic_target(path, work_root)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    work_root: str | Path,
) -> Path:
    """Atomically write JSONL next to the destination."""

    target, temporary = _atomic_target(path, work_root)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def stable_json_hash(value: object) -> str:
    """Return a deterministic SHA-256 hash for JSON-compatible content."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
