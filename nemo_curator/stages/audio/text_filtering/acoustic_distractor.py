# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Acoustic distractor generation stage for contextual ASR (CPU-only).

Appends phonetically-similar words/phrases to the ``distractor_terms``
list produced by :class:`ContextualASRExtractionStage`.  Semantic
distractors from the LLM teach the model not to copy hints blindly;
acoustic distractors additionally teach the model to disambiguate
phonetically-confusable words.

For each entity in ``fine_context_terms``:

1. G2P the full phrase via :mod:`phonemizer` (espeak-ng backend) into an
   IPA phoneme token list.
2. Compute Normalized Phonetic Distance (NPD)
   ``editdistance(query, candidate) / len(query)`` against every entry in
   a precomputed phoneme vocabulary loaded at ``setup()``.
3. Filter by ``min_npd < NPD < max_npd`` to drop both trivially-identical
   matches and too-distant ones.
4. Exclude vocabulary entries already present in ``fine_context_terms``
   or the existing (semantic) ``distractor_terms``.
5. Take the top ``per_entity_top_k`` candidates by ascending NPD.

Candidates collected across all entities of a sample are merged,
deduplicated, sorted by NPD, capped at ``max_acoustic_distractors``, and
appended to ``distractor_terms``.  The combined list is then capped at
``max_total_distractors``.

The precomputed phoneme vocabulary is produced offline by
``scripts/build_phoneme_vocab.py``.  Build it once per target language.
``phoneme_vocab_path`` may point either at a single ``{word: [phonemes]}``
JSON file (one language, used for every sample) or at a **directory** of
``phoneme_vocab_<lang>.json`` files — in directory mode every file is loaded
and the per-sample ``source_lang`` selects the matching vocab, so a single
stage instance can serve a multi-language job.

This stage is CPU-only — no GPU or LLM is required at runtime.

Language handling
-----------------
The stage needs an espeak-ng language code to G2P entities.  In order
of precedence:

1. ``language`` (if set on the stage) — used for all samples.
2. ``source_lang`` from the manifest (display name like ``"English"``
   or ISO-639-1 code like ``"en"``) — mapped to an espeak code.
3. ``default_source_lang`` — used when neither of the above resolves.

When a sample's language cannot be mapped to a supported espeak code,
the stage records ``unsupported_language`` in the additional_notes and
leaves the existing ``distractor_terms`` untouched.

Dependencies
------------
- ``phonemizer`` (``pip install phonemizer``)
- ``espeak-ng`` system package (``apt install espeak-ng``)
- ``editdistance`` (already a transitive project dependency)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.stages.audio.pipeline_utils import LANG_CODE_TO_NAME, set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

try:
    from phonemizer.backend import EspeakBackend as _EspeakBackend
    from phonemizer.separator import Separator as _Separator

    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    _EspeakBackend = None  # type: ignore[assignment]
    _Separator = None  # type: ignore[assignment]

try:
    import editdistance as _editdistance

    EDITDISTANCE_AVAILABLE = True
except ImportError:
    EDITDISTANCE_AVAILABLE = False
    _editdistance = None  # type: ignore[assignment]


# Display-name → espeak-ng language code.  Covers the 32 languages
# supported by the design (see ``plan/acoustic_distractors_plan.md``).
_LANG_TO_ESPEAK: dict[str, str] = {
    "Arabic": "ar",
    "Bulgarian": "bg",
    "Chinese": "cmn",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en-us",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr-fr",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Maltese": "mt",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swedish": "sv",
    "Thai": "th",
    "Ukrainian": "uk",
}

_WORD_BOUNDARY_MARKER = "|"


def _normalize_lang_to_espeak(value: Any) -> str | None:  # noqa: ANN401
    """Map a manifest ``source_lang`` value to an espeak-ng language code.

    Accepts either a display name (``"English"``) or an ISO-639-1 code
    (``"en"``).  Returns ``None`` when no mapping is known.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in _LANG_TO_ESPEAK:
        return _LANG_TO_ESPEAK[text]
    title = text.title()
    if title in _LANG_TO_ESPEAK:
        return _LANG_TO_ESPEAK[title]
    display = LANG_CODE_TO_NAME.get(text.lower())
    if display and display in _LANG_TO_ESPEAK:
        return _LANG_TO_ESPEAK[display]
    return None


def _npd(query: list[str], candidate: list[str]) -> float:
    if not query:
        return 1.0
    if _editdistance is None:
        return 1.0
    return _editdistance.eval(query, candidate) / len(query)


# Extra neighbors cached beyond per_entity_top_k so per-utterance exclusion (the
# entity's own terms + existing distractors) can be applied at the call site while
# still leaving enough candidates — lets the NN result be cached per (language, entity).
_SEARCH_CACHE_MARGIN = 16


def _vocab_search(
    query_phonemes: list[str],
    len_index: dict[int, list[tuple[str, list[str]]]],
    *,
    min_npd: float,
    max_npd: float,
    top_k: int,
) -> list[tuple[str, float]]:
    """Nearest-neighbor search over a length-bucketed phoneme vocab.

    NPD = editdistance(query, candidate) / len(query), and edit distance is at least
    the length difference, so a candidate can satisfy ``npd < max_npd`` only when its
    phoneme length lies in ``[q_len*(1-max_npd), q_len*(1+max_npd)]``. We therefore scan
    only those length buckets instead of the full (~141k-word) vocab. Word exclusion is
    applied by the caller so this result can be cached per (language, entity). Returns up
    to ``top_k`` ``(word, npd)`` pairs with the smallest NPD in ``(min_npd, max_npd)``.
    """
    if not query_phonemes or top_k <= 0:
        return []
    q_len = len(query_phonemes)
    len_lo = max(1, int(q_len * (1.0 - max_npd)))
    len_hi = max(1, int(q_len * (1.0 + max_npd)) + 1)

    hits: list[tuple[str, float]] = []
    for clen in range(len_lo, len_hi + 1):
        bucket = len_index.get(clen)
        if not bucket:
            continue
        for word, phonemes in bucket:
            d = _npd(query_phonemes, phonemes)
            if min_npd < d < max_npd:
                hits.append((word, d))

    hits.sort(key=lambda kv: kv[1])
    return hits[:top_k]


@dataclass
class AcousticDistractorStage(ProcessingStage[AudioTask, AudioTask]):
    """Append phonetically-similar distractors to ``context_asr.distractor_terms``.

    CPU-only stage.  Loads a precomputed phoneme vocabulary at
    ``setup()`` and, for each task, G2Ps every entity in
    ``fine_context_terms``, searches the vocab for phonetically similar
    words by Normalized Phonetic Distance (NPD), and appends the top
    candidates to ``distractor_terms`` (capped at
    ``max_total_distractors``).

    On samples with no extraction dict, an empty entity list, or an
    unsupported language, the stage is a no-op (existing
    ``distractor_terms`` are preserved).  A per-stage note is written
    via :func:`set_note`.

    Args:
        context_key: Manifest key holding the extraction dict produced
            by :class:`ContextualASRExtractionStage`.
        source_lang_key: Manifest key holding the per-sample source
            language.  Accepts display names (``"English"``) or ISO
            codes (``"en"``).
        default_source_lang: Fallback used when ``source_lang_key`` is
            missing or empty on a sample.
        language: Optional explicit espeak-ng code (e.g. ``"en-us"``).
            When set, used for all samples and the per-sample
            ``source_lang`` is ignored.  Use this when you know the
            entire dataset is a single language.
        phoneme_vocab_path: Path produced by ``scripts/build_phoneme_vocab.py``.
            Either a single ``{word: [phonemes]}`` JSON file (single-language,
            applied to all samples) or a directory of ``phoneme_vocab_<lang>.json``
            files (multi-language; per-sample ``source_lang`` selects the vocab).
            Required.
        max_acoustic_distractors: Maximum acoustic distractors appended
            per sample (combined across all entities of that sample).
        max_total_distractors: Cap on the combined ``distractor_terms``
            list (semantic + acoustic) after merging.
        per_entity_top_k: Top-K candidates retained per source entity
            before cross-entity merging.
        min_npd: Lower NPD bound — entries closer than this are
            considered too-similar (typically the entity itself or a
            near-duplicate).
        max_npd: Upper NPD bound — entries farther than this are
            considered acoustically unrelated.
        notes_key: Key holding the ``additional_notes`` dict that
            :func:`set_note` writes into.
        num_workers_override: Explicit worker count for Xenna.
    """

    name: str = "AcousticDistractor"
    context_key: str = "context_asr"
    source_lang_key: str = "source_lang"
    default_source_lang: str = "English"
    language: str | None = None
    phoneme_vocab_path: str = ""
    max_acoustic_distractors: int = 8
    max_total_distractors: int = 16
    per_entity_top_k: int = 3
    min_npd: float = 0.1
    max_npd: float = 0.5
    notes_key: str = "additional_notes"
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 256

    _vocab_items: list[tuple[str, list[str]]] = field(default_factory=list, init=False, repr=False)
    _vocab_by_lang: dict[str, list[tuple[str, list[str]]]] = field(default_factory=dict, init=False, repr=False)
    _g2p_cache: dict[tuple[str, str], list[str]] = field(default_factory=dict, init=False, repr=False)
    # One reusable EspeakBackend per espeak language code, per actor process. phonemizer's
    # top-level phonemize() builds a fresh EspeakBackend on EVERY call, which copies
    # libespeak-ng.so to a tempdir and dlopen()s it without ever dlclose()-ing — so per-term
    # calls leak VMA mappings until the process hits vm.max_map_count (65530) and every
    # subsequent mmap fails with "failed to map segment from shared object". Reusing one
    # backend per language collapses thousands of lib copies to one.
    _espeak_backends: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # Length-bucketed vocab index per language (built lazily): phoneme-length -> items,
    # so _vocab_search scans only the valid length window instead of all ~141k words.
    _len_index_cache: dict[str, dict[int, list[tuple[str, list[str]]]]] = field(
        default_factory=dict, init=False, repr=False
    )
    # Per-(language, entity) NN cache. Entities repeat heavily across utterances, so caching
    # the raw neighbor list (excluded words filtered at the call site) skips the scan on repeats.
    _search_cache: dict[tuple[str, str], list[tuple[str, float]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _n_processed: int = field(default=0, init=False, repr=False)
    _n_appended: int = field(default=0, init=False, repr=False)

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        pass

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if not PHONEMIZER_AVAILABLE:
            msg = (
                "phonemizer is required for AcousticDistractorStage. "
                "Install it with `pip install phonemizer` and the espeak-ng system package."
            )
            raise ImportError(msg)
        if not EDITDISTANCE_AVAILABLE:
            msg = "editdistance is required for AcousticDistractorStage. `pip install editdistance`."
            raise ImportError(msg)
        if not self.phoneme_vocab_path:
            msg = "AcousticDistractorStage requires phoneme_vocab_path to be set."
            raise ValueError(msg)

        vocab_path = Path(self.phoneme_vocab_path)
        if not vocab_path.exists():
            msg = f"AcousticDistractorStage: phoneme vocab path not found: {vocab_path}"
            raise FileNotFoundError(msg)

        if vocab_path.is_dir():
            # Directory mode: load every ``phoneme_vocab_<lang>.json`` and key it
            # by the espeak code that a sample of that language resolves to, so the
            # per-sample ``source_lang`` selects the right vocab. Lets one stage
            # instance serve a multi-language job. The filename code is the ISO
            # code the file was built for (e.g. ``en`` → built with espeak ``en-us``),
            # so it is normalised through the same map ``_resolve_language`` uses.
            files = sorted(vocab_path.glob("phoneme_vocab_*.json"))
            if not files:
                msg = f"AcousticDistractorStage: no phoneme_vocab_*.json files in directory {vocab_path}"
                raise FileNotFoundError(msg)
            for fpath in files:
                code = fpath.stem[len("phoneme_vocab_") :]
                espeak = _normalize_lang_to_espeak(code)
                if not espeak:
                    logger.warning(
                        "{}: skipping vocab file with unmappable language code {!r} ({})",
                        self.name,
                        code,
                        fpath.name,
                    )
                    continue
                items = self._load_vocab_file(fpath)
                self._vocab_by_lang[espeak] = items
                logger.info(
                    "{}: loaded {} entries for {} (espeak={}) from {}",
                    self.name,
                    len(items),
                    code,
                    espeak,
                    fpath.name,
                )
            if not self._vocab_by_lang:
                msg = f"AcousticDistractorStage: no usable phoneme_vocab_*.json files under {vocab_path}"
                raise ValueError(msg)
            logger.info(
                "{}: directory mode — {} language(s) loaded: {}",
                self.name,
                len(self._vocab_by_lang),
                ",".join(sorted(self._vocab_by_lang)),
            )
        else:
            self._vocab_items = self._load_vocab_file(vocab_path)
            logger.info(
                "{}: loaded {} phoneme vocab entries from {} (language={})",
                self.name,
                len(self._vocab_items),
                vocab_path,
                self.language or "(per-sample source_lang)",
            )

    @staticmethod
    def _load_vocab_file(path: Path) -> list[tuple[str, list[str]]]:
        """Load one ``{word: [phonemes]}`` JSON into a filtered item list."""
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            msg = f"Phoneme vocab must be a JSON object mapping word→[phonemes]; got {type(raw).__name__} ({path})."
            raise TypeError(msg)
        return [
            (str(word), [str(p) for p in phonemes])
            for word, phonemes in raw.items()
            if isinstance(phonemes, list) and phonemes
        ]

    def teardown(self) -> None:
        if self._n_processed:
            logger.info(
                "{}: processed {} samples, appended acoustic distractors to {} ({:.1f}%)",
                self.name,
                self._n_processed,
                self._n_appended,
                100.0 * self._n_appended / self._n_processed,
            )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.context_key, self.source_lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.context_key]

    def _resolve_language(self, task: AudioTask) -> str | None:
        if self.language:
            return self.language
        raw = task.data.get(self.source_lang_key) or self.default_source_lang
        return _normalize_lang_to_espeak(raw)

    def _vocab_for_language(self, language: str) -> list[tuple[str, list[str]]]:
        """Return the vocab items for ``language``.

        Directory mode: look up the per-language vocab (empty list if the
        directory has no file for this language). Single-file mode: the one
        loaded vocab is used for every sample (legacy single-language behaviour).
        """
        if self._vocab_by_lang:
            return self._vocab_by_lang.get(language, [])
        return self._vocab_items

    def _get_espeak_backend(self, language: str) -> Any:  # noqa: ANN401
        """Return a reused EspeakBackend for ``language`` (built once per actor).

        Avoids phonemizer's top-level ``phonemize()``, which constructs a new
        backend — copying + dlopen()-ing libespeak-ng.so — on every call and
        leaks VMA mappings until ``vm.max_map_count`` is exhausted.
        """
        backend = self._espeak_backends.get(language)
        if backend is None:
            backend = _EspeakBackend(
                language,
                preserve_punctuation=False,
                with_stress=False,
            )
            self._espeak_backends[language] = backend
        return backend

    def _g2p(self, text: str, language: str) -> list[str]:
        key = (language, text)
        cached = self._g2p_cache.get(key)
        if cached is not None:
            return cached
        phonemes: list[str] = []
        if text and _EspeakBackend is not None and _Separator is not None:
            try:
                backend = self._get_espeak_backend(language)
                sep = _Separator(phone=" ", word=f" {_WORD_BOUNDARY_MARKER} ", syllable="")
                out = backend.phonemize([text], separator=sep, strip=True)
                ipa = out[0] if out else ""
                phonemes = [tok for tok in ipa.split() if tok and tok != _WORD_BOUNDARY_MARKER]
            except Exception as exc:  # noqa: BLE001
                logger.warning("{}: phonemize failed for {!r} ({}): {}", self.name, text, language, exc)
                phonemes = []
        self._g2p_cache[key] = phonemes
        return phonemes

    def _get_len_index(self, language: str) -> dict[int, list[tuple[str, list[str]]]]:
        """Lazily build + cache the phoneme-length -> items index for ``language``."""
        idx = self._len_index_cache.get(language)
        if idx is None:
            idx = {}
            for word, phonemes in self._vocab_for_language(language):
                idx.setdefault(len(phonemes), []).append((word, phonemes))
            self._len_index_cache[language] = idx
        return idx

    def _search_neighbors(self, entity: str, language: str) -> list[tuple[str, float]]:
        """Top NN candidates for ``entity`` (NOT excluded-filtered), cached per (language, entity)."""
        key = (language, entity)
        cached = self._search_cache.get(key)
        if cached is not None:
            return cached
        query = self._g2p(entity, language)
        if not query:
            result: list[tuple[str, float]] = []
        else:
            result = _vocab_search(
                query,
                self._get_len_index(language),
                min_npd=self.min_npd,
                max_npd=self.max_npd,
                top_k=self.per_entity_top_k + _SEARCH_CACHE_MARGIN,
            )
        self._search_cache[key] = result
        return result

    def _generate_acoustic_distractors(
        self,
        fine_terms: list[str],
        existing_distractors: list[str],
        language: str,
    ) -> list[str]:
        """Return up to ``max_acoustic_distractors`` words from the vocab."""
        if not fine_terms:
            return []

        excluded = {w.lower() for w in fine_terms} | {w.lower() for w in existing_distractors}

        scored: dict[str, float] = {}
        for entity in fine_terms:
            taken = 0
            for word, dist in self._search_neighbors(entity, language):
                if word.lower() in excluded:
                    continue
                prev = scored.get(word)
                if prev is None or dist < prev:
                    scored[word] = dist
                taken += 1
                if taken >= self.per_entity_top_k:
                    break

        ranked = sorted(scored.items(), key=lambda kv: kv[1])
        return [w for w, _ in ranked[: self.max_acoustic_distractors]]

    def _merge_distractors(self, existing: list[str], acoustic: list[str]) -> list[str]:
        seen: set[str] = {w.lower() for w in existing}
        merged: list[str] = list(existing)
        for word in acoustic:
            if word.lower() in seen:
                continue
            seen.add(word.lower())
            merged.append(word)
            if len(merged) >= self.max_total_distractors:
                break
        return merged[: self.max_total_distractors]

    def _process_one(self, task: AudioTask) -> None:
        extraction = task.data.get(self.context_key)
        if not isinstance(extraction, dict):
            set_note(task.data, self.name, "no_extraction", self.notes_key)
            return

        fine_terms = extraction.get("fine_context_terms") or []
        if not isinstance(fine_terms, list) or not fine_terms:
            set_note(task.data, self.name, "no_fine_terms", self.notes_key)
            return

        language = self._resolve_language(task)
        if not language:
            set_note(task.data, self.name, "unsupported_language", self.notes_key)
            return

        vocab_items = self._vocab_for_language(language)
        if not vocab_items:
            # Directory mode with no vocab file for this language.
            set_note(task.data, self.name, f"no_vocab_for_language:{language}", self.notes_key)
            return

        raw_existing = extraction.get("distractor_terms") or []
        existing = [str(t) for t in raw_existing] if isinstance(raw_existing, list) else []

        acoustic = self._generate_acoustic_distractors(
            [str(t) for t in fine_terms],
            existing,
            language,
        )

        self._n_processed += 1
        if not acoustic:
            set_note(task.data, self.name, "no_acoustic_candidates", self.notes_key)
            return

        extraction["distractor_terms"] = self._merge_distractors(existing, acoustic)
        self._n_appended += 1
        set_note(task.data, self.name, f"appended={len(acoustic)}", self.notes_key)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            self._process_one(task)
        logger.debug("{}: batch of {} tasks", self.name, len(tasks))
        return tasks
