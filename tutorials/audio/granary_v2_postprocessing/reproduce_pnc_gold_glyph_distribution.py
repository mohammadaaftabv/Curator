"""Reproduce the Indic PnC gold-glyph tables and frozen 12k hashes.

The inputs are the 24 downloadable 1,000-row, per-language gold JSONL shards
and the frozen PnC language-rule JSON. The script writes the complete 168-row
language-by-glyph table plus dataset/language summaries and verifies the exact
serialization used by the development and held-out runtime manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

LANGUAGES = ("as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur")
LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}
ROLES = (
    "statement/command terminal",
    "direct-question mark",
    "exclamation",
    "comma",
    "semicolon",
    "colon",
    "ellipsis",
)
ROWS_PER_LANGUAGE = 1000
COHORTS = {
    "development": {
        "dataset": "Development — ASR-style prompt-finetuning 12k",
        "pattern": "pnc_development12k_{language}_1000.jsonl",
        "expected_sha256": "7bbf5bff8b4ea0bdac13a9c737202c19e0f0592fc95d49c4e14e1e730ee29d52",
    },
    "heldout": {
        "dataset": "Held-out — ASR-style multilingual test 12k",
        "pattern": "pnc_heldout12k_{language}_1000.jsonl",
        "expected_sha256": "db2e2acf0428deac958a7fbf8b3a83d5bd34ecb11bed9e1118753e2fbf05e97a",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_rules(path: Path) -> dict[str, list[tuple[str, str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if tuple(raw) != LANGUAGES:
        message = f"Unexpected rule language order: {tuple(raw)}"
        raise ValueError(message)
    result: dict[str, list[tuple[str, str]]] = {}
    for language, prose in raw.items():
        glyphs = prose.split("`")[1::2]
        if len(glyphs) != len(ROLES) or any(len(glyph) != 1 for glyph in glyphs):
            message = f"Expected seven one-code-point glyphs for {language}: {glyphs}"
            raise ValueError(message)
        result[language] = [(ROLES[index], glyph) for index, glyph in enumerate(glyphs)]
    return result


def runtime_record(row: dict[str, Any], *, language: str, heldout: bool) -> dict[str, Any]:
    output = dict(row)
    reference_source_id = str(row["source_id"])
    output.update(
        {
            "_skipme": "",
            "abbreviated_text": row["text_unpunctuated"],
            "audio_filepath": "harvested_data/youtube/audios/ooSDkEWptyw.opus",
            "container": "harvested_data",
            "download_audio_filepath": "youtube/audios/ooSDkEWptyw.opus",
            "duration": 1 if heldout else 1.0,
            "reference_source_id": reference_source_id,
            "source_id": f"{language}:{reference_source_id}",
            "source_lang": language,
            "transport_only_audio": True,
        }
    )
    if heldout:
        output["corpus"] = "pnc_eval_heldout_structural12k"
    return output


def serialized_runtime_row(row: dict[str, Any], *, heldout: bool) -> bytes:
    kwargs: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if not heldout:
        kwargs["separators"] = (",", ":")
    return (json.dumps(row, **kwargs) + "\n").encode("utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def measure_language(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    language: str,
    rules: list[tuple[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    literal: Counter[str] = Counter()
    literal_rows: Counter[str] = Counter()
    target: Counter[str] = Counter()
    target_rows: Counter[str] = Counter()
    exact: Counter[str] = Counter()
    outside: Counter[str] = Counter()
    primary_total = 0
    skip_rows = 0
    allowed = {glyph for _, glyph in rules}
    for row in rows:
        skip_rows += int(bool(row.get("_skipme", False)))
        gold = row["text_punctuated"]
        for glyph in allowed:
            count = gold.count(glyph)
            literal[glyph] += count
            literal_rows[glyph] += int(bool(count))
        seen_targets: set[str] = set()
        for event in row["punctuation_events"]:
            if event.get("action") != "remove" or not event.get("primary_scored", False):
                continue
            primary_total += 1
            canonical = event.get("canonical") or event.get("character")
            if canonical not in allowed:
                outside[str(canonical)] += 1
                continue
            target[canonical] += 1
            seen_targets.add(canonical)
            exact[canonical] += int(event.get("character") == canonical)
        for glyph in seen_targets:
            target_rows[glyph] += 1

    literal_total = sum(literal.values())
    target_total = sum(target.values())
    details = [
        {
            "Dataset": dataset,
            "Language code": language,
            "Language": LANGUAGE_NAMES[language],
            "Gold rows": len(rows),
            "Role": role,
            "Allowed glyph": glyph,
            "Unicode": f"U+{ord(glyph):04X} {unicodedata.name(glyph)}",
            "Literal gold occurrences": literal[glyph],
            "Gold rows with literal glyph": literal_rows[glyph],
            "Literal row coverage": literal_rows[glyph] / len(rows),
            "Literal share of allowed glyphs": literal[glyph] / literal_total if literal_total else 0.0,
            "Canonical scored target events": target[glyph],
            "Rows with scored target": target_rows[glyph],
            "Target row coverage": target_rows[glyph] / len(rows),
            "Share of mapped target events": target[glyph] / target_total if target_total else 0.0,
            "Exact-glyph removed events": exact[glyph],
            "Canonicalized alias/artifact events": target[glyph] - exact[glyph],
            "Target observed?": "Yes" if target[glyph] else "No",
        }
        for role, glyph in rules
    ]
    summary = {
        "Dataset": dataset,
        "Language code": language,
        "Language": LANGUAGE_NAMES[language],
        "Gold rows": len(rows),
        "Skip rows": skip_rows,
        "Allowed glyphs": " ".join(glyph for _, glyph in rules),
        "Allowed glyphs observed as targets": sum(bool(target[glyph]) for glyph in allowed),
        "Allowed glyphs absent as targets": sum(not target[glyph] for glyph in allowed),
        "Absent target glyphs": " ".join(glyph for _, glyph in rules if not target[glyph]) or "—",
        "Literal allowed-glyph occurrences": literal_total,
        "Canonical mapped target events": target_total,
        "All primary scored events": primary_total,
        "Primary events outside mapping": sum(outside.values()),
        "Outside-mapping canonical glyphs": " | ".join(f"{glyph}: {count}" for glyph, count in sorted(outside.items()))
        or "—",
    }
    return summary, details


def analyze_cohort(
    *,
    cohort: str,
    cohort_dir: Path,
    rules: dict[str, list[tuple[str, str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cohort_details = COHORTS[cohort]
    heldout = cohort == "heldout"
    runtime_digest = hashlib.sha256()
    language_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for language in LANGUAGES:
        path = cohort_dir / cohort_details["pattern"].format(language=language)
        rows = read_jsonl(path)
        if len(rows) != ROWS_PER_LANGUAGE:
            message = f"{path}: expected {ROWS_PER_LANGUAGE} rows, found {len(rows)}"
            raise ValueError(message)
        for row in rows:
            if row.get("language_code") != language:
                message = f"{path}: row declares {row.get('language_code')!r}"
                raise ValueError(message)
            runtime = runtime_record(row, language=language, heldout=heldout)
            runtime_digest.update(serialized_runtime_row(runtime, heldout=heldout))
        summary, details = measure_language(
            rows,
            dataset=cohort_details["dataset"],
            language=language,
            rules=rules[language],
        )
        language_rows.append(summary)
        detail_rows.extend(details)

    actual_hash = runtime_digest.hexdigest()
    expected_hash = cohort_details["expected_sha256"]
    if actual_hash != expected_hash:
        message = f"{cohort} runtime SHA-256 mismatch: {actual_hash} != {expected_hash}"
        raise ValueError(message)
    runtime_hash = {
        "actual_sha256": actual_hash,
        "expected_sha256": expected_hash,
        "match": True,
        "language_order": list(LANGUAGES),
        "serialization": (
            "json.dumps(ensure_ascii=False, sort_keys=True, separators=(',', ':')) + newline"
            if not heldout
            else "json.dumps(ensure_ascii=False, sort_keys=True) + newline"
        ),
    }
    dataset_row = {
        "Dataset": cohort_details["dataset"],
        "Languages": len(language_rows),
        "Gold rows": sum(row["Gold rows"] for row in language_rows),
        "Language-symbol pairs": len(detail_rows),
        "Observed target pairs": sum(row["Allowed glyphs observed as targets"] for row in language_rows),
        "Absent target pairs": sum(row["Allowed glyphs absent as targets"] for row in language_rows),
        "Literal allowed-glyph occurrences": sum(row["Literal allowed-glyph occurrences"] for row in language_rows),
        "Canonical mapped target events": sum(row["Canonical mapped target events"] for row in language_rows),
        "Primary events outside mapping": sum(row["Primary events outside mapping"] for row in language_rows),
    }
    return dataset_row, language_rows, detail_rows, runtime_hash


def main() -> None:
    args = parse_args()
    rules = parse_rules(args.rules)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_dirs = {"development": args.development_dir, "heldout": args.heldout_dir}
    analyses = {
        cohort: analyze_cohort(cohort=cohort, cohort_dir=cohort_dirs[cohort], rules=rules) for cohort in COHORTS
    }
    dataset_rows = [analysis[0] for analysis in analyses.values()]
    language_rows = [row for analysis in analyses.values() for row in analysis[1]]
    detail_rows = [row for analysis in analyses.values() for row in analysis[2]]
    runtime_hashes = {cohort: analysis[3] for cohort, analysis in analyses.items()}

    write_csv(args.output_dir / "glyph_detail_full_168_rows.csv", detail_rows, list(detail_rows[0]))
    write_csv(args.output_dir / "language_summary.csv", language_rows, list(language_rows[0]))
    write_csv(args.output_dir / "dataset_summary.csv", dataset_rows, list(dataset_rows[0]))
    report = {
        "rules": str(args.rules),
        "rules_sha256": sha256_file(args.rules),
        "runtime_hashes": runtime_hashes,
        "detail_rows": len(detail_rows),
        "language_rows": len(language_rows),
        "dataset_rows": len(dataset_rows),
    }
    (args.output_dir / "reproduction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
