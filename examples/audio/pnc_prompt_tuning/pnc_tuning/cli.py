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

"""Command-line interface for the Indic PNC prompt-tuning toolkit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pnc_tuning.config import ExperimentConfig, load_config
from pnc_tuning.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    ensure_within_work_root,
    expand_input_paths,
    iter_jsonl,
    load_json,
    stable_json_hash,
)
from pnc_tuning.languages import LANGUAGE_SPECS
from pnc_tuning.normalization import CommonYamlNormalizer
from pnc_tuning.nvidia_client import NvidiaClient
from pnc_tuning.pipeline import (
    aggregate_candidates,
    attach_transcripts,
    build_subset,
    calibrate_judges,
    experiment_summary,
    generate_candidates,
    import_candidates,
    judge_candidates,
    pairwise_compare,
    promotion_report,
    score_reference_rows,
    validate_candidates,
)
from pnc_tuning.prompts import load_prompt
from pnc_tuning.routing import resolve_panel

_REQUIRED_DEMONSTRATIONS = 2


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Experiment JSON configuration.")


def _rows(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl([path]))


def _named_prompts(values: list[str], language_values: list[str]) -> dict[str, dict[str, str]]:
    prompts: dict[str, dict[str, str]] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            msg = f"Prompt must use NAME=PATH syntax: {value!r}"
            raise ValueError(msg)
        if name in prompts:
            msg = f"Duplicate prompt name: {name}"
            raise ValueError(msg)
        prompts[name] = {"*": load_prompt(path)}
    for value in language_values:
        selector, separator, path = value.partition("=")
        name, language_separator, language = selector.partition(":")
        if not separator or not language_separator or not name or language not in LANGUAGE_SPECS or not path:
            msg = f"Language prompt must use NAME:LANG=PATH syntax: {value!r}"
            raise ValueError(msg)
        if name not in prompts:
            msg = f"Language override refers to undefined prompt: {name}"
            raise ValueError(msg)
        prompts[name][language] = load_prompt(path)
    return prompts


def _demonstrations(path: str | None) -> dict[str, str] | None:
    if path is None:
        return None
    raw = load_json(path)
    rendered = {}
    for language, examples in raw.items():
        if language not in LANGUAGE_SPECS or not isinstance(examples, list):
            msg = f"Demonstrations must map a supported language to a JSON array: {language!r}"
            raise TypeError(msg)
        if len(examples) != _REQUIRED_DEMONSTRATIONS:
            msg = f"Demonstrations for {language!r} must contain exactly two gold examples"
            raise ValueError(msg)
        blocks = []
        for index, example in enumerate(examples, start=1):
            if not isinstance(example, dict) or "raw" not in example or "restored" not in example:
                msg = f"Demonstration {language}[{index}] must contain raw and restored"
                raise TypeError(msg)
            blocks.append(
                f"Example {index}\n"
                f"<transcript>{example['raw']}</transcript>\n"
                f"<restored>{example['restored']}</restored>"
            )
        rendered[language] = "\n\n".join(blocks)
    return rendered


def _client(config: ExperimentConfig, cache_dir: str) -> NvidiaClient:
    return NvidiaClient(config.nvidia, work_root=config.work_root, cache_dir=cache_dir)


def _normalizer(config: ExperimentConfig) -> CommonYamlNormalizer:
    return CommonYamlNormalizer.load(config.common_yaml_path)


def _available_models(
    *,
    client: NvidiaClient,
    snapshot_path: str | None,
) -> set[str]:
    if snapshot_path is None:
        return client.list_models()
    snapshot = load_json(snapshot_path)
    models = snapshot.get("models")
    if not isinstance(models, list) or not all(isinstance(item, str) for item in models):
        msg = "Model snapshot must contain a string array named 'models'"
        raise TypeError(msg)
    return set(models)


def _calibrated_routes(path: str | None) -> set[str] | None:
    if path is None:
        return None
    report = load_json(path)
    return {route for route, value in report.items() if isinstance(value, dict) and value.get("route_enabled") is True}


def _write_json(path: str, value: object, config: ExperimentConfig) -> None:
    target = atomic_write_json(path, value, config.work_root)
    print(target)


def _write_jsonl(path: str, rows: list[dict[str, Any]], config: ExperimentConfig) -> None:
    target = atomic_write_jsonl(path, rows, config.work_root)
    print(target)


def _cmd_discover_models(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    client = _client(config, args.cache_dir)
    models = client.list_models()
    resolution = {}
    for language in config.sampling.languages:
        try:
            routes = resolve_panel(
                available_models=models,
                roles=config.judge_roles,
                language=language,
                allow_partial=args.allow_partial_panel,
            )
            resolution[language] = [route.to_dict() for route in routes]
        except RuntimeError as exc:
            resolution[language] = {"error": str(exc)}
    _write_json(
        args.output,
        {
            "endpoint": config.nvidia.base_url,
            "models": sorted(models),
            "configured_panel_resolution": resolution,
        },
        config,
    )


def _cmd_attach_transcripts(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    metadata_paths = expand_input_paths(config.input_paths)
    transcript_paths = expand_input_paths(args.transcripts)
    rows, report = attach_transcripts(
        iter_jsonl(metadata_paths),
        iter_jsonl(transcript_paths),
        config=config,
        transcript_id_field=args.transcript_id_field,
        transcript_text_field=args.transcript_text_field,
        transcript_language_field=args.transcript_language_field,
        transcript_reference_field=args.transcript_reference_field,
        transcript_complete_field=args.transcript_complete_field,
    )
    atomic_write_jsonl(args.output, rows, config.work_root)
    _write_json(args.report, report, config)


def _cmd_build_subset(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = expand_input_paths(args.input or config.input_paths)
    selected, report = build_subset(iter_jsonl(paths), config=config)
    atomic_write_jsonl(args.output, selected, config.work_root)
    _write_json(args.report, report, config)


def _cmd_verify_contract(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    normalizer = _normalizer(config)
    report = normalizer.contract_report(config.sampling.languages)
    _write_json(args.output, report, config)
    normalizer.require_language_coverage(config.sampling.languages)


def _cmd_import_candidates(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output, report = import_candidates(
        _rows(args.subset),
        _rows(args.results),
        result_id_field=args.result_id_field,
        candidate_field=args.candidate_field,
        prompt_id=args.prompt_id,
        generator_model=args.generator_model,
    )
    atomic_write_jsonl(args.output, output, config.work_root)
    _write_json(args.report, report, config)


def _cmd_generate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prompts = _named_prompts(args.prompt, args.language_prompt)
    client = _client(config, args.cache_dir)
    output = generate_candidates(
        _rows(args.input),
        prompt_templates=prompts,
        generator_models=args.generator_model,
        client=client,
        demonstrations=_demonstrations(args.demonstrations),
        max_tokens=args.max_tokens,
    )
    _write_jsonl(args.output, output, config)


def _cmd_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    normalizer = _normalizer(config)
    normalizer.require_language_coverage(config.sampling.languages)
    output = validate_candidates(
        _rows(args.input),
        allowed_punctuation=config.allowed_punctuation,
        normalizer=normalizer,
    )
    _write_jsonl(args.output, output, config)


def _cmd_judge(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    client = _client(config, args.cache_dir)
    models = _available_models(client=client, snapshot_path=args.models_snapshot)
    output = judge_candidates(
        _rows(args.input),
        config=config,
        judge_template=load_prompt(args.judge_prompt),
        client=client,
        available_models=models,
        calibrated_routes=_calibrated_routes(args.calibration),
        allow_partial_panel=args.allow_partial_panel,
        max_tokens=args.max_tokens,
    )
    _write_jsonl(args.output, output, config)


def _cmd_aggregate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output = aggregate_candidates(_rows(args.candidates), _rows(args.judgments))
    _write_jsonl(args.output, output, config)


def _cmd_pairwise(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    client = _client(config, args.cache_dir)
    output = pairwise_compare(
        _rows(args.input),
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        judge_model=args.judge_model,
        pairwise_template=load_prompt(args.pairwise_prompt),
        client=client,
        max_tokens=args.max_tokens,
    )
    _write_jsonl(args.output, output, config)


def _cmd_calibrate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _write_json(
        args.output,
        calibrate_judges(_rows(args.labels), _rows(args.judgments), config=config),
        config,
    )


def _cmd_score(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _write_json(
        args.output,
        score_reference_rows(_rows(args.input), allowed_punctuation=config.allowed_punctuation),
        config,
    )


def _cmd_summarize(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _write_json(args.output, experiment_summary(_rows(args.input)), config)


def _cmd_promotion_report(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _write_json(
        args.output,
        promotion_report(
            _rows(args.input),
            baseline_prompt=args.baseline_prompt,
            candidate_prompt=args.candidate_prompt,
            allowed_punctuation=config.allowed_punctuation,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            noninferiority_margin=args.noninferiority_margin,
        ),
        config,
    )


def _cmd_make_label_sheet(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output = []
    for row in _rows(args.input):
        if args.only_review and not bool(dict(row.get("panel", {})).get("needs_human_review", True)):
            continue
        output.append(
            {
                "candidate_key": row.get("candidate_key", ""),
                "id": row["id"],
                "language": row["language"],
                "text": row["text"],
                "text_common": row.get("text_common", ""),
                "candidate_raw": row.get("candidate_raw", ""),
                "candidate_common": row.get("candidate_common", row["candidate"]),
                "candidate": row["candidate"],
                "prompt_id": row["prompt_id"],
                "generator_model": row["generator_model"],
                "validation": row.get("validation", {}),
                "panel": row.get("panel", {}),
                "human_overall": "",
                "human_rubric": {
                    "content_preservation": "",
                    "language_script_preservation": "",
                    "sentence_termination": "",
                    "intra_sentence_punctuation": "",
                    "capitalization": "",
                    "completeness_handling": "",
                    "overall": "",
                },
                "human_error_categories": [],
                "human_corrected_text": "",
                "human_notes": "",
            }
        )
    _write_jsonl(args.output, output, config)


def _cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    normalizer = _normalizer(config)
    normalizer.require_language_coverage(config.sampling.languages)
    output_dir = ensure_within_work_root(args.output_dir, config.work_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _named_prompts(args.prompt, args.language_prompt)
    client = _client(config, str(Path(args.output_dir) / "cache"))

    input_paths = expand_input_paths(args.input or config.input_paths)
    subset, subset_report = build_subset(iter_jsonl(input_paths), config=config)
    candidates = generate_candidates(
        subset,
        prompt_templates=prompts,
        generator_models=args.generator_model,
        client=client,
        demonstrations=_demonstrations(args.demonstrations),
        max_tokens=args.generation_max_tokens,
    )
    validated = validate_candidates(
        candidates,
        allowed_punctuation=config.allowed_punctuation,
        normalizer=normalizer,
    )
    models = _available_models(client=client, snapshot_path=args.models_snapshot)
    judgments = judge_candidates(
        validated,
        config=config,
        judge_template=load_prompt(args.judge_prompt),
        client=client,
        available_models=models,
        calibrated_routes=_calibrated_routes(args.calibration),
        allow_partial_panel=args.allow_partial_panel,
        max_tokens=args.judge_max_tokens,
    )
    aggregated = aggregate_candidates(validated, judgments)
    scores = score_reference_rows(aggregated, allowed_punctuation=config.allowed_punctuation)
    summary = experiment_summary(aggregated)

    atomic_write_jsonl(output_dir / "01_subset.jsonl", subset, config.work_root)
    atomic_write_json(output_dir / "01_subset_report.json", subset_report, config.work_root)
    atomic_write_jsonl(output_dir / "02_candidates.jsonl", candidates, config.work_root)
    atomic_write_jsonl(output_dir / "03_validated.jsonl", validated, config.work_root)
    atomic_write_jsonl(output_dir / "04_judgments.jsonl", judgments, config.work_root)
    atomic_write_jsonl(output_dir / "05_aggregated.jsonl", aggregated, config.work_root)
    atomic_write_json(output_dir / "06_reference_scores.json", scores, config.work_root)
    atomic_write_json(output_dir / "07_summary.json", summary, config.work_root)
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "config": str(Path(args.config).resolve()),
            "config_sha256": stable_json_hash(load_json(args.config)),
            "input_files": [str(path) for path in input_paths],
            "prompt_ids": sorted(prompts),
            "prompt_sha256": {
                name: {
                    language: stable_json_hash({"text": text}) for language, text in sorted(language_templates.items())
                }
                for name, language_templates in sorted(prompts.items())
            },
            "generator_models": args.generator_model,
            "normalization_profile": {
                "path": str(normalizer.path),
                "sha256": normalizer.sha256,
                "rule_count": len(normalizer.rules),
                "generator_insertions": config.allowed_punctuation,
            },
            "available_model_count": len(models),
            "calibration_report": args.calibration,
            "counts": {
                "subset": len(subset),
                "candidates": len(candidates),
                "judgments": len(judgments),
                "aggregated": len(aggregated),
            },
        },
        config.work_root,
    )
    print(output_dir)


def _parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        prog="pnc-tuning",
        description="Reproducible Indic punctuation-and-capitalization prompt experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover-models", help="Snapshot live NVIDIA model IDs and routes.")
    _add_config(discover)
    discover.add_argument("--output", default="artifacts/models_snapshot.json")
    discover.add_argument("--cache-dir", default="cache")
    discover.add_argument("--allow-partial-panel", action="store_true")
    discover.set_defaults(handler=_cmd_discover_models)

    attach = subparsers.add_parser("attach-transcripts", help="Join transcript text to read-only metadata.")
    _add_config(attach)
    attach.add_argument("--transcripts", nargs="+", required=True)
    attach.add_argument("--output", default="data/enriched_manifest.jsonl")
    attach.add_argument("--report", default="data/enriched_manifest_report.json")
    attach.add_argument("--transcript-id-field", default="id")
    attach.add_argument("--transcript-text-field", default="text")
    attach.add_argument("--transcript-language-field", default="language")
    attach.add_argument("--transcript-reference-field", default="reference")
    attach.add_argument("--transcript-complete-field", default="complete")
    attach.set_defaults(handler=_cmd_attach_transcripts)

    subset = subparsers.add_parser("build-subset", help="Create deterministic group-disjoint splits.")
    _add_config(subset)
    subset.add_argument("--input", nargs="+", help="Override config input paths.")
    subset.add_argument("--output", default="artifacts/subset.jsonl")
    subset.add_argument("--report", default="artifacts/subset_report.json")
    subset.set_defaults(handler=_cmd_build_subset)

    contract = subparsers.add_parser(
        "verify-contract",
        help="Verify that the pinned common.yaml preserves every configured target language.",
    )
    _add_config(contract)
    contract.add_argument("--output", default="artifacts/common_yaml_contract.json")
    contract.set_defaults(handler=_cmd_verify_contract)

    imported = subparsers.add_parser("import-candidates", help="Import Curator or baseline output unchanged.")
    _add_config(imported)
    imported.add_argument("--subset", required=True)
    imported.add_argument("--results", required=True)
    imported.add_argument("--result-id-field", default="id")
    imported.add_argument("--candidate-field", required=True)
    imported.add_argument("--prompt-id", required=True)
    imported.add_argument("--generator-model", required=True)
    imported.add_argument("--output", default="artifacts/imported_candidates.jsonl")
    imported.add_argument("--report", default="artifacts/imported_candidates_report.json")
    imported.set_defaults(handler=_cmd_import_candidates)

    generate = subparsers.add_parser("generate", help="Generate all prompt/model candidates.")
    _add_config(generate)
    generate.add_argument("--input", required=True)
    generate.add_argument("--prompt", action="append", required=True, metavar="NAME=PATH")
    generate.add_argument("--language-prompt", action="append", default=[], metavar="NAME:LANG=PATH")
    generate.add_argument("--demonstrations", help="Per-language gold demonstrations JSON for P2.")
    generate.add_argument("--generator-model", action="append", required=True)
    generate.add_argument("--output", default="artifacts/candidates.jsonl")
    generate.add_argument("--cache-dir", default="cache/generation")
    generate.add_argument("--max-tokens", type=int, default=512)
    generate.set_defaults(handler=_cmd_generate)

    validate = subparsers.add_parser("validate", help="Apply strict Unicode-preservation gates.")
    _add_config(validate)
    validate.add_argument("--input", required=True)
    validate.add_argument("--output", default="artifacts/validated.jsonl")
    validate.set_defaults(handler=_cmd_validate)

    judge = subparsers.add_parser("judge", help="Run the resolved heterogeneous judge panel.")
    _add_config(judge)
    judge.add_argument("--input", required=True)
    judge.add_argument("--judge-prompt", required=True)
    judge.add_argument("--models-snapshot")
    judge.add_argument("--calibration", help="Calibration JSON; only enabled language:model routes run.")
    judge.add_argument("--output", default="artifacts/judgments.jsonl")
    judge.add_argument("--cache-dir", default="cache/judging")
    judge.add_argument("--allow-partial-panel", action="store_true")
    judge.add_argument("--max-tokens", type=int, default=800)
    judge.set_defaults(handler=_cmd_judge)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate hard gates and panel votes.")
    _add_config(aggregate)
    aggregate.add_argument("--candidates", required=True)
    aggregate.add_argument("--judgments", required=True)
    aggregate.add_argument("--output", default="artifacts/aggregated.jsonl")
    aggregate.set_defaults(handler=_cmd_aggregate)

    pairwise = subparsers.add_parser("pairwise", help="Run position-swapped A/B judging.")
    _add_config(pairwise)
    pairwise.add_argument("--input", required=True)
    pairwise.add_argument("--prompt-a", required=True)
    pairwise.add_argument("--prompt-b", required=True)
    pairwise.add_argument("--judge-model", required=True)
    pairwise.add_argument("--pairwise-prompt", required=True)
    pairwise.add_argument("--output", default="artifacts/pairwise.jsonl")
    pairwise.add_argument("--cache-dir", default="cache/pairwise")
    pairwise.add_argument("--max-tokens", type=int, default=300)
    pairwise.set_defaults(handler=_cmd_pairwise)

    calibrate = subparsers.add_parser("calibrate", help="Compare judge decisions with human labels.")
    _add_config(calibrate)
    calibrate.add_argument("--labels", required=True)
    calibrate.add_argument("--judgments", required=True)
    calibrate.add_argument("--output", default="artifacts/calibration.json")
    calibrate.set_defaults(handler=_cmd_calibrate)

    score = subparsers.add_parser("score", help="Compute punctuation-event reference metrics.")
    _add_config(score)
    score.add_argument("--input", required=True)
    score.add_argument("--output", default="artifacts/reference_scores.json")
    score.set_defaults(handler=_cmd_score)

    summarize = subparsers.add_parser("summarize", help="Summarize prompt outcomes by language/model.")
    _add_config(summarize)
    summarize.add_argument("--input", required=True)
    summarize.add_argument("--output", default="artifacts/summary.json")
    summarize.set_defaults(handler=_cmd_summarize)

    promotion = subparsers.add_parser("promotion-report", help="Create a paired bootstrap decision record.")
    _add_config(promotion)
    promotion.add_argument("--input", required=True)
    promotion.add_argument("--baseline-prompt", required=True)
    promotion.add_argument("--candidate-prompt", required=True)
    promotion.add_argument("--output", default="artifacts/promotion_report.json")
    promotion.add_argument("--bootstrap-samples", type=int, default=2000)
    promotion.add_argument("--seed", type=int, default=1234)
    promotion.add_argument("--noninferiority-margin", type=float, default=0.01)
    promotion.set_defaults(handler=_cmd_promotion_report)

    labels = subparsers.add_parser("make-label-sheet", help="Create a human-labeling JSONL template.")
    _add_config(labels)
    labels.add_argument("--input", required=True)
    labels.add_argument("--output", default="artifacts/human_labels.jsonl")
    labels.add_argument("--only-review", action="store_true")
    labels.set_defaults(handler=_cmd_make_label_sheet)

    run = subparsers.add_parser("run", help="Run subset, generation, gates, judging, and reports.")
    _add_config(run)
    run.add_argument("--input", nargs="+", help="Override config input paths.")
    run.add_argument("--prompt", action="append", required=True, metavar="NAME=PATH")
    run.add_argument("--language-prompt", action="append", default=[], metavar="NAME:LANG=PATH")
    run.add_argument("--demonstrations", help="Per-language gold demonstrations JSON for P2.")
    run.add_argument("--generator-model", action="append", required=True)
    run.add_argument("--judge-prompt", required=True)
    run.add_argument("--models-snapshot")
    run.add_argument("--calibration", help="Calibration JSON; only enabled language:model routes run.")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--allow-partial-panel", action="store_true")
    run.add_argument("--generation-max-tokens", type=int, default=512)
    run.add_argument("--judge-max-tokens", type=int, default=800)
    run.set_defaults(handler=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
