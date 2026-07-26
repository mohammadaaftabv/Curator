# Indic PNC prompt-tuning toolkit

This review-only toolkit implements the experiment described in the private
Google Doc. It does not change the production
`nemo_curator/stages/audio/text_filtering/prompts/pnc_prompt.md`.

## Non-negotiable PNC contract

The only target is the Granary v2 profile at
`tutorials/audio/granary_v2_postprocessing/common.yaml`. There is no native
punctuation profile and no second normalizer.

- The **generator** receives raw ASR text. It may insert only `.`, `,`, `?`, or
  `!` and may make only the explicitly permitted Latin-script case changes.
  It must not perform quote/dash mapping, bracket removal, character
  whitelisting, whitespace cleanup, translation, transliteration, correction,
  or any other normalization.
- The raw generator output first passes the strict Unicode preservation gate.
- The **normalizer** is deterministic. Only after the raw gate passes, the
  toolkit applies the repository's exact `common.yaml` rules using the same
  order and final whitespace cleanup as `RegexSubstitutionStage`.
- Judges, reference metrics, pairwise comparison, and promotion reports use
  the common-normalized input, candidate, and reference. Artifacts retain the
  raw candidate separately as `candidate_raw`.
- Every run records the exact `common.yaml` SHA-256. Configuration cannot
  replace the file or expand the four-mark generator insertion set.

The currently pinned `common.yaml` does **not** whitelist the Odia Unicode
block (`U+0B00–U+0B7F`). Its whitelist turns Odia text into whitespace. Because
Odia is one of the 12 ground-truth targets, `verify-contract`, `validate`, and
the guarded end-to-end run fail closed. Do not add a local workaround. Update
and repin the authoritative `common.yaml` itself, then rerun the contract
check.

It provides:

- a P0–P4 prompt registry;
- deterministic, group-disjoint sampling;
- a transcript-overlay join that never alters source metadata;
- NVIDIA OpenAI-compatible model discovery, calls, retries, and hash caching;
- strict code-point preservation checks for Indic/Urdu Unicode text;
- language-aware judge routing and structured response validation;
- position-swapped pairwise judging;
- human-label exports, per-route calibration, reference scoring, and summaries;
- a Draco launcher that pins every cache, temporary file, log, and artifact to
  `/lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning`.

## Safety and current data precondition

On Draco, the CLI permits a Lustre work root only when it is exactly:

```text
/lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning
```

Input manifests are read-only and may be elsewhere. Every CLI output is checked
against the configured work root and written atomically beside its destination.
The source directory is never changed.

The requested `135640` source contains 192,162 metadata rows, all currently
marked Hindi, and every `text_original` value is empty. It cannot be sent to a
PNC model as-is. Produce ASR transcripts or extract the approved subtitle
tracks, save only the necessary derived manifest under the work root, and then
run this toolkit. Other languages require their corresponding source harvests.

## Layout

```text
config.example.json        dataset fields, splits, routes, calibration thresholds
prompt_registry.json       P0–P4 file registry and activation conditions
prompts/                   generator, absolute-judge, and pairwise-judge prompts
pnc_tuning/                dependency-light Python implementation
scripts/run_draco_review.sh guarded P0/P1/P3 review run
tests/                     offline unit and CLI tests
```

`openai`, already a Curator dependency, is imported only for online chat calls.
All sampling, validation, routing, aggregation, calibration, and tests are
offline.

## Transcript overlay

Create a small JSONL supplied by ASR or annotation. Each row must contain the
metadata `id`, transcript `text`, and language code. `reference` and `complete`
are optional:

```json
{"id":"source-row-id","text":"raw unpunctuated transcript","language":"hi","reference":"human punctuated reference","complete":true}
```

Join it to the read-only metadata:

```bash
export PYTHONPATH=examples/audio/pnc_prompt_tuning
python3 -m pnc_tuning attach-transcripts \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --transcripts /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/data/transcripts.jsonl \
  --output data/enriched_manifest.jsonl \
  --report data/enriched_manifest_report.json
```

The command rejects duplicate transcript IDs and reports unmatched IDs. It
copies only the configured ID, group, language, duration, text, optional
reference, and completeness fields into the approved work root; source URLs,
audio paths, and unrelated metadata are not copied.

## Review-first workflow

1. Verify the exact normalizer and all 12 scripts before any model call:

```bash
python3 -m pnc_tuning verify-contract \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --output artifacts/common_yaml_contract.json
```

This currently writes the report and exits nonzero because Odia is not
preserved by the pinned file.

2. After the authoritative file passes, snapshot the live model catalog and
resolved routes:

```bash
python3 -m pnc_tuning discover-models \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --output artifacts/models_snapshot.json
```

3. Build frozen, hash-selected, group-disjoint splits:

```bash
python3 -m pnc_tuning build-subset \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --input /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/data/enriched_manifest.jsonl \
  --output artifacts/subset.jsonl \
  --report artifacts/subset_report.json
```

4. Generate P0/P1 candidates with the same fixed generation model:

```bash
python3 -m pnc_tuning generate \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --input /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/subset.jsonl \
  --prompt p0=examples/audio/pnc_prompt_tuning/prompts/p0_current.md \
  --prompt p1=examples/audio/pnc_prompt_tuning/prompts/p1_strict.md \
  --generator-model YOUR_FIXED_GENERATOR_MODEL \
  --output artifacts/candidates.jsonl
```

5. Apply the raw-output hard gate, then the exact common.yaml normalizer,
judge only gate-pass common-normalized outputs, and aggregate:

```bash
python3 -m pnc_tuning validate \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --input /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/candidates.jsonl \
  --output artifacts/validated.jsonl

python3 -m pnc_tuning judge \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --input /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/validated.jsonl \
  --judge-prompt examples/audio/pnc_prompt_tuning/prompts/judge_absolute.md \
  --models-snapshot /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/models_snapshot.json \
  --output artifacts/judgments.jsonl

python3 -m pnc_tuning aggregate \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --candidates /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/validated.jsonl \
  --judgments /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/judgments.jsonl \
  --output artifacts/aggregated.jsonl
```

The configured panel uses Qwen3.5-397B-A17B with a runtime Qwen fallback,
Sarvam-M only for the ten languages named by its card, and Nemotron 3 Ultra as
a policy arbiter rather than a universal native-language judge. Required
missing routes fail closed unless `--allow-partial-panel` is explicitly used.

## Human calibration

Create a label sheet. `--only-review` produces the disagreement/failure queue;
omit it to create the full random-audit/calibration sheet:

```bash
python3 -m pnc_tuning make-label-sheet \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --input /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/aggregated.jsonl \
  --output artifacts/human_labels.jsonl
```

After double annotation and adjudication, fill `human_overall`,
`human_rubric`, error categories, corrected text, and notes. Then calculate
per-language/model agreement, false-accept, false-reject, and rubric confusion:

```bash
python3 -m pnc_tuning calibrate \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --labels /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/human_labels.jsonl \
  --judgments /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/judgments.jsonl \
  --output artifacts/calibration.json
```

On later development/test runs, pass `--calibration artifacts/calibration.json`
to `judge` or `run`. Only language/model routes that meet all configured
thresholds are used. With no calibrated vote, aggregation returns `review`.
Never use locked-test labels to tune thresholds or prompts.

## Conditional prompt variants

- P2 is allowed only after two adjudicated development examples exist for each
  evaluated language. Supply a private JSON object mapping each language to
  exactly two `{raw, restored}` examples via `--demonstrations`.
- P3 makes the reconstruction check more explicit.
- P4 is a two-template Brahmic/Urdu experiment. Invoke it with:

```text
--prompt p4=prompts/p4_brahmic.md \
--language-prompt p4:ur=prompts/p4_urdu.md
```

Do not create twelve prompts unless measured per-language failures justify
them. Examples must never come from the locked test set.

## Pairwise and reference reports

Position-swapped pairwise comparison:

```bash
python3 -m pnc_tuning pairwise \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --input /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/validated.jsonl \
  --prompt-a p0 --prompt-b p1 \
  --judge-model qwen/qwen3.5-397b-a17b \
  --pairwise-prompt examples/audio/pnc_prompt_tuning/prompts/judge_pairwise.md \
  --output artifacts/p0_vs_p1_pairwise.jsonl
```

`score` computes boundary-aligned punctuation precision/recall/F1 on
common-normalized candidates and references. `summarize` reports hard-gate pass
rate, panel outcomes, review load, and judge parsing failures by language,
prompt, and generator.

To evaluate output produced by the existing Curator pipeline or by
Cadence/Cadence-Fast, first run that output through the exact pinned
`common.yaml`, then import it onto the same frozen subset. Native-punctuation
benchmarking is out of scope:

```bash
python3 -m pnc_tuning import-candidates \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --subset /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/subset.jsonl \
  --results /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/cadence_output.jsonl \
  --candidate-field pnc_text \
  --prompt-id cadence --generator-model ai4bharat-cadence \
  --output artifacts/cadence_candidates.jsonl \
  --report artifacts/cadence_import_report.json
```

After validation and human-reference attachment, build a paired statistical
decision record:

```bash
python3 -m pnc_tuning promotion-report \
  --config examples/audio/pnc_prompt_tuning/config.example.json \
  --input /lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning/artifacts/locked_test_candidates.jsonl \
  --baseline-prompt p0 --candidate-prompt p1 \
  --bootstrap-samples 2000 \
  --output artifacts/p1_vs_p0_promotion.json
```

The report blocks promotion on any candidate preservation failure, absence of
paired human references, a language whose lower confidence bound falls below
the negative non-inferiority margin, or a macro F1 gain whose 95% interval is
not positive.
Run the same comparison against the `cadence` prompt ID. It produces a decision
record only; it never edits `pnc_prompt.md`.

## One guarded Draco run

From the Draco clone:

```bash
export NVIDIA_API_KEY='...'
export GENERATOR_MODEL='the-fixed-production-relevant-model-id'
export PNC_PYTHON='/path/to/curator-python-3.11-or-newer'
bash examples/audio/pnc_prompt_tuning/scripts/run_draco_review.sh
```

The script runs `verify-contract` before reading the NVIDIA key or making a
model call, so the current Odia incompatibility stops execution. After that
contract passes, the key is read from the environment and is never written.
The script runs P0/P1/P3 only, uses a live model snapshot, and creates a review
queue. It does not run P2 without adjudicated examples, does not run a Cadence
baseline, does not edit the production prompt, and does not promote a winner
automatically.
The Draco login-node `python3` currently reports 3.9, while this Curator branch
requires Python 3.11–3.13; set `PNC_PYTHON` to the reviewed Curator environment.

## Review gates before online execution

- Confirm the transcript field actually reaching PNC after merge/entity
  recovery, plus language, source-group, duration, and completeness fields.
- Fix and repin the authoritative `common.yaml` so it preserves Odia; the
  twelve-language contract check must pass. Do not create a separate profile.
- Review prompt text, calibration thresholds, data-governance approval, model
  licenses, endpoint rate limits, and cost.
- Add a Cadence/Cadence-Fast result only after exact common.yaml normalization,
  using the same frozen IDs.
- Require 100% preservation, per-language human calibration, and locked-test
  non-regression before copying any candidate into production.
