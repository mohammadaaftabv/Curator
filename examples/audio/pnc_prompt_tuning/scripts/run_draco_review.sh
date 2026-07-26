#!/usr/bin/env bash

set -euo pipefail

WORK_ROOT="/lustre/fsw/portfolios/edgeai/users/aaftabv/pnc_prompt_finetuning"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

case "${REPO_ROOT}/" in
  "${WORK_ROOT}/"*) ;;
  *)
    echo "Refusing to run: repository is outside ${WORK_ROOT}: ${REPO_ROOT}" >&2
    exit 2
    ;;
esac

CONFIG_PATH="${PNC_CONFIG_PATH:-${REPO_ROOT}/examples/audio/pnc_prompt_tuning/config.example.json}"
INPUT_MANIFEST="${PNC_INPUT_MANIFEST:-${WORK_ROOT}/data/enriched_manifest.jsonl}"
RUN_ID="${PNC_RUN_ID:-review_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${WORK_ROOT}/runs/${RUN_ID}"
RUNTIME_ROOT="${WORK_ROOT}/runtime"
TOOLKIT="${REPO_ROOT}/examples/audio/pnc_prompt_tuning"
PNC_PYTHON="${PNC_PYTHON:-python3}"

case "${RUN_DIR}/" in
  "${WORK_ROOT}/"*) ;;
  *)
    echo "Refusing output path outside ${WORK_ROOT}: ${RUN_DIR}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${INPUT_MANIFEST}" ]]; then
  echo "Missing transcript-enriched manifest: ${INPUT_MANIFEST}" >&2
  echo "Run the attach-transcripts phase first; the source metadata has empty text_original." >&2
  exit 2
fi

if ! "${PNC_PYTHON}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "PNC_PYTHON must point to Python 3.11 or newer; got: $("${PNC_PYTHON}" --version 2>&1)" >&2
  exit 2
fi

mkdir -p \
  "${RUNTIME_ROOT}/tmp" \
  "${RUNTIME_ROOT}/xdg-cache" \
  "${RUNTIME_ROOT}/hf-cache" \
  "${RUNTIME_ROOT}/torch-cache" \
  "${RUNTIME_ROOT}/triton-cache"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${TOOLKIT}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg-cache"
export HF_HOME="${RUNTIME_ROOT}/hf-cache"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_ROOT}/torch-cache"
export TRITON_CACHE_DIR="${RUNTIME_ROOT}/triton-cache"

"${PNC_PYTHON}" -m pnc_tuning verify-contract \
  --config "${CONFIG_PATH}" \
  --output "${WORK_ROOT}/artifacts/common_yaml_contract.json"

: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY without writing it to a file.}"
: "${GENERATOR_MODEL:?Set GENERATOR_MODEL to the fixed PNC generation model ID.}"

"${PNC_PYTHON}" -m pnc_tuning discover-models \
  --config "${CONFIG_PATH}" \
  --output "${RUN_DIR}/models_snapshot.json"

"${PNC_PYTHON}" -m pnc_tuning run \
  --config "${CONFIG_PATH}" \
  --input "${INPUT_MANIFEST}" \
  --prompt "p0=${TOOLKIT}/prompts/p0_current.md" \
  --prompt "p1=${TOOLKIT}/prompts/p1_strict.md" \
  --prompt "p3=${TOOLKIT}/prompts/p3_reconstruction.md" \
  --generator-model "${GENERATOR_MODEL}" \
  --judge-prompt "${TOOLKIT}/prompts/judge_absolute.md" \
  --models-snapshot "${RUN_DIR}/models_snapshot.json" \
  --output-dir "${RUN_DIR}"

"${PNC_PYTHON}" -m pnc_tuning make-label-sheet \
  --config "${CONFIG_PATH}" \
  --input "${RUN_DIR}/05_aggregated.jsonl" \
  --only-review \
  --output "${RUN_DIR}/08_review_queue.jsonl"

echo "Review artifacts: ${RUN_DIR}"
