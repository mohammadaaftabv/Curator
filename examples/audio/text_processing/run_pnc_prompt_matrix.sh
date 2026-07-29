#!/usr/bin/env bash
#
# Run the unchanged text pipeline four times over the same mixed-language input.
# Each row keeps its existing source_lang and data flow; only the selected PnC
# prompt version and its per-row language block change.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 INPUT_MANIFEST OUTPUT_ROOT [run_text_pipeline.py arguments ...]" >&2
    exit 2
fi

input_manifest=$1
output_root=$2
shift 2

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python}

for prompt_version in p0 p1 p2 p3; do
    "${python_bin}" "${script_dir}/run_text_pipeline.py" \
        --input_manifest "${input_manifest}" \
        --output_dir "${output_root}/${prompt_version}" \
        --enable_pnc \
        --pnc_prompt_version "${prompt_version}" \
        "$@"
done
