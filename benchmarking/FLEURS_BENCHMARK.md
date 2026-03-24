# FLEURS Audio Pipeline Benchmark

A dedicated benchmark for the FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) audio curation pipeline. The benchmark script lives at `benchmarking/scripts/audio_fleurs_benchmark.py` and runs through the full NeMo Curator benchmarking framework.

## How It Works

The benchmark script:
1. Downloads the FLEURS dataset for a given language/split via `CreateInitialManifestFleursStage`
2. Runs ASR inference with an NeMo FastConformer model (`InferenceAsrNemoStage`)
3. Computes pairwise WER between reference text and ASR predictions
4. Measures audio duration, filters by WER threshold
5. Converts to document format and writes JSONL output
6. Collects per-stage profiling metrics via `TaskPerfUtils`

## Running Benchmarks

The benchmarking framework is designed to run inside Docker. See `benchmarking/README.md` for Docker setup instructions that apply to all benchmarks.

**Prerequisites:**
- Docker with NVIDIA container toolkit
- NeMo Curator repository checked out
- GPU required for ASR inference

**Step 1: Build the benchmarking Docker image (one-time):**

```bash
cd /path/to/Curator
bash benchmarking/tools/build_docker.sh --tag-as-latest
```

**Step 2: Disable the Slack sink for local runs.**

The shared `benchmarking/nightly-benchmark.yaml` has the Slack sink enabled for CI.
For local testing, temporarily disable it:

```yaml
sinks:
  - name: slack
    enabled: false   # <-- change from true to false
```

**Step 3: Run the FLEURS benchmark:**

```bash
docker run --rm --net=host --shm-size=8g --gpus all \
  -v $(pwd):/opt/Curator \
  --entrypoint bash nemo_curator_benchmarking:latest \
  -c "cd /opt/Curator && python benchmarking/scripts/audio_fleurs_benchmark.py \
    --benchmark-results-path /tmp/fleurs_bench \
    --scratch-output-path /tmp/fleurs_scratch \
    --executor ray_data --gpus 1"
```

> **Remember** to re-enable the Slack sink (`enabled: true`) before pushing.

## Benchmark Configuration

The FLEURS benchmark entries are defined in `benchmarking/nightly-benchmark.yaml`:

```yaml
entries:
  - name: audio_fleurs_xenna
    script: audio_fleurs_benchmark.py
    args: >-
      --benchmark-results-path={session_entry_dir}
      --scratch-output-path={session_entry_dir}/scratch
      --model-name=nvidia/stt_hy_fastconformer_hybrid_large_pc
      --lang=hy_am
      --split=train
      --wer-threshold=5.5
      --gpus=1

  - name: audio_fleurs_raydata
    script: audio_fleurs_benchmark.py
    args: >-
      --benchmark-results-path={session_entry_dir}
      --scratch-output-path={session_entry_dir}/scratch
      --model-name=nvidia/stt_hy_fastconformer_hybrid_large_pc
      --lang=hy_am
      --split=train
      --wer-threshold=5.5
      --gpus=1
      --executor=ray_data
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--benchmark-results-path` | Required | Directory for output files |
| `--scratch-output-path` | Required | Scratch directory for downloaded data |
| `--model-name` | `nvidia/stt_hy_fastconformer_hybrid_large_pc` | NeMo ASR model name |
| `--lang` | `hy_am` | FLEURS language code |
| `--split` | `dev` | Dataset split (`train`, `dev`, `test`) |
| `--wer-threshold` | `5.5` | WER threshold for filtering |
| `--executor` | `xenna` | `xenna` or `ray_data` |
| `--gpus` | `1` | Number of GPUs for ASR inference |

## Benchmark Results

Results from running on a single workstation with `ray_data` executor, `dev` split:

**Machine specs:**
- CPU: Intel Core i9-9900KF @ 3.60GHz (8 cores / 16 threads)
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 3080 Ti 12 GB
- OS: Ubuntu 20.04, Linux 5.15

**FLEURS Armenian (`hy_am`), dev split (50 entries after WER filter):**

| Metric | Value |
|--------|-------|
| Output tasks | 50 |
| Execution time | 40.40s |
| Throughput (tasks/sec) | 1.24 |

## Stage-wise Profiling

Per-stage metrics are collected via `TaskPerfUtils.aggregate_task_metrics()` and surfaced in `metrics.json` with keys like `task_<stage>_<metric>_<agg>`.

**FLEURS Armenian dev split (ray_data, 1 GPU):**

| Stage | process_time mean | process_time sum | idle_time sum | items |
|-------|-------------------|------------------|---------------|-------|
| CreateInitialManifestFleurs | 5.9174s | 295.87s | 0.00s | 0 |
| ASR_inference | 0.2225s | 11.12s | 3.63s | 790 |
| GetPairwiseWerStage | 0.0000s | 0.0012s | 1.16s | 50 |
| GetAudioDurationStage | 0.0005s | 0.0244s | 1.13s | 50 |
| PreserveByValueStage | 0.0000s | 0.0005s | 0.02s | 50 |
| AudioToDocumentStage | 0.0003s | 0.0134s | 0.26s | 50 |
| jsonl_writer | 0.0004s | 0.0209s | 0.02s | 50 |

**Top bottleneck:** `CreateInitialManifestFleurs` dominates with 295.87s of total process_time
because it downloads and extracts the FLEURS dataset from HuggingFace on the first run (cached
on subsequent runs). `ASR_inference` is the second most expensive stage at 11.12s total, running
GPU-accelerated NeMo FastConformer inference. All other stages are negligible (<0.03s total).

## Output Files

The benchmark produces three files in `--benchmark-results-path`:

| File | Description |
|------|-------------|
| `params.json` | All pipeline parameters for reproducibility |
| `metrics.json` | `is_success`, `time_taken_s`, `throughput_tasks_per_sec`, per-stage profiling metrics |
| `tasks.pkl` | Pickled task objects for `TaskPerfUtils` aggregation |
