#!/usr/bin/env bash
# Full T3 queue: run manually after the R2 small-check review, with machine idle.
# Usage from repo root: bash all_mixed/evaluation/run_latency_v2.sh
# Optional: PY=/path/to/python bash all_mixed/evaluation/run_latency_v2.sh
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$task_root"
task_python="${PY:-/home/abaragithan/miniconda3/envs/routenet-fermi/bin/python}"
task_output="results/dec1/latency_v2.json"
if [[ -e "$task_output" ]]; then
    echo "Output exists: $task_output. Preserve it before starting a new benchmark." >&2
    exit 1
fi
task_run_dir="results/dec1/latency_v2_runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p results/dec1
mkdir "$task_run_dir"
mapfile -t task_models < <("$task_python" all_mixed/evaluation/latency_v2.py --list-models)
task_results=()
for task_model in "${task_models[@]}"; do
    task_name="${task_model%%:*}"
    for task_threads in default 1; do
        task_file="$task_run_dir/${task_name}_${task_threads}.json"
        "$task_python" -u all_mixed/evaluation/latency_v2.py \
            --model "$task_model" --threads "$task_threads" --out "$task_file" \
            2>&1 | tee "$task_run_dir/${task_name}_${task_threads}.log"
        task_results+=("$task_file")
    done
done
"$task_python" all_mixed/evaluation/latency_v2.py \
    --merge "${task_results[@]}" --out "$task_output"
