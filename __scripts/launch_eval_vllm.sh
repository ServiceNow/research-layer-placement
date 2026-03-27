#!/usr/bin/env bash
set -euo pipefail

# Evaluate tasks against native vLLM servers (fixed architecture models).
# No placement switching — just runs all tasks once per model.
#
# Usage:
#   bash __scripts/launch_eval_vllm.sh \
#     --name nemotron_12b \
#     --url https://116e75a4-....job.toolkit-sp.yul201.service-now.com \
#     --model nvidia/NVIDIA-Nemotron-Nano-12B-v2
#
#   # Filter tasks:
#   bash __scripts/launch_eval_vllm.sh \
#     --name nemotron_12b \
#     --url https://job-url.com \
#     --model nvidia/NVIDIA-Nemotron-Nano-12B-v2 \
#     --tasks gpqa,mmlu_pro,ifeval
#
#   # Dry run:
#   bash __scripts/launch_eval_vllm.sh --name test --url http://localhost:8000 --model my-model --dry-run

DRY_RUN=""
NAME=""
URL=""
MODEL=""
TASKS_FILTER=""
OUTPUT_ROOT="outputs/evals"
CONCURRENT="32"
TIMEOUT="28000"
EXTRA_OVERRIDES=""
NOHUP_DIR="${NOHUP_DIR:-nohup_outs}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tasks) TASKS_FILTER="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --concurrent) CONCURRENT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA_OVERRIDES="$*"; break ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$NAME" || -z "$URL" || -z "$MODEL" ]]; then
  cat >&2 <<'USAGE'
Usage: launch_eval_vllm.sh --name <name> --url <vllm_url> --model <model_name> [options]

Required:
  --name            Output subdirectory under output-root/
  --url             vLLM server URL (e.g., https://job-id.job.toolkit-sp.yul201.service-now.com)
  --model           Model name as served by vLLM (e.g., nvidia/NVIDIA-Nemotron-Nano-12B-v2)

Optional:
  --tasks           Comma-separated task filter (evaluator names from eval_suite)
                    Valid: aime25,aime24,math500,gsm8k,ruler,niah,mmlu,swde,fda,gpqa,mmlu_pro,ifeval
  --concurrent      Max concurrent requests (default: 32)
  --timeout         Request timeout in seconds (default: 28000)
  --output-root     Output root directory (default: outputs/evals)
  --dry-run         Print command without executing
  -- <overrides>    Extra Hydra overrides passed verbatim
USAGE
  exit 1
fi

OUTPUT_BASE="${OUTPUT_ROOT}/${NAME}"
mkdir -p "$NOHUP_DIR"

# All known evaluator names in eval_suite
ALL_EVAL_NAMES=(aime25 aime24 math500 gsm8k ruler niah mmlu swde fda gpqa mmlu_pro ifeval)

echo "============================================"
echo "vLLM Eval"
echo "============================================"
echo "Name:        ${NAME}"
echo "Output:      ${OUTPUT_BASE}"
echo "URL:         ${URL}"
echo "Model:       ${MODEL}"
echo "Concurrent:  ${CONCURRENT}"
echo "Timeout:     ${TIMEOUT}s"
echo "Tasks:       ${TASKS_FILTER:-all (${#ALL_EVAL_NAMES[@]} evaluators)}"
echo "============================================"
echo ""

# Build command
cmd=(
  python -m place_layers
  "+command=evaluate"
  "+evaluators=eval_suite"
  "+backend=vllm"
  "backend.client.base_url=${URL}"
  "backend.client.model=${MODEL}"
  "backend.client.timeout_seconds=${TIMEOUT}"
  "backend.client.max_concurrent_requests=${CONCURRENT}"
  "placement=null"
)

# Task filter: remove evaluators not in the filter via Hydra ~ deletion
if [[ -n "$TASKS_FILTER" ]]; then
  IFS=',' read -ra FILTER_TASKS <<< "$TASKS_FILTER"
  for eval_name in "${ALL_EVAL_NAMES[@]}"; do
    local_keep=false
    for ft in "${FILTER_TASKS[@]}"; do
      if [[ "$ft" == "$eval_name" ]]; then
        local_keep=true
        break
      fi
    done
    if [[ "$local_keep" == "false" ]]; then
      cmd+=("~evaluators.configs.${eval_name}")
    fi
  done
fi

# Extra Hydra overrides
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  read -r -a extras <<< "$EXTRA_OVERRIDES"
  cmd+=("${extras[@]}")
fi

logfile="${NOHUP_DIR}/nohup_vllm_${NAME}.out"

if [[ -n "$DRY_RUN" ]]; then
  echo "[dry-run] OUTPUT_BASE=\"${OUTPUT_BASE}\" ${cmd[*]}"
  echo "[dry-run] Log: ${logfile}"
  exit 0
fi

echo "Log: ${logfile}"
echo "Running: ${cmd[*]}"
echo ""

nohup env OUTPUT_BASE="${OUTPUT_BASE}" bash -c "${cmd[*]}" > "${logfile}" 2>&1 &
PID=$!

echo "Launched. PID: ${PID}"
echo "Log: ${logfile}"
