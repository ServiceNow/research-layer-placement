#!/usr/bin/env bash
set -euo pipefail

# Evaluate per-layer "single KDA swap" architecture variants.
#
# Default behavior:
# - Queries a running place_layers server (/info) for num_layers and mixer_names
# - For each layer i, builds a placement with REPLACE_MIXER at i and BASE_MIXER elsewhere
# - Runs evaluation on MMLU tasks (likelihood evaluator by default)
#
# Example (REST backend against a running place_layers server):
#   OUTPUT_BASE=outputs/mmlu_kda \
#   SERVER_URL="http://0.0.0.0:8080" \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   MAX_JOBS=4 MAX_PROBLEMS_PER_TASK=10 \
#   bash _scripts/eval_all_layers.sh

# SERVER_URL="${SERVER_URL:-http://0.0.0.0:8080}"

TASK_CFG="${TASK_CFG:-mmlu}"                 # e.g. mmlu, mmlu_quick, mmlu_test
EVALUATOR_CFG="${EVALUATOR_CFG:-likelihood}"   # e.g. likelihood, generative
GENERATION_CFG="${GENERATION_CFG:-greedy}" # e.g. greedy, nucleus
COMMAND="${COMMAND:-evaluate}"
PLACEMENT_CFG="${PLACEMENT_CFG:-}"
# USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-true}" # whether to use chat templates in the evaluator (if supported by the evaluator)

# Architecture variant spec
BASE_MIXER="${BASE_MIXER:-attention}"        # attention or sliding_window
REPLACE_MIXER="${REPLACE_MIXER:-kda}"        # kda
NUM_LAYERS="${NUM_LAYERS:-48}"              # optional, queried from server if not set
START_LAYER="${START_LAYER:-0}"        # optional, default 0
LAYOUTS_PATH="${LAYOUTS_PATH:-}" # optional, if set, should be a JSONL file with {"layout": [...]} per line. If set, overrides START_LAYER and NUM_LAYERS and iterates over layouts in the file instead.

DNS="${DNS:-http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id}"                     # whether SERVER_URL is a DNS name
IDS=("${IDS:-200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 300 301 302 303 304 305 306 307 308 309 310 311}") # whether SERVER_URL is a DNS name
PORT="${PORT:-8080}"
# construct SERVER_URL: for each ID in IDS, make DNS-ID:PORT
SERVER_URL=""
for id in ${IDS[@]}; do
  if [[ -z "$SERVER_URL" ]]; then
    SERVER_URL="${DNS}-${id}:${PORT}"
  else
    SERVER_URL="${SERVER_URL}\",\"${DNS}-${id}:${PORT}"
  fi
done
SERVER_URL="\"${SERVER_URL}\""


# Backend overrides (REST server pool)
MAX_CONCURRENT_PER_SERVER="${MAX_CONCURRENT_PER_SERVER:-8}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"

# Runtime
MAX_JOBS="1"                    # number of concurrent eval jobs
MAX_PROBLEMS_PER_TASK="${MAX_PROBLEMS_PER_TASK:-null}" # optional, e.g. 20

RUN_PREFIX="${RUN_PREFIX:-${TASK_CFG}_${EVALUATOR_CFG}_${BASE_MIXER}_swap_${REPLACE_MIXER}}"



OUTPUT_BASE="${OUTPUT_BASE:-outputs/mmlu_kda_sweep_${BASE_MIXER}_to_${REPLACE_MIXER}}"
mkdir -p "$OUTPUT_BASE"

info_json="$(curl -fsS "${SERVER_URL%/}/info" || true)"
if [[ -z "$info_json" && -z "$NUM_LAYERS" ]]; then
  echo "ERROR: Failed to GET ${SERVER_URL%/}/info and NUM_LAYERS is not set." >&2
  echo "Set NUM_LAYERS=<n> or point SERVER_URL at a place_layers server." >&2
  exit 1
fi


echo "Output base: $OUTPUT_BASE"
echo "Server:      $SERVER_URL"
echo "Task cfg:    $TASK_CFG"
echo "Evaluator:   $EVALUATOR_CFG"
echo "Num layers:  $NUM_LAYERS"
echo "Base mixer:  $BASE_MIXER"
echo "Replace:     $REPLACE_MIXER"
echo "Max jobs:    $MAX_JOBS"

tasks_str="${TASK_CFG//,/ }"
read -r -a TASKS <<<"$tasks_str"
if (( ${#TASKS[@]} == 0 )); then
  echo "ERROR: TASK_CFG is empty. Provide one or more task names." >&2
  exit 1
fi
echo "Tasks:       ${TASKS[*]}"

make_placement() {
  local idx="$1"
  NUM_LAYERS="$NUM_LAYERS" BASE_MIXER="$BASE_MIXER" REPLACE_MIXER="$REPLACE_MIXER" IDX="$idx" \
    python - <<'PY'
import json
import os

n = int(os.environ["NUM_LAYERS"])
base = os.environ["BASE_MIXER"]
rep = os.environ["REPLACE_MIXER"]
i = int(os.environ["IDX"])

if i < 0 or i > n:
    raise SystemExit(f"idx out of range: {i} for n={n}")

p = [base] * n
i -= 1 # -1 is all base
if i >= 0:
    p[i] = rep
print(json.dumps(p))
PY
}

make_placement_config() {
  local idx="$1"
  # if LAYOUTS_PATH is given, extrat file {idx}.yaml 
  local cfg_name="layer_${idx}"                                                                                                                                                                                                                            
  # If LAYOUTS_PATH is given, use pre-made layout files instead of generating                                                                                                                                                  
  if [[ -n "${LAYOUTS_PATH:-}" ]]; then                                                                                                                                                                                                    
      echo "$cfg_name"                                                                                                                                                                                                           
     return                                                                                                                                                                                                                     
  fi 

  local placement_configs_root="${OUTPUT_BASE}/_placement_configs"
  local cfg_dir="${placement_configs_root}/placement"
  local cfg_file="${cfg_dir}/${cfg_name}.yaml"

  mkdir -p "$cfg_dir"
  if [[ ! -f "$cfg_file" ]]; then
    local placement
    placement="$(make_placement "$idx")"
    {
      echo "# @package _global_"
      echo "# Auto-generated by _scripts/eval_all_layers.sh"
      echo "placement: ${placement}"
    } >"$cfg_file"
  fi

  echo "$cfg_name"
}

run_one() {
  local idx="$1"
  local placement_cfg
  if [[ -n "$PLACEMENT_CFG" ]]; then
    placement_cfg=${PLACEMENT_CFG}
  else
    placement_cfg="$(make_placement_config "$idx")"
    echo "Layer $idx placement config: $placement_cfg"
  fi

  for task in "${TASKS[@]}"; do
    local cmd=(
      python -m place_layers
      "+command=${COMMAND}"
      "+evaluator=${EVALUATOR_CFG}"
      +backend=rest
      "+task=${task}"
      "+generation=${GENERATION_CFG}"
      "+placement=${placement_cfg}"
      "backend.max_concurrent_per_server=${MAX_CONCURRENT_PER_SERVER}"
      "backend.timeout_seconds=${TIMEOUT_SECONDS}"
      "backend.servers=[${SERVER_URL}]"
    )

    # Add custom searchpath only when using generated/layout configs (not PLACEMENT_CFG)
    if [[ -z "$PLACEMENT_CFG" ]]; then
      local placement_configs_root_abs
      if [[ -n "$LAYOUTS_PATH" ]]; then
        placement_configs_root_abs="$LAYOUTS_PATH"
      else
        placement_configs_root_abs="$(
        P="${OUTPUT_BASE}/_placement_configs" python - <<'PY'
import os
print(os.path.abspath(os.environ["P"]))
PY
        )"
      fi
      cmd+=("hydra.searchpath=[file://${placement_configs_root_abs}]")
    fi

    if [[ -n "${MAX_PROBLEMS_PER_TASK}" ]]; then
      cmd+=("task.max_problems_per_task=${MAX_PROBLEMS_PER_TASK}")
    fi

    echo "[$idx/$((NUM_LAYERS))][task=${task}] ${cmd[*]}"
    OUTPUT_BASE="$OUTPUT_BASE" "${cmd[@]}"
  done
}

running=0
for ((i=START_LAYER; i<=NUM_LAYERS; i++)); do
  run_one "$i"
  #  &
  # running=$((running+1))
  # if (( running >= MAX_JOBS )); then
  #   wait -n
  #   running=$((running-1))
  # fi
  sleep 5
done

wait
echo "Done."
