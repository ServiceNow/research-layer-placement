#!/usr/bin/env bash
set -euo pipefail

# Batch evaluation orchestrator: evaluate multiple placements sequentially,
# with all tasks running in parallel sharing ALL servers within each placement.
#
# Instead of statically allocating server subsets to tasks (launch_eval_suite.sh),
# this script gives ALL servers to ALL tasks via a multi-evaluator config.
# When a fast task finishes, remaining tasks naturally absorb the freed bandwidth.
#
# Usage:
#   bash __scripts/launch_eval_batch.sh \
#     --name apriel2_1p6/comparison \
#     --placements all-attention,every2nd-kda,optimized-v3 \
#     --ids 300-349
#
#   # Filter to specific tasks:
#   bash __scripts/launch_eval_batch.sh \
#     --name apriel2_1p6/math_only \
#     --placements all-attention \
#     --ids 300-309 \
#     --tasks aime25,aime24,math500
#
#   # Read placements from file (one per line, # comments allowed):
#   bash __scripts/launch_eval_batch.sh \
#     --name apriel2_1p6/sweep \
#     --placements-file placements.txt \
#     --ids 300-349
#
#   # Dry run to verify commands:
#   bash __scripts/launch_eval_batch.sh \
#     --name test --placements all-attention --ids 300-304 --dry-run
#
#   # Extra Hydra overrides after --:
#   bash __scripts/launch_eval_batch.sh \
#     --name test --placements all-attention --ids 300-319 \
#     -- evaluators.configs.mmlu.max_problems_per_task=10

DNS="${DNS:-http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id}"
PORT="${PORT:-8080}"
DRY_RUN=""

NAME=""
PLACEMENTS=""
PLACEMENTS_FILE=""
IDS_RANGE=""
TASKS_FILTER=""
OUTPUT_ROOT="outputs/evals"
CONCURRENT="4"
TIMEOUT="28000"
MATH_MAX_TOKENS=""
MMLU_MAX_PROBLEMS=""
EXTRA_OVERRIDES=""
NOHUP_DIR="${NOHUP_DIR:-nohup_outs}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --placements) PLACEMENTS="$2"; shift 2 ;;
    --placements-file) PLACEMENTS_FILE="$2"; shift 2 ;;
    --ids) IDS_RANGE="$2"; shift 2 ;;
    --dns) DNS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --tasks) TASKS_FILTER="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --concurrent) CONCURRENT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --math-max-tokens) MATH_MAX_TOKENS="$2"; shift 2 ;;
    --mmlu-max-problems) MMLU_MAX_PROBLEMS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA_OVERRIDES="$*"; break ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# --- Validation ---
if [[ -z "$NAME" || -z "$IDS_RANGE" ]]; then
  cat >&2 <<'USAGE'
Usage: launch_eval_batch.sh --name <name> --ids <range> --placements <p1,p2,...> [options]

Required:
  --name            Output subdirectory under outputs/evals/
  --ids             Server ID range (e.g., 300-319 or 300,305,310-315)
  --placements      Comma-separated placement config names
  --placements-file File with one placement per line (alternative to --placements)

Optional:
  --tasks           Comma-separated task filter (evaluator names from eval_suite)
                    Valid: aime25,aime24,math500,gsm8k,ruler,niah,mmlu,swde,fda
  --concurrent      Max concurrent requests per server (default: 4)
  --timeout         Request timeout in seconds (default: 28000)
  --math-max-tokens Override math500 max_new_tokens (e.g., 15000)
  --mmlu-max-problems Override MMLU max_problems_per_task (e.g., 10)
  --output-root     Output root directory (default: outputs/evals)
  --dns             DNS prefix (default: $DNS env var)
  --port            Server port (default: 8080)
  --dry-run         Print commands without executing
  -- <overrides>    Extra Hydra overrides passed verbatim
USAGE
  exit 1
fi

if [[ -z "$PLACEMENTS" && -z "$PLACEMENTS_FILE" ]]; then
  echo "ERROR: Must specify --placements or --placements-file" >&2
  exit 1
fi

# --- Build placement list ---
PLACEMENT_LIST=()
if [[ -n "$PLACEMENTS" ]]; then
  IFS=',' read -ra PLACEMENT_LIST <<< "$PLACEMENTS"
fi
if [[ -n "$PLACEMENTS_FILE" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"        # strip comments
    line="${line// /}"        # strip whitespace
    [[ -n "$line" ]] && PLACEMENT_LIST+=("$line")
  done < "$PLACEMENTS_FILE"
fi

if (( ${#PLACEMENT_LIST[@]} == 0 )); then
  echo "ERROR: No placements specified" >&2
  exit 1
fi

# --- Build server URL list ---
ALL_IDS=()
IFS=',' read -ra ID_PARTS <<< "$IDS_RANGE"
for part in "${ID_PARTS[@]}"; do
  if [[ "$part" == *-* ]]; then
    IDS_START="${part%-*}"
    IDS_END="${part#*-}"
    for ((i=IDS_START; i<=IDS_END; i++)); do
      ALL_IDS+=("$i")
    done
  else
    ALL_IDS+=("$part")
  fi
done

SERVER_URL=""
for id in "${ALL_IDS[@]}"; do
  if [[ -z "$SERVER_URL" ]]; then
    SERVER_URL="\"${DNS}-${id}:${PORT}\""
  else
    SERVER_URL="${SERVER_URL},\"${DNS}-${id}:${PORT}\""
  fi
done
SERVER_LIST="[${SERVER_URL}]"

NUM_SERVERS=${#ALL_IDS[@]}
OUTPUT_BASE="${OUTPUT_ROOT}/${NAME}"

mkdir -p "$NOHUP_DIR"

# --- Print summary ---
echo "============================================"
echo "Batch Eval Orchestrator"
echo "============================================"
echo "Name:        ${NAME}"
echo "Output:      ${OUTPUT_BASE}"
echo "Servers:     ${NUM_SERVERS} (IDs: ${IDS_RANGE})"
echo "Concurrent:  ${CONCURRENT}/server"
echo "Timeout:     ${TIMEOUT}s"
echo "Placements:  ${#PLACEMENT_LIST[@]}"
for p in "${PLACEMENT_LIST[@]}"; do
  echo "  - ${p}"
done
echo "Tasks:       ${TASKS_FILTER:-all (${#ALL_EVAL_NAMES[@]} evaluators)}"
echo "============================================"
echo ""

# --- All known evaluator names in eval_suite ---
ALL_EVAL_NAMES=(aime25 aime24 math500 gsm8k ruler niah mmlu swde fda gpqa mmlu_pro ifeval)

# --- Build command for a placement ---
build_cmd() {
  local placement="$1"

  local cmd=(
    python -m place_layers
    "+command=evaluate"
    "+evaluators=eval_suite"
    "+backend=rest"
    "+placement=${placement}"
    "backend.servers=${SERVER_LIST}"
    "backend.max_concurrent_per_server=${CONCURRENT}"
    "backend.timeout_seconds=${TIMEOUT}"
  )

  # Task filter: remove evaluators not in the filter via Hydra ~ deletion
  if [[ -n "$TASKS_FILTER" ]]; then
    IFS=',' read -ra FILTER_TASKS <<< "$TASKS_FILTER"
    for eval_name in "${ALL_EVAL_NAMES[@]}"; do
      local keep=false
      for ft in "${FILTER_TASKS[@]}"; do
        if [[ "$ft" == "$eval_name" ]]; then
          keep=true
          break
        fi
      done
      if [[ "$keep" == "false" ]]; then
        cmd+=("~evaluators.configs.${eval_name}")
      fi
    done
  fi

  # Optional per-evaluator overrides
  if [[ -n "$MATH_MAX_TOKENS" ]]; then
    cmd+=("evaluators.configs.math500.max_new_tokens=${MATH_MAX_TOKENS}")
  fi
  if [[ -n "$MMLU_MAX_PROBLEMS" ]]; then
    cmd+=("evaluators.configs.mmlu.max_problems_per_task=${MMLU_MAX_PROBLEMS}")
  fi

  # Extra Hydra overrides
  if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    read -r -a extras <<< "$EXTRA_OVERRIDES"
    cmd+=("${extras[@]}")
  fi

  echo "${cmd[@]}"
}

# --- Main loop: sequential placements, each as nohup background job ---
NOHUP_PREFIX="${NOHUP_DIR}/nohup_$(echo "$NAME" | tr '/' '_')"
total=${#PLACEMENT_LIST[@]}
idx=0

for placement in "${PLACEMENT_LIST[@]}"; do
  idx=$((idx + 1))
  echo "========== Placement ${idx}/${total}: ${placement} =========="

  cmd_str="$(build_cmd "$placement")"
  logfile="${NOHUP_PREFIX}_${placement}.out"

  if [[ -n "$DRY_RUN" ]]; then
    echo "[dry-run] OUTPUT_BASE=\"${OUTPUT_BASE}\" ${cmd_str}"
    echo "[dry-run] Log: ${logfile}"
    echo ""
    continue
  fi

  echo "Log: ${logfile}"
  echo "Running: ${cmd_str}"
  echo ""

  # Launch as nohup background job and wait for completion before next placement
  nohup env OUTPUT_BASE="${OUTPUT_BASE}" bash -c "${cmd_str}" > "${logfile}" 2>&1 &
  PID=$!
  echo "PID: ${PID}"

  # Wait for this placement to finish before starting the next
  if wait "$PID"; then
    echo "Placement ${placement} complete (exit 0)."
  else
    EXIT_CODE=$?
    echo "WARNING: Placement ${placement} exited with code ${EXIT_CODE}. Check ${logfile}" >&2
    echo "Continuing to next placement..."
  fi
  echo ""
done

echo "============================================"
echo "All ${total} placements processed."
echo "Results: ${OUTPUT_BASE}/"
echo "Logs:    ${NOHUP_PREFIX}_*.out"
echo "============================================"
