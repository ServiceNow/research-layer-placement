#!/usr/bin/env bash
set -euo pipefail

# Launch a full eval suite (all tasks) for a given model/placement/ID range.
#
# Usage:
#   bash __scripts/launch_eval_suite.sh \
#     --name apriel2_1p6_frm1p5_distil_annealed_cpt40k/surogate_search_lieklihood_6.4cost_cpt40k \
#     --placement surogate_search_lieklihood_6.4cost_cpt40k \
#     --ids 300-319
#
#   bash __scripts/launch_eval_suite.sh \
#     --name apriel2_1p6_frm1p5_distil_annealed_cpt10k/all_attn \
#     --placement all-attention \
#     --ids 320-339
#
# The --ids range is split across tasks with no overlap:
#   AIME25 (5), AIME24 (6), MATH (3), GSM8K (2), RULER (1), NIAH (1), MMLU (1), Extraction (1)
#   Total: 20 IDs required.
#
# To relaunch only specific tasks (e.g. after a failure):
#   bash __scripts/launch_eval_suite.sh \
#     --name foo/bar --placement all-attention --ids 300-319 \
#     --tasks aime25,mmlu,extraction

DNS="${DNS:-http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id}"
PORT="${PORT:-8080}"
DRY_RUN=""

# Parse arguments
NAME="ap1p6_frm1p5_distil_annealed_cpt10k_mixo_gs/all_attn"
PLACEMENT="all-attention"
IDS_RANGE="300-339"
TASKS_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --placement) PLACEMENT="$2"; shift 2 ;;
    --ids) IDS_RANGE="$2"; shift 2 ;;
    --dns) DNS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --tasks) TASKS_FILTER="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$NAME" || -z "$PLACEMENT" || -z "$IDS_RANGE" ]]; then
  echo "Usage: $0 --name <output_name> --placement <placement_cfg> --ids <start>-<end>" >&2
  echo "  --name        Output subdirectory under outputs/evals/ (e.g. model/variant)" >&2
  echo "  --placement   Hydra placement config name (e.g. all-attention)" >&2
  echo "  --ids         Server ID range, e.g. 300-319 (exactly 20 IDs required)" >&2
  echo "  --dns         DNS prefix (default: \$DNS)" >&2
  echo "  --port        Server port (default: 8080)" >&2
  echo "  --tasks       Comma-separated task filter: aime25,aime24,math500,gsm8k,ruler,niah,mmlu,extraction" >&2
  echo "                (default: all tasks)" >&2
  echo "  --dry-run     Print commands without executing" >&2
  exit 1
fi

# Parse ID range into array
IDS_START="${IDS_RANGE%-*}"
IDS_END="${IDS_RANGE#*-}"
ALL_IDS=()
for ((i=IDS_START; i<=IDS_END; i++)); do
  ALL_IDS+=("$i")
done

NUM_IDS=${#ALL_IDS[@]}
if (( NUM_IDS < 9 )); then
  echo "ERROR: Need at least 9 IDs (one per task), got $NUM_IDS ($IDS_RANGE)" >&2
  exit 1
fi

# Allocate IDs to tasks proportionally (no overlaps).
# Weights: AIME25=5, AIME24=6, MATH=3, GSM8K=2, RULER=1, NIAH=1, MMLU=1, Extraction=1
# Each task gets at least 1, the rest are distributed by weight.
TASK_NAMES=(aime25 aime24 math500 gsm8k ruler_qa_squad niah mmlu extraction)
TASK_WEIGHTS=(5 6 3 2 1 1 1 1)
TOTAL_WEIGHT=0
for w in "${TASK_WEIGHTS[@]}"; do TOTAL_WEIGHT=$((TOTAL_WEIGHT + w)); done

# Distribute: each gets max(1, floor(NUM_IDS * weight / TOTAL_WEIGHT)), then assign remainders
TASK_COUNTS=()
assigned=0
for w in "${TASK_WEIGHTS[@]}"; do
  c=$(( NUM_IDS * w / TOTAL_WEIGHT ))
  (( c < 1 )) && c=1
  TASK_COUNTS+=("$c")
  assigned=$((assigned + c))
done
# Distribute remaining IDs to heaviest tasks first
remainder=$((NUM_IDS - assigned))
i=0
while (( remainder > 0 )); do
  TASK_COUNTS[$i]=$(( ${TASK_COUNTS[$i]} + 1 ))
  remainder=$((remainder - 1))
  i=$(( (i + 1) % ${#TASK_COUNTS[@]} ))
done

# Slice IDs for each task
offset=0
for i in "${!TASK_NAMES[@]}"; do
  count=${TASK_COUNTS[$i]}
  eval "IDS_${TASK_NAMES[$i]^^}=\"${ALL_IDS[*]:$offset:$count}\""
  offset=$((offset + count))
done

# Map uppercase names back to variables
IDS_AIME25="${IDS_AIME25}"
IDS_AIME24="${IDS_AIME24}"
IDS_MATH="${IDS_MATH500}"
IDS_GSM8K="${IDS_GSM8K}"
IDS_RULER="${IDS_RULER_QA_SQUAD}"
IDS_NIAH="${IDS_NIAH}"
IDS_MMLU="${IDS_MMLU}"
IDS_EXTRACTION="${IDS_EXTRACTION}"

OUTPUT_BASE="outputs/evals/${NAME}"
NOHUP_PREFIX="nohup_outs/nohup_$(echo "$NAME" | tr '/' '_')"

mkdir -p nohup_outs

echo "Eval suite: ${NAME}"
echo "Placement:  ${PLACEMENT}"
echo "Output:     ${OUTPUT_BASE}"
echo "IDs:        ${IDS_RANGE} (${NUM_IDS} servers)"
echo "Tasks:      ${TASKS_FILTER:-all}"
echo ""
echo "  AIME25:     ${IDS_AIME25}"
echo "  AIME24:     ${IDS_AIME24}"
echo "  MATH:       ${IDS_MATH}"
echo "  GSM8K:      ${IDS_GSM8K}"
echo "  RULER:      ${IDS_RULER}"
echo "  NIAH:       ${IDS_NIAH}"
echo "  MMLU:       ${IDS_MMLU}"
echo "  Extraction: ${IDS_EXTRACTION}"
echo ""

should_run() {
  local suffix="$1"
  if [[ -z "$TASKS_FILTER" ]]; then
    return 0
  fi
  [[ ",$TASKS_FILTER," == *",$suffix,"* ]]
}

launch() {
  local task="$1" evaluator="$2" generation="$3" ids="$4" timeout="$5" concurrent="$6" suffix="$7"
  shift 7
  # Remaining args are extra env vars

  if ! should_run "$suffix"; then
    echo "Skipping ${suffix} (not in --tasks filter)"
    return
  fi

  local cmd=(
    nohup env
    OUTPUT_BASE="${OUTPUT_BASE}"
    DNS="${DNS}"
    IDS="${ids}"
    PORT="${PORT}"
    TIMEOUT_SECONDS="${timeout}"
    PLACEMENT_CFG="${PLACEMENT}"
    NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER="${concurrent}"
    TASK_CFG="${task}" EVALUATOR_CFG="${evaluator}" GENERATION_CFG="${generation}"
  )
  # Append extra env vars
  for extra in "$@"; do
    cmd+=("$extra")
  done
  cmd+=(
    bash __scripts/eval_all_layers.sh
  )

  local outfile="${NOHUP_PREFIX}_${suffix}.out"

  if [[ -n "$DRY_RUN" ]]; then
    echo "[dry-run] ${cmd[*]} > ${outfile} 2>&1 &"
  else
    echo "Launching ${suffix}..."
    "${cmd[@]}" > "${outfile}" 2>&1 &
  fi
}

launch aime25_strip_end generative greedy_very_very_long "$IDS_AIME25" 12000 4 aime25
launch aime24_strip_end generative greedy_very_very_long "$IDS_AIME24" 12000 4 aime24
launch math_500_strip_end generative greedy_very_long "$IDS_MATH" 2500 4 math500
launch gsm8k_strip_end generative greedy_long "$IDS_GSM8K" 1800 4 gsm8k
launch ruler_qa_squad generative_extraction_ruller.yaml greedy_short "$IDS_RULER" 1800 15 ruler_qa_squad
launch niah_single_2 generative_extraction_ruller_long.yaml greedy_short "$IDS_NIAH" 1800 15 niah
launch mmlu likelihood greedy_long "$IDS_MMLU" 1800 10 mmlu "MAX_PROBLEMS_PER_TASK=10"
launch swde,fda extraction extraction "$IDS_EXTRACTION" 1800 4 extraction

echo ""
echo "All tasks launched. Logs: ${NOHUP_PREFIX}_*.out"
