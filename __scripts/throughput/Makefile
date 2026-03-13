DRY_RUN ?= 0
ifeq ($(DRY_RUN), 1)
	_DRY_RUN_PREFIX := @echo '
	_DRY_RUN_SUFFIX := '
else
	_DRY_RUN_PREFIX :=
	_DRY_RUN_SUFFIX :=
endif

# vLLM CLI inside container (use system-installed vllm)
_VLLM := /usr/local/bin/vllm
_WORKDIR := $(PWD)

# CUDA compat library path (needed for driver compatibility)
_LD_LIBRARY_PATH := /usr/local/cuda/compat/lib.real:/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Cluster settings
TOOLKIT_PROFILE := yul201
EVAL_JOB_ACCOUNT := snow.research.afm
IMAGE := registry.toolkit-sp.yul201.service-now.com/snow.research.afm/interactive-toolkit:25.12-py3-vllm014rc1
DATA_OBJS := --data snow.plato.slam.keys:/mnt/keys \
             --data snow.plato.slam.checkpoints:/mnt/checkpoints \
             --data snow.plato.slam.datasets:/mnt/datasets \
             --data snow.home.oleksiy_ostapenko:/home/toolkit \
             --data snow.plato.slam.hf_home:/mnt/hf_home

# Run naming
EXP_ARGS__RUN_NAME ?= vllm_bench_throughput
RUN_TIMESTAMP := $(shell date +%Y_%m_%d_%H_%M_%S)
RUN_NAME := $(EXP_ARGS__RUN_NAME)_$(RUN_TIMESTAMP)

# Resources
CPU ?= 4
CPU_MEM ?= 128
GPU ?= 1
GPU_MEM ?= 80

# vLLM throughput benchmark command
COMPILATION_CONFIG ?= {\"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}
JOB_LAUNCH_COMMAND_THROUGHPUT := bench throughput \
    --model ${MODEL} \
    --tokenizer ${TOKENIZER} \
    --max-model-len ${MAX_MODEL_LEN} \
    --random-input-len ${INPUT_LEN} --random-output-len ${OUTPUT_LEN} \
    --load-format dummy \
    --trust_remote_code \
    --output-json ${OUTPUT_JSON} \
    --num-prompts ${NUM_PROMPTS} \
    --dtype bfloat16 \
    --gpu-memory-utilization ${GPU_MEM_UTILIZATION} \
    --tensor-parallel-size ${TP} \
	--no-enable-prefix-caching \
	--compilation-config '${COMPILATION_CONFIG}'


#     --block-size 16 \
# 	--max-num-seqs 256 \
# vLLM serve command
SERVE_PORT ?= 8000
JOB_LAUNCH_COMMAND_SERVE := serve ${MODEL} \
    --tokenizer ${TOKENIZER} \
    --max-model-len ${MAX_MODEL_LEN} \
    --trust_remote_code \
    --dtype bfloat16 \
    --block-size 16 \
    --gpu-memory-utilization ${GPU_MEM_UTILIZATION} \
    --tensor-parallel-size ${TP} \
    --port ${SERVE_PORT}

.PHONY: run-vllm-serve
run-vllm-serve:
	${_DRY_RUN_PREFIX} \
	EAI_PROFILE=${TOOLKIT_PROFILE} eai job submit \
	--preemptable \
	--restartable \
	--image ${IMAGE} \
	--cpu ${CPU} \
	--mem ${CPU_MEM} \
	--gpu ${GPU} \
	--gpu-mem ${GPU_MEM} \
	--name ${RUN_NAME}_serve \
	--account ${EVAL_JOB_ACCOUNT} \
	--workdir ${_WORKDIR} \
	--env HF_HOME=/mnt/hf_home \
	--env VLLM_USE_V1=1 \
	--compilation-config '{"cache_dir": ""}' \
	--env PYTHONUSERBASE=/home/toolkit/.local \
	--env LD_LIBRARY_PATH=${_LD_LIBRARY_PATH} \
	${DATA_OBJS} \
	-- ${_VLLM} ${JOB_LAUNCH_COMMAND_SERVE} ${_DRY_RUN_SUFFIX}

.PHONY: run-vllm-throughput
run-vllm-throughput:
	${_DRY_RUN_PREFIX} \
	EAI_PROFILE=${TOOLKIT_PROFILE} eai job submit \
	--preemptable \
	--restartable \
	--image ${IMAGE} \
	--cpu ${CPU} \
	--mem ${CPU_MEM} \
	--gpu ${GPU} \
	--gpu-mem ${GPU_MEM} \
	--name ${RUN_NAME} \
	--account ${EVAL_JOB_ACCOUNT} \
	--workdir ${_WORKDIR} \
	--env HF_HOME=/mnt/hf_home \
	--env VLLM_USE_V1=1 \
	--env APRIEL2_DEBUG_SYNC=0 \
	--env VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER} \
	--env VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND} \
	--env PYTHONUSERBASE=/home/toolkit/.local \
	--env LD_LIBRARY_PATH=${_LD_LIBRARY_PATH} \
	${DATA_OBJS} \
	-- bash -c "mkdir -p $$(dirname ${OUTPUT_JSON}) && ${_VLLM} ${JOB_LAUNCH_COMMAND_THROUGHPUT}" ${_DRY_RUN_SUFFIX}
