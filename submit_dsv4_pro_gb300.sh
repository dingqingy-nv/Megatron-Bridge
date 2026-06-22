#!/bin/bash
# DeepSeek-V4-Pro FULL 61L on GB300 (256 GPUs / 64 nodes), MXFP8,
# full-iteration CUDA graph + fused DSA + cuteDSL fused grouped MLP + FGO.
#
# Mirrors the mcore-direct reference run A (job 2109264, ~22.13 s/iter ->
# 967 calibrated TF/s/GPU): PP4/VPP4/EP64, GBS4096, MTP=1, hash-MoE=3,
# SwiGLU clamp=10, layout Et*4|(tttt|)*14tmL, recompute mla_up_proj+mhc,
# fine-grained activation offload (core_attn,attn_proj, max-inflight 2).
# All of that is encoded in the perf recipe deepseek_v4_pro (gb300) + the
# DEEPSEEK_V4_PRO_PRETRAIN_CONFIG_GB300_FP8_MX_V1 workload base; the bridge
# derives hash/clamp/MTP/csa-ratios from the DeepSeek-V4-Pro HF config.
#
# Same plain-python / venv / bind-mount pattern as submit_dsv4_pro_proxy_gb200.sh.
# Extra vs the proxy: NVTE_CPU_OFFLOAD_V1=1 (run A's FGO CPU-offload path).
# bf16 main-grads is the MBridge default (no override needed).
#
# Usage:
#   DRY=1 bash submit_dsv4_pro_gb300.sh   # print sbatch, don't submit
#   bash     submit_dsv4_pro_gb300.sh

source /lustre/fsw/coreai_dlalgo_llm/dingqingy/venvs/mbridge/bin/activate

ROOTDIR=$(dirname "$PWD")
export NEMORUN_HOME=$ROOTDIR/.nemo_run
MBRIDGE=$PWD
MCORE_DEV=${MCORE_DEV:-$ROOTDIR/megatron-lm-dev-9d46c924d-clean}

CONT=${CONT:-/lustre/fsw/coreai_dlalgo_llm/dingqingy/sqsh/nemo_fht_dsa.sqsh}
CONT_NAME=$(basename "$CONT" .sqsh)
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
MODEL=deepseek
MODEL_RECIPE=deepseek_v4_pro
NUM_GPUS=${NUM_GPUS:-256}       # 64 GB300 nodes
DTYPE=fp8_mx
GPU=gb300
PARTITION=${PARTITION:-${GPU}}
PROJ=dingqingy-dsv4-pro
EXP=${MODEL_RECIPE}-${GPU}-${DTYPE}-fullitercg-fuseddsa-fgo-${CONT_NAME}

SRC_MOUNT="$MBRIDGE/src/megatron:/opt/Megatron-Bridge/src/megatron"
MCORE_MOUNT="$MCORE_DEV/megatron:/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron"
MOUNTS="$SRC_MOUNT,$MCORE_MOUNT"

DRY_FLAG=()
[[ -n "$DRY" ]] && DRY_FLAG=(-d)

PYTHONPATH=$MBRIDGE/src:$MBRIDGE/scripts/performance:$PYTHONPATH \
python -m scripts.performance.setup_experiment \
  -m $MODEL -mr $MODEL_RECIPE --task pretrain --num_gpus $NUM_GPUS \
  -a $ACCOUNT -p $PARTITION -l $NEMORUN_HOME -i $CONT --hf_token $HF_TOKEN \
  -t "01:10:00" \
  -wdk $WB_TOKEN -wdp $PROJ -wdj $EXP \
  --gpu $GPU -c $DTYPE -cv v1 \
  -gn 4 -ms 40 \
  -en --profiling_start_step 18 --profiling_stop_step 19 \
  --profiling_ranks 0,64,128,192 --export_nsys_sqlite \
  -E NVTE_CPU_OFFLOAD_V1=1 \
  -E NCCL_NET_GDR_LEVEL=PHB \
  -E NCCL_NET_GDR_C2C=1 \
  -cm "$MOUNTS" \
  "${DRY_FLAG[@]}"
