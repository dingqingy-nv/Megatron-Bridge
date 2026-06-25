#!/bin/bash
# DeepSeek-V4-Pro FULL 61L on RUBIN / VR200 (256 GPUs / 64 nodes), MXFP8,
# full-iteration CUDA graph + fused DSA + cuteDSL fused grouped MLP + FGO (CPU activation offload).
#
# Same workload/parallelism as the validated GB300 full Pro: PP4/VPP4/EP64, GBS4096, MTP=1,
# layout Et*4|(tttt|)*14tmL, recompute mla_up_proj+mhc, fine-grained activation offloading of
# core_attn/attn_proj. Encoded in recipe `deepseek_v4_pro` + base
# DEEPSEEK_V4_PRO_PRETRAIN_CONFIG_VR200_FP8_MX_V1 (= the GB300 full config). The vr200 builder
# (deepseek_v4_pro_pretrain_config_vr200) adds the Rubin-only SwiGLU-clamp fix.
#
# Full-Pro-only env flags (mirrors submit_dsv4_pro_gb300.sh):
#   - NVTE_CPU_OFFLOAD_V1=1   REQUIRED: fine-grained activation offloading raises a ValueError
#                             without it (TE>=2.10 guard, bridge training/config.py:1859).
#   - NCCL_NET_GDR_LEVEL=PHB
#   - NCCL_NET_GDR_C2C=1      help the FGO CPU-offload transport. MBridge auto-sets these ONLY
#                             for gpu==gb200 (executors.py:110), NOT vr200 -> pass explicitly.
# The two grouped-GEMM flags (CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL, NVTE_GROUPED_LINEAR_SINGLE_PARAM)
# are intentionally OMITTED -- proven inert on Rubin (jobs 136489 EP8, 136509 EP64).
#
# Usage:
#   DRY=1 bash submit_dsv4_pro_256gpu_vr200.sh   # print sbatch, don't submit (inspect first!)
#   bash     submit_dsv4_pro_256gpu_vr200.sh

VENV=${VENV:-/lustre/fsw/coreai_dlalgo_llm/dingqingy/2604release/bridge_env}
source "$VENV/bin/activate"

ROOTDIR=$(dirname "$PWD")
export NEMORUN_HOME=$ROOTDIR/.nemo_run
MBRIDGE=$PWD
MCORE_DEV=${MCORE_DEV:-$ROOTDIR/megatron-lm-dev}

# Validated DSv4-on-Rubin container (trained the 3L toy 134695 + 8L proxy 136509).
CONT=${CONT:-/lustre/fsw/coreai_dlalgo_llm/dingqingy/sqsh/nemo_mlperf_dsv4_ready.sqsh}
CONT_NAME=$(basename "$CONT" | tr ':/' '__')
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
MODEL=deepseek
MODEL_RECIPE=${MODEL_RECIPE:-deepseek_v4_pro}    # full 61L (not the proxy)
NUM_GPUS=${NUM_GPUS:-256}        # 64 vr200 nodes x 4 GPUs
DTYPE=fp8_mx
GPU=vr200
PARTITION=${PARTITION:-batch-xdr}   # TS5 is in batch-xdr
CONSTRAINT=${CONSTRAINT:-ts5}       # pin the TS5 rack
CV=${CV:-v1}
PROJ=dingqingy-dsv4-pro-rubin
EXP=${MODEL_RECIPE}-${GPU}-${DTYPE}-fullitercg-fuseddsa-fgo-${CONT_NAME}

MB_DEST=${MB_DEST:-/workspace/Megatron-Bridge}
SRC_MOUNT="$MBRIDGE/src/megatron:$MB_DEST/src/megatron"
MCORE_MOUNT="$MCORE_DEV/megatron:$MB_DEST/3rdparty/Megatron-LM/megatron"
MOUNTS="$SRC_MOUNT,$MCORE_MOUNT"

DRY_FLAG=()
[[ -n "$DRY" ]] && DRY_FLAG=(-d)

PYTHONPATH=$MBRIDGE/src:$MBRIDGE/scripts/performance:$PYTHONPATH \
python -m scripts.performance.setup_experiment \
  -m $MODEL -mr $MODEL_RECIPE --task pretrain --num_gpus $NUM_GPUS \
  -a $ACCOUNT -p $PARTITION -l $NEMORUN_HOME -i $CONT --hf_token $HF_TOKEN \
  -t "01:10:00" \
  -wdk $WB_TOKEN -wdp $PROJ -wdj $EXP \
  --gpu $GPU -c $DTYPE -cv $CV \
  -gn 4 -ms 20 \
  -E NVTE_CPU_OFFLOAD_V1=1 \
  -E NCCL_NET_GDR_LEVEL=PHB \
  -E NCCL_NET_GDR_C2C=1 \
  --additional_slurm_params "constraint=$CONSTRAINT" \
  -cm "$MOUNTS" \
  "${DRY_FLAG[@]}"
