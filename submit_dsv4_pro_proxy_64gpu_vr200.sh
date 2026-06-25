#!/bin/bash
# DeepSeek-V4-Pro 8L PROXY scale-up on RUBIN / VR200 (64 GPUs / 16 nodes),
# MXFP8 + full-iteration CUDA graph + fused DSA + cuteDSL fused grouped MLP.
#
# This is the NVL-domain scale-up sibling of submit_dsv4_pro_proxy_vr200.sh (the 3L/8-GPU
# fail-fast pipeclean that passed as job 134695). Same plain-python / venv / bind-mount
# pattern; the ONLY differences vs the 3L toy script are:
#   - NUM_GPUS  : 64 (16 vr200 nodes x 4) instead of 8.
#   - CV        : v2 -> deepseek_v4_pro_proxy_pretrain_config_vr200 selects 8 layers,
#                 TP1/PP1/EP64, GBS2048 (mirrors the GB200 single-NVL72 proxy).
#   - CONT      : defaults to the VALIDATED container nemo_mlperf_dsv4_ready.sqsh
#                 (the 3L toy script still defaults to the older nemo_rubin_dsv4.sqsh;
#                 the 134695 run overrode CONT to the ready sqsh).
#   - MB_DEST   : /workspace/Megatron-Bridge (the MLPerf-based ready container's layout).
# Everything else is identical to the toy run: batch-xdr partition + constraint=ts5,
# dev-mcore + bridge bind-mounts.
#
# NOTE: the two grouped-GEMM env flags the original toy script carried
# (CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL, NVTE_GROUPED_LINEAR_SINGLE_PARAM) are
# intentionally OMITTED -- proven inert by control job 136489 (3L toy, both removed,
# 20/20, 0 nan/0 skipped). NVTE_GROUPED_LINEAR_SINGLE_PARAM was a misattribution; neither
# is in the MLPerf or Blackwell recipes. (Caveat: 136489 was EP8; this is EP64.)
#
# Usage:
#   DRY=1 bash submit_dsv4_pro_proxy_64gpu_vr200.sh   # print sbatch, don't submit (inspect first!)
#   bash     submit_dsv4_pro_proxy_64gpu_vr200.sh

# --- host launcher venv: setup_experiment needs only nemo_run (recipe is built inside the
# container at runtime), so any light venv with nemo_run works. Reusing the toy run's. ---
VENV=${VENV:-/lustre/fsw/coreai_dlalgo_llm/dingqingy/2604release/bridge_env}
source "$VENV/bin/activate"

ROOTDIR=$(dirname "$PWD")
export NEMORUN_HOME=$ROOTDIR/.nemo_run
MBRIDGE=$PWD

# Megatron-LM `dev` ToT, bind-mounted over the container's bundled mcore.
MCORE_DEV=${MCORE_DEV:-$ROOTDIR/megatron-lm-dev}

# Validated DSv4-on-Rubin container (MLPerf DSv3 base + the 4 overlays; trained 134695 @ 20/20).
CONT=${CONT:-/lustre/fsw/coreai_dlalgo_llm/dingqingy/sqsh/nemo_mlperf_dsv4_ready.sqsh}
CONT_NAME=$(basename "$CONT" | tr ':/' '__')
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
MODEL=deepseek
MODEL_RECIPE=${MODEL_RECIPE:-deepseek_v4_pro_proxy}
NUM_GPUS=${NUM_GPUS:-64}          # 16 vr200 nodes x 4 GPUs
DTYPE=fp8_mx
GPU=vr200
PARTITION=${PARTITION:-batch-xdr}   # TS5 is in batch-xdr (NOT a gpu-named partition)
CONSTRAINT=${CONSTRAINT:-ts5}       # pin the TS5 rack
# Skip suspect NVL blocks nvlblk10 (137332 NCCL hang) + nvlblk15 (137363 rank process-exit). Clear with EXCLUDE=.
EXCLUDE=${EXCLUDE:-hecate[0163-0180,0253-0270]}
SLURM_PARAMS="constraint=$CONSTRAINT"
[[ -n "$EXCLUDE" ]] && SLURM_PARAMS="$SLURM_PARAMS;exclude=$EXCLUDE"
CV=${CV:-v2}                        # v2 -> 16 layers on 64 GPUs (TP1/PP1/EP64)
PROJ=dingqingy-dsv4-pro-proxy-rubin
EXP=${MODEL_RECIPE}-${GPU}-${DTYPE}-fullitercg-fuseddsa-${CV}-${CONT_NAME}

# Where the container expects the bridge tree. The MLPerf-based ready container uses
# /workspace/Megatron-Bridge (set MB_DEST to overlay OUR code there).
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
  -t "00:40:00" \
  -wdk $WB_TOKEN -wdp $PROJ -wdj $EXP \
  --gpu $GPU -c $DTYPE -cv $CV \
  -gn 4 -ms 20 \
  --additional_slurm_params "$SLURM_PARAMS" \
  -cm "$MOUNTS" \
  "${DRY_FLAG[@]}"
