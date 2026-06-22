#!/bin/bash
# DeepSeek-V4-Pro 8L PROXY pipeclean on GB200 (64 GPUs / 16 nodes), MXFP8,
# full-iteration CUDA graph + fused DSA + cuteDSL fused grouped MLP.
#
# Mirrors verify_dsv3_fp8mx_gb200_clean.sh: plain `python -m
# scripts.performance.setup_experiment` from the mbridge venv, with the dev
# Megatron-LM backbone bind-mounted over the container's bundled mcore.
#
# Stack:
#   - MBridge : this worktree (dingqingy/dsv4-pro-proxy-perf, live r0.5.0 +
#               the DSv4-Pro recipe). recipe `deepseek_v4_pro_proxy`.
#   - mcore   : megatron-lm-dev-9d46c924d-clean (dev backbone, fused DSA).
#   - container: nemo_fht_dsa.sqsh (nemo:26.06.rc7 + flash_mla + cudnn-fe 1.24
#               + cuteDSL 4.5.0 + TE 2.17 + fast-hadamard-transform).
#
# cuteDSL fused grouped MLP (NVTE_CUTEDSL_FUSED_GROUPED_MLP=1) and the full-iter
# CG alloc-conf env are auto-set by perf_plugins because the workload base config
# sets cuda_graph_impl="full_iteration" + cutedsl_fused_grouped_mlp=True.
#
# Usage:
#   DRY=1 bash submit_dsv4_pro_proxy_gb200.sh   # print sbatch, don't submit
#   bash     submit_dsv4_pro_proxy_gb200.sh

source /lustre/fsw/coreai_dlalgo_llm/dingqingy/venvs/mbridge/bin/activate

ROOTDIR=$(dirname "$PWD")
export NEMORUN_HOME=$ROOTDIR/.nemo_run
MBRIDGE=$PWD
MCORE_DEV=${MCORE_DEV:-$ROOTDIR/megatron-lm-dev-9d46c924d-clean}

CONT=${CONT:-/lustre/fsw/coreai_dlalgo_llm/dingqingy/sqsh/nemo_fht_dsa.sqsh}
CONT_NAME=$(basename "$CONT" .sqsh)
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
MODEL=deepseek
MODEL_RECIPE=deepseek_v4_pro_proxy
NUM_GPUS=${NUM_GPUS:-64}        # 16 GB200 nodes
DTYPE=fp8_mx
GPU=gb200
PARTITION=${PARTITION:-${GPU}}
PROJ=dingqingy-dsv4-pro-proxy
EXP=${MODEL_RECIPE}-${GPU}-${DTYPE}-fullitercg-fuseddsa-${CONT_NAME}

SRC_MOUNT="$MBRIDGE/src/megatron:/opt/Megatron-Bridge/src/megatron"
MCORE_MOUNT="$MCORE_DEV/megatron:/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron"
MOUNTS="$SRC_MOUNT,$MCORE_MOUNT"

DRY_FLAG=()
[[ -n "$DRY" ]] && DRY_FLAG=(-d)

PYTHONPATH=$MBRIDGE/src:$MBRIDGE/scripts/performance:$PYTHONPATH \
python -m scripts.performance.setup_experiment \
  -m $MODEL -mr $MODEL_RECIPE --task pretrain --num_gpus $NUM_GPUS \
  -a $ACCOUNT -p $PARTITION -l $NEMORUN_HOME -i $CONT --hf_token $HF_TOKEN \
  -t "00:40:00" \
  -wdk $WB_TOKEN -wdp $PROJ -wdj $EXP \
  --gpu $GPU -c $DTYPE -cv v1 \
  -gn 4 -ms 20 \
  -cm "$MOUNTS" \
  "${DRY_FLAG[@]}"
