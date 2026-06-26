#!/bin/bash
# DeepSeek-V4-Pro "first+last rank" PROXY on RUBIN / VR200 (128 GPUs / 32 nodes = 2 NVL blocks),
# MXFP8 + full-iter CUDA graph + fused DSA + cuteDSL grouped MLP + FGO.  recipe deepseek_v4_pro_proxy, cv=v3.
#
# 29 layers over TP1/PP2/VPP4/EP64 + MTP=1 (base DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_VR200_FP8_MX_V3,
# layout "Et*4|(t*4|)*6tmL"): reproduces the full 61L Pro's FIRST rank (embed+16L) and LAST rank
# (13L+MTP+loss), adding exactly ONE cross-NVL-domain PP boundary so its point-to-point cost can be
# measured at half scale. Compare to: the 16L/PP1/1-domain probe (no PP comm) and the 256-GPU full run.
#
# Full-Pro-style env (this proxy carries FGO, like the full run):
#   - NVTE_CPU_OFFLOAD_V1=1   REQUIRED (FGO ValueError guard, bridge config.py:1859).
#   - NCCL_NET_GDR_LEVEL=PHB, NCCL_NET_GDR_C2C=1  (FGO transport; not auto-set for vr200).
# Grouped-GEMM flags omitted (proven inert, jobs 136489/136509).
#
# Usage:
#   DRY=1 bash submit_dsv4_pro_proxy_128gpu_vr200.sh
#   bash     submit_dsv4_pro_proxy_128gpu_vr200.sh

VENV=${VENV:-/lustre/fsw/coreai_dlalgo_llm/dingqingy/2604release/bridge_env}
source "$VENV/bin/activate"

# Resolve paths from THIS script's location (not $PWD) so it runs from any cwd.
MBRIDGE="${MBRIDGE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # Megatron-Bridge checkout (script lives at its root)
ROOTDIR="$(dirname "$MBRIDGE")"
cd "$MBRIDGE" || exit 1                                             # so `python -m scripts.performance...` resolves
export NEMORUN_HOME="${NEMORUN_HOME:-$ROOTDIR/.nemo_run}"
# Pinned mcore dev commit (NVIDIA/Megatron-LM `dev`, DSv4 mHC fused kernels). The bind-mounted
# clone must be at this commit to reproduce; override MCORE_DEV, or set MCORE_COMMIT= to bypass.
MCORE_COMMIT=${MCORE_COMMIT:-9d46c924dce3818f2b5f894f7380712c780d1801}
MCORE_DEV=${MCORE_DEV:-$ROOTDIR/megatron-lm-dev}
if [ -n "$MCORE_COMMIT" ] && [ -d "$MCORE_DEV/.git" ] && \
   [ "$(git -C "$MCORE_DEV" rev-parse HEAD 2>/dev/null)" != "$MCORE_COMMIT" ]; then
  echo "WARNING: $MCORE_DEV not at pinned commit $MCORE_COMMIT" >&2
  echo "         fix: git -C '$MCORE_DEV' fetch origin && git -C '$MCORE_DEV' checkout $MCORE_COMMIT" >&2
fi

CONT=${CONT:-/lustre/fsw/coreai_dlalgo_llm/dingqingy/sqsh/nemo_mlperf_dsv4_ready.sqsh}
CONT_NAME=$(basename "$CONT" | tr ':/' '__')
ACCOUNT=${ACCOUNT:-coreai_dlalgo_llm}
MODEL=deepseek
MODEL_RECIPE=${MODEL_RECIPE:-deepseek_v4_pro_proxy}
NUM_GPUS=${NUM_GPUS:-128}         # 32 vr200 nodes x 4 GPUs = 2 NVL blocks
DTYPE=fp8_mx
GPU=vr200
PARTITION=${PARTITION:-batch-xdr}
CONSTRAINT=${CONSTRAINT:-ts5}
# Skip suspect NVL blocks nvlblk10 (137332 NCCL hang) + nvlblk15 (137363 rank process-exit). Clear with EXCLUDE=.
EXCLUDE=${EXCLUDE:-hecate[0163-0180,0253-0270]}
SLURM_PARAMS="constraint=$CONSTRAINT"
[[ -n "$EXCLUDE" ]] && SLURM_PARAMS="$SLURM_PARAMS;exclude=$EXCLUDE"
CV=${CV:-v3}                      # v3 -> 29L / PP2 / VPP4 / 128 GPU + MTP
PROJ=dingqingy-dsv4-pro-proxy-rubin
EXP=${MODEL_RECIPE}-${GPU}-${DTYPE}-fullitercg-fuseddsa-fgo-${CV}-${CONT_NAME}

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
  -t "00:50:00" \
  -wdk $WB_TOKEN -wdp $PROJ -wdj $EXP \
  --gpu $GPU -c $DTYPE -cv $CV \
  -gn 4 -ms 20 \
  -E NVTE_CPU_OFFLOAD_V1=1 \
  -E NCCL_NET_GDR_LEVEL=PHB \
  -E NCCL_NET_GDR_C2C=1 \
  --additional_slurm_params "$SLURM_PARAMS" \
  -cm "$MOUNTS" \
  "${DRY_FLAG[@]}"
