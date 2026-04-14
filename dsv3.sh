ROOTDIR=$(dirname "$PWD")

export NEMORUN_HOME=$ROOTDIR/.nemo_run
MBRIDGE=$ROOTDIR/Megatron-Bridge
# MEGATRONPATH=$ROOTDIR/megatron-lm


# CONT=nvcr.io/nvidian/nemo:26.04.rc2
CONT=$ROOTDIR/cont_save/nemo-26.04.rc2.sqsh

if [[ "$CONT" == nvcr.io* ]]; then
  CONT_NAME="${CONT#*nemo:}"
elif [[ "$CONT" == *.sqsh ]]; then
  CONT_NAME=$(basename "$CONT" .sqsh)
else
  CONT_NAME=$(basename "$CONT")
fi
ACCOUNT=coreai_dlalgo_llm

MODEL=deepseek
MODEL_RECIPE=deepseek_v3
NUM_GPUS=256

DTYPE=fp8_mx

for GPU in gb200; do
PARTITION=batch
PROJ=dingqingy-3-25

# BINDING="$MBRIDGE/scripts/performance:/opt/Megatron-Bridge/scripts/performance"
# BINDING="$MBRIDGE/src:/opt/Megatron-Bridge/src"

# for MBS in 4; do
# for EP in 64; do
EXP=${MODEL_RECIPE}-v2-${GPU}-${DTYPE}-${CONT_NAME}
# PYTHONPATH=$MEGATRONPATH:$MBRIDGE/src:$MBRIDGE/scripts/performance:$PYTHONPATH  \
PYTHONPATH=$MBRIDGE/src:$MBRIDGE/scripts/performance:$PYTHONPATH  \
python -m scripts.performance.setup_experiment -m $MODEL -mr $MODEL_RECIPE --task pretrain --num_gpus $NUM_GPUS \
-a $ACCOUNT -p $PARTITION -l $NEMORUN_HOME -i $CONT -hf $HF_TOKEN -t "00:30:00" \
-wdk $WB_TOKEN -wdp $PROJ -wdj $EXP \
--gpu $GPU \
-c $DTYPE \
-cv v2 \
-gn 4 -ms 3 \
--gres gpu:4 \
-ce NCCL_DEBUG=INFO,NCCL_DEBUG_FILE=/nemo_run/nccl_debug_%h_%p.log,DUMP_ENV=1 \
dist.nccl_communicator_config_path=$(readlink -f $MBRIDGE/scripts/performance/nccl_communicator_config.yaml)
done
