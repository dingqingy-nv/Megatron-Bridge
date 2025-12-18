ROOTDIR=$(dirname "$PWD")

export NEMORUN_HOME=$ROOTDIR/.nemo_run
MBRIDGE=$ROOTDIR/Megatron-Bridge
MEGATRONPATH=$ROOTDIR/megatron-lm

CONT=$ROOTDIR/cont_save/nemo2511.sqsh
ACCOUNT=coreai_dlalgo_llm
GPU=gb200
DTYPE=fp8_mx

PARTITION=$GPU

# Qwen 3
PROJ=dingqingy-12-18
EXP=2511-${GPU}-${DTYPE}-nsys
PYTHONPATH=$MEGATRONPATH:$MBRIDGE/src:$MBRIDGE/scripts/performance:$PYTHONPATH  \
python -m scripts.performance.setup_experiment -m qwen3 -s 235b_a22b --task pretrain --num_gpus 64 \
-a $ACCOUNT -p $PARTITION -l $NEMORUN_HOME -i $CONT -hf $HF_TOKEN -t "00:40:00" \
-wdk $WB_TOKEN -wdp $PROJ -wdj $EXP \
-cm "$MBRIDGE:/opt/Megatron-Bridge,$MEGATRONPATH:/opt/megatron-lm" \
--gpu $GPU \
-en \
-c $DTYPE \
-gn 4 
# -c bf16 \
