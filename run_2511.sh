ROOTDIR=$(dirname "$PWD")

export NEMORUN_HOME=$ROOTDIR/.nemo_run
MBRIDGE=$ROOTDIR/Megatron-Bridge
MEGATRONPATH=$ROOTDIR/megatron-lm

CONT=$ROOTDIR/cont_save/nemo2511.sqsh
ACCOUNT=coreai_dlalgo_llm
PARTITION=batch

# Qwen 3
PROJ=dingqingy-12-17
EXP=2511-fp8_mx-torchprof
PYTHONPATH=$MEGATRONPATH:$MBRIDGE/src:$MBRIDGE/scripts/performance:$PYTHONPATH  \
python -m scripts.performance.setup_experiment -m qwen3 -s 235b_a22b --task pretrain --num_gpus 64 \
-a $ACCOUNT -p $PARTITION -l $NEMORUN_HOME -i $CONT -hf $HF_TOKEN -t "00:40:00" \
-wdk $WB_TOKEN -wdp $PROJ -wdj $EXP \
-cm "$MBRIDGE:/opt/Megatron-Bridge,$MEGATRONPATH:/opt/megatron-lm" \
--gpu b200 \
-en \
-c fp8_mx \
-gn 8 
# -c bf16 \
