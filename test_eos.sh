ROOTDIR=$(dirname "$PWD")
export NEMORUN_HOME=$ROOTDIR/.nemo_run
MBRIDGE=$ROOTDIR/Megatron-Bridge
MEGATRONPATH=$ROOTDIR/megatron-lm

CONT=$ROOTDIR/cont_save/nemo2511rc7.sqsh
ACCOUNT=coreai_dlalgo_llm
PARTITION=batch

PROJ=dingqingy-12-23-kimi-k2
EXP=eos-29-layers-PP8

# kimi k2
PYTHONPATH=$MEGATRONPATH:$MBRIDGE/src:$MBRIDGE/scripts/performance:$PYTHONPATH  \
python -m scripts.performance.setup_experiment -m kimi -mr kimi_k2 --task pretrain --num_gpus 1024 \
-a $ACCOUNT -p $PARTITION -l $NEMORUN_HOME -i $CONT -hf $HF_TOKEN -t "00:30:00" \
-wdk $WB_TOKEN -wdp $PROJ -wdj $EXP \
-cm "$MBRIDGE:/opt/Megatron-Bridge,$MEGATRONPATH:/opt/megatron-lm" \
--gpu h100 -ms 20 \
-gn 8 