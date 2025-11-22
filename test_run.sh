export NEMORUN_HOME=/lustre/fsw/coreai_dlalgo_llm/dingqingy/MOE/.nemo_run
MBRIDGE=/lustre/fsw/coreai_dlalgo_llm/dingqingy/MOE/Megatron-Bridge
MEGATRONPATH=/lustre/fsw/coreai_dlalgo_llm/dingqingy/MOE/megatron-lm
# CONT=nvcr.io/nvidian/nemo:25.11.rc5
# CONT=nvcr.io/nvidia/nemo:25.09
CONT=gitlab-master.nvidia.com:5005/dl/joc/nemo-ci/aot_build_muon_new/train:pipe.38348673-x86
# CONT=/lustre/fsw/coreai_dlalgo_llm/dingqingy/MOE/Megatron-Bridge/containers/nemo25.09-update.sqsh
ACCOUNT=coreai_dlalgo_llm
PARTITION=batch

# # DSV3
# PYTHONPATH=$MBRIDGE/src:$MBRIDGE/scripts/performance:$PYTHONPATH  \
PYTHONPATH=$MEGATRONPATH:$MBRIDGE/src:$MBRIDGE/scripts/performance:$PYTHONPATH  \
python -m scripts.performance.setup_experiment -m deepseek -s v3 --task pretrain --num_gpus 256 \
-a $ACCOUNT -p $PARTITION -l $NEMORUN_HOME -i $CONT -hf $HF_TOKEN -t "00:40:00" \
-cm "$MBRIDGE:/opt/Megatron-Bridge,$MEGATRONPATH:/opt/megatron-lm" \
--gpu b200 \
-gn 8 -en \
-c fp8_cs

# -cm "$MBRIDGE:/opt/Megatron-Bridge" \
