MBRIDGE=/lustre/fsw/coreai_dlalgo_llm/dingqingy/MOE/Megatron-Bridge
MEGATRONPATH=/lustre/fsw/coreai_dlalgo_llm/dingqingy/MOE/megatron-lm

SRC_CONT=/lustre/fsw/coreai_dlalgo_llm/gdeng/cont_save/nemo2511rc7.sqsh
DEST_CONT=/lustre/fsw/coreai_dlalgo_llm/dingqingy/MOE/cont_save/nemo2511rc7.sqsh
# CONT=gitlab-master.nvidia.com:5005/dl/joc/nemo-ci/aot_build_muon_new/train:pipe.38348673-x86
# CONT=/lustre/fsw/coreai_dlalgo_llm/dingqingy/MOE/Megatron-Bridge/containers/nemo25.09-update.sqsh
ACCOUNT=coreai_dlalgo_llm
PARTITION=batch

# read-only
# srun -A $ACCOUNT -N1 -p $PARTITION --container-image=$CONT -J $ACCOUNT-test. --container-mounts="$MBRIDGE:/opt/Megatron-Bridge,$MEGATRONPATH:/opt/megatron-lm" --pty bash

# persistent
srun -A $ACCOUNT -N1 -p $PARTITION --container-image=$SRC_CONT --container-save=$DEST_CONT -J $ACCOUNT-test. --container-mounts="$MBRIDGE:/opt/Megatron-Bridge,$MEGATRONPATH:/opt/megatron-lm" --pty bash
