ROOTDIR=$(dirname "$PWD")
MBRIDGE=$ROOTDIR/Megatron-Bridge
MEGATRONPATH=$ROOTDIR/megatron-lm
SRC_CONT=nvcr.io/nvidia/nemo:25.11
DEST_CONT=$ROOTDIR/cont_save/nemo2511.sqsh
# CONT=gitlab-master.nvidia.com:5005/dl/joc/nemo-ci/aot_build_muon_new/train:pipe.38348673-x86
# CONT=/lustre/fsw/coreai_dlalgo_llm/dingqingy/MOE/Megatron-Bridge/containers/nemo25.09-update.sqsh
ACCOUNT=coreai_dlalgo_llm
# only on eos
# PARTITION=interactive
PARTITION=batch

# read-only
srun -A $ACCOUNT -N1 -p $PARTITION --container-image=$DEST_CONT -J $ACCOUNT-test. --container-mounts="$MBRIDGE:/opt/Megatron-Bridge,$MEGATRONPATH:/opt/megatron-lm" --pty bash

# persistent
# srun -A $ACCOUNT -N1 -p $PARTITION --container-image=$SRC_CONT --container-save=$DEST_CONT -J $ACCOUNT-test. --container-mounts="$MBRIDGE:/opt/Megatron-Bridge,$MEGATRONPATH:/opt/megatron-lm" --pty bash

# direct local save
# srun -A $ACCOUNT -N1 -p $PARTITION --container-image=$SRC_CONT --container-save=$DEST_CONT -J $ACCOUNT-test. true
