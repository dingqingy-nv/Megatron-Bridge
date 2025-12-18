ROOTDIR=$(dirname "$PWD")
MBRIDGE=$ROOTDIR/Megatron-Bridge
MEGATRONPATH=$ROOTDIR/megatron-lm
SRC_CONT=nvcr.io/nvidia/nemo:25.11
DEST_CONT=$ROOTDIR/cont_save/nemo2511.sqsh

ACCOUNT=coreai_dlalgo_llm
# only on eos
# PARTITION=interactive
# PARTITION=batch
PARTITION=gb200

# # direct local save: for faster container startup
# srun -A $ACCOUNT -N1 -p $PARTITION --container-image=$SRC_CONT --container-save=$DEST_CONT -J $ACCOUNT-test. true

# read-only, but can update mount. For example, build mock dataset
srun -A $ACCOUNT -N1 -p $PARTITION --container-image=$DEST_CONT -J $ACCOUNT-test. --container-mounts="$MBRIDGE:/opt/Megatron-Bridge,$MEGATRONPATH:/opt/megatron-lm" --pty bash

# persistent, update the container. For example, install emerging optimizers.
# srun -A $ACCOUNT -N1 -p $PARTITION --container-image=$SRC_CONT --container-save=$DEST_CONT -J $ACCOUNT-test. --container-mounts="$MBRIDGE:/opt/Megatron-Bridge,$MEGATRONPATH:/opt/megatron-lm" --pty bash
