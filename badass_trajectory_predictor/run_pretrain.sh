#!/bin/bash

# Submit the pretraining job
PRETRAIN_JOB_ID=$(sbatch --parsable run_pretrain.sh)
if [ -z "$PRETRAIN_JOB_ID" ]; then
    echo "Failed to submit pretrain job"
    exit 1
fi
echo "Submitted pretrain job with ID $PRETRAIN_JOB_ID"

# Submit the training job, dependent on the pretraining job completing
TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$PRETRAIN_JOB_ID run_train.sh)
if [ -z "$TRAIN_JOB_ID" ]; then
    echo "Failed to submit train job"
    exit 1
fi
echo "Submitted train job with ID $TRAIN_JOB_ID, dependent on pretrain job $PRETRAIN_JOB_ID"

# Submit the fine-tuning job, dependent on the training job completing
FINETUNE_JOB_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB_ID run_finetune.sh)
if [ -z "$FINETUNE_JOB_ID" ]; then
    echo "Failed to submit finetune job"
    exit 1
fi
echo "Submitted finetune job with ID $FINETUNE_JOB_ID, dependent on train job $TRAIN_JOB_ID"
