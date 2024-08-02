#!/bin/bash

# Submit the pretraining job
PRETRAIN_JOB_ID=$(sbatch --parsable run_pretrain.sh)
echo "Submitted pretrain job with ID $PRETRAIN_JOB_ID"

# Submit the training job, dependent on the pretraining job completing
TRAIN_JOB_ID=$(sbatch --parsable run_train.sh)
echo "Submitted train job with ID $TRAIN_JOB_ID"

# Submit the fine-tuning job, dependent on both the pretraining and training jobs completing
FINETUNE_JOB_ID=$(sbatch --parsable --dependency=afterok:$PRETRAIN_JOB_ID run_finetune.sh)
echo "Submitted finetune job with ID $FINETUNE_JOB_ID, dependent on pretrain job $PRETRAIN_JOB_ID"

