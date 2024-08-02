#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem-per-gpu=30G
#SBATCH --error=sbatch/pretrain_%A_%a.err
#SBATCH --output=sbatch/pretrain_%A_%a.out
#SBATCH --array=0-43  # Array indices for 44 jobs

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
