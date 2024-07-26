#!/bin/bash

# Job ID of the pretraining job
PRETRAIN_JOB_ID=50360

# Function to check if the job is still running
check_job_status() {
    local job_id=$1
    squeue --job $job_id --noheader | awk '{print $1}'
}

# Wait for the pretraining job to complete
while [ "$(check_job_status $PRETRAIN_JOB_ID)" ]; do
    echo "Job $PRETRAIN_JOB_ID is still running. Waiting..."
    sleep 60  # Wait for 1 minute before checking again
done

echo "Job $PRETRAIN_JOB_ID has completed. Starting fine-tuning..."

# Start the fine-tuning job
sbatch run_finetune.sh
