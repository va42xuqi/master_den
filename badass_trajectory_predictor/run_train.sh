#!/bin/bash

#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem-per-gpu=30G
#SBATCH --error=sbatch/train_from_scratch_%A_%a.err
#SBATCH --output=sbatch/train_from_scratch_%A_%a.out
#SBATCH --array=0-41  # Array indices for 42 jobs

# Define arrays for architectures and scenes
ARCHS=("ostf" "oslstm" "oslmu" "one_layer_linear" "two_layer_linear" "os_bitnet" "uni_lstm" "uni_lmu" "uni_bitnet" "uni_trafo" "pos_lstm" "vel_lstm" "pos_lmu" "vel_lmu" "pos_bitnet" "vel_bitnet" "pos_trafo" "vel_trafo" "pos_1l_linear" "vel_1l_linear" "pos_2l_linear" "vel_2l_linear")
SCENES=("SOC" "NBA")

# Calculate indices
NUM_ARCHS=${#ARCHS[@]}
NUM_SCENES=${#SCENES[@]}
TASK_ID=$SLURM_ARRAY_TASK_ID
ARCH_IDX=$((TASK_ID % NUM_ARCHS))
SCENE_IDX=$(((TASK_ID / NUM_ARCHS) % NUM_SCENES))

ARCH=${ARCHS[$ARCH_IDX]}
SCENE=${SCENES[$SCENE_IDX]}

# Run training from scratch
srun --gres=gpu:1 python train.py --arch=$ARCH --scene=$SCENE --mode=train
