#!/bin/bash
# Which GPU does a job get? Decides which model size fits. Output: cluster/logs/vq-gpu_<id>.out
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=vq-gpu
#SBATCH --output=cluster/logs/%x_%j.out
#SBATCH --error=cluster/logs/%x_%j.err

hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
