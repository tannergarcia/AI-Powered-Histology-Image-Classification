#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --output=logs/train_model_%j.out
#SBATCH --error=logs/train_model_%j.err
#SBATCH --mail-user=user@ufl.edu  # insert your email to receive notifications, or remove this line
#SBATCH --mail-type=ALL
#SBATCH --time=06:00:00                # Adjust as needed
#SBATCH --mem=64G                     # Adjust memory as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=4

module load tensorflow/2.7.0

python resnet.py
