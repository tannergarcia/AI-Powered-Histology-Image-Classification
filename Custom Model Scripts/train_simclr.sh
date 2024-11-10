#!/bin/bash

#SBATCH --job-name=simclr_training               # Job name
#SBATCH --output=logs/simclr_training_%j.out      # Standard output log (%j will be replaced by the job ID)
#SBATCH --error=logs/simclr_training_%j.err       # Standard error log
#SBATCH --time=48:00:00                           # Maximum runtime (adjust as needed)
#SBATCH --mem=32G                                 # Memory allocation (adjust as needed)
#SBATCH --ntasks=1                                # Number of tasks
#SBATCH --cpus-per-task=8                         # Number of CPU cores per task
#SBATCH --partition=gpu                           # Use the GPU partition (update based on your SLURM config)
#SBATCH --gres=gpu:a100:1                         # making sure to use the nvidia one

# Load necessary modules (adjust based on your environment)
ml pytorch/2.2.0        # Load PyTorch module with GPU support

cd /home/francokrepel/blue/vabfmc/data/working/d.uriartediaz/francokrepel/project-root

export PYTHONPATH="${PYTHONPATH}:/home/francokrepel/blue/vabfmc/data/working/d.uriartediaz/francokrepel/project-root"

# Run the SimCLR training script
python scripts/train_simclr.py # --epochs 100 --batch_size 256 --learning_rate 0.001

