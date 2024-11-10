#!/bin/bash

#SBATCH --job-name=feature_extraction               # Job name
#SBATCH --output=logs/feature_extraction_%j.out      # Standard output log (%j will be replaced by the job ID)
#SBATCH --error=logs/feature_extraction_%j.err       # Standard error log
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

echo "Running feature extraction..."
python scripts/extract_features.py