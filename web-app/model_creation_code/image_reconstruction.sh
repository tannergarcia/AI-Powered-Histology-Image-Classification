#!/bin/bash

#SBATCH --job-name=image_reconstruction
#SBATCH --output=logs/image_reconstruction_%j.out
#SBATCH --error=logs/image_reconstruction_%j.err
#SBATCH --mail-user=tannergarcia@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00                # Adjust as needed
#SBATCH --mem=64G                     # Adjust memory as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

date;hostname;pwd
mkdir -p logs

/blue/vabfmc/data/working/tannergarcia/DermHisto/conda/envs/image_splitting/bin/python 1792x1792_image_reconstruction.py
