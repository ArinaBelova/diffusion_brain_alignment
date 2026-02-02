#!/bin/bash

#SBATCH --job-name=handle-dvc

#SBATCH --mail-type=ALL

#SBATCH --mail-user=<arina.belova@hhi.fraunhofer.de>

#SBATCH --output=output_logs/%j_%x.out

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2

#SBATCH --gpus=0

#SBATCH --mem=32G


apptainer exec --nv ./cluster/diffusion-brain.sif \
bash -c "python -m dvc"