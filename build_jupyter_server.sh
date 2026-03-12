#!/bin/bash

# SLURM job configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arina.belovar@hhi.fraunhofer.de
#SBATCH --output=out_jupyter_server/%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00

# Check if image exists
if [ ! -f 'cluster/jupyter-diffusion-brain.sif' ]
then
    apptainer build --nv cluster/jupyter-diffusion-brain.sif cluster/jupyter_environment_yml_to_sif.def
fi

# Run container sbalign_jupyter
apptainer run --nv cluster/jupyter-diffusion-brain.sif &

# Monitor output for job information
while true
do
    # Locate the Jupyter runtime directory and extract the most recent server address and token
    JUPYTER_RUNTIME_DIR=~/.local/share/jupyter/runtime/
    MOST_RECENT_JSON=$(ls -t "$JUPYTER_RUNTIME_DIR"jpserver-*.json | head -n 1)
    JN_SERVER=$(grep -Eo '"url": "[^"]*' "$MOST_RECENT_JSON" | sed 's/"url": "//')
    TOKEN=$(grep -Eo '"token": "[^"]*' "$MOST_RECENT_JSON" | sed 's/"token": "//')
    FULL_URL="${JN_SERVER}tree?token=${TOKEN}"
    
    CPU_USAGE=$(top -b -n 1 -u belova | awk 'NR>7 { sum += $9; } END { print sum; }')
    MEM_USAGE=$(top -b -n 1 -u belova | awk 'NR>7 { sum += $10; } END { print sum; }')
    TIME_LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o %L)
    JUPYTER_KERNELS=$(ps -ef | grep 'jupyter/runtime/kernel' | grep -v 'grep' | wc -l)
    
    echo "========================================"
    echo "         System Monitoring Report       "
    echo "========================================"
    echo "Jupyter Information:"
    echo "  Server Address       : $FULL_URL"
    echo "  Running Kernels      : $JUPYTER_KERNELS"
    echo "----------------------------------------"
    echo "Resource Usage:"
    echo "  CPU Usage [%]        : $CPU_USAGE"
    echo "  Memory Usage [%]     : $MEM_USAGE"
    echo "----------------------------------------"
    echo "Job Information:"
    echo "  Time Left            : $TIME_LEFT"
    echo "========================================"
    sleep 5  # Wait 5 seconds
done