#!/bin/bash

#SBATCH --job-name=2d-no-gpu-baseline-train-ann-brain-diffusion

#SBATCH --mail-type=ALL

#SBATCH --mail-user=<arina.belova@hhi.fraunhofer.de>

#SBATCH --output=output_logs/%j_%x.out

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=5

#SBATCH --gpus=0

#SBATCH --mem=32G

#####################################################################################


# meta-data
DATE=`date`
# set output directories
OUTPUTFOLDER="$SLURM_JOB_NAME-$SLURM_JOB_ID"
OUTPUTPATH_JOB="/opt/output"
SUBMIT_DIR=`pwd`
OUTPUTPATH_LOCAL="$SUBMIT_DIR/runs/$OUTPUTFOLDER"
# create temporary output directory
#source "/etc/slurm/local_job_dir.sh"
export LOCAL_JOB_DIR=/data/local/jobs/${SLURM_JOB_ID}
mkdir -p "${LOCAL_JOB_DIR}/job_results"

nvidia-smi

# # for wandb certificates to work:
export SSL_CERT_FILE=${SLURM_SUBMIT_DIR}/cacert.pem
export APPTAINERENV_PYCORTEX_STORE="/data/datapool3/datasets/nsd_betas_condavg/"
# Train
apptainer exec --nv --bind ${LOCAL_JOB_DIR},src:/opt/app/src,/data/datapool3/datasets/nsd_betas_condavg/,my_pycortex_db:/opt/conda/envs/diffusion_brain/share/pycortex/db \
--env PYTHONPATH=/opt/app/src \
./cluster/diffusion-brain.sif \
bash -c "python ${SLURM_SUBMIT_DIR}/src/diffusion_brain/scripts/train_sklearn.py --config ${SLURM_SUBMIT_DIR}/src/diffusion_brain/configs/brain/2d_sklearn_config_train.yaml \
--jobid ${SLURM_JOB_NAME}-${SLURM_JOB_ID} --override data.roi=$1 data.ann_model_weights=$2 data.ann_model=$3 data.roi_file=$4"


echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo " CONTENTS                 PATH                                                  "
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"
echo " job_results              $OUTPUTPATH_LOCAL"
echo " .out file                ${SUBMIT_DIR}/runs/${SLURM_JOB_ID}"
echo "________________________________________________________________________________"