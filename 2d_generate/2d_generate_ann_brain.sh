#!/bin/bash

#SBATCH --job-name=2d-cluster-generate-ann-brain-diffusion

#SBATCH --mail-type=ALL

#SBATCH --mail-user=<arina.belova@hhi.fraunhofer.de>

#SBATCH --output=output_logs/%j_%x.out

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16

#SBATCH --gpus=1

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
export LOCAL_JOB_DIR=/data/local/jobs/${SLURM_JOB_ID}
mkdir -p "${LOCAL_JOB_DIR}/job_results"

# # for wandb certificates to work:
export SSL_CERT_FILE=${SLURM_SUBMIT_DIR}/cacert.pem
export APPTAINERENV_NETRC=/data/cluster/users/belova/.netrc
export APPTAINERENV_WANDB_CONFIG_DIR=/data/cluster/users/belova/
export APPTAINERENV_WANDB_DIR=${SLURM_SUBMIT_DIR}
# Train
apptainer exec --nv --bind /data/cluster/users/belova/.netrc,${LOCAL_JOB_DIR},src:/opt/app/src,src:/opt/app/src,/data/datapool3/datasets/nsd_betas_condavg/,/data/datapool3/datasets/full_nsd_betas/,my_pycortex_db:/opt/conda/envs/diffusion_brain/share/pycortex/db \
--env PYTHONPATH=/opt/app/src,CUDA_LAUNCH_BLOCKING=1 \
./cluster/diffusion-brain.sif \
bash -c "python ${SLURM_SUBMIT_DIR}/src/diffusion_brain/scripts/generate.py --config ${SLURM_SUBMIT_DIR}/src/diffusion_brain/configs/brain/2d_config_generate.yaml  \
--jobid ${SLURM_JOB_NAME}-${SLURM_JOB_ID} --override data.roi=$1 data.ann_model_weights=$2 data.ann_model=$3 data.roi_file=$4 validation.guidance_scale=$5 model.run_id=$6 validation.ode=$7 validation.n_steps=$8 data.grid_resolution_2d=$9" 

# information about the outputs of the script
echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo " CONTENTS                 PATH                                                  "
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"
echo " job_results              $OUTPUTPATH_LOCAL"
echo " .out file                ${SUBMIT_DIR}/runs/${SLURM_JOB_ID}"
echo "________________________________________________________________________________"