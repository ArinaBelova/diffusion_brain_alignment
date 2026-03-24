#!/bin/bash

#SBATCH --job-name=train-ann-brain-diffusion

#SBATCH --mail-type=ALL

#SBATCH --mail-user=<arina.belova@hhi.fraunhofer.de>

#SBATCH --output=output_logs/%j_%x.out

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --gpus=1

#SBATCH --mem=40G

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

#CUDA_VISIBLE_DEVICES=0,1

echo "${SLURM_GPUS_ON_NODE} GPUs are visible to the job, and will be used for training"

# # for wandb certificates to work:
export SSL_CERT_FILE=${SLURM_SUBMIT_DIR}/cacert.pem
export APPTAINERENV_NETRC=/data/cluster/users/belova/.netrc
export APPTAINERENV_WANDB_CONFIG_DIR=/data/cluster/users/belova/
export APPTAINERENV_WANDB_DIR=${SLURM_SUBMIT_DIR}
export APPTAINERENV_PYCORTEX_STORE="/data/datapool3/datasets/nsd_betas_condavg/"
export NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
export MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=lo
export APPTAINERENV_MASTER_ADDR=${MASTER_ADDR}
export APPTAINERENV_MASTER_PORT=${MASTER_PORT}
export APPTAINERENV_NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY}
export APPTAINERENV_GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}
# Train
apptainer exec --nv --bind /data/cluster/users/belova/.netrc,${LOCAL_JOB_DIR},src:/opt/app/src,/data/datapool3/datasets/nsd_betas_condavg/,my_pycortex_db:/opt/conda/envs/diffusion_brain/share/pycortex/db \
--env PYTHONPATH=/opt/app/src \
./cluster/diffusion-brain.sif \
bash -c "python ${SLURM_SUBMIT_DIR}/src/diffusion_brain/scripts/train.py --config ${SLURM_SUBMIT_DIR}/src/diffusion_brain/configs/brain/config_train.yaml --jobid ann-brain-cluster-${SLURM_JOB_ID}"

#bash -c "torchrun --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${SLURM_SUBMIT_DIR}/src/diffusion_brain/scripts/train.py --config ${SLURM_SUBMIT_DIR}/src/diffusion_brain/configs/brain/config_train.yaml --jobid ann-brain-cluster-${SLURM_JOB_ID}"


echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo " CONTENTS                 PATH                                                  "
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"
echo " job_results              $OUTPUTPATH_LOCAL"
echo " .out file                ${SUBMIT_DIR}/runs/${SLURM_JOB_ID}"
echo "________________________________________________________________________________"
