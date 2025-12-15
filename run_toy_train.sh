#!/bin/bash

#SBATCH --job-name=train-toy-diffusion

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
#source "/etc/slurm/local_job_dir.sh"
export LOCAL_JOB_DIR=/data/local/jobs/${SLURM_JOB_ID}
mkdir -p "${LOCAL_JOB_DIR}/job_results"
#export APPTAINER_BINDPATH="${APPTAINER_BINDPATH},${LOCAL_JOB_DIR}"
#cp -r ${SLURM_SUBMIT_DIR}/cache_datasets ${LOCAL_JOB_DIR}

# Launch the apptainer image with --nv for nvidia support. Two bind mounts are used:
# - One for the ImageNet dataset and
# - One for the results (e.g. checkpoint data that you may store in $LOCAL_JOB_DIR on the node
# For debugging disable wandb
# apptainer exec --nv --bind ${LOCAL_JOB_DIR} \
# ./Reflected-Diffusion/cluster/reflected.sif \
# wandb enabled; 

#CUDA_VISIBLE_DEVICES=0,1

# # for wandb certificates to work:
export SSL_CERT_FILE=${SLURM_SUBMIT_DIR}/cacert.pem
# Train
apptainer exec --nv --bind ${LOCAL_JOB_DIR},src:/opt/app/src \
--env PYTHONPATH=/opt/app/src \
./cluster/diffusion-brain.sif \
bash -c "python ${SLURM_SUBMIT_DIR}/src/diffusion_brain/scripts/train.py --config ${SLURM_SUBMIT_DIR}/src/diffusion_brain/configs/mnist/config_train.yaml --jobid toy-cluster"
#bash -c "python -m diffusion_brain.scripts.train --config ${SLURM_SUBMIT_DIR}/src/diffusion_brain/configs/mnist/config_train.yaml --jobid toy-cluster"

#source /opt/conda/bin/activate diffusion-brain
# poetry install --no-root
# poetry env use /opt/conda/envs/diffusion-brain/bin/python

# poetry run python ...
# python scripts/train.py --config configs/mnist/config_train.yaml --jobid toy
#python -m torch.distributed.launch --nproc_per_node=2  --master_port=1111 ${SLURM_SUBMIT_DIR}/GOUB/codes/tasks/deraining/train.py -opt=GOUB/codes/tasks/deraining/options/train.yml --launcher=pytorch

#wandb login "7d001f095c395e6c0fc4c23d85c2e9831b089ea7"

# copying results from local
# mkdir -p $OUTPUTPATH_LOCAL
# cp -r ${LOCAL_JOB_DIR}/job_results/* ${SUBMIT_DIR}/GOUB/job_results
# rm -r ${LOCAL_JOB_DIR}/job_results
# also copy output
#cp "${SUBMIT_DIR}/runs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.out" "${SUBMIT_DIR}/runs/${SLURM_JOB_ID}"

# information about the outputs of the script
echo "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
echo " CONTENTS                 PATH                                                  "
echo "――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――"
echo " job_results              $OUTPUTPATH_LOCAL"
echo " .out file                ${SUBMIT_DIR}/runs/${SLURM_JOB_ID}"
echo "________________________________________________________________________________"