# Set to "unaveraged" to use full repetitions (3x per image), or "averaged" for averaged betas
DATA_VARIANT="averaged"  # "averaged" or "unaveraged"

ROIS=(5 18) #(18 5)
ROI_FILES=("streams" "HCP_MMP1") #("HCP_MMP1" "streams") #
CONDITIONING_STRENGTH=(5) # 3 was used in the last 2d trainig
# WEIGHTS=("ResNet50_Weights.IMAGENET1K_V2" "ResNet101_Weights.IMAGENET1K_V2" "ResNet152_Weights.IMAGENET1K_V2")
# MODELS=("resnet50" "resnet101" "resnet152")
ODES=(0) # 0 for sampling without ODE, 1 for sampling with ODE
WEIGHTS=("IMAGENET1K_SWAG_E2E_V1") #("ResNet50_Weights.IMAGENET1K_V2")
MODELS=("vit_b_16") #("resnet50")
TRAINED_MODEL_FILES=(266004 279633) #(280181 280179 280182 280180) #(280182 280180) #(266004 279633) #(266004 280181 280182) #(211379 210760) #(197894 197972) # 201512 (197894 197972)
NUM_GENERATION_STEPS=(100) # 200 500 1000 250 400 500)
GRID_RESOLUTIONS=(1.0) # in mm, for 2d case only, it will determine the grid size based on the bounding box of the ROI vertices;
THRESHOLDING=("none") # none / static / dynamic; only used for unaveraged variant
SUBJS=("subj01" "subj01") #("subj02" "subj02" "subj05" "subj05")

# Select sbatch script based on data variant
if [ "$DATA_VARIANT" == "unaveraged" ]; then
    SBATCH_SCRIPT="2d_generate/2d_unaveraged_generate_ann_brain.sh"
elif [ "$DATA_VARIANT" == "averaged" ]; then
    SBATCH_SCRIPT="2d_generate/2d_generate_ann_brain.sh"
else
    echo "ERROR: DATA_VARIANT must be 'averaged' or 'unaveraged', got '$DATA_VARIANT'"
    exit 1
fi

echo "Using data variant: $DATA_VARIANT (script: $SBATCH_SCRIPT)"

nvidia-smi

#for trained_model_file in "${TRAINED_MODEL_FILES[@]}"; do
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    weight="${WEIGHTS[$i]}"
    for j in "${!ROIS[@]}"; do
        trained_model_file="${TRAINED_MODEL_FILES[$j]}"
        echo "Generation for model trained with run id $trained_model_file"
        roi="${ROIS[$j]}"
        roi_file="${ROI_FILES[$j]}"
        subj="${SUBJS[$j]}"
        for cond_strength in "${CONDITIONING_STRENGTH[@]}"; do
            for ode in "${ODES[@]}"; do
                for num_steps in "${NUM_GENERATION_STEPS[@]}"; do
                    for grid_res in "${GRID_RESOLUTIONS[@]}"; do
                        for thresholding in "${THRESHOLDING[@]}"; do
                            echo "Submitting job for model $model with weights $weight and ROI $roi with ROI file $roi_file, subject $subj and conditioning strength $cond_strength; Sampling with ODE: $ode; Number of generation steps: $num_steps; Grid resolution for 2d case: $grid_res mm; thresholding: $thresholding"
                            sbatch -p gpu1 $SBATCH_SCRIPT $roi $weight $model $roi_file $cond_strength $trained_model_file $ode $num_steps $grid_res $thresholding $subj
                        done
                    done
                done
            done
        done
    done
done