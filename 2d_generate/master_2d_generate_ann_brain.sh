ROIS=(5) #(18 5)
ROI_FILES=("streams") #("HCP_MMP1" "streams") #
CONDITIONING_STRENGTH=(20)
# WEIGHTS=("ResNet50_Weights.IMAGENET1K_V2" "ResNet101_Weights.IMAGENET1K_V2" "ResNet152_Weights.IMAGENET1K_V2")
# MODELS=("resnet50" "resnet101" "resnet152")
ODES=(0) # 0 for sampling without ODE, 1 for sampling with ODE
WEIGHTS=("IMAGENET1K_SWAG_E2E_V1") #("ResNet50_Weights.IMAGENET1K_V2")
MODELS=("vit_b_16") #("resnet50")
TRAINED_MODEL_FILES=(223400) #(211379 210760) #(197894 197972) # 201512 (197894 197972)
NUM_GENERATION_STEPS=(10 100) # 250 400 500)
GRID_RESOLUTIONS=(1.0) # in mm, for 2d case only, it will determine the grid size based on the bounding box of the ROI vertices;

#for trained_model_file in "${TRAINED_MODEL_FILES[@]}"; do
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    weight="${WEIGHTS[$i]}"
    for j in "${!ROIS[@]}"; do
        trained_model_file="${TRAINED_MODEL_FILES[$j]}"
        echo "Generation for model trained with run id $trained_model_file"
        roi="${ROIS[$j]}"
        roi_file="${ROI_FILES[$j]}"
        for cond_strength in "${CONDITIONING_STRENGTH[@]}"; do
            for ode in "${ODES[@]}"; do  
                for num_steps in "${NUM_GENERATION_STEPS[@]}"; do
                    for grid_res in "${GRID_RESOLUTIONS[@]}"; do
                        echo "Submitting job for model $model with weights $weight and ROI $roi with ROI file $roi_file and conditioning strength $cond_strength; Sampling with ODE: $ode; Number of generation steps: $num_steps; Grid resolution for 2d case: $grid_res mm"
                        sbatch -p gpu3,gpu4 2d_generate/2d_generate_ann_brain.sh $roi $weight $model $roi_file $cond_strength $trained_model_file $ode $num_steps $grid_res
                    done
                done
            done
        done
    done
done