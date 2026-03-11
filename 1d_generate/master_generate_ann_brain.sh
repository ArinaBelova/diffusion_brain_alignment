ROIS=(5) #(18 5)
ROI_FILES=("streams") #("HCP_MMP1" "streams") #
CONDITIONING_STRENGTH=(20)
# WEIGHTS=("ResNet50_Weights.IMAGENET1K_V2" "ResNet101_Weights.IMAGENET1K_V2" "ResNet152_Weights.IMAGENET1K_V2")
# MODELS=("resnet50" "resnet101" "resnet152")
ODES=(0) # 0 for sampling without ODE, 1 for sampling with ODE
WEIGHTS=("IMAGENET1K_SWAG_E2E_V1") #("ResNet50_Weights.IMAGENET1K_V2")
MODELS=("vit_b_16") #("resnet50")
TRAINED_MODEL_FILES=(218947) #(211379 210760) #(197894 197972) # 201512 (197894 197972)
NUM_GENERATION_STEPS=(10 100) # 250 400 500)

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
                echo "Submitting job for model $model with weights $weight and ROI $roi with ROI file $roi_file and conditioning strength $cond_strength"
                echo "Sampling with ODE: $ode"
                for num_steps in "${NUM_GENERATION_STEPS[@]}"; do
                    echo "Number of generation steps: $num_steps"
                    #sbatch --export=ALL,ROI=$roi,ANN_MODEL_WEIGHTS=$weight,ANN_MODEL=$model,ROI_FILE=$roi_file,CONDITIONING_STRENGTH=$cond_strength,ODE=$ode,NUM_STEPS=$num_steps run_ann_brain_generate.sh
                    sbatch -p testing 1d_generate/generate_ann_brain.sh $roi $weight $model $roi_file $cond_strength $trained_model_file $ode $num_steps
                done
            done
        done
    done
done
#done    