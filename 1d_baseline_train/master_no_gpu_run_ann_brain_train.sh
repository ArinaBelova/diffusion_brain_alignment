ROIS=(5 18)
ROI_FILES=("streams" "HCP_MMP1") # "HCP_MMP1"
# WEIGHTS=("ResNet50_Weights.IMAGENET1K_V2" "ResNet101_Weights.IMAGENET1K_V2" "ResNet152_Weights.IMAGENET1K_V2")
# MODELS=("resnet50" "resnet101" "resnet152")

WEIGHTS=("IMAGENET1K_SWAG_E2E_V1")  # IMAGENET1K_V2
MODELS=("vit_b_16") # resnet50

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    weight="${WEIGHTS[$i]}"
    
    echo "Running with model=$model, weights=$weight"
    for j in "${!ROIS[@]}"; do
        roi="${ROIS[$j]}"
        roi_file="${ROI_FILES[$j]}"
        echo "Submitting job for ROI $roi with ROI file $roi_file"
        sbatch -p testing 1d_baseline_train/no_gpu_run_ann_brain_train.sh $roi $weight $model $roi_file
    done
done