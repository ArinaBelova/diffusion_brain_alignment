ROIS=(1 2 3 4 5 6 7 )
# WEIGHTS=("ResNet50_Weights.IMAGENET1K_V2" "ResNet101_Weights.IMAGENET1K_V2" "ResNet152_Weights.IMAGENET1K_V2")
# MODELS=("resnet50" "resnet101" "resnet152")

WEIGHTS=("ResNet50_Weights.IMAGENET1K_V2")
MODELS=("resnet50")

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    weight="${WEIGHTS[$i]}"
    
    echo "Running with model=$model, weights=$weight"
    for roi in "${ROIS[@]}"; do
        echo "Submitting job for ROI $roi"
        sbatch -p testing no_gpu_run_ann_brain_train.sh $roi $weight $model
    done
done