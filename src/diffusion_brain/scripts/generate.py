import torch
import os
import re

from diffusion_brain.utils.diffusivity import generate_samples, get_diffusion
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.visualise import visualise_and_save_results
from diffusion_brain.models import set_model


def generate_sample_loop(args):
    print("Setting up diffusion process...", flush=True)
    diffusion_process = get_diffusion(args, device=DEVICE)
    generated_samples = {}

    if str(args.model.which).lower() == "all":
        print("Testing all the models in the model folder...")
        model_files = [f for f in os.listdir(args.model.input_folder) if f.endswith('.pth')]
    else:
        print(f"Testing the model at step {args.model.which}")
        if str(args.model.which).lower() == "final":
            model_files = ["model_final.pth"]
        else:
            model_files = [f"model_step_{args.model.which}.pth"]    

    for model_file in model_files:
        print("Setting up the model: ", model_file)
        model_path = os.path.join(args.model.input_folder, model_file)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model = set_model(args)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Sometimes the checkpoint IS the state_dict itself
            model.load_state_dict(checkpoint)
        print(f"Starting generation. We generate label {args.validation.label_to_generate}")
        generated_samples_one_model = generate_samples(args.validation.batch_size, model, diffusion_process, args, device=DEVICE) * 255
        generated_samples_one_model = torch.clip(generated_samples_one_model, 0, 255)    
        print("max value of generated samples: ", generated_samples_one_model.max().item())
        print("min value of generated samples: ", generated_samples_one_model.min().item())

        filename = os.path.basename(model_path)
        match = re.search(r"(step_\d+|final)", filename)
        tag = match.group(1) if match else filename
        generated_samples[f"{tag}"] = generated_samples_one_model

    return generated_samples

def main(): 
    args = parse_args_and_setup_wandb()
    generated_samples = generate_sample_loop(args)
    
    print(f"Generating the visualisations with guidance scale {args.validation.guidance_scale})")
    for model_name, generated_samples_per_model in generated_samples.items():
        print("Visualising results for model at step: ", model_name)
        visualise_and_save_results(generated_samples_per_model, step=model_name, args=args)

if __name__ == "__main__":
    # Set here a global device variable for the whole training script:
    global DEVICE 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()   