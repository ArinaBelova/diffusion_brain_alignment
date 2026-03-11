import torch
import os
import re
import wandb
import numpy as np

from diffusion_brain.utils.diffusivity import generate_samples, get_diffusion
from diffusion_brain.utils.setup import parse_args_and_setup_wandb
from diffusion_brain.utils.visualise import visualise_and_save_results, pyplot_brain
from diffusion_brain.models import set_model
from diffusion_brain.models.autoencoder import LinearAutoencoder, get_linear_autoencoder
from diffusion_brain.data_utils import get_dataloader

_STEP_TAG_RE = re.compile(r"step_(\d+)")


def _infer_wandb_step(model_name, args, model_dir=None):
    match = _STEP_TAG_RE.search(str(model_name))
    if match:
        return int(match.group(1))

    if str(model_name).lower() == "final" and model_dir and os.path.isdir(model_dir):
        step_nums = []
        for fname in os.listdir(model_dir):
            m = _STEP_TAG_RE.search(fname)
            if m:
                step_nums.append(int(m.group(1)))
        if step_nums:
            return args.validation.final_model_num #max(step_nums) + 1  # final model is one step after the last checkpoint

    return None


def _model_file_sort_key(filename):
    name = str(filename)
    if "final" in name.lower():
        return (2, float("inf"), name)
    match = _STEP_TAG_RE.search(name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, float("inf"), name)

def _infer_model_dims_from_dataloader(dataloader):
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None:
        fmri_data = getattr(dataset, "fmri_data", None)
        activations = getattr(dataset, "activations", None)
        if fmri_data is not None and activations is not None:
            return fmri_data.shape[1:], activations.shape[-1] # to cater for 2D cases

    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        fmri_signal, cond_signal = batch[0], batch[1]
        return fmri_signal.shape[1:], cond_signal.shape[-1]

    raise ValueError("Expected dataloader to yield (fmri_signal, cond) pairs.")

def generate_sample_loop_toy(args):
    print("Setting up diffusion process...", flush=True)
    diffusion_process = get_diffusion(args, device=DEVICE)
    generated_samples = {}

    if str(args.model.which).lower() == "all":
        print("Testing all the models in the model folder...")
        model_files = [f for f in os.listdir(args.model.input_folder) if f.endswith('.pth')]
        model_files = sorted(model_files, key=_model_file_sort_key)
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

def generate_sample_loop(args):
    print("Setting up true samples and conditions dataloader for comparison...", flush=True)
    print("In generation we use the data from subject: ", args.data.subj)
    _, gen_dataloader = get_dataloader(args)
    input_size, cross_attention_dim = _infer_model_dims_from_dataloader(gen_dataloader)
    args.model.input_size = tuple(input_size) # was int(input_size)
    args.model.cross_attention_dim = int(cross_attention_dim)

    args.model.input_folder = args.model.input_folder + "-" + str(args.model.run_id) 

    print("Setting up diffusion process...", flush=True)
    diffusion_process = get_diffusion(args, device=DEVICE)
    generated_samples = {}
    true_fmri_per_model = {}

    if str(args.model.which).lower() == "all":
        print("Testing all the models in the model folder...")
        model_files = [f for f in os.listdir(args.model.input_folder) if f.endswith('.pth')]
        model_files = sorted(model_files, key=_model_file_sort_key)
    else:
        print(f"Testing the model at step {args.model.which}")
        if str(args.model.which).lower() == "final":
            model_files = ["model_final.pth"]
        else:
            model_files = [f"model_step_{args.model.which}.pth"]    

    for model_file in model_files:
        print("Setting up the model: ", model_file)
        model_path = os.path.join(args.model.input_folder, model_file)
        print("model path is ", model_path)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model = set_model(args)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Sometimes the checkpoint IS the state_dict itself
            model.load_state_dict(checkpoint)

        model.eval()  # set to eval mode for generation
        print("model device: ", next(model.parameters()).device)
        print("len dataloader.dataset is ", len(gen_dataloader.dataset))

        generated_samples_list = []
        true_fmri_list = []
        
        for idx, (true_fmri, cond) in enumerate(gen_dataloader): 
            cond = cond.to(DEVICE)
            
            print(f"Generating samples with guidance strength {args.validation.guidance_scale} for batch {idx+1} out of {len(gen_dataloader)}...", flush=True)   
            print(f"In this generation procedure we will average across {args.validation.average_over_num_runs} runs")
            
            generated_samples_one_model_one_cond = []
            for _ in range(args.validation.average_over_num_runs):
                generated_samples_one_model_one_cond_one_time = generate_samples(args.validation.batch_size, model, diffusion_process, args, cond=cond, device=DEVICE) 
                generated_samples_one_model_one_cond.append(generated_samples_one_model_one_cond_one_time)
            generated_samples_one_model_one_cond = torch.stack(generated_samples_one_model_one_cond, dim=0).mean(dim=0) # averaging across repetitions of the same generation, conditioned on the same ANN signal

            print("Shape of generated samples after stacking average_over_num_runs: ", generated_samples_one_model_one_cond.shape)

            autoencoder = get_linear_autoencoder(args)
            if autoencoder is not None:
                autoencoder.to(DEVICE)
                with torch.no_grad():
                    z = generated_samples_one_model_one_cond.squeeze(1)
                    generated_samples_one_model_one_cond = autoencoder.decoder(z)

            print("max value of generated samples: ", generated_samples_one_model_one_cond.max().item())
            print("min value of generated samples: ", generated_samples_one_model_one_cond.min().item())
            
            generated_samples_one_model_one_cond = generated_samples_one_model_one_cond.squeeze(1)
            print("Dimension of generaetd samples: ", generated_samples_one_model_one_cond.shape, flush=True)

            generated_samples_list.append(generated_samples_one_model_one_cond.cpu())
            true_fmri_list.append(true_fmri.cpu())   

        # Concatenate all batches
        generated_samples_one_model = torch.cat(generated_samples_list, dim=0)
        true_fmri_concat = torch.cat(true_fmri_list, dim=0)

        print("Shape of generated samples after concatenating batches: ", generated_samples_one_model.shape, flush=True)
        print("Shape of true fMRI after concatenating batches: ", true_fmri_concat.shape, flush=True)

        filename = os.path.basename(model_path)
        match = re.search(r"(step_\d+|final)", filename)
        tag = match.group(1) if match else filename
        generated_samples[f"{tag}"] = generated_samples_one_model
        true_fmri_per_model[f"{tag}"] = true_fmri_concat

    return generated_samples, true_fmri_per_model

def main(): 
    args = parse_args_and_setup_wandb()
    wandb.define_metric("model_step")
    wandb.define_metric("*", step_metric="model_step")
    if args.data.data_name == "ann-brain":
        print("Using ANN-Brain dataset, we will generate samples and visualise them on the brain surface.")
        generated_samples, true_fmri_per_model = generate_sample_loop(args)
        print(f"Generating the visualisations with guidance scale {args.validation.guidance_scale})")
        for model_name, generated_samples_per_model in generated_samples.items():
            print("Visualising results for model at step: ", model_name)
            step_num = _infer_wandb_step(model_name, args, model_dir=args.model.input_folder)
            true_fmri = true_fmri_per_model[model_name]
            visualise_and_save_results(
                generated_samples_per_model, 
                true_fmri=true_fmri,
                step=model_name, 
                args=args,
                step_num=step_num,
            )
            # no f-string in the name as i want to have all the models in the slide bar in wandb
            #pyplot_brain(generated_samples_per_model.mean(axis=0), args=args, savename=f"generated_samples_mean", figpath=f"{args.validation.output_folder}/{args.jobid}", save_type='png', step_num=step_num)
    else:
        print(f"Using {args.data.data_name} dataset, we will generate samples and visualise them as images.")
        generated_samples = generate_sample_loop_toy(args)
    
        print(f"Generating the visualisations with guidance scale {args.validation.guidance_scale})")
        for model_name, generated_samples_per_model in generated_samples.items():
            print("Visualising results for model at step: ", model_name)
            step_num = _infer_wandb_step(model_name, args, model_dir=args.model.input_folder)
            visualise_and_save_results(generated_samples_per_model, step=model_name, args=args, step_num=step_num)

if __name__ == "__main__":
    # Set here a global device variable for the whole training script:
    global DEVICE 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # "cpu"
    main()   
