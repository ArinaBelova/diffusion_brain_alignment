Currently, the project is still in heavy prototyping phase, so most of up-to-date development is in "inter-subject-experiments" branch.
This is subject to change in the future.

# To setup an environment to run the code:

1. conda create -n test_brain_alignment python=3.9 
2. conda activate test_brain_alignment
3. pip install -r requirements.txt
4. pip install -e .

# Train
From the project root directory run:
python ./src/diffusion_brain/scripts/train.py --config ./src/diffusion_brain/configs/toy/config_train.yaml --jobid toy-cluster-some-random-number-to-differentiate-the-jobs-in-wandb

If you want to use wandb you need to setup your project and to change the code in wandb.init(id=run_id, name=run_id, project=args.wandb.project_name, config=vars(args)) in src/diffusion_brain/scripts/train.py script, namely to give it your own args.wandb.project_name.
