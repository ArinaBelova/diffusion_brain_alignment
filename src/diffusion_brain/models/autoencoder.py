import torch
import torch.nn as nn
import os

class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def get_linear_autoencoder(args):
    if args.model.use_autoencoder and os.path.exists(os.path.join(args.model.ae_save_path, args.data.roi_file, args.model.ae_name + f"roi_{str(args.data.roi)}" + ".pth")):
        print("Using linear autoencoder for dimensionality reduction of the generated samples.")
        autoencoder = LinearAutoencoder(input_dim=args.model.input_size, latent_dim=args.model.ae_latent_dim)
        return autoencoder
    else:
        return None