import torch
import numpy as np
from torch.utils.data import Dataset

def get_toy_dataloader(args):
    dataset = EightGaussianConditional(
        n_samples=args.train.dataset_size,
        n_centers=args.model.num_classes, 
        random_state=42, 
        label_type=args.train.label_type,  # 'index' or 'coordinates'
        radius=args.data.radius,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train.batch_size,
        shuffle=True,
    )
    return dataloader, None

class ToyDataset(Dataset):
    def __init__(self, n_samples, generator=None, normalize=True, random_state=None, **kwargs):
        self.n_samples = n_samples
        self.generator = generator
        self.normalize = normalize
        if random_state is not None:
            torch.manual_seed(random_state)

        self.data, self.labels = self.sample_data(**kwargs)
        if normalize:
            self.data = self.data / torch.max(torch.max(self.data), -torch.min(self.data))
            self.data = (self.data - torch.mean(self.data, axis=0, keepdim=True))

        # self.data = self.data.detach().cpu().numpy()
        # self.labels = self.labels.detach().cpu().numpy()

    def sample_data(self, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    # torch.from_numpy(self.data[idx]).long(), torch.from_numpy(self.labels[idx]).long()

class EightGaussianConditional(ToyDataset):
    def __init__(self, n_samples, n_centers, std=0.1, radius=20.0, random_state=None, label_type='index'):
        """
        n_samples: number of datapoints/Gaussians
        std: standard deviation of each Gaussian
        radius: distance of centers from the origin
        label_type: 'index' for numeric labels (0, 1, 2, ...), 
                    'coordinates' for center coordinates as labels.
        """
        self.label_type = label_type
        self.std = radius / 40 # Scale std with radius
        self.radius = radius
        self.n_centers = n_centers
        super().__init__(n_samples, normalize=False, random_state=random_state)

    def sample_data(self):
        # ---------------------------------------------------
        # 1. Define fixed centers (8 directions on a circle)
        # ---------------------------------------------------
        angles = torch.linspace(0, 2 * np.pi, self.n_centers + 1)[:-1]
        centers = torch.stack(
            [
                self.radius * torch.cos(angles),
                self.radius * torch.sin(angles),
            ],
            dim=1,
        )

        # ---------------------------------------------------
        # 2. Sample class indices uniformly
        # ---------------------------------------------------
        labels_idx = torch.randint(0, len(centers), (self.n_samples,))
        selected_centers = centers[labels_idx]

        # ---------------------------------------------------
        # 3. Sample isotropic Gaussian noise
        # ---------------------------------------------------
        noise = torch.randn(self.n_samples, 2)

        # ---------------------------------------------------
        # 4. Generate samples
        #    x = μ_k + σ ε
        # ---------------------------------------------------
        samples = selected_centers + self.std * noise

        # ---------------------------------------------------
        # 5. Labels
        # ---------------------------------------------------
        if self.label_type == "index":
            labels = labels_idx
        elif self.label_type == "coordinates":
            labels = selected_centers
        else:
            raise ValueError("label_type must be 'index' or 'coordinates'")

        return samples, labels
    
    # def sample_data(self):
    #     z = torch.randn(self.n_samples, 2)
    #     scale = 4
    #     sq2 = 1 / np.sqrt(2)
    #     centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (sq2, sq2), (-sq2, sq2), (sq2, -sq2), (-sq2, -sq2)]
    #     centers = torch.tensor([(scale * x, scale * y) for x, y in centers])

    #     # Randomly assign each sample to a Gaussian blob
    #     center_indices = torch.randint(len(centers), size=(self.n_samples,))
    #     selected_centers = centers[center_indices]
    #     gaussians = sq2 * (0.5 * z + selected_centers)

    #     # Generate labels based on the label_type
    #     if self.label_type == 'index':
    #         labels = center_indices  # Numeric labels (0, 1, 2, ...)
    #     elif self.label_type == 'coordinates':
    #         labels = selected_centers  # Center coordinates as labels
    #     else:
    #         raise ValueError("Invalid label_type. Choose 'index' or 'coordinates'.")

    #     return gaussians, labels 