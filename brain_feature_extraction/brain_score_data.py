import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm


def get_neural_data(region, dataset='dicarlo', image_type='original', data_root = None,loader_kwargs=None):

    assert region in ['V4', 'IT']

    if dataset == 'dicarlo':
        identifier = 'majajhong2015'
        df = get_brainscore(identifier=identifier, region=region, data_root=data_root)
    else:
        raise Exception(f'No such dataset with dataset {dataset} and region {region}!')

    # Load images vs responses data_loader
    if loader_kwargs is None:
        loader_kwargs = {'batch_size': 800,
                         'shuffle': False,
                         'num_workers': 4,
                         'pin_memory': True,
                         'onehot': True,
                         'labels_from': 'neural_activity'
                         }

    # Get the DataLoader for Neural Responses and the entire ordered data
    data_loader_neural, label_map = get_dataloader(df, **loader_kwargs)
    data_loader_neural = LoaderTORCH(data_loader_neural, cuda=True)
    images, responses = get_ordered_data(data_loader_neural())

    print(images.shape, responses.shape)

    labels = {'responses': responses
              }

    return data_loader_neural, images, labels


def get_dataloader(df,
                   labels_from='categories',
                   onehot=False,
                   mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225),
                   **loader_kwargs):

    image_files = df.image_files
    image_files = np.array(image_files.to_list())

    if labels_from == 'categories':
        image_categories = df.image_categories
        label_names = image_categories
        task = 'categorical'
    elif labels_from == 'names':
        image_names = df.image_names.values
        label_names = image_names
        task = 'categorical'
    elif labels_from == 'neural_activity':
        label_names = ""
        task = 'regression'
    else:
        raise Exception

    target_transform = None
    if task == 'categorical':
        label_map = {lb: label for lb, label in zip(set(label_names), range(len(label_names)))}
        labels = np.vectorize(label_map.get)(label_names)
        if onehot:
            def target_transform(y): return torch.eye(len(label_map))[y]
    else:
        # task == 'regression':
        label_map = None
        labels = np.stack(df.mean_responses).astype(np.float32)

    transform = [transforms.Resize((224, 224), antialias=True),
                 transforms.ConvertImageDtype(torch.float32),
                 ]
    if mean is not None and std is not None:
        transform += [transforms.Normalize(mean=mean, std=std)]
    transform = transforms.Compose(transform)

    ds = BrainscoreImageDataset(image_files, labels,
                                transform=transform,
                                target_transform=target_transform)

    data_loader = DataLoader(ds, **loader_kwargs)

    return data_loader, label_map


class BrainscoreImageDataset(Dataset):
    def __init__(self, images, img_labels, transform=None, target_transform=None):
        self.images = images
        self.img_labels = img_labels
        self.transform = transform
        self.target_transform = target_transform
        # DiCarlo dataset images are file paths
        self.images = np.array([str(file) for file in self.images])
        self.images = [read_image(str(file), ImageReadMode(3)) for file in self.images]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class LoaderTORCH:
    def __init__(self, loader, cuda=False):
        self.loader = loader
        self.cuda = cuda

    def __call__(self):
        return self.torch_loader(self.loader, self.cuda)

    def torch_loader(self, loader, cuda=False):
        for data in tqdm(loader, total=len(loader), desc="Batch"):

            if cuda:
                yield data[0].cuda(), data[1].cuda()
            else:
                yield data[0], data[1]


def get_ordered_data(data_loader):

    X, Y = [], []
    for x, y in data_loader:
        x, y = x.cpu(), y.cpu()
        X += [x]
        Y += [y]

    images = torch.cat(X)
    labels = torch.cat(Y)

    return images, labels


def get_brainscore(identifier="majajhong2015", region='IT', data_root=None):

    if region == 'V4':
        df = pd.read_pickle(f'{data_root}/df_majajhong2015_V4.pkl')
    else:
        df = pd.read_pickle(f'{data_root}/df_majajhong2015_IT.pkl')
    return df
