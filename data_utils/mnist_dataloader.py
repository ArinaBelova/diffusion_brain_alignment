import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import utils.diffusivity as diffusivity

def get_mnist_dataloader(args):
    #image_size = args.train.image_size if hasattr(args.train, 'image_size') else 32 #28

    # do the additional padding to have 32x32 images for UNet digestion
    # transforms.Resize(image_size),\
    transform = transforms.Compose([transforms.ToTensor(),\
                                    transforms.Pad(2),\
                                    transforms.Normalize([0.5],[0.5])]) #Normalize to -1,1
    
    train_set = torchvision.datasets.MNIST(root=args.data.data_root, train=True,
                                        download=True, transform=transform)
    val_set = torchvision.datasets.MNIST(root=args.data.data_root, train=False, transform=transform)

    train_batch_size = args.train.batch_size if hasattr(args.train, 'batch_size') else 64
    val_batch_size = args.validation.batch_size if hasattr(args.validation, 'batch_size') else 1

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                              shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size,
                                              shuffle=True, num_workers=2)

    return train_loader, val_loader                                          




# Some old helper functions:
def load_mnist():
    image_size = 28
    transform = transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #Normalize to -1,1
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    batch_size = 256
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    return image_size, trainloader, trainset

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print("numpy image shape: ", npimg.shape)
    plt.figure(figsize=[20, 20])
    plt.imshow(npimg[0], cmap='gray') # as torchvision.utils.make_grid returns a 3-channel tensor
    plt.show()
    
def visualize_forward_sde(X_0):
    n_grid_points = 10
    # X_0 = torch.stack([X_0]*n_grid_points) # making a batch out of a single image ?
    # time_vec = torch.linspace(0,1,n_grid_points)**2
    process = diffusivity.VPSDE()
    
    X_t, _ = diffusivity.run_forward_sde(process=process, x_0=X_0, n_steps=n_grid_points)
    imshow(torchvision.utils.make_grid(X_t.view(-1, 1, 28, 28), nrow=len(X_t), padding=0, normalize=False)) # as we have 1 channel greyscale images 

def execute_visualisation(img_index=20130):
    _, _, trainset = load_mnist()
    X_0 = trainset.__getitem__(img_index)[0].squeeze()
    visualize_forward_sde(X_0)

def main():
    execute_visualisation()

if __name__ == "__main__":
    main()    