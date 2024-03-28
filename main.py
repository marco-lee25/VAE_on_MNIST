from torchvision.utils import make_grid
from tqdm import tqdm
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model.model import *

'''
KL divergence allow us to evaluate the difference between distributions.
KL == 0 -> same distributions
'''
def kl_divergence_loss(q_dist):
    return kl_divergence(
        q_dist, Normal(torch.zeros_like(q_dist.mean), torch.ones_like(q_dist.stddev))
    ).sum(-1)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.axis('off')
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())

if __name__ == "__main__":
    # Distance between input and generated image.
    reconstruction_loss = nn.BCELoss(reduction='sum')
    transform=transforms.Compose([
    transforms.ToTensor(),
    ])
    mnist_dataset = datasets.MNIST('.', train=True, transform=transform, download=True)
    train_dataloader = DataLoader(mnist_dataset, shuffle=True, batch_size=1024)
    device = 'cuda'
    vae = VAE().to(device)
    vae_opt = torch.optim.Adam(vae.parameters(), lr=0.002)
    for epoch in range(10):
        print(f"Epoch {epoch}")
        time.sleep(0.5)
        for images, _ in tqdm(train_dataloader):
            images = images.to(device)
            vae_opt.zero_grad() # Clear out the gradients
            recon_images, encoding = vae(images)
            loss = reconstruction_loss(recon_images, images) + kl_divergence_loss(encoding).sum()
            loss.backward()
            vae_opt.step()

        plt.subplot(1,2,1)
        show_tensor_images(images)
        plt.title("True")
        plt.subplot(1,2,2)
        show_tensor_images(recon_images)
        plt.title("Reconstructed")
        plt.show()
    
