import torch

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def to_gpu(ngpu):

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    return device


def loss_plot(G_losses, D_losses, out):

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out + 'loss_plot.png')


def image_grid(dataloader, img_list, device, out):

    real_batch = next(iter(dataloader))

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
               :64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig(out + 'real_fake_grid.png')
