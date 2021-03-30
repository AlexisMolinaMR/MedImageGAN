import yaml
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from data.image_preprocessing import dataLoader
from utils.utils import to_gpu, loss_plot, image_grid
from utils.metrics import compute_metrics
from models.dcgan import weights_init, Generator, Generator_256, Discriminator, Discriminator_256, Discriminator_SN_256, Discriminator_SN, training_loop


def parseyaml():

    with open(sys.argv[1]) as ctrl_file:
        params = yaml.load(ctrl_file, Loader=yaml.FullLoader)

    return params


def main():

    params = parseyaml()

    dataloader = dataLoader(
        path=params['path'], image_size=params['image_size'], batch_size=params['batch_size'],
        workers=params['loader_workers'])

    device = to_gpu()

    netG = Generator(ngpu=params['n_gpu'], nz=params['latent_vector'],
                     ngf=params['gen_feature_maps'], nc=params['number_channels']).to(device)

    if (device.type == 'cuda') and (params['n_gpu'] > 1):
        netG = nn.DataParallel(netG, list(range(params['n_gpu'])))

    netG.apply(weights_init)

    print(netG)

    if params['arch'] == 'DCGAN':

        if params['image_size'] == 64:

            netG = Generator(ngpu=params['n_gpu'], nz=params['latent_vector'],
                             ngf=params['gen_feature_maps'], nc=params['number_channels']).to(device)

            netD = Discriminator(params['n_gpu'], nc=params['number_channels'],
                                 ndf=params['dis_feature_maps']).to(device)

        elif params['image_size'] == 256:

            netG = Generator_256(ngpu=params['n_gpu'], nz=params['latent_vector'],
                                 ngf=params['gen_feature_maps'], nc=params['number_channels']).to(device)

            netD = Discriminator_256(params['n_gpu'], nc=params['number_channels'],
                                     ndf=params['dis_feature_maps']).to(device)

    elif params['arch'] == 'SNGAN':

        if params['image_size'] == 64:

            netG = Generator(ngpu=params['n_gpu'], nz=params['latent_vector'],
                             ngf=params['gen_feature_maps'], nc=params['number_channels']).to(device)

            netD = Discriminator_SN(params['n_gpu'], nc=params['number_channels'],
                                    ndf=params['dis_feature_maps']).to(device)

        elif params['image_size'] == 256:

            netG = Generator_256(ngpu=params['n_gpu'], nz=params['latent_vector'],
                                 ngf=params['gen_feature_maps'], nc=params['number_channels']).to(device)

            netD = Discriminator_SN_256(params['n_gpu'], nc=params['number_channels'],
                                        ndf=params['dis_feature_maps']).to(device)

    if (device.type == 'cuda') and (params['n_gpu'] > 1):
        netG = nn.DataParallel(netG, list(range(params['n_gpu'])))

    if (device.type == 'cuda') and (params['n_gpu'] > 1):
        netD = nn.DataParallel(netD, list(range(params['n_gpu'])))

    netG.apply(weights_init)
    netD.apply(weights_init)

    print(netG)
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(params['image_size'],
                              params['latent_vector'], 1, 1, device=device)

    optimizerD = optim.Adam(netD.parameters(), lr=params['learning_rate'], betas=(
        params['beta_adam'], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=params['learning_rate'], betas=(
        params['beta_adam'], 0.999))

    G_losses, D_losses, img_list, img_list_only = training_loop(num_epochs=params['num_epochs'], dataloader=dataloader,
                                                                netG=netG, netD=netD, device=device, criterion=criterion, nz=params[
                                                                    'latent_vector'],
                                                                optimizerG=optimizerG, optimizerD=optimizerD, fixed_noise=fixed_noise, out=params['out'])

    loss_plot(G_losses=G_losses, D_losses=D_losses, out=params['out'] + params['run'] + '_')

    image_grid(dataloader=dataloader, img_list=img_list,
               device=device, out=params['out'] + params['run'] + '_')

    compute_metrics(real=next(iter(dataloader)), fakes=img_list_only, out=params['out'])


if __name__ == "__main__":
    main()
