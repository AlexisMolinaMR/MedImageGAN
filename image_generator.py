import yaml
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from data.image_preprocessing import dataLoader
from utils.utils import to_gpu, loss_plot, image_grid
from utils.metrics import compute_metrics
from models.dcgan import weights_init, Generator,  Generator_128, Generator_256
from models.dcgan import Discriminator,  Discriminator_128, Discriminator_SN_128, Discriminator_256, Discriminator_SN_256, Discriminator_SN, training_loop


def parseyaml():

    with open(sys.argv[1]) as ctrl_file:
        params = yaml.load(ctrl_file, Loader=yaml.FullLoader)

    return params


def main():

    params = parseyaml()

    if params['arch'] == 'Generator':

        device = to_gpu(ngpu=params['n_gpu'])

        if params['image_size'] == 64:

            netG = Generator(ngpu=0, nz=256,
                             ngf=64, nc=64).to(device)

        elif params['image_size'] == 128:

            netG = Generator_128(ngpu=0, nz=256,
                                 ngf=64, nc=64).to(device)

        elif params['image_size'] == 256:

            netG = Generator_256(ngpu=0, nz=256,
                                 ngf=64, nc=64).to(device)

        netG.apply(weights_init)
        netG.load_state_dict(torch.load(params['path']))

        for i in range(params['quantity']):

            fixed_noise = torch.randn(64, 256, 1, 1, device=device)
            fakes = netG(fixed_noise)

            for j in range(len(fakes)):
                save_image(fakes[j], params['out'] + params['run'] +
                           '_' + str(i) + '_' + str(j) + '_img.png')

    else:

        dataloader = dataLoader(
            path=params['path'], image_size=params['image_size'], batch_size=params['batch_size'],
            workers=params['loader_workers'])

        device = to_gpu(ngpu=params['n_gpu'])

        if params['arch'] == 'DCGAN':

            if params['image_size'] == 64:

                netG = Generator(ngpu=params['n_gpu'], nz=params['latent_vector'],
                                 ngf=params['gen_feature_maps'], nc=params['number_channels']).to(device)

                netD = Discriminator(params['n_gpu'], nc=params['number_channels'],
                                     ndf=params['dis_feature_maps']).to(device)

            elif params['image_size'] == 128:

                netG = Generator_128(ngpu=params['n_gpu'], nz=params['latent_vector'],
                                     ngf=params['gen_feature_maps'], nc=params['number_channels']).to(device)

                netD = Discriminator_128(params['n_gpu'], nc=params['number_channels'],
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

            elif params['image_size'] == 128:

                netG = Generator_128(ngpu=params['n_gpu'], nz=params['latent_vector'],
                                     ngf=params['gen_feature_maps'], nc=params['number_channels']).to(device)

                netD = Discriminator_SN_128(params['n_gpu'], nc=params['number_channels'],
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

        if params['learning_rate'] >= 1:

            optimizerD = optim.Adam(netD.parameters(), lr=0.0002 * params['learning_rate'], betas=(
                params['beta_adam'], 0.999))
            optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(
                params['beta_adam'], 0.999))

        else:

            optimizerD = optim.Adam(netD.parameters(), lr=params['learning_rate'], betas=(
                params['beta_adam'], 0.999))
            optimizerG = optim.Adam(netG.parameters(), lr=params['learning_rate'], betas=(
                params['beta_adam'], 0.999))

        G_losses, D_losses, img_list, img_list_only = training_loop(num_epochs=params['num_epochs'], dataloader=dataloader,
                                                                    netG=netG, netD=netD, device=device, criterion=criterion, nz=params[
                                                                        'latent_vector'],
                                                                    optimizerG=optimizerG, optimizerD=optimizerD, fixed_noise=fixed_noise, out=params['out'] + params['run'] + '_')

        loss_plot(G_losses=G_losses, D_losses=D_losses, out=params['out'] + params['run'] + '_')

        image_grid(dataloader=dataloader, img_list=img_list,
                   device=device, out=params['out'] + params['run'] + '_')

        compute_metrics(real=next(iter(dataloader)), fakes=img_list_only,
                        size=params['image_size'], out=params['out'] + params['run'] + '_')


if __name__ == "__main__":
    main()
