from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils import data
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
from pathlib import Path
import models.HDF5Dataset as H

import models.dcgan as dcgan
import models.mlp as mlp

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=5, help='input image channels')
    parser.add_argument('--nz', type=int, default=1024, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--full' , action='store_true', help='switch to full all layers 30 x 30 x 30 ')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=5, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    opt = parser.parse_args()
    print(opt)

    cudnn.benchmark = True
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    ## Load and make them iterable
    loader_params = {'batch_size': opt.batchSize, 'shuffle': True, 'num_workers': 6}
    path = '/beegfs/desy/user/eren/WassersteinGAN/data/gamma-fullG.hdf5'
    d = H.HDF5Dataset(path, '30x30/layers')
    e = H.HDF5Dataset(path, '30x30/energy')
    dataloader_layer  = data.DataLoader(d, **loader_params)
    dataloader_energy = data.DataLoader(e, **loader_params)

    data_layer = iter(dataloader_layer)
    data_energy = iter(dataloader_energy)


    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = int(opt.nc)
    n_extra_layers = int(opt.n_extra_layers)

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu, "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)    

    ## create generator 
    netG = dcgan.DCGAN_G(nc, ngf, nz)

    # write out generator config to generate images together wth training checkpoints (.pth)
    #generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu, "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    #with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
    #    gcfg.write(json.dumps(generator_config)+"\n")

    netG.apply(weights_init)
    if opt.netG != '': # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    ## create critics (i.e Discriminator)
    netD = dcgan.DCGAN_D(opt.imageSize, nc, ndf)
    netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    ## layers
    input_layer = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)

    ## incoming energy
    input_energy = torch.FloatTensor(opt.batchSize,1)
    

    
    noise = torch.FloatTensor(opt.batchSize, nz)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    if torch.cuda.is_available():
        netD.cuda()
        netG.cuda()
        input_energy = input_energy.cuda()
        input_layer = input_layer.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    
    

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

    gen_iterations = 0
    for epoch in range(opt.niter):
        data_layer = iter(dataloader_layer)
        data_energy = iter(dataloader_energy)
        i = 0
        while i < len(dataloader_layer):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 150
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader_layer):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(-0.01, 0.01)

                ### input size matters. Reshape if we want 30x30
                if opt.full :
                    layer = data_layer.next()
                else :
                    tmp = data_layer.next()      ## [Bs, 30, 30 , 30 ]
                    layer = torch.sum(tmp, dim=1)
                    layer = layer.unsqueeze(1)  ## [Bs, 1, 30 , 30 ]
                    
                energy = data_energy.next()
                i += 1
                
                #print ("Updating D network, step: {}".format(j))
                
                # train with real
                ## layers
                real_cpu = layer

                ## incoming energy
                real_cpu_e = energy

                netD.zero_grad()
                batch_size = real_cpu.size(0)
                
                if torch.cuda.is_available():
                    real_cpu = real_cpu.cuda()
                    real_cpu_e = real_cpu_e.cuda()
                
                ## input layers
                input_layer.resize_as_(real_cpu.float()).copy_(real_cpu.float())
                
                ## input energy
                input_energy.resize_as_(real_cpu_e.float()).copy_(real_cpu_e.float())
 
		## quick and dirt fix for imp parameter 
                impoint = torch.zeros([batch_size,2]) 
                 
                if torch.cuda.is_available():
                    inputv_layer = Variable(input_layer.cuda())
                    inputv_e = Variable(input_energy.cuda())
                    inputv_imp = Variable(impoint.cuda()) ## input impact point
                else :
                    inputv_layer = Variable(input_layer)
                    inputv_e = Variable(input_energy)
                    inputv_imp = Variable(impoint) ## input impact point
                    
		
                #print (epoch, inputv_e.shape)
                #print (epoch, inputv_imp.shape) 
                errD_real = netD(inputv_layer, inputv_e, inputv_imp)
                
                
                # train with fake
                noise.resize_(batch_size, nz).normal_(0, 1)
                input_energy.resize_(batch_size, 1).uniform_(10, 100)
                
                if torch.cuda.is_available():
                    inputv_e = Variable(input_energy.cuda())
                    noisev = Variable(noise.cuda(), volatile = True) # totally freeze netG
                else :
                    inputv_e = Variable(input_energy)
                    noisev = Variable(noise, volatile = True) # totally freeze netG
                
                fake = netG(noisev, inputv_e, inputv_imp)
                
                if torch.cuda.is_available():
                    inputv_layer = Variable(fake.cuda())
                else :
                    inputv_layer = Variable(fake)
                    
                errD_fake = netD(inputv_layer, inputv_e, inputv_imp)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            #print ("Updating G network")
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(batch_size, nz).normal_(0, 1)
            input_energy.resize_(batch_size, 1).uniform_(10, 100)
            if torch.cuda.is_available():
                noisev = Variable(noise.cuda())
                inputv_e = Variable(input_energy.cuda())
                inputv_imp = Variable(impoint.cuda())  ## input impact point
            else :
                noisev = Variable(noise)
                inputv_e = Variable(input_energy)
                inputv_imp = Variable(impoint)  ## input impact point
            
            
            
            fake = netG(noisev, inputv_e, inputv_imp)
            
            if torch.cuda.is_available():
                fake = fake.cuda()
                
            errG = netD(fake, inputv_e, inputv_imp)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1    
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, opt.niter, i, len(dataloader_layer), gen_iterations,
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))


        # do checkpointing
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))


    
