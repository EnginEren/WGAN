import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
import models.HDF5Dataset as H
from torch.utils import data

class DCGAN_D(nn.Module):
    """ 
    discriminator component of WGAN

    """

    def __init__(self, isize, nc, ndf):
        super(DCGAN_D, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        ## linear layers
        self.cond1 = torch.nn.Linear(3, 10)
        self.cond2 = torch.nn.Linear(10, isize*isize)
        
        ### convolution
        self.conv1 = torch.nn.Conv2d(nc+1, ndf*8, kernel_size=3, stride=1, padding=1)
        ## batch-normalization
        self.bn1 = torch.nn.BatchNorm2d(ndf*8)
        ## convolution
        self.conv2 = torch.nn.Conv2d(ndf*8, ndf*4, kernel_size=3, stride=1, padding=1)
        ## batch-normalization
        self.bn2 = torch.nn.BatchNorm2d(ndf*4)
        #convolution
        self.conv3 = torch.nn.Conv2d(ndf*4, ndf*2, kernel_size=3, stride=1, padding=1)
        ## batch-normalization
        self.bn3 = torch.nn.BatchNorm2d(ndf*2)
        #convolution
        self.conv4 = torch.nn.Conv2d(ndf*2, ndf, kernel_size=3, stride=1, padding=1)
        
        # Read-out layer : ndf * isize * isize input features, ndf output features 
        self.fc1 = torch.nn.Linear(ndf * isize * isize, 1)
        
    def forward(self, x, energy, impactPoint):
        
        ## conditioning on energy and impact parameter
        t = F.leaky_relu(self.cond1(torch.cat((energy, impactPoint), 1)), 0.2, inplace=True)
        t = F.leaky_relu(self.cond2(t))
        
        ## reshape into two 2D
        t = t.view(-1, 1, self.isize, self.isize)
        
        ## concentration with input : 31 (30layers + 1 cond) x 30 x 30
        x = torch.cat((x, t), 1)
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.conv4(x), 0.2, inplace=True)
        #Size changes from (nc+1, 30, 30) to (ndf, 30, 30)

        
        x = x.view(-1, self.ndf * self.isize * self.isize)
        # Size changes from (ndf, 30, 30) to (1, ndf * 30 * 30) 
        #Recall that the -1 infers this dimension from the other given dimension


        # Read-out layer 
        x = self.fc1(x)
        
        x = x.mean(0)
        return x.view(1)



class DCGAN_G(nn.Module):
    """ 
    generator component of WGAN

    """
    def __init__(self, nc, ngf, z):
        super(DCGAN_G, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z = z
        
        self.cond1 = torch.nn.Linear(self.z+3, 50)
        self.cond2 = torch.nn.Linear(50, 10*10*ngf)
        
        ## deconvolution
        self.deconv1 = torch.nn.ConvTranspose2d(ngf, ngf*2, kernel_size=3, stride=3, padding=1)
        ## batch-normalization
        self.bn1 = torch.nn.BatchNorm2d(ngf*2)
        ## deconvolution
        self.deconv2 = torch.nn.ConvTranspose2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1)
        ## batch-normalization
        self.bn2 = torch.nn.BatchNorm2d(ngf*4)
        # deconvolution
        self.deconv3 = torch.nn.ConvTranspose2d(ngf*4, ngf*8, kernel_size=3, stride=2, padding=1)
        ## batch-normalization
        self.bn3 = torch.nn.BatchNorm2d(ngf*8)
        
        ## convolution 
        self.conv0 = torch.nn.Conv2d(ngf*8, 1, kernel_size=3, stride=4, padding=1)
        ## batch-normalisation
        self.bn0 = torch.nn.BatchNorm2d(1)
        
        ## convolution 
        self.conv1 = torch.nn.Conv2d(nc, ngf*4, kernel_size=3, stride=1, padding=1)
        ## batch-normalisation
        self.bn01 = torch.nn.BatchNorm2d(ngf*4)
        
        ## convolution 
        self.conv2 = torch.nn.Conv2d(ngf*4, ngf*8, kernel_size=3, stride=1, padding=1)
        ## batch-normalisation
        self.bn02 = torch.nn.BatchNorm2d(ngf*8)
        
        ## convolution 
        self.conv3 = torch.nn.Conv2d(ngf*8, ngf*4, kernel_size=3, stride=1, padding=1)
        ## batch-normalisation
        self.bn03 = torch.nn.BatchNorm2d(ngf*4)
        
        ## convolution 
        self.conv4 = torch.nn.Conv2d(ngf*4, nc, kernel_size=3, stride=1, padding=1)
    
        
        
    def forward(self, noise, energy, impactPoint):
        
        layer = []
         ### need to do generated 30 layers, hence the loop!
        for i in range(self.nc):     
            ## conditioning on energy, impact parameter and noise
            x = F.leaky_relu(self.cond1(torch.cat((energy, impactPoint, noise), 1)), 0.2, inplace=True)
            x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
            
            ## change size for deconv2d network. Image is 10x10
            x = x.view(-1,self.ngf,10,10)        

            ## apply series of deconv2d and batch-norm
            x = F.leaky_relu(self.bn1(self.deconv1(x, output_size=[x.size(0), x.size(1) , 30, 30])), 0.2, inplace=True) 
            x = F.leaky_relu(self.bn2(self.deconv2(x, output_size=[x.size(0), x.size(1) , 60, 60])), 0.2, inplace=True)
            x = F.leaky_relu(self.bn3(self.deconv3(x, output_size=[x.size(0), x.size(1) , 120, 120])), 0.2, inplace=True)                         
            
            ##Image is 120x120
            
            ## one standard conv and batch-norm layer (I dont know why :) )
            x = F.leaky_relu(self.bn0(self.conv0(x)), 0.2, inplace=True)

            layer.append(x)
        
       
        ## concentration of the layers
        x = torch.cat([layer[i] for l in range(self.nc)], 1)

        ## Further apply series of conv and batch norm layers 
        x = F.leaky_relu(self.bn01(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn02(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn03(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)

        return x
