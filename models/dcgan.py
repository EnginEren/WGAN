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


        ### convolution
        self.conv1 = torch.nn.Conv2d(nc, ndf*8, kernel_size=4, stride=2, padding=1, bias=False)
        ## batch-normalization
        self.bn1 = torch.nn.BatchNorm2d(ndf*8)
        ## convolution
        self.conv2 = torch.nn.Conv2d(ndf*8, ndf*4, kernel_size=4, stride=1, padding=1, bias=False)
        ## batch-normalization
        self.bn2 = torch.nn.BatchNorm2d(ndf*4)
        #convolution
        self.conv3 = torch.nn.Conv2d(ndf*4, ndf*2, kernel_size=4, stride=1, padding=1, bias=False)
        ## batch-normalization
        self.bn3 = torch.nn.BatchNorm2d(ndf*2)
        #convolution
        self.conv4 = torch.nn.Conv2d(ndf*2, ndf, kernel_size=4, stride=2, padding=1, bias=False)

        # Read-out layer : ndf * 2 * 2 input features, ndf output features 
        self.fc1 = torch.nn.Linear((ndf * 6 * 6)+1, 50)
        self.fc2 = torch.nn.Linear(50, 25)
        self.fc3 = torch.nn.Linear(25, 1)
        
    def forward(self, x, energy):
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True) # 15 x 15
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, inplace=True) # 14 x 14
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, inplace=True) # 13 x 13
        x = F.leaky_relu(self.conv4(x), 0.2, inplace=True)  # 6x6

        #After series of convlutions --> size changes from (nc, 30, 30) to (ndf, 6, 6)

        
        x = x.view(-1, self.ndf * 6 * 6) 
        x = torch.cat((x, energy), 1)
        
        # Size changes from (ndf, 30, 30) to (1, (ndf * 6 * 6) + 1) 
        #Recall that the -1 infers this dimension from the other given dimension


        # Read-out layer 
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        x = self.fc3(x)
        
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
        
        ## linear projection
        self.cond = torch.nn.Linear(self.z, 5*5*ngf*8)
        
        ## deconvolution
        self.deconv1 = torch.nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=2, stride=3, padding=1, bias=False)
        ## batch-normalization
        self.bn1 = torch.nn.BatchNorm2d(ngf*4)
        ## deconvolution
        self.deconv2 = torch.nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=2, stride=2, padding=1, bias=False)
        ## batch-normalization
        self.bn2 = torch.nn.BatchNorm2d(ngf*2)
        # deconvolution
        self.deconv3 = torch.nn.ConvTranspose2d(ngf*2, ngf, kernel_size=6, stride=1, padding=1, bias=False)
        ## batch-normalization
        self.bn3 = torch.nn.BatchNorm2d(ngf)
        # deconvolution
        self.deconv4 = torch.nn.ConvTranspose2d(ngf, 1, kernel_size=8, stride=1, padding=1, bias=False)
        

        
        
    def forward(self, noise):
        
        layer = []
        ## need to do generate N layers, hence the loop!
        for i in range(self.nc):     
    
            #noise 
            x = F.leaky_relu(self.cond(noise), 0.2, inplace=True)

            ## change size for deconv2d network. Image is 5x5
            x = x.view(-1,self.ngf*8,5,5)        

            ## apply series of deconv2d and batch-norm
            x = F.leaky_relu(self.bn1(self.deconv1(x, output_size=[x.size(0), x.size(1) , 12, 12])), 0.2, inplace=True) 
            x = F.leaky_relu(self.bn2(self.deconv2(x, output_size=[x.size(0), x.size(1) , 22, 22])), 0.2, inplace=True)
            x = F.leaky_relu(self.bn3(self.deconv3(x, output_size=[x.size(0), x.size(1) , 25, 25])), 0.2, inplace=True)                         
            x = F.relu(self.deconv4(x, output_size=[x.size(0), x.size(1) , 30, 30])) 

            ##Image is 30x30 now
            
           

            layer.append(x)
        
       
        ## concentration of the layers
        x = torch.cat([layer[i] for l in range(self.nc)], 1)


        return x
