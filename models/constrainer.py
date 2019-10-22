import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
import models.HDF5Dataset as H
from torch.utils import data

class energyRegressor(nn.Module):
    """ 
    Energy regressor of WGAN. 

    """

    def __init__(self, nc):
        super(energyRegressor, self).__init__()
        self.nc = nc
        
        ## 3d conv layers
        self.conv1 = torch.nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm3d(1)
        self.conv2 = torch.nn.Conv3d(1, 16, kernel_size=(6,6,3), stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm3d(16)
        self.conv3 = torch.nn.Conv3d(16, 32, kernel_size=(6,6,3), stride=1, padding=0)
        self.bn3 = torch.nn.BatchNorm3d(32)
        self.conv4 = torch.nn.Conv3d(32, 32, kernel_size=(6,6,3), stride=1, padding=0)
        self.bn4 = torch.nn.BatchNorm3d(32)
        self.conv5 = torch.nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding=0)
        self.bn5 = torch.nn.BatchNorm3d(64)
        self.conv6 = torch.nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding=0)
        self.bn6 = torch.nn.BatchNorm3d(64)
        
        ## FC layers
        self.fc1 = torch.nn.Linear(64 * 18 * 9 * 9, 50)
        self.fc2 = torch.nn.Linear(50, 1)
        
    def forward(self, x):
        #input shape :  [30, 30, 30]
        ## reshape the input: expand one dim
        x = x.unsqueeze(1)
        
        ## image [30, 30, 30]
        ### convolution adn batch normalisation
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.2, inplace=True)
        ## shape [9, 9, 18]
        
        ## flatten for FC
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))
        
        ## pass to FC layers
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return x
    