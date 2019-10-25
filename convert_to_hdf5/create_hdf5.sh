#!/bin/bash

#SBATCH --partition=all
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --time=24:00:00     
#SBATCH --job-name  hdf5
#SBATCH --output    hdf5-%N-%j.out            # File to which STDOUT will be written
#SBATCH --error     hdf5-%N-%j.err            # File to which STDERR will be written

# go to directory
cd /beegfs/desy/user/eren/WassersteinGAN/convert_to_hdf5

# start 0
python create_hdf5.py --ncpu 16 --rootfile ../data/calo_hits-50GeV.root --branch photonSIM 

exit 0;



