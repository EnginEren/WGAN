# WGAN

Code is taken from `https://github.com/martinarjovsky/WassersteinGAN` and modified for our needs

if you run it in maxwell node (CPU) :

(Assuming you are in `/beegfs/desy/user/`)
 
`docker run -it --rm -v $PWD:/home engineren/ml:pytorch python main.py --imageSize 16 --batchSize 512  \
       --workers 3  --experiment test \`


if you run with GPUs in maxwell :

  1. `singularity instance start --bind $PWD:/home --nv docker://engineren/ml:pytorch testPy`
  2. `singularity run instance://testPy python main.py --imageSize 16 --batchSize 512 --experiment test --cuda`
