import torch
import torchani
from torchani.units import HARTREE_TO_KCALMOL
from ANI_load_functions import ANIRepulsionEnsemble

path1x = '/home/nick/From_yinuo/runs_1x_smooth/' # change this
device = torch.device("cpu")
functional = "wb97x"
basis_set = "631gd"
n_models = 6
ensemble_1x = ANIRepulsionEnsemble(path1x, functional, basis_set, n_models).float()

species = torch.tensor([[6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1]])
coordinates = torch.tensor([[[-0.9949, -1.2415,  0.4576],
         [-1.8726, -0.1363, -0.0685],
         [-1.3382,  1.0712, -0.2418],
         [ 0.1076,  1.2830,  0.1221],
         [ 0.3584, -1.2507, -0.2764],
         [ 1.0214,  0.1475, -0.3934],
         [-0.8231, -1.0883,  1.5341],
         [-1.4785, -2.2193,  0.3596],
         [-2.9165, -0.3370, -0.3002],
         [-1.9257,  1.9053, -0.6203],
         [ 0.1941,  1.3270,  1.2200],
         [ 1.0395, -1.9468,  0.2295],
         [ 0.1928, -1.6574, -1.2806],
         [ 0.4782,  2.2453, -0.2489],
         [ 2.3727,  0.1948,  0.3198],
         [ 2.2519, -0.0217,  1.3893],
         [ 2.8382,  1.1836,  0.2296],
         [ 3.0684, -0.5442, -0.0938],
         [ 1.1959,  0.3448, -1.4588]]])
_, e_1x, qbc_1x = ensemble_1x.energies_qbcs((species, coordinates))
breakpoint()
