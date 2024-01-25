# minimal-nnp-training.py
import warnings
warnings.filterwarnings("ignore", message="cuaev not installed")
warnings.filterwarnings("ignore", message="mnp not installed")

import torch
import torchani
import os
import math

elements = ('H', 'C', 'N', 'O', 'S', 'F', 'Cl')
self_energies = [] # FIX THIS LINE

device = torch.device('cpu')

# Need to find the ani-2x dataset to batch here, not in torchani code

aev_computer = torchani.AEVComputer.like_2x()
energy_shifter = torchani.utils.EnergyShifter(None)

aev_dim = aev_computer.aev_length

H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 256),
    torch.nn.CELU(0.1),
    torch.nn.Linear(256, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 224),
    torch.nn.CELU(0.1),
    torch.nn.Linear(224, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 1)
)
'''
S_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

F_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

Cl_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)
'''

nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
# Un-comment this line when you have 2x dataset for training, --> ### S_network, F_network, Cl_network])
print(nn)