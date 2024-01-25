import torch
import torchani
from torchani.datasets import ANIDataset
import ase
from ase.optimize import BFGS
from pathlib import Path

device = torch.device('cpu')
ani2x = torchani.models.ANI2x().to(device)

ds_path = Path.cwd() / 'First_DSs/ANI-2x-first-conformers.h5'
ds = ANIDataset(locations=(ds_path), names=('ANI-2x first'))

ch4 = ds['CH4']
species = ch4['species']
coord = ch4['coordinates']
inp = (species, coord)

print(inp)

from WIP_geometry import optimizer
