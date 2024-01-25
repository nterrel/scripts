import torch
import torchani
from torchani.datasets import ANIDataset
from pathlib import Path
import ase
from ase.optimize import BFGS

ds_path = Path('/home/nick/Datasets/ani1x/ANI-1x-wB97X-631Gd.h5')
ds = ANIDataset(locations=ds_path,names='1x')
ds = ds['CH4']
print(len(ds))
