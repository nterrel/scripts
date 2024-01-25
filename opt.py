import torch
import torchani
from torchani.datasets import ANIDataset
import ase
from ase.optimize import LBFGS
import time
from pathlib import Path

ds_path = Path('/home/nick/First_DSs/ANI-1x-first-conformers.h5')
ds = ANIDataset(locations=ds_path,names='1x first')
device = torch.device('cuda')
ani2x = torchani.models.ANI2x(periodic_table_index=True).double()
calculator = ani2x.to(device).ase()
count = 1
program_starts = time.time()

for i in ds.keys():
    species = ds[i]['species']
    coord = ds[i]['coordinates']

    molecule = ase.Atoms(numbers=species[0].numpy(),positions=coord[0].numpy())
    print(molecule)
    molecule.calc = calculator
    dyn = LBFGS(molecule)
    dyn.run(fmax=0.1)
    print(count)
    count += 1
    #if count > 10:
    #    break
    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))
    #break
