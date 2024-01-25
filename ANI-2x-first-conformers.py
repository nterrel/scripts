import torch
from torchani.datasets import ANIDataset
from pathlib import Path

ani2x_data = Path('/home/nick/Datasets/ani2x/ALL_COMBINED.h5')        # Location of ANI-1x dataset h5 file
ani2x_first = Path('/home/nick/ANI-2x-first-conformers.h5')  # Location of first conformers only dataset h5 file

if ani2x_first.exists():
    Path.unlink

ds = ANIDataset(locations=(ani2x_data), names=('2x-combined'))
print(ds.properties)
print(ds.grouping)

ds_new = ANIDataset(locations=(ani2x_first), names=('ANI-2x-first-conformers'), create=True) 
print(ds_new.grouping)
# Empty dataset has no properties

### Everything above this works ###
for formula in list(ds.keys()):
    conformer = ds.get_conformers(formula, [0])         # Must do (formula, [0]) rather than (formula, 0) to have a properly shaped dictionary as input 
                                                        # ** When adding 1 conformer at a time **
    #print({k: v.shape for k, v in conformer.items()})  # The line Ignacio added to check the input shape
    # conformer = {k: v.unsqueeze(0) for k, v in conformer.items()}
    ds_new.append_conformers(formula, conformer)

print(ds_new)
