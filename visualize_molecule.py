import torch
import torchani
from torchani.datasets import ANIDataset
import ase
from pathlib import Path

device = torch.device('cpu')
ani2x = torchani.models.ANI2x(periodic_table_index=True)

ds_path = Path.cwd() / 'ANI-1x-first-conformers.h5'
ds = ANIDataset(locations=(ds_path), names=('1x-first'))

# weird molecule i wanted to look at
c2h8n6o2 = ds['C2H8N6O2']
species = c2h8n6o2['species']
coordinates = c2h8n6o2['coordinates']
inp = (species, coordinates)

# qbc info about this one
print('QBC factor:',ani2x.energies_qbcs(inp).qbcs)
print('Species:',species[0])
print('Atomic rhos:',ani2x.atomic_qbcs(inp).ae_stdev)

# this is the only stuff you need for ase.visualize.view, documentation has a lot of options
#   can configure to open vmd or avogadro, or plot with matplotlib instead of using the ase gui
from ase import Atoms
from ase.visualize import view
molecule = Atoms(numbers=species[0], positions=coordinates[0])
view(molecule)
