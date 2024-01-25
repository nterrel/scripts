import torch
import torchani
from torchani.datasets import ANIDataset
from pathlib import Path
import os
import numpy as np
import pandas as pd
import math
from torchani.units import hartree2kcalmol
from tqdm import tqdm
import sys


device = torch.device('cpu')
ani2x = torchani.models.ANI2x().to(device)
calculator = ani2x.ase()

ds_path = Path.cwd() / 'Datasets/ani2x'
combined_paths = []
for i in os.listdir(ds_path):
    i = str(ds_path)+'/'+i
    combined_paths.append(i)
ds_opt_path = Path.cwd() / 'ANI-2x-optimized.h5'
ds_2x = ANIDataset(locations=combined_paths, names=('1x', '2x-dimers', '2x-heavy'))

#ds_opt = ANIDataset(locations=(ds_opt_path), names='ANI-2x-optimized')

sys.exit('Stop')

count = 0
formula_list = []
species_list = []
avg_ae_list = []
stdev_list = []
qbc_list = []

for index, conformer in tqdm(enumerate(ds.iter_conformers())):
    species = conformer['species'][None]
    coordinates = conformer['coordinates'][None]
    assert len(species) == len(coordinates)
    ani_input = (species, coordinates)
    _, energies, qbc_factor = ani2x.energies_qbcs(ani_input)
    species_index, ae = ani2x.atomic_energies(ani_input, average=False, with_SAEs=False)
    avg_ae = ae.mean(0).tolist()
    stdev_ae = ae.std(0).tolist()
    for atoms in avg_ae[0]:
        formula_list.append(formula)
        qbc_list.append(qbc_factor.detach().numpy())
    species_list.extend(species.tolist()[0])
    avg_ae_list.extend(avg_ae[0])
    stdev_list.extend(stdev_ae[0])

    count += 1
    print('iter #', count)
    if count == 100:
        break



df['Species'] = species_list
df['Avg_AE'] = avg_ae_list
df['Stdev'] = stdev_list
df['Formula'] = formula_list
df['QBC'] = qbc_list
print(df)
df.to_csv('df_2x_full.csv')
