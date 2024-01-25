import torch
import torchani
from torchani.datasets import ANIDataset
from torchani.units import hartree2kcalmol
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ani2x = torchani.models.ANI2x().to(device)

ds_path = Path('/home/nick/Datasets/comp6v1/')

ds = torchani.datasets.COMP6v1(root=ds_path, download=False)

atomic_df = pd.DataFrame(columns=['Species','Dataset forces','Members forces','Mean forces','Stdev forces'])
species_list = []
ds_forces_list = []
members_forces_list = []
mean_force_list = []
stdev_force_list = []

for index, conformer in tqdm(enumerate(ds.iter_conformers())):
    stop = 1e6
    if index >= stop: break
    species = conformer['species'][None].to(device)
    coordinates = conformer['coordinates'][None].to(device)
    ani_input = (species, coordinates)
    expected_forces = hartree2kcalmol(conformer['forces'])
    predicted_forces = hartree2kcalmol(ani2x.members_forces(ani_input).model_forces)
    predicted_forces = predicted_forces.transpose(0,1)
    members_forces_list.extend(predicted_forces.tolist())
    mean_predicted_forces = predicted_forces.mean(1).tolist()
    stdev_forces = predicted_forces.std(1).tolist()
    species_list.extend(species.tolist()[0])
    ds_forces_list.extend(expected_forces.tolist())
    mean_force_list.extend(mean_predicted_forces)
    stdev_force_list.extend(stdev_forces)


    
atomic_df['Species'] = species_list
atomic_df['Dataset forces'] = ds_forces_list
atomic_df['Members forces'] = members_forces_list
atomic_df['Mean forces'] = mean_force_list
atomic_df['Stdev forces'] = stdev_force_list


print(atomic_df)

atomic_df = atomic_df.set_index('Species')
atomic_df.to_parquet('comp6v1_forces_df.pq')
