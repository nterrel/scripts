import torch
import torchani
from torchani.datasets import ANIDataset
from torchani.units import hartree2kcalmol
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time

_start = time.perf_counter()

device = torch.device('cpu')
ani2x = torchani.models.ANI2x().to(device)

ds_path = Path.cwd() / '/home/nick/First_DSs/ANI-1x-first-conformers.h5'

ds = ANIDataset(locations=(ds_path), names=('ANI-1x-first'))
print(len(ds))

def mean_squared_error(expected_forces, predicted_forces):
    squared_diff = (expected_forces - predicted_forces) ** 2
    mse_loss = torch.sum(squared_diff) / expected_forces.numel()
    return mse_loss

atomic_df = pd.DataFrame(columns=['Species','Dataset forces','Mean forces','Stdev forces'])
species_list = []
ds_forces_list = []
mean_force_list = []
stdev_force_list = []
count = 0

#with ds.keep_open('r') as read_ds:
pbar = tqdm(total=ds.num_conformers)
for index, conformer in tqdm(enumerate(ds.iter_conformers())):
    count += 1
    print(count)
    species = conformer['species'][None].to(device)
    coordinates = conformer['coordinates'][None].to(device)
    ani_input = (species, coordinates)
    expected_forces = hartree2kcalmol(conformer['forces'].to(device))
    predicted_forces = hartree2kcalmol(ani2x.members_forces(ani_input).model_forces)
    mean_predicted_forces = predicted_forces.mean(0)
    stdev_forces = predicted_forces.std(0)

    species_list.extend(species.tolist()[0])
    ds_forces_list.extend(expected_forces.tolist())
    #members_forces_list.extend([predicted_forces.tolist()])
    mean_force_list.extend(mean_predicted_forces.tolist())
    stdev_force_list.extend(stdev_forces.tolist())
    pbar.update(len(coordinates))
    print(pbar)
    if count > 100: break
    


    
atomic_df['Species'] = species_list
atomic_df['Dataset forces'] = ds_forces_list
atomic_df['Mean forces'] = mean_force_list
atomic_df['Stdev forces'] = stdev_force_list



print(atomic_df)
atomic_df.to_csv('temp.csv')
pbar.close()

print(f'Time elapsed: {time.perf_counter() - _start} s')