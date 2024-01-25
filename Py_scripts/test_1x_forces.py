import torch
import torchani
from torchani.datasets import ANIDataset
from torchani.units import hartree2kcalmol
from pathlib import Path
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ani2x = torchani.models.ANI2x().to(device)

ds_path = Path.cwd() / '/home/nick/First_DSs/ANI-1x-first-conformers.h5'

ds = ANIDataset(locations=(ds_path), names=('ANI-1x-first'))
print(len(ds))

def mean_squared_error(expected_forces, predicted_forces):
    squared_diff = (expected_forces - predicted_forces) ** 2
    mse_loss = torch.sum(squared_diff) / expected_forces.numel()
    return mse_loss

df = pd.DataFrame(columns=['Formula','Dataset forces','Members forces','Mean forces','Stdev forces'])
formula_list = []
ds_forces_list = []
members_forces_list = []
mean_force_list = []
stdev_force_list = []

for index, formula in tqdm(enumerate(ds.keys())):
    stop = 3200
    if index >= stop: break
    conformer = ds.get_conformers(formula)
    print(formula)
    species = conformer['species'].to(device)
    coordinates = conformer['coordinates'].to(device)
    ani_input = (species, coordinates)
    expected_forces = hartree2kcalmol(conformer['forces'].to(device))
    
    predicted_forces = hartree2kcalmol(ani2x.members_forces(ani_input).model_forces)
    mean_predicted_forces = predicted_forces.mean(0)
    stdev_forces = predicted_forces.std(0)

    formula_list.append(formula)
    ds_forces_list.extend(expected_forces.tolist())
    members_forces_list.extend([predicted_forces.tolist()])
    mean_force_list.extend([mean_predicted_forces.tolist()])
    stdev_force_list.extend([stdev_forces.tolist()])
    
df['Formula'] = formula_list
df['Dataset forces'] = ds_forces_list
df['Members forces'] = members_forces_list
df['Mean forces'] = mean_force_list
df['Stdev forces'] = stdev_force_list


print(df)
df.to_csv('df_1x-first_forces.csv')