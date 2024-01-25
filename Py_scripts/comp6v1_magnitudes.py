import torch
import torchani
from torchani.units import hartree2kcalmol
from pathlib import Path
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ani2x = torchani.models.ANI2x().to(device)

ds_path = Path('/home/nick/Datasets/comp6v1/')

ds = torchani.datasets.COMP6v1(root=ds_path, download=False)

magnitudes_df = pd.DataFrame(columns=['Species','Dataset magnitudes', 'Members magnitudes', 'Mean magnitudes', 'Stdev magnitudes', 'Difference'])

species_list = []
dataset_magnitudes_list = []
members_magnitudes_list = []
mean_magnitudes_list = []
stdev_magnitudes_list = []

for index, conformer in tqdm(enumerate(ds.iter_conformers())):
    stop = 2e6 
    if index >= stop: break
    species = conformer['species'][None].to(device)
    coordinates = conformer['coordinates'][None].to(device)
    ani_input = (species, coordinates)
    expected_forces = hartree2kcalmol(conformer['forces'])
    expected_magnitude = expected_forces.norm(dim=-1)
    predicted_forces = hartree2kcalmol(ani2x.members_forces(ani_input).model_forces)
    predicted_forces = predicted_forces.transpose(0,1)

    predicted_magnitudes = predicted_forces.norm(dim=-1)
    stdev_magnitudes = predicted_magnitudes.std(-1)
    
    members_magnitudes_list.extend(predicted_magnitudes.tolist())
    mean_predicted_magnitude = predicted_forces.mean(1).norm(dim=-1)
    
    species_list.extend(species.tolist()[0])
    dataset_magnitudes_list.extend(expected_magnitude.tolist())
    mean_magnitudes_list.extend(mean_predicted_magnitude.tolist())
    stdev_magnitudes_list.extend(stdev_magnitudes.tolist())

    print(len(species_list))

magnitudes_df['Species'] = species_list
magnitudes_df['Dataset magnitudes'] = dataset_magnitudes_list
magnitudes_df['Members magnitudes'] = members_magnitudes_list
magnitudes_df['Mean magnitudes'] = mean_magnitudes_list
magnitudes_df['Stdev magnitudes'] = stdev_magnitudes_list
magnitudes_df['Difference'] = abs(magnitudes_df['Dataset magnitudes']-magnitudes_df['Mean magnitudes'])


print(magnitudes_df)

magnitudes_df = magnitudes_df.set_index('Species')
magnitudes_df.to_parquet('mag_df.pq')
