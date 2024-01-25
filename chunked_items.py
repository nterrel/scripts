import time
from tqdm import tqdm
import torch
from torchani.datasets import COMP6v1
import torchani
from torchani.units import hartree2kcalmol

ds = COMP6v1(root='/home/nick/Datasets/comp6v1_B973c/', basis_set='def2mTZVP', functional='B973c')
device = torch.device('cuda')
ani2x = torchani.models.ANI2x().to(device)


members_magnitudes_list = []
species_list = []
dataset_magnitudes_list = []
mean_magnitudes_list = []
stdev_magnitudes_list = []



_start = time.perf_counter()
with ds.keep_open("r") as rods:
    pbar = tqdm(total=ds.num_conformers)
    for group, j, conformer in rods.chunked_items(max_size=1500):
        species = conformer["species"].to(device)
        coordinates = conformer["coordinates"].to(device)
        print(coordinates.shape)
        coordinates.requires_grad_(True)
        ani_input = (species, coordinates)
        print(ani_input)
        expected_forces = hartree2kcalmol(conformer['forces'])
        expected_magnitude = expected_forces.norm(dim=-1)
        print(expected_magnitude)
        break
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
        pbar.update(len(coordinates))
    pbar.close()



print(f"Time elapsed: {time.perf_counter() - _start} s")