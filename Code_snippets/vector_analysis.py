# Make a function to compare force vectors to reference:

import torch
import torchani
from pathlib import Path
from torchani.datasets import ANIDataset
from torchani.units import hartree2kcalmol


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ani2x = torchani.models.ANI2x().to(device).double()
ds_path = Path('/home/nick/First_DSs/ANI-1x-first-conformers.h5')
ds = ANIDataset(locations=ds_path, names='1x-first')

ch4_forces = ds['CH4']['forces'].to(device)
print(ch4_forces)
print(ch4_forces.shape)

normalized_ch4_forces = torch.nn.functional.normalize(ch4_forces, dim=-1)
print(normalized_ch4_forces)
print(normalized_ch4_forces.shape)

assert normalized_ch4_forces == ch4_forces
'''
ch4_species = ds['CH4']['species'].to(device)
ch4_coord = ds['CH4']['coordinates'].to(device)
inp = (ch4_species, ch4_coord)
members_forces = ani2x.members_forces(inp).model_forces
print('mem_forces:\n', hartree2kcalmol(members_forces))

print(members_forces.shape)

norm_members_forces = torch.linalg.norm(members_forces, dim=(1, 2), keepdim=True)
print('normalized mem_forces:\n', hartree2kcalmol(norm_members_forces))

assert members_forces == norm_members_forces
# split_forces = ch4_forces.split(1, dim=0)
norm_forces = ch4_forces / ch4_forces.norm(dim=1)[:, None]
print(hartree2kcalmol(norm_forces))

#print(ch4_forces, '\n', split_forces)
#print(ch4_forces.shape, members_forces.shape)

# print(hartree2kcalmol(ani2x.force_qbcs(inp).mean_force))
"""
for i in members_forces:
    i = i.split(1, dim=0)
    print(i)
    print()
    for j in ch4_forces:
        j = j.split(3, dim=0)
        print(j)
        print(torch.dot(i,j))
        break
    # print(torch.matmul(i, ch4_forces))
"""
# def find_theta():
# Need to pull forces from dataset and predict using members_forces / forceQBCs functions
'''
