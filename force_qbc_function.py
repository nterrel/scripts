# write a functions to produce:
#   the forces from each model
#   stdev in those forces
#   mean force
#   coefficient of variation in forces (stdev / mean)
#   ?

import torch
import torchani
import math
import numpy as np
from pathlib import Path
from torchani.datasets import ANIDataset
from torchani.units import hartree2kcalmol

import ase
from ase.optimize import BFGS

device = torch.device('cpu')
ani1x = torchani.models.ANI1x().to(device).double()
ani2x = torchani.models.ANI2x().to(device).double()

calculator_1x = ani1x.ase()
calculator_2x = ani2x.ase()

ds_path = Path('/home/nick/First_DSs/ANI-1x-first-conformers.h5')
ds = ANIDataset(locations=ds_path,names='1x_first')

ch4_species = ds['CH4']['species']
ch4_coord = ds['CH4']['coordinates']
ch4 = (ch4_species,ch4_coord)

ase_ch4 = ase.Atoms(numbers=ch4_species[0].detach().numpy(),positions=ch4_coord[0].detach().numpy())
ase_ch4.calc = calculator_1x
dyn = BFGS(ase_ch4)
dyn.run(fmax = 0.0005)

ch4_1x = torch.tensor(ase_ch4.positions[None])
ch4_1x.requires_grad=True
inp_1x = (ch4[0],ch4_1x)

print('1x:\n')

members_energies_1x = ani1x.members_energies(inp_1x).energies
sqrt_n = math.sqrt(len(ch4_species[0]))
print('1x QBC:',hartree2kcalmol(members_energies_1x.std(0)/sqrt_n).item(),'kcal/mol')

forces_1x = []
for i in members_energies_1x:
    derivative = torch.autograd.grad(i,inp_1x[1],retain_graph=True)[0]
    force = -derivative
    #print(force)
    forces_1x.append(force)
    
forces_1x = hartree2kcalmol(torch.cat(forces_1x, dim=0))
mean_force = forces_1x.mean(0)
stdev_force = forces_1x.std(0)
print('Mean force:\n',mean_force)
print('Number of force vectors:\n',len(forces_1x[0]))
print('Std in forces\n',stdev_force,'(kcal/mol)/angstrom')

coef_var_forces = stdev_force / mean_force

print('Coefficient of variation in forces:\n',coef_var_forces)

print(coef_var_forces.mean(),coef_var_forces.std())

print('2x:\n')


ase_ch4 = ase.Atoms(numbers=ch4_species[0].detach().numpy(),positions=ch4_coord[0].detach().numpy())
ase_ch4.calc = calculator_2x
dyn = BFGS(ase_ch4)
dyn.run(fmax = 0.0005)

ch4_2x = torch.tensor(ase_ch4.positions[None])
ch4_2x.requires_grad=True
inp_2x = (ch4[0],ch4_2x)

members_energies_2x = ani2x.members_energies(inp_2x).energies
print('2x QBC:',hartree2kcalmol(members_energies_2x.std(0)/sqrt_n).item(),'kcal/mol')

forces_2x = []
for i in members_energies_2x:
    derivative = torch.autograd.grad(i,inp_2x[1],retain_graph=True)[0]
    force = -derivative
    #print(force)
    forces_2x.append(force)
    
forces_2x = hartree2kcalmol(torch.cat(forces_2x, dim=0))
mean_force = forces_2x.mean(0)
stdev_force = forces_2x.std(0)
print('Mean force:\n',mean_force)
print('Number of force vectors:\n',len(forces_2x[0]))
print('Std in forces\n',stdev_force,'(kcal/mol)/angstrom')

coef_var_forces = stdev_force / mean_force

print('Coefficient of variation in forces:\n',coef_var_forces)

print(coef_var_forces.mean(),coef_var_forces.std())