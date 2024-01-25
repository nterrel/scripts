from ase.visualize import view
from ase.optimize import BFGS
import ase
import torch
import torchani
from torchani.datasets import ANIDataset
from pathlib import Path
import math
from torchani.units import hartree2kcalmol
from torchani.utils import PERIODIC_TABLE


def tensor_from_xyz(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        num_atoms = int(lines[0])
        coordinates = []
        species = []
        element, a, b, c = lines[2].split()
        cell = torch.diag(torch.tensor([float(a), float(b), float(c)]))
        for line in lines[2:]:
            values = line.split()
            if values:
                s = values[0].strip()
                x = float(values[1])
                y = float(values[2])
                z = float(values[3])
                coordinates.append([x, y, z])
                species.append(PERIODIC_TABLE.index(s))
        coordinates = torch.tensor(coordinates)
        species = torch.tensor(species, dtype=torch.long)
        assert coordinates.shape[0] == num_atoms
        assert species.shape[0] == num_atoms
    return species, coordinates

# Create a function which computes the stdev, coefficient of variation of forces


device = torch.device('cpu')
ani2x = torchani.models.ANI2x().to(device).double()

ds_path = Path('/home/nick/First_DSs/ANI-1x-first-conformers.h5')
ds = ANIDataset(locations=ds_path, names=('1x_first'))

epoxide = tensor_from_xyz(Path('/home/nick/m-epoxide.xyz'))
epoxide[1].requires_grad = True
inp = (epoxide[0][None], epoxide[1][None])
print(inp)

members_energies = ani2x.members_energies(inp).energies
sqrt_n = math.sqrt(len(epoxide[0]))
print('QBC:', hartree2kcalmol(members_energies.std(0) / sqrt_n).item(), 'kcal/mol')

forces = []
for i in members_energies:
    derivative = torch.autograd.grad(i, inp[1], retain_graph=True)[0]
    force = -derivative
    # print(force)
    forces.append(force)

forces = hartree2kcalmol(torch.cat(forces, dim=0))
mean_force = forces.mean(0)
stdev_force = forces.std(0)

print('mean force:\n', mean_force)
print('number of force vectors:\n', len(forces[0]))
print('Std in forces\n', stdev_force, '(kcal/mol)/angstrom')

coef_var_forces = stdev_force / mean_force

print('Coefficient of variation in forces:\n', coef_var_forces)

print(coef_var_forces.mean(), coef_var_forces.std())

#atomic_qbcs = hartree2kcalmol(ani2x.atomic_qbcs(inp).ae_stdev)
#print('number of atomic energy stdev:',len(atomic_qbcs[0]))
#print('Std atomic energies:\n',atomic_qbcs,'kcal/mol')

calculator = ani2x.ase()
ase_epoxide = ase.Atoms(numbers=inp[0][0].detach(
).numpy(), positions=inp[1][0].detach().numpy())
ase_epoxide.calc = calculator

print(ase_epoxide.positions[None])

dyn = BFGS(ase_epoxide)
dyn.run(fmax=0.0001)

updated_coord = torch.tensor(ase_epoxide.positions[None])
updated_coord.requires_grad = True
new_inp = (epoxide[0][None], updated_coord)
print(new_inp)

members_energies = ani2x.members_energies(new_inp).energies
print('QBC:', hartree2kcalmol(members_energies.std(0)/sqrt_n).item(), 'kcal/mol')

new_forces = []

for i in members_energies:
    derivative = torch.autograd.grad(i, new_inp[1], retain_graph=True)[0]
    force = -derivative
    # print(force)
    new_forces.append(force)
new_forces = hartree2kcalmol(torch.cat(new_forces, dim=0))
mean_force = new_forces.mean(0)
stdev_force = new_forces.std(0)

print('mean force:\n', mean_force)
print('number of force vectors:\n', len(forces[0]))
print('Std in forces\n', stdev_force, '(kcal/mol)/angstrom')

print(stdev_force.mean(), stdev_force.std())

coef_var_forces = stdev_force / mean_force
print('Coefficient of variation in forces:\n', coef_var_forces)
print(coef_var_forces.mean(), coef_var_forces.std())
