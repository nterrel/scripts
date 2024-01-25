import torch
import torchani
from torchani.datasets import ANIDataset
from torchani.utils import PERIODIC_TABLE
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds_path = Path('/home/nick/First_DSs/ANI-1x-first-conformers.h5')
ds = ANIDataset(locations=ds_path, names='1x-first')
ani2x = torchani.models.ANI2x().to(device).double()


def tensor_from_xyz(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        num_atoms = int(lines[0])
        coordinates = []
        species = []
        element, a, b, c = lines[2].split()
        # cell = torch.diag(torch.tensor([float(a), float(b), float(c)]))
        for line in lines[2:]:
            values = line.split()
            if values:
                s = values[0].strip()
                x = float(values[1])
                y = float(values[2])
                z = float(values[3])
                coordinates.append([x, y, z])
                species.append(PERIODIC_TABLE.index(s))
        coordinates = torch.tensor(coordinates, device=device).requires_grad_()
        species = torch.tensor(species, dtype=torch.long, device=device)
        assert coordinates.shape[0] == num_atoms
        assert species.shape[0] == num_atoms
    return species, coordinates


epoxide = tensor_from_xyz(Path('/home/nick/m-epoxide.xyz'))
inp = (epoxide[0][None], epoxide[1][None])
print(inp)

print(ani2x.members_forces(inp))
print(ani2x.force_qbcs(inp))
