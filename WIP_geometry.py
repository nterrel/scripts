import copy

import torch
from torch import Tensor
import torchani
from torchani.models import ANI2x
from torchani.datasets import ANIDataset
from pathlib import Path
from tqdm.auto import tqdm

device = torch.device('cpu')
ds_path = Path.cwd() / 'First_DSs/ANI-1x-first-conformers.h5'

# Change these if desired, assigned in the __init__ function for the optimizer
ds = ANIDataset(locations=(ds_path), names=('1x-first'))
ani2x = ANI2x().to(device)
n_steps = 250
learning_rate = 0.001


class optimizer:
    def __init__(self,
                 model: torchani.models,
                 dataset: ANIDataset, 
                 n_steps,
                 learning_rate):
        self.model = ani2x
        self.dataset = ds
        self.n_steps = n_steps
        self.lr = learning_rate

    def iterate_dataset(ds):
        for k, v in tqdm(ds.items()):
            initial_coordinates = v['coordinates']
            coordinates = copy.deepcopy(initial_coordiantes)
            coordinates.requires_grad=True
            species = v['species']
            ani_input = (species, coordinates)
            geom_opt(ani_input, n_steps)

    def geom_opt(ani_input, n_steps):
        optimizer = torch.optim.Adam([ani_input], lr=learning_rate)
        for epoch in range(self.n_steps):
            _, energy = ani2x(ani_input)
            energy.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f'Epoch {epoch}: E={energy.item()}')
        

if __name__ == '__main__':
    optimizer()
    for k, v in tqdm(ds.items()):           # tqdm( ... ) gives a progress bar
        initial_coordinates = v['coordinates']
        coordinates = copy.deepcopy(initial_coordinates)
        coordinates.requires_grad = True
        species = v['species']
        ani_input = (species, coordinates)
        
        #_species, ae, std = ani2x.atomic_qbcs(ani_input)

        lr = 0.001
        n_epochs = 250
        optimizer = torch.optim.Adam([coordinates], lr=lr)



# NEED TO ADD:
    # Initialize new (empty) dataset for storing minimized structures
    # Move 'k, v in ds.items():' for loop into its own function "iterate_ds(ds)"
    # Move optimzer loop to its own function "geom_opt"


    # When finished, the statement 'if __name__ == '__main__':' should only execute other functions, not define any loops
