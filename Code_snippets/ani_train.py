# Customized training script adapted from nnp_training_low_memory.py

import torch
import torchani
import os
import math
from pathlib import  Path
from torchani.datasets import ANIDataset
import torch.utils.tensorboard
import tqdm
import pkbar

from torchani.transforms import AtomicNumbersToIndices
from torchani.units import hartree2kcalmol

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = Path.cwd()
ds_path = Path.joinpath(path, '/home/nick/Datasets/ani2x/')
path_to_1x = Path.joinpath(ds_path, 'ANI-1x-wB97X-631Gd.h5')
path_to_2x_dimers = Path.joinpath(ds_path,'ANI-2x_dimers-wB97X-631Gd.h5')
path_to_2x_heavy = Path.joinpath(ds_path,'ANI-2x_heavy-wB97X-631Gd.h5') 
ds = ANIDataset(locations=(path_to_1x,
                           path_to_2x_dimers,
                           path_to_2x_heavy),
                names=('1x_data',
                       '2x_dimers',
                       '2x_heavy'))

batch_size = 2560

batched_dataset_path = './batched_dataset_2x'
folds = True

if not Path(batched_dataset_path).resolve().is_dir():
    if not folds:
        torchani.datasets.create_batched_dataset(ds,
                                                 dest_path = batched_dataset_path,
                                                 batch_size=batch_size,
                                                 splits={'training':0.8, 'validation':0.2})
    else:
        torchani.datasets.create_batched_dataset(ds,
                                                 dest_path=batched_dataset_path,
                                                 batch_size=batch_size,
                                                 folds=5)
if not folds:
    training = torchani.datasets.ANIBatchedDataset(batched_dataset_path, split='training')
    validation = torchani.datasets.ANIBatchedDataset(batched_dataset_path, split='validation')

else:
    training = torchani.datasets.ANIBatchedDataset(batched_dataset_path, split='training0')
    validation = torchani.datasets.ANIBatchedDataset(batched_dataset_path, split='validation0')

cache = False
if not cache:
    training = torch.utils.data.DataLoader(training,
                                           shuffle=True,
                                           num_workers=2,
                                           prefetch_factor=2,
                                           pin_memory=True,
                                           batch_size=None) # Make sure not to batch twice
    validation = torch.utils.data.DataLoader(validation,
                                             shuffle=False,
                                             num_workers=2,
                                             prefetch_factor=2,
                                             pin_memory=True,
                                             batch_size=None)
elif cache:
    training = torch.utils.data.DataLoader(training.cache(),
                                           shuffle=True,
                                           batch_size=None)
    validation = torch.utils.data.DataLoader(validation.cache(),
                                             shuffle=False,
                                             batch_size=None)

elements = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']
functional = 'wb97x'
basis_set = '631gd'

self_energies = torchani.utils.sorted_gsaes(elements, functional, basis_set)

transform = torchani.transforms.Compose([AtomicNumbersToIndices(elements),
                                         SubtractSAE(elements, self_energies)]).to(device)

estimate_saes = False
if estimate_saes:
    from torchani.transforms import calculate_saes
    saes, _ = calculate_saes(training, elements, mode='sgd')   
    print(saes)
    transform = torchani.transforms.Compose([AtomicNumbersToIndices(elements), SubtractSAE(elements, saes)]).to(device)

aev_computer = torchani.AEVComputer.like_2x(use_cuda_extension=True)
aev_dim = aev_computer.aev_length


# Think about defining networks how ping does: 
# def atomic_network(layer_dimensions: List (per atom)):
#   assert len(layers_dimensions) == 5
#   return torch.nn.Sequential(
#       torch.nn.Linear(layer_dimensions[0], layer_dimensions[1], bias=False),
#       torch.nn.GELU(),
#       ...
#   )
H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 256, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(256, 192, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(192, 160, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.linear(160, 1, bias=False)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 224, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(224, 192, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(192, 160, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(160, 1, bias=False)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 192, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(192, 160, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(160, 128, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(128, 1, bias=False) 
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 192, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(192, 160, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(160, 128, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(128, 1, bias=False)
)

S_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(160, 128, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(128, 96, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(96, 1, bias=False)
)

F_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(160, 128, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(128, 96, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(96, 1, bias=False)
)

Cl_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(160, 128, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(128, 96, bias=False),
    torch.nn.GELU(0.1),
    torch.nn.Linear(96, 1, bias=False)
)

nn = torchani.ANIModel([H_network, C_network, N_network, O_network, S_network, F_network, Cl_network])
print(nn)

def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

model = torchani.nn.Sequential(aev_computer, nn).to(device)

AdamW = torch.optim.AdamW([
        # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[2].weight], 'weight_decay': 0.00001},
    {'params': [H_network[4].weight], 'weight_decay': 0.000001},
    {'params': [H_network[6].weight]},
    # C networks
    {'params': [C_network[0].weight]},
    {'params': [C_network[2].weight], 'weight_decay': 0.00001},
    {'params': [C_network[4].weight], 'weight_decay': 0.000001},
    {'params': [C_network[6].weight]},
    # N networks
    {'params': [N_network[0].weight]},
    {'params': [N_network[2].weight], 'weight_decay': 0.00001},
    {'params': [N_network[4].weight], 'weight_decay': 0.000001},
    {'params': [N_network[6].weight]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[2].weight], 'weight_decay': 0.00001},
    {'params': [O_network[4].weight], 'weight_decay': 0.000001},
    {'params': [O_network[6].weight]},
    # S networks
    {'params': [S_network[0].weight]},
    {'params': [S_network[2].weight], 'weight_decay': 0.00001},
    {'params': [S_network[4].weight], 'weight_decay': 0.000001},
    {'params': [S_network[6].weight]},
    # F networks
    {'params': [F_network[0].weight]},
    {'params': [F_network[2].weight], 'weight_decay': 0.00001},
    {'params': [F_network[4].weight], 'weight_decay': 0.000001},
    {'params': [F_network[6].weight]},
    # Cl networks
    {'params': [Cl_network[0].weight]},
    {'params': [Cl_network[2].weight], 'weight_decay': 0.00001},
    {'params': [Cl_network[4].weight], 'weight_decay': 0.000001},
    {'params': [Cl_network[6].weight]},
])

# Put SGD optimizer here if using biases, but this is written to not use

AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)

latest_checkpoint = 'latest.pt'

if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    # Would need to include SGD stuff here too, if using biases

def validate():
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    model.train(False)
    with torch.no_grad():
        for properties in validation:
            properties = {k: v.to(device, non_blocking=True) for k, v in properties.items()}
            properties = transform(properties)
            species = properties['species']
            coordinates = properties['coordinates']
            true_energies = properties['energies'].float()
            _, predicted_energies = model((species, coordinates))
            count += predicted_energies.shape[0]
    model.train(True)
    return hartree2kcalmol(math.sqrt(total_mse / count))

tensorboard = torch.utils.tensorboard.SummaryWriter()

mse = torch.nn.MSELoss(reduction='none')

print('Training starting from epoch:', AdamW_scheduler.last_epoch + 1)
max_epochs = 800    # This is just a guess, should be converged by then, I'd hope.
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'best.pt'

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    print('RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)

    learning_rate = AdamW.param_groups[0]['lr']
    if learning_rate < early_stopping_learning_rate:
        break

    # Checkpoint:
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse)
    # Do the same for SGD here, if using biases

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):
        properties = {k: v.to(device, non_blocking=True) for k, v in properties.items()}
        properties = transform(properties)
        species = properties['species']
        coordinates = properties['coordinates'].float()
        true_energies = properties['energies'].float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates))

        loss = (mse(predicted_energies, true_energies)/ num_atoms.sqrt()).mean()

        AdamW.zero_grad()
        loss.backward()
        AdamW.step()

        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
    }, latest_checkpoint)

