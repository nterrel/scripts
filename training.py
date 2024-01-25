import torch
import torchani
import os
import math
from pathlib import Path
import torch.utils.tensorboard
import tqdm
import pkbar
from torchani.units import hartree2kcalmol
from torchani.transforms import AtomicNumbersToIndices, SubtractSAE

device = torch.device('cuda')

h5_path = Path('/home/nick/Datasets/ani2x/ANI-1x-wB97X-631Gd.h5')
batched_dataset_path = './batched_dataset_1x'

if not Path(batched_dataset_path).resolve().is_dir():
    torchani.datasets.create_batched_dataset(h5_path, dest_path=batched_dataset_path, batch_size=2560, folds=5)

training = torchani.datasets.ANIBatchedDataset(batched_dataset_path, split='training0')
validation = torchani.datasets.ANIBatchedDataset(batched_dataset_path, split='validation0')

cache = True
if not cache:
    training = torch.utils.data.DataLoader(training, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True, batch_size=None)
    validation = torch.utils.data.DataLoader(validation, shuffle=False, num_workers=2, prefetch_factor=2, pin_memory=True, batch_size=None)

elif cache: 
    training = torch.utils.data.DataLoader(training.cache(), shuffle=True, batch_size=None)
    validation = torch.utils.data.DataLoader(validation.cache(), shuffle=False, batch_size=None)

elements = ('H', 'C', 'N', 'O')
self_energies = [-0.499321200000, -37.83383340000, -54.57328250000, -75.04245190000]
transform = torchani.transforms.Compose([AtomicNumbersToIndices(elements), SubtractSAE(elements, self_energies)]).to(device)

aev_computer = torchani.AEVComputer.like_1x(use_cuda_extension=True)
aev_dim = aev_computer.aev_length

H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 144),
    torch.nn.CELU(0.1),
    torch.nn.Linear(144, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
print(nn)


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)


nn.apply(init_params)

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
])

AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)

latest_checkpoint = 'latest.pt'

if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])

def validate():
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    model.train(False)
    with torch.no_grad():
        for properties in validation:
            properties = {k: v.to(device, non_blocking=True) for k,v in properties.items()}
            properties = transform(properties)
            species = properties['species']
            coordinates = properties['coordinates'].float()
            true_energies = properties['energies'].float()
            _, predicted_energies = model((species, coordinates))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]
    model.train(True)
    return hartree2kcalmol(math.sqrt(total_mse / count))

tensorboard = torch.utils.tensorboard.SummaryWriter()

mse = torch.nn.MSELoss(reduction='none')

print('Training starting from epoch', AdamW_scheduler.last_epoch + 1)
max_epochs = 1500
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'best.pt'

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    print('RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)
    learning_rate = AdamW.param_groups[0]['lr']
    
    if learning_rate < early_stopping_learning_rate:
        break

    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    for i, properties in tqdm.tqdm(
            enumerate(training),
            total=len(training),
            desc='epoch {}'.format(AdamW_scheduler.last_epoch)
    ):
        properties = {k: v.to(device, non_blocking=True) for k, v in properties.items()}
        properties = transform(properties)
        species = properties['species']
        coordinates = properties['coordinates'].float()
        true_energies = properties['energies'].float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates))

        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

        AdamW.zero_grad()
        loss.backward()
        AdamW.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
    }, latest_checkpoint)
