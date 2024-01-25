import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from aiqm.torchani import ENSEMBLE_MODELS
from aiqm.torchani.datasets import ANI1Loader
from aiqm.torchani.utils import AverageMeter
from tqdm import tqdm

# Define the ensemble size
ensemble_size = 4

# Load the ANI1 dataset
ani1_data = ANI1Loader('ani1x', 256)

# Define the data loader
dataloader = DataLoader(ani1_data, batch_size=32, shuffle=True)

# Define the models in the ensemble
models = []
for i in range(ensemble_size):
    model = ENSEMBLE_MODELS['ANI-1ccx']
    model = nn.DataParallel(model)
    model = model.to(device)
    models.append(model)

# Define the optimizer and learning rate scheduler
optimizers = [AdamW(model.parameters(), lr=1e-4) for model in models]
schedulers = [ReduceLROnPlateau(optimizer, factor=0.5, patience=10) for optimizer in optimizers]

# Define the custom loss function
def ensemble_loss(outputs, targets, models, weights):
    loss = 0
    for i, model in enumerate(models):
        loss += weights[i] * nn.functional.mse_loss(outputs[i], targets)
    return loss

# Define the training loop
for epoch in range(100):
    # Set models to train mode
    for model in models:
        model.train()

    # Train for one epoch
    for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = [model(inputs) for model in models]
        loss = ensemble_loss(outputs, targets, models, [1/ensemble_size]*ensemble_size)

        # Compute gradients and update parameters
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

    # Set models to eval mode
    for model in models:
        model.eval()

    # Evaluate on the validation set and update learning rate
    val_loss_meter = AverageMeter()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = [model(inputs) for model in models]
            val_loss = ensemble_loss(outputs, targets, models, [1/ensemble_size]*ensemble_size)
            val_loss_meter.update(val_loss.item(), inputs.size(0))
    val_loss = val_loss_meter.avg
    for scheduler in schedulers:
        scheduler.step(val_loss)
