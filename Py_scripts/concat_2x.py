import torch
import torchani
from pathlib import Path
from torchani.datasets import ANIDataset
from torchani.datasets.utils import concatenate

ds_paths = (Path('/home/nick/Datasets/ani2x/ANI-1x-wB97X-631Gd.h5'), Path('/home/nick/Datasets/ani2x/ANI-2x_dimers-wB97X-631Gd.h5'), Path('/home/nick/Datasets/ani2x/ANI-2x_heavy-wB97X-631Gd.h5'))
ds = ANIDataset(locations=(ds_paths), names=('1x','2x_dimers','2x_heavy'))
concat_path = Path('/home/nick/Datasets/ani2x/ALL_COMBINED.h5')
ds = concatenate(ds, concat_path, delete_originals=False)
