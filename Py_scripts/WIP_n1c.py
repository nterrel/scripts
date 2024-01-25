import torch
import torchani
from pathlib import Path
import numpy as np
import pandas as pd
import math
from torchani.datasets import ANIDataset
from tqdm import tqdm

from ANI2x_n1c.ani2x_nc1 import CustomEnsemble

device = torch.device('cuda')
model = CustomEnsemble(periodic_table_index=True, return_option=3).to(device)
print(model)

ds_path = Path('/home/nick/First_DSs/ANI-1x-first-conformers.h5/')
ds = ANIDataset(locations=(ds_path), names=('1x first'))
print(ds)

from ch4 import min_ch4
print(model(min_ch4))

df = pd.DataFrame(columns=['Formula','Species','Avg_AE (Hartree)','Stdev (Hartree)','QBC (Hartree)'])

count = 0

formula_list = []
species_list = []
avg_ae_list = []
stdev_list = []
qbc_list = []

for index, formula in tqdm(enumerate(ds.keys())):
    conformer = ds.get_conformers(formula)
    species = conformer['species'].to(device)
    coordinates = conformer['coordinates'].to(device)
    assert len(species) == len(coordinates)
    ani_input = (species,coordinates)
    #print(ani_input)
    species_index, mol_e, ae = model(ani_input)
    num_atoms = (species_index >= 0).sum(dim=1, dtype=mol_e.dtype)
    #print(num_atoms)

    avg_ae = ae.mean(0).tolist()
    stdev_ae = ae.std(0).tolist()
    variance = ae.var(0).tolist()
    coef_var = (ae.std(0)/abs(ae.mean(0))).tolist()

    qbc = mol_e.std(0, unbiased=True)
    #print(qbc)
    qbc_factor = (qbc / num_atoms.sqrt())
    #kcal_qbc_factor = hartree2kcalmol(qbc / num_atoms.sqrt())
    #print('QBC factor:',qbc_factors.item(),'kcal/mol')

    for atoms in avg_ae[0]:
        formula_list.append(formula)
        # Want a chemical formula and qbc factor attached to each atom type
        qbc_list.append(qbc_factor.cpu().detach().numpy())


    species_list.extend(species.tolist()[0])
    avg_ae_list.extend(avg_ae[0])
    stdev_list.extend(stdev_ae[0])



df['Formula'] = formula_list
df['Species'] = species_list
df['Avg_AE (Hartree)']  = avg_ae_list
df['Stdev (Hartree)']   = stdev_list
df['QBC (Hartree)'] = qbc_list
print(df)
df.to_csv('df_1x_n1c.csv')
