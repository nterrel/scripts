import torch
import torchani
from torchani.datasets import ANIDataset
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from torchani.units import hartree2kcalmol

device = torch.device('cpu')
ani2x = torchani.models.ANI2x()

dataset_path = Path.cwd() / '/home/nick/ANI-2x-first-conformers.h5'
ds = ANIDataset(locations=(dataset_path), names=('ANI-2x-first-conformers'))
print('Conformers in dataset', len(ds.keys()))

df = pd.DataFrame(columns=['Formula','Species','Avg_AE','Stdev','QBC'])
count = 0
formula_list = []
species_list = []
avg_ae_list = []
stdev_list = []
qbc_list = []


for index, formula in enumerate(ds.keys()):
    #print('Molecular formula:\n',formula)
    
    conformer = ds.get_conformers(formula)    
    species = conformer['species']
    #print(species)
    coordinates = conformer['coordinates']
    ani_input = (species,coordinates)
    #mol_e = ani2x(ani_input).energies
    #print('Molecular energy:\n',mol_e.detach().item(), 'Hartree')
    _, energies, qbc_factor = ani2x.energies_qbcs(ani_input)
        
    species_index, ae = ani2x.atomic_energies(ani_input, average=False, with_SAEs=False)
    #print('Atomic energy contributions (no SAEs):\n',ae.detach(),'in Hartree')
    
    avg_ae = ae.mean(0).tolist()
    #print('Average atomic energy contribution:\n',avg_ae[0],'in Hartree')
    
    stdev_ae = ae.std(0).tolist()
    #print('Stdev in atomic energies across the ensemble:\n',stdev_ae.detach()[0],'in Hartree')
    
    coef_var = (ae.std(0)/abs(ae.mean(0))).tolist()
    #print('Coefficient of variation in atomic energies:\n',coef_var[0])
    
    for atoms in avg_ae[0]:
        formula_list.append(formula)
        # Want a chemical formula and qbc factor attached to each atom type
        qbc_list.append(qbc_factor.detach().numpy())
    
    #print(formula_list)
    species_list.extend(species.tolist()[0])
    #print(species_list)
    avg_ae_list.extend(avg_ae[0])
    #print(avg_ae_list)
    stdev_list.extend(stdev_ae[0])
    #print(stdev_list)
    
    
    # Counter to stop iterating (test that it works):
    #count += 1
    #if count == 25:
    #    break


df['Species'] = species_list
df['Avg_AE'] = avg_ae_list
df['Stdev'] = stdev_list
df['Formula'] = formula_list
df['QBC'] = qbc_list
print(df)
df.to_csv('df_2x.csv')
