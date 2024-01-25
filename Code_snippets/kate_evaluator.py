import torch
import torchani
import os
import sys
from typing import Tuple, NamedTuple, Optional, Sequence
from torch.nn import Module
from copy import deepcopy

class personal_evaluator:
    def __init__(self,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            wkdir: str=None,
            checkpoint: str='best', 
            netlike1x: bool=False,
            netlike2x: bool=False,
            functional: str = 'wb97x',
            basis_set: str = '631gd',
            constants = None,
            elements = None,
            gsae_dat: str = None,
            activation: Optional[Module] = None,
            bias: bool = False,
            classifier_out: int = 1, 
            personal: bool = True): 
        if netlike1x == True:
            self.constants = '/data/khuddzu/torchani_sandbox/torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params'
            self.elements = ['H', 'C', 'N', 'O']
        elif netlike2x == True:
            self.constants = '/data/khuddzu/torchani_sandbox/torchani/resources/ani-2x_8x/rHCNOSFCl-5.1R_16-3.5A_a8-4.params'
            self.elements = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']
        else:
            self.constants = constants
            self.elements = elements
        self.device = device
        self.wkdir = wkdir
        self.checkpoint = '{}/{}.pt'.format(wkdir, checkpoint)
        self.netlike1x = netlike1x
        self.netlike2x = netlike2x
        self.functional = functional
        self.basis_set = basis_set
        self.gsae_dat = gsae_dat
        self.activation = activation
        self.bias = bias
        self.classifier_out = classifier_out
        self.personal = personal
        #self.model = '{}/model.py'.format(wkdir)

    def AEV_Computer(self):
        consts = torchani.neurochem.Constants(self.constants)
        aev_computer = torchani.AEVComputer(**consts)
        return aev_computer

    def Energy_Shifter(self):
        if self.gsae_dat:
            _, energy_shifter= torchani.neurochem.load_sae(self.gsae_dat, return_dict=True)
        else:
            energy_shifter = torchani.utils.sorted_gsaes(self.elements, self.functional, self.basis_set)
        assert len(energy_shifter) == len(self.elements), "There must be a mistake with what atoms you are trying to use. The length of the EnergyShifter does not match the Elements"
        return energy_shifter

    def standard(self, dims: Sequence[int]):
        r"""Makes a standard ANI style atomic network"""
        if self.activation is None:
            activation = torch.nn.CELU(0.1)
        else:
            activation = self.activation

        dims = list(deepcopy(dims))
        layers = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.extend([torch.nn.Linear(dim_in, dim_out, bias=self.bias), self.activation])
        # final layer is a linear classifier that is always appended
        layers.append(torch.nn.Linear(dims[-1], self.classifier_out, bias=self.bias))

        assert len(layers) == (len(dims) - 1) * 2 + 1
        return torch.nn.Sequential(*layers)


    def like_1x(self, atom: str = 'H', aev_dim=384, **kwargs):
        r"""Makes a sequential atomic network like the one used in the ANI-1x model"""
        dims_for_atoms = {'H': (aev_dim, 160, 128, 96),
                          'C': (aev_dim, 144, 112, 96),
                          'N': (aev_dim, 128, 112, 96),
                          'O': (aev_dim, 128, 112, 96)}
        return self.standard(dims_for_atoms[atom], **kwargs)

    def like_2x(self, atom: str = 'H', aev_dim=1008, **kwargs):
        r"""Makes a sequential atomic network like the one used in the ANI-2x model"""
        dims_for_atoms = {'H': (aev_dim, 256, 192, 160),
                          'C': (aev_dim, 224, 192, 160),
                          'N': (aev_dim, 192, 160, 128),
                          'O': (aev_dim, 192, 160, 128),
                          'S': (aev_dim, 160, 128, 96),
                          'F': (aev_dim, 160, 128, 96),
                          'Cl': (aev_dim, 160, 128, 96)}
        return self.standard(dims_for_atoms[atom], **kwargs)
    
    def setup_nets(self, aevsize):
        modules = []
        if self.netlike1x == True:
            for a in self.elements:
                network = self.like_1x(a, aev_dim = aevsize)
                modules.append(network)
        if self.netlike2x == True:
            for a in self.elements:
                network = self.like_2x(a, aev_dim = aevsize)
                modules.append(network)
        return modules

    def model_creator(self, aev_computer):
        modules = self.setup_nets(aev_computer.aev_length)
        if self.personal == True:
            sys.path.append(self.wkdir)
            from model import ANIModelAIM
            nn = ANIModelAIM(modules, aev_computer)
            print(ANIModelAIM)
            checkpoint = torch.load(self.checkpoint)
            nn.load_state_dict(checkpoint['model'],  strict=False)
            model = torch.nn.Sequential(nn).to(self.device)
        else:
            nn = torchani.ANIModel(modules)
            checkpoint = torch.load(self.checkpoint)
            nn.load_state_dict(checkpoint['model'],  strict=False)
            model = torchani.nn.Sequential(aev_computer, nn).to(self.device)
        return model, nn
    
    def model_builder(self):
        aev_computer = self.AEV_Computer()
        energy_shifter = self.Energy_Shifter()
        model, nn = self.model_creator(aev_computer)
        return aev_computer, energy_shifter, model, nn


"""
    def ta_sp_benchmark(data_path):
    #dataset = torchani.data.load(data_path,additional_properties=('forces','wb97x_dz.hirshfeld_charges',))
    mae_averager_energy = Averager()
    mae_averager_relative_energy = Averager()
    mae_averager_force = Averager()
    mae_averager_charges = Averager()
    rmse_averager_energy = Averager()
    rmse_averager_relative_energy = Averager()
    rmse_averager_force = Averager()
    rmse_averager_charges = Averager()
    for i in tqdm.tqdm(dataset, position=0, desc="dataset"):
        # read
        coordinates = torch.tensor(i['coordinates'], device=device, requires_grad=True)
        print(coordinates.shape)
        species = species_to_tensor(i['species'])
        print(species.shape)
        species = species_to_tensor(i['species']) \
                .unsqueeze(0).expand(coordinates.shape[0], -1)
        energies = torch.tensor(i['energies'], device=device)
        forces = torch.tensor(i['forces'], device=device)
        charges = torch.tensor(i['hirshfeld'], device=device)
        #compute
        _, predicted_energies, predicted_atomic_energies, predicted_charges, excess_charge, coulomb = model((species, coordinates))
        #print(predicted_charges)
        _, model_energies = energy_shifter((species, predicted_energies))
        model_forces, = torch.autograd.grad(model_energies.sum(), coordinates)
        ediff = energies - model_energies
        relative_ediff = relative_energies(energies) - \
            relative_energies(model_energies)
        fdiff = forces.flatten() - model_forces.flatten()
        cdiff = charges.flatten() - predicted_charges.flatten()
        # update
        mae_averager_energy.update(ediff.abs())
        mae_averager_relative_energy.update(relative_ediff.abs())
        mae_averager_force.update(fdiff.abs())
        mae_averager_charges.update(cdiff.abs())
        rmse_averager_energy.update(ediff ** 2)
        rmse_averager_relative_energy.update(relative_ediff ** 2)
        rmse_averager_force.update(fdiff ** 2)
        rmse_averager_charges.update(cdiff ** 2)
    mae_energy = hartree2kcalmol(mae_averager_energy.compute())
    rmse_energy = hartree2kcalmol(math.sqrt(rmse_averager_energy.compute()))
    mae_relative_energy = hartree2kcalmol(mae_averager_relative_energy.compute())
    rmse_relative_energy = hartree2kcalmol(math.sqrt(rmse_averager_relative_energy.compute()))
    mae_force = hartree2kcalmol(mae_averager_force.compute())
    rmse_force = hartree2kcalmol(math.sqrt(rmse_averager_force.compute()))
    mae_charges = mae_averager_charges.compute()
    rmse_charges= math.sqrt(rmse_averager_charges.compute())
    print("Energy:", mae_energy, rmse_energy)
    print("Relative Energy:", mae_relative_energy, rmse_relative_energy)
    print("Forces:", mae_force, rmse_force)
    print("Charges:", mae_charges, rmse_charges)
"""
