import torch
import torchani
from pathlib import Path
from torchani.transforms import AtomicNumbersToIndices, SubtractSAE
from typing import Tuple, NamedTuple, Optional, Sequence
from torch.nn import Module
from copy import deepcopy
from models.nets import ANIModelAIM
import math
import torch.utils.tensorboard
import os
import shutil
from .loss import MTLLoss
import tqdm
import datetime

class personal_trainer:
    """
    Kate's Personal training class
    Keep track of species order in elements list, which is fed into setting up network architecture.
    """
    def __init__(self, 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            netlike1x: bool=False, 
            netlike2x: bool=False, 
            functional: str = 'wb97x', 
            basis_set: str = '631gd',
            forces : bool=False, 
            charges : bool=False, 
            typed_charges : bool=False,
            dipole : bool=False, 
            constants = None, 
            elements = None, 
            gsae_dat: str = None, 
            batch_size: int = 2048, 
            ds_path: str = None,
            h5_path: str = None, 
            include_properties: list=['energies', 'species', 'coordinates', 'forces'], 
            logdir: str = None, 
            projectlabel: str = None,
            train_file  = os.path.abspath(__file__), 
            now = None, 
            data_split = {'training': 0.8, 'validation': 0.2}, 
            activation: Optional[Module] = None,
            bias: bool = False,
            classifier_out: int = 1,
            num_tasks : int = 1, 
            personal: bool = True, 
            weight_decay: list = [6.1E-5, None, None, None], 
            lr_factor: int = 0.7, 
            lr_patience: int = 14, 
            lr_threshold: int = 0.0006, 
            max_epochs: int = 2000,
            early_stopping_learning_rate: int=1.0E-7, 
            restarting: bool = False):
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
        self.netlike1x = netlike1x
        self.netlike2x = netlike2x
        self.functional = functional
        self.basis_set = basis_set
        self.forces = forces
        self.charges = charges
        self.typed_charges = typed_charges
        self.dipole = dipole
        self.gsae_dat = gsae_dat
        self.batch_size = batch_size
        self.ds_path = ds_path
        self.h5_path = h5_path
        self.include_properties = include_properties
        self.logdir = logdir
        self.projectlabel = projectlabel
        self.now = now
        self.data_split = data_split
        self.activation = activation
        self.bias = bias
        self.classifier_out = classifier_out
        self.personal = personal
        self.weight_decay = weight_decay
        self.factor = lr_factor
        self.patience = lr_patience
        self.threshold = lr_threshold
        self.max_epochs = max_epochs
        self.earlylr = early_stopping_learning_rate
        self.restarting = restarting
        self.num_tasks = num_tasks
        self.train_file = train_file

    
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

    def datasets_loading(self, energy_shifter):
        # ds_path can either be a path or None
        # if it is a path, it can either exist or not
        # if it is None -> In memory
        # if it is an existing path -> use it
        # if it is a nonoe existing path -> create it, and then use it
        in_memory = self.ds_path is None
        transform = torchani.transforms.Compose([AtomicNumbersToIndices(self.elements), SubtractSAE(self.elements, energy_shifter)])
        if in_memory:
            learning_sets = torchani.datasets.create_batched_dataset(self.h5_path,
                                        include_properties=self.include_properties,
                                        batch_size=self.batch_size,
                                        inplace_transform=transform,
                                        shuffle_seed=123456789,
                                        splits=self.data_split, direct_cache=True)
            training = torch.utils.data.DataLoader(learning_sets['training'],
                                               shuffle=True,
                                               num_workers=1,
                                               prefetch_factor=2,
                                               pin_memory=True,
                                               batch_size=None)
            validation= torch.utils.data.DataLoader(learning_sets['validation'],
                                                 shuffle=False,
                                                 num_workers=1,
                                                 prefetch_factor=2, pin_memory=True, batch_size=None)
        else:
            if not Path(self.ds_path).resolve().is_dir():
                h5 = torchani.datasets.ANIDataset.from_dir(self.h5_path)
                torchani.datasets.create_batched_dataset(h5,
                                                 dest_path=self.ds_path,
                                                 batch_size=self.batch_size,
                                                 include_properties=self.include_properties,
                                                 splits = self.data_split) 
            # This below loads the data if dspath exists
            training = torchani.datasets.ANIBatchedDataset(self.ds_path, transform=transform, split='training')
            validation = torchani.datasets.ANIBatchedDataset(self.ds_path, transform=transform, split='validation')
            training = torch.utils.data.DataLoader(training,
                                           shuffle=True,
                                           num_workers=1,
                                           prefetch_factor=2,
                                           pin_memory=True,
                                           batch_size=None)
            validation = torch.utils.data.DataLoader(validation,
                                             shuffle=False,
                                             num_workers=1,
                                             prefetch_factor=2,
                                             pin_memory=True,
                                             batch_size=None)
        return training, validation
   
    def standard(self, dims: Sequence[int]):
        r"""Makes a standard ANI style atomic network"""
        if self.activation is None:
            activation = torch.nn.GELU()
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
        if self.netlike1x:
            for a in self.elements:
                network = self.like_1x(a, aev_dim = aevsize)
                modules.append(network)
        if self.netlike2x:
            for a in self.elements:
                network = self.like_2x(a, aev_dim = aevsize)
                modules.append(network)
        return modules

    def init_params(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, a=1.0)
            if self.bias:
                torch.nn.init.zeros_(m.bias)
    
    def model_creator(self, aev_computer):
        modules = self.setup_nets(aev_computer.aev_length)
        if self.personal == True:
            from models.nets import ANIModelAIM
            nn = ANIModelAIM(modules, aev_computer)
            nn.apply(self.init_params)
            model = nn.to(self.device)
        else:
            nn = torchani.ANIModel(modules)
            nn.apply(self.init_params)
            model = torchani.nn.Sequential(aev_computer, nn).to(self.device)
        return nn, model, modules
    
    def AdamWOpt_build(self, modules, weight_decay):
        params = []
        for mod in modules:
            for i in range(4):
                if weight_decay[i]:
                    params.append({'params': [mod[i+i].weight], 'weight_decay': weight_decay[i]})
                else:
                    params.append({'params': [mod[i+i].weight]})
        AdamW = torch.optim.AdamW(params)
        return AdamW
    
    def LR_Plat_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=self.factor, patience= self.patience, threshold=self.threshold)
        return scheduler

    def pt_setup(self):
        #setup directories so you have one for each model 'date_project/model0/best and latest and tensorboard 
        date = self.now.strftime("%Y%m%d_%H%M")
        log = '{}{}_{}'.format(self.logdir, date, self.projectlabel)
        assert os.path.isdir(log)==False, "Oops! This project sub-directory already exists."
        if not os.path.isdir(log):
            print("creating your log sub-directory")
            os.makedirs(log)
        training_writer = torch.utils.tensorboard.SummaryWriter(log_dir=log + '/train')
        latest_checkpoint = '{}/latest.pt'.format(log)
        best_checkpoint = '{}/best.pt'.format(log)
        shutil.copy(self.train_file, '{}/trainer.py'.format(log))
        if self.personal == True:
            shutil.copy('models/nets.py', '{}/model.py'.format(log))
        return log, training_writer, latest_checkpoint, best_checkpoint

    def save_model(self, nn, optimizer, energy_shifter, checkpoint, lr_scheduler):
        torch.save({
            'model': nn.state_dict(),
            'AdamW': optimizer.state_dict(),
            'self_energies': energy_shifter, 
            'AdamW_scheduler':  lr_scheduler
            }, checkpoint)
    
    def restart_train(self, latest_checkpoint, nn, optimizer, lr_scheduler):
        if os.path.isfile(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint)
            nn.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['AdamW'])
            lr_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])

        
    def eA2debeye(x):
        return x/0.20819434
        
    def validate(self, validation, model):
        valdict = {}
        mse_sum = torch.nn.MSELoss(reduction='sum')
        mse = torch.nn.MSELoss(reduction='none')
        total_energy_mse = 0.0
        count = 0 
        if self.charges == True:
            total_charge_mse = 0.0
            total_excess_mse = 0.0
        if self.forces == True:
            total_force_mse = 0.0
        if self.dipole == True:
            total_dipole_mse = 0.0
        if self.typed_charges == True:
            type_charge_mse = 0.0
        for properties in validation:
            species = properties['species'].to(self.device)
            coordinates = properties['coordinates'].to(self.device).float().requires_grad_(True)
            true_energies = properties['energies'].to(self.device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            if self.typed_charges == True:
                true_charges = properties['mbis_charges'].to(self.device).float()
            if self.personal == True:
                if self.typed_charges == True:
                    _, predicted_energies, predicted_atomic_energies, predicted_charges, excess_charge, coulomb, correction = model((species, coordinates))
            else:
                if self.dipole == True:
                    raise TypeError ('Published ANI does not currently support dipoles.')
                if self.charges == True:
                    raise TypeError ('Published ANI does not currently support charge prediction.')
                _, predicted_energies = model((species, coordinates))
            count += predicted_energies.shape[0]
            total_energy_mse += mse_sum(predicted_energies, true_energies).item()
            if self.typed_charges == True:
                type_charge_mse += mse_sum(predicted_charges.sum(dim=1), true_charges.sum(dim=1)).item() 
        energy_rmse = torchani.units.hartree2kcalmol(math.sqrt(total_energy_mse / count))
        valdict['energy_rmse']=energy_rmse
        if self.forces == True:
            force_rmse = torchani.units.hartree2kcalmol(math.sqrt(total_force_mse / count))
            valdict['force_rmse']=force_rmse
        if self.dipole == True:
            dipole_rmse = self.eA2debeye(math.sqrt(total_dipole_mse / count))
            valdict['dipole_rmse']=dipole_rmse
        if self.charges == True:
            charge_rmse = math.sqrt(total_charge_mse / count)
            valdict['charge_rmse'] = charge_rmse
        if self.typed_charges == True: 
            type_charge_rmse = math.sqrt(type_charge_mse / count)
            valdict['typed_charge'] = type_charge_rmse
        return valdict

    def trainer(self):
        """
        for i in range(3):
            manual seed
            #looking at checkpoint setup function
        """
        aev_computer = self.AEV_Computer()
        energy_shifter = self.Energy_Shifter()
        training, validation = self.datasets_loading(energy_shifter)
        nn, model, modules = self.model_creator(aev_computer)
        AdamW = self.AdamWOpt_build(modules, self.weight_decay)
        LRscheduler = self.LR_Plat_scheduler(AdamW)
        logdir, training_writer, latest_pt, best_pt = self.pt_setup()
        shutil.copyfile('/data/khuddzu/personal_trainer/personal_trainer/protocol.py', '{}/protocol.py'.format(logdir))
        if self.num_tasks > 1:
            mtl = MTLLoss(num_tasks=self.num_tasks).to(self.device)
            AdamW.param_groups[0]['params'].append(mtl.log_sigma)  #avoids LRdecay problem
        best = 1e3
        mse = torch.nn.MSELoss(reduction='none')
        print("training starting from epoch", LRscheduler.last_epoch + 1)
        for _ in range(LRscheduler.last_epoch + 1, self.max_epochs):
            valrmse = self.validate(validation, model)
            for k, v in valrmse.items():
                training_writer.add_scalar(k, v, LRscheduler.last_epoch)
            learning_rate = AdamW.param_groups[0]['lr']
            if learning_rate < self.earlylr:
                break
            
            #best checkpoint
            if valrmse['energy_rmse'] < best: 
            #if LRscheduler.is_better(valrmse['energy_rmse'], LRscheduler.best):
                print('Saving the model, epoch={}, RMSE = {}'.format((LRscheduler.last_epoch + 1), valrmse['energy_rmse']))
                self.save_model(nn, AdamW, energy_shifter, best_pt, LRscheduler)
                for k, v in valrmse.items():
                    training_writer.add_scalar('best_{}'.format(k), v, LRscheduler.last_epoch)
                best = valrmse['energy_rmse']
            LRscheduler.step(valrmse['energy_rmse'])
            for i, properties in tqdm.tqdm(
                enumerate(training),
                total=len(training),
                desc="epoch {}".format(LRscheduler.last_epoch)
            ):
                ##Get Properties##
                species = properties['species'].to(self.device)
                coordinates = properties['coordinates'].to(self.device).float().requires_grad_(True)
                true_energies = properties['energies'].to(self.device).float()
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                if self.forces == True:
                    true_forces = properties['forces'].to(self.device).float()
                if self.dipole == True:
                    true_dipoles = properties['dipole'].to(self.device).float()
                ## Compute predicted ##
                if self.personal == True:
                    if self.dipole == True:
                        _, predicted_energies, predicted_atomic_energies, predicted_charges, excess_charge, coulomb, predicted_dipole = model((species, coordinates))
                    if self.charges == True:
                        initial_charges = properties['am1bcc_charges'].to(self.device)
                        _, predicted_energies, predicted_atomic_energies, predicted_charges, init_charge, excess_charge, coulomb = model((species, coordinates), initial_charges)
                    if self.typed_charges == True:
                        true_charges = properties['mbis_charges'].to(self.device)
                        _, predicted_energies, predicted_atomic_energies, predicted_charges, excess_charge, coulomb, correction = model((species, coordinates))
                    else:
                        _, predicted_energies, predicted_atomic_energies, predicted_charges, excess_charge, coulomb  = model((species, coordinates))
                else:
                    if self.dipole == True:
                        raise TypeError ('Published ANI does not currently support dipoles.')
                    if self.charges == True:
                        raise TypeError ('Published ANI does not currently support charge prediction.')
                    _, predicted_energies = model((species, coordinates))
                if self.forces == True:
                    forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                ##Get loss##
                energy_loss = (mse(predicted_energies, true_energies) /num_atoms.sqrt()).mean()
                if self.typed_charges == True: 
                    charge_loss = (mse(predicted_charges,true_charges).sum(dim=1)/num_atoms).mean()
                if self.charges == True:
                    total_charge_loss = torch.sum((((predicted_charges-init_charge)**2).sum(dim=1))/num_atoms).mean() 
                if self.forces == True:
                    force_loss = (mse(true_forces, forces).sum(dim=(1, 2)) / (3.0 * num_atoms)).mean()
                if self.dipole == True:
                    dipole_loss = (torch.sum((mse(predicted_dipoles, true_dipoles))/3.0, dim=1) / num_atoms.sqrt()).mean()
                ####FIX THIS#####
                if self.forces == True and self.dipole == True:
                    loss = mtl(energy_loss, force_loss, dipole_loss)
                elif self.forces == True:
                    loss = mtl(energy_loss, force_loss)
                elif self.dipole == True:
                    loss = mtl(energy_loss, dipole_loss)
                elif self.charges == True:
                    loss = energy_loss
                    #loss = energy_loss + ((1)*total_charge_loss)
                elif self.typed_charges ==True:
                    print('EL:', energy_loss)
                    print('QL:',(1/300)*charge_loss)
                    
                    loss = energy_loss + (1/300)*charge_loss
                    print('Total:', loss)
                else:
                    loss = energy_loss
                ##BackProp##
                AdamW.zero_grad()
                loss.backward()
                AdamW.step()
                training_writer.add_scalar('batch_loss', loss, LRscheduler.last_epoch * len(training) + i)
                training_writer.add_scalar('learning_rate', learning_rate, LRscheduler.last_epoch)
            self.save_model(nn, AdamW, energy_shifter, latest_pt, LRscheduler)



       
