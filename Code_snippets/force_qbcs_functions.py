# My stuff for force_qbcs branch

class AtomicQBCs(NamedTuple):
    species: Tensor
    energies: Tensor
    ae_stdev: Tensor

class ForceQBCs(NamedTuple):
    species: Tensor
    energies: Tensor
    mean_force: Tensor
    stdev_force: Tensor

    def members_energies(self, species_coordinates: Tuple[Tensor, Tensor],
                         cell: Optional[Tensor] = None,
                         pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted energies of all member modules
        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled
        Returns:
            species_energies: species and members energies for the given configurations
                shape of energies is (M, C), where M is the number of modules in the ensemble.
        """
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        species, members_energies = self.atomic_energies(species_coordinates, cell=cell, pbc=pbc, average=False)
        return SpeciesEnergies(species, members_energies.sum(-1))

# Need to add these two classes, two functions to BuiltInModel, and copy the original members_energies to there as well
    def atomic_qbcs(self, species_coordinates: Tuple[Tensor, Tensor],
                    cell: Optional[Tensor] = None,
                    pbc: Optional[Tensor] = None,
                    average: bool = False, 
                    with_SAEs: bool = False,
                    unbiased: bool = True) -> AtomicQBCs:
        '''
        Largely does the same thing as the atomic_energies function, but with a different set of default inputs. 
        Returns standard deviation in atomic energy predictions across the ensemble. 
        '''
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        species_coordinates = self._maybe_convert_species(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        atomic_energies = self.neural_networks._atomic_energies(species_aevs)

        if atomic_energies.dim() == 2:
            atomic_energies = atomic_energies.unsqueeze(0)

        ae_stdev = atomic_energies.std(0, unbiased=unbiased)

        if average:
            atomic_energies = atomic_energies.mean(0)

        # Want to return with GSAEs, but that can wait
        if with_SAEs:
            atomic_energies += self.energy_shifter._atomic_saes(species_coordinates[0])
            #atomic_energies += self.energy_shifter.with_gsaes(species_coordinates[0], 'wb97x', '631gd')

        return AtomicQBCs(species_coordinates[0], atomic_energies, ae_stdev)

    def force_qbcs(self, species_coordinates: Tuple[Tensor, Tensor],
                   cell: Optional[Tensor] = None,
                   pbc: Optional[Tensor] = None,
                   average: bool = False) -> ForceQBCs:
        assert isinstance(self.neural_networks, Ensemble), "Your model doesn't have an ensemble of networks"
        species_coordinates[1].requires_grad=True
        #species_coordinates = self._maybe_convert_species(species_coordinates)     # This is only needed if periodic_table_index=False
        members_energies = self.members_energies(species_coordinates, cell, pbc).energies
        forces = []

        for energy in members_energies:
            derivative = torch.autograd.grad(energy,species_coordinates[1],retain_graph=True)[0]
            force = -derivative
            forces.append(force)
        forces = torch.cat(forces, dim=0)
        mean_force = forces.mean(0)
        stdev_force = forces.std(0)

        return ForceQBCs(species_coordinates[0], members_energies, mean_force, stdev_force)