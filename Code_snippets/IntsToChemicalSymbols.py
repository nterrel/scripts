import torch


class IntsToChemicalSymbols(torch.nn.Module):
    r"""Helper that can be called to convert tensor or list of integers to list of chemical symbol strings

    On initialization the class should be supplied with either a list or torch.Tensor.
    Returned is a list of corresponding chemical symbol strings, of equal dimension to the input.

    Usage example:

        #species list used for indexing
        elements = ['H','C','N','O','S','F', 'Cl']

        # A 1-D Tenosr, however the dimension can be larger
        species = torch.Tensor([3, 0, 0, -1, -1, -1])

        IntsToChemicalSymbols(elements, species)

        Output:
            ['O', 'H', 'H']

    Arguments:
        elements: list of species in your model, used for indexing
        species: list or tensor of species integer values you wish to convert

    """
    # _dummy: Tensor
    rev_species: Dict[int, str]

    def __init__(self, all_species: Sequence[str]):
        super().__init__()
        self.rev_species = {i: s for i, s in enumerate(all_species)}
        # dummy tensor to hold output device
        # self.register_buffer('_dummy', torch.empty(0), persistent=False)

    def forward(self, species) -> Tensor:
        r"""Convert species from list or Tensor of integers to list of strings of equal dimension"""
        if torch.is_tensor(species):
            species = species.detach().numpy()
        if len(species.shape) > 1:
            rev = [[self.rev_species[x] for x in l if x != -1] for l in species]
        else:
            rev = [self.rev_species[x] for x in species if x != -1]
        return rev

    def __len__(self):
        return len(self.rev_species)
