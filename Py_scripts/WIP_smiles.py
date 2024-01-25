from torchani.io import tensor_to_xyz
def create_estimated_smiles(species, coordinates, canonicalize=True):
    r"""Creates estimated smiles, requires openbabel

    Canonicalization requires rdkit
    """
    temp = tempfile.NamedTemporaryFile(mode='r+', suffix='.xyz')
    tensor_to_xyz(temp.name, (species, coordinates), truncate_output_file=True)
    mol = next(pybel.readfile('xyz', temp.name))
    temp.close()
    smiles_raw = mol.write(format='smi')
    smiles = smiles_raw.split()[0].strip()
    if canonicalize:
        try:
            conformer = Chem.MolFromSmiles(smiles, sanitize=True)
            if conformer:
                smiles = Chem.MolToSmiles(conformer)
            else:
                # If conversion fails we don't sanitize, and we send the
                # and then we send the conformer through a series of
                # operations to try to get some meaningful output
                conformer = Chem.MolFromSmiles(smiles, sanitize=False)
                if conformer:
                    conformer.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(
                        conformer,
                        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                        | Chem.SanitizeFlags.SANITIZE_KEKULIZE
                        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                        catchErrors=True)
                # Two ifs are necessary since the previous process may have
                # invalidated the conformer
                if conformer:
                    smiles = Chem.MolToSmiles(conformer)
                else:
                    smiles = ''
        except Exception:
            smiles = ''
    return smiles
