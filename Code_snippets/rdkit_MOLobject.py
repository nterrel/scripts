def create_RdMOL_object(species, coordinates, canonicalize=True, rm_H=False):
    r"""Creates estimated smiles, requires openbabel
    
    Canonicalization requires rdkit
    """
    temp = tempfile.NamedTemporaryFile(mode='r+', suffix='.xyz')
    tensor_to_xyz(temp.name, (species, coordinates), truncate_output_file=True)
    mol = next(pybel.readfile('xyz', temp.name))
    temp.close()
    mol_raw = mol.write(format='mol')
    if canonicalize:
        try:
            conformer = Chem.MolFromMolBlock(mol_raw, sanitize=True, removeHs=rm_H)
            if conformer:
                rdmol = conformer
            else:
                # If conversion fails we don't sanitize, and we send the
                # and then we send the conformer through a series of
                # operations to try to get some meaningful output
                conformer = Chem.MolFromMolBlock(mol_raw, sanitize=False, removeHs=rm_H)
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
                    rdmol = Chem.MolFromMolBlock(mol_raw, removeHs=rm_H)
                else:
                    rdmol = None
        except Exception:
            rdmol = None
    return rdmol