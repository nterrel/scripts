from Bio import PDB

def get_unique_elements(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    elements = set()

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_name = atom.get_name()
                    element = ''.join(filter(str.isalpha, atom_name))  # Extract letters only
                    elements.add(element)

    return sorted(list(elements))

if __name__ == "__main__":
    pdb_file = "AF-A0A562SP41_fixer.pdb"
    unique_elements = get_unique_elements(pdb_file)

    print("Unique Elements:")
    for element in unique_elements:
        print(element)
