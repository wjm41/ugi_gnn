import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # removes annoying RDKit warnings


def decompose_ugi_molecule_into_components(smiles_or_mol):

    # ugi into acid, amine, aldehyde, isocyanide
    decomp_smarts = '[*:1][C](=[O])[N]([*:2])[C]([*:3])([*:4])[C](=[O])[N]([#1])[*:5]>>[#6:1][C](=[O])[O][#1].[#6:2][N]([#1])[#1].[#6:3][C](=[O])[#1:4].[#6:5][N+]#[C-]'
    decomp_rxn = rdChemReactions.ReactionFromSmarts(decomp_smarts)

    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    elif isinstance(smiles_or_mol, Chem.rdchem.Mol):
        mol = smiles_or_mol
    else:
        raise TypeError(
            'smiles_or_mol must be a string or an rdkit.Chem.rdchem.Mol object.')

    reactant = (Chem.AddHs(mol),)

    parts = decomp_rxn.RunReactants(reactant)
    components = []
    for part in parts:
        for j in part:
            try:
                x = Chem.MolToSmiles(Chem.RemoveHs(j))
                if x not in components:
                    components.append(x)
            except:
                continue

    if len(components) != 4:
        return None

    else:
        return components
