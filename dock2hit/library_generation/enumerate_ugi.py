from typing import List
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # removes annoying RDKit warnings


def create_ugi_from_component(acid: Chem.rdchem.Mol,
                              amine: Chem.rdchem.Mol,
                              aldehyde: Chem.rdchem.Mol,
                              isocyanide: Chem.rdchem.Mol) -> Chem.rdchem.Mol:

    # acid, amine, aldehyde, isocyanide into ugi
    generate_smarts = '[#6:1][C](=[O])[O][#1].[#6:2][N]([#1])[#1].[#6:3][C](=[O])[#1:4].[#6:5][N+]#[C-]>>[*:1][C](=[O])[N]([*:2])[C]([*:3])([*:4])[C](=[O])[N]([#1])[*:5]'
    ugi_rxn = rdChemReactions.ReactionFromSmarts(generate_smarts)

    reactants = [acid, amine, aldehyde, isocyanide]

    reactants = tuple([Chem.AddHs(reactant) for reactant in reactants])

    ugi_mol = ugi_rxn.RunReactants(reactants)[0]
    ugi_smiles = Chem.MolToSmiles(Chem.RemoveHs(ugi_mol))
    return ugi_smiles


def generate_ugi_library(list_of_acids,
                         list_of_amines,
                         list_of_aldehydes,
                         list_of_isocyanides) -> List:

    ugi_library = []
    # loop over all lists
    for aci in list_of_acids:
        for ami in list_of_amines:
            for al in list_of_aldehydes:
                for iso in list_of_isocyanides:
                    ugi_library.append(
                        create_ugi_from_component(aci, ami, al, iso))

    return ugi_library
