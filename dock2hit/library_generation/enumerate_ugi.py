import logging
from typing import List
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # removes annoying RDKit warnings


def ugi_generation_rxn() -> rdChemReactions.ChemicalReaction:

    acid_smarts = '[#6:1][C](=[O])[O][#1]'
    # amine_smarts = '[#6:2][NX3;H2,H1;!$(NC=O);!$(NS(=O)=O)]'
    amine_smarts = '[#6:2][NX3;H2,H1;!$(N*=*)]'
    aldehyde_smarts = '[#6:3][C](=[O])[#1:4]'
    isocyanide_smarts = '[#6:5][N+]#[C-]'
    ugi_smarts = '[*:1][C](=[O])[N]([*:2])[C]([*:3])([*:4])[C](=[O])[N]([#1])[*:5]'

    ugi_generation_smarts = f'{acid_smarts}.{amine_smarts}.{aldehyde_smarts}.{isocyanide_smarts}>>{ugi_smarts}'
    ugi_rxn = rdChemReactions.ReactionFromSmarts(ugi_generation_smarts)

    return ugi_rxn


def create_ugi_from_component(acid: Chem.rdchem.Mol,
                              amine: Chem.rdchem.Mol,
                              aldehyde: Chem.rdchem.Mol,
                              isocyanide: Chem.rdchem.Mol,
                              ugi_rxn: rdChemReactions.ChemicalReaction = None) -> Chem.rdchem.Mol:

    if ugi_rxn is None:
        ugi_rxn = ugi_generation_rxn()

    reactants = [acid, amine, aldehyde, isocyanide]

    reactants = tuple([Chem.AddHs(reactant) for reactant in reactants])

    ugi_products = ugi_rxn.RunReactants(reactants)
    if len(ugi_products) > 1:
        product_smiles = []
        for product in ugi_products:
            product_smiles.append(Chem.MolToSmiles(Chem.RemoveHs(product[0])))
        assert len(set(product_smiles)
                   ) == 1, f'Ugi generation produced more than one product:\n{product_smiles}\nwhen using amine:\n{Chem.MolToSmiles(amine)}\n'
    elif len(ugi_products) == 0:
        logging.warning(
            f'No product produced for amine:\n{Chem.MolToSmiles(amine)}')
        return None
        # raise ValueError(
        #     f'Ugi generation produced no products with amine:\n{Chem.MolToSmiles(amine)}')
    ugi_mol = ugi_products[0][0]
    ugi_smiles = Chem.MolToSmiles(Chem.RemoveHs(ugi_mol))
    return ugi_smiles


def generate_ugi_library(list_of_acids: List,
                         list_of_amines,
                         list_of_aldehydes,
                         list_of_isocyanides) -> List:
    ugi_rxn = ugi_generation_rxn()
    ugi_library = []
    total_len = len(list_of_acids) * len(list_of_amines) * \
        len(list_of_aldehydes) * len(list_of_isocyanides)
    with tqdm(total=total_len) as progress_bar:
        # loop over all lists
        for aci in list_of_acids:
            for ami in list_of_amines:
                for al in list_of_aldehydes:
                    for iso in list_of_isocyanides:
                        created_ugi = create_ugi_from_component(
                            aci, ami, al, iso, ugi_rxn=ugi_rxn)
                        ugi_library.append(created_ugi)

                        progress_bar.update(1)

    return ugi_library
