
from pathlib import Path
from joblib import Parallel, delayed
import logging
from typing import Union

from fire import Fire
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import PandasTools
from rdkit.Chem import rdChemReactions
from mpi4py import MPI
from useful_rdkit_utils import add_molecule_and_errors

from dock2hit.library_generation.decompose_ugi import decompose_ugi_molecule_into_components
from dock2hit.library_generation.enumerate_ugi import generate_ugi_library


def replace_isocyanide() -> rdChemReactions.ChemicalReaction:

    best_mol_substruct = '[NX3]([*:1])(C([C:2](=O)[NX3;H1])c1cccnc1)C(=O)c1cocn1'
    amine_smarts = '[NX3;!$(N*=*):3]([H:4])[*:5]'

    new_ugi_smarts = '[NX3]([*:1])(C([C:2](=O)[N:3][*:5])c1cccnc1)C(=O)c1cocn1'
    replace_isocyanide_smarts = f'{best_mol_substruct}.{amine_smarts}>>{new_ugi_smarts}'
    replacement_rxn = rdChemReactions.ReactionFromSmarts(
        replace_isocyanide_smarts)

    return replacement_rxn


def replace_isocyanide_for_single_ugi(input_ugi_mol: Union[str, Mol],
                                      df_amine: pd.DataFrame,
                                      replacement_rxn: rdChemReactions.ChemicalReaction = None) -> pd.DataFrame:

    if replacement_rxn is None:
        replacement_rxn = replace_isocyanide()

    if isinstance(input_ugi_mol, str):
        input_ugi_mol = Chem.MolFromSmiles(input_ugi_mol)

    df_this_ugi = df_amine.copy()

    input_ugi_mol = Chem.AddHs(input_ugi_mol)
    for index, amine_row in df_this_ugi.iterrows():

        reactants = tuple([input_ugi_mol, Chem.AddHs(amine_row.mol)])

        possible_products = replacement_rxn.RunReactants(reactants)
        if len(possible_products) == 0:
            logging.warning(
                f'No product produced for amine:\n{amine_row["ID_x"]}')
            df_this_ugi.loc[index, 'ugi'] = None
        else:
            product = Chem.RemoveHs(possible_products[0][0])
            df_this_ugi.loc[index, 'ugi'] = Chem.MolToSmiles(product)

    return df_this_ugi


def load_enamine_amines(test: bool = False,):
    enamine_dir = '/rds-d2/user/wjm41/hpc-work/datasets/Ugis/datasets/enamine_library_generation/'
    primary_sdf = enamine_dir + 'Enamine_Primary_Amines_37221cmpds_20220404.sdf'
    secondary_sdf = enamine_dir + 'Enamine_Secondary_Amines_25592cmpds_20220404.sdf'

    df_amines = PandasTools.LoadSDF(
        primary_sdf, idName='rdkit_ID', smilesName='SMILES', molColName='mol')[['ID', 'SMILES', 'mol']]

    if test:
        df_amines = df_amines.head()
    else:
        df_secondary = PandasTools.LoadSDF(
            secondary_sdf, idName='rdkit_ID', smilesName='SMILES', molColName='mol')[['ID', 'SMILES', 'mol']]
        df_amines = pd.concat([df_amines, df_secondary]).reset_index(drop=True)
    return df_amines


def slice_indices_according_to_rank_and_size(my_rank: int, mpi_size: int, object_to_slice: int):

    if not isinstance(object_to_slice, int):
        length_of_object_to_slice = len(object_to_slice)
    else:
        length_of_object_to_slice = object_to_slice

    mpi_borders = np.linspace(
        0, length_of_object_to_slice, mpi_size + 1).astype('int')

    border_low = mpi_borders[my_rank]
    border_high = mpi_borders[my_rank+1]
    return border_low, border_high


def enumerate_isocyanides_for_library_of_ugis(input_df: str,
                                              test: bool = False,):

    replacement_rxn = replace_isocyanide()

    df_amines = load_enamine_amines(test)
    df_amines = df_amines.rename(
        columns={'ID': 'ID_x', 'SMILES': 'SMILES_x'})

    df_new_lib = []
    for index, ugi_row in tqdm(input_df.iterrows(), total=len(input_df)):

        df_this_ugi = replace_isocyanide_for_single_ugi(
            ugi_row['mol'], df_amines, replacement_rxn)
        df_this_ugi['ID_y'] = ugi_row['ID']
        df_this_ugi['SMILES_y'] = ugi_row['SMILES']
        df_new_lib.append(df_this_ugi)

    df_new_lib = pd.concat(df_new_lib).dropna().drop(columns=['mol'])
    print(
        f'Ugi library generated, original length {len(df_amines)}, final length: {len(df_new_lib)}')
    return df_new_lib


def return_results(row, df_amines, rxn):
    df_this_ugi = replace_isocyanide_for_single_ugi(
        row['mol'], df_amines, rxn)
    df_this_ugi['ID_y'] = row['ID']
    df_this_ugi['SMILES_y'] = row['SMILES']
    return df_this_ugi


def joblib_enumeration(input_smiles_file: str,
                       output_file: str,
                       n_jobs: int = 2,
                       test: bool = False):

    df_ugi = pd.read_csv(input_smiles_file)

    if test:
        df_ugi = df_ugi.head()

    add_molecule_and_errors(df_ugi, smiles_col='ugi', mol_col_name='mol')

    replacement_rxn = replace_isocyanide()
    enamine_dir = '/rds-d2/user/wjm41/hpc-work/datasets/Ugis/datasets/enamine_library_generation/'
    primary_sdf = enamine_dir + 'Enamine_Primary_Amines_37221cmpds_20220404.sdf'
    secondary_sdf = enamine_dir + 'Enamine_Secondary_Amines_25592cmpds_20220404.sdf'

    df_amines = PandasTools.LoadSDF(
        primary_sdf, idName='rdkit_ID', smilesName='SMILES', molColName='mol')[['ID', 'SMILES', 'mol']]

    df_secondary = PandasTools.LoadSDF(
        secondary_sdf, idName='rdkit_ID', smilesName='SMILES', molColName='mol')[['ID', 'SMILES', 'mol']]
    df_amines = pd.concat([df_amines, df_secondary])

    df_amines = df_amines.rename(
        columns={'ID': 'ID_x', 'SMILES': 'SMILES_x'})

    print('Beginning joblib implementation:')
    df_new_lib = Parallel(n_jobs=n_jobs)(
        delayed(return_results)(ugi_row, df_amines, replacement_rxn) for index, ugi_row in tqdm(df_ugi.iterrows(), total=len(df_ugi)))
    df_new_lib = pd.concat(df_new_lib).dropna().drop(columns=['mol'])

    print('Beginning simple serial implementation:')
    df_new_lib = enumerate_isocyanides_for_library_of_ugis(df_ugi)
    print(
        f'Ugi library generated, original length {len(df_amines)}, final length: {len(df_new_lib)}')

    df_new_lib.to_csv(output_file, index=False)

    return


def mpi_enumeration(input_smiles_file: str,
                    output_file: str,
                    test: bool = False,
                    gather: bool = False):

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    df_ugi = pd.read_csv(input_smiles_file)

    if test:
        df_ugi = df_ugi.head()
    add_molecule_and_errors(df_ugi, smiles_col='ugi', mol_col_name='mol')

    my_start_index, my_end_index = slice_indices_according_to_rank_and_size(
        mpi_rank, mpi_size, len(df_ugi))

    my_ugis = df_ugi.iloc[my_start_index:my_end_index]
    my_results = enumerate_isocyanides_for_library_of_ugis(
        my_ugis, test)

    if gather:
        mpi_comm.Barrier()

        df_new_lib = mpi_comm.gather(my_results, root=0)

        if mpi_rank == 0:
            pd.concat(df_new_lib).to_csv(output_file, index=False)
    else:
        output_dir = '/'.join(output_file.split('/')[:-1])
        my_output_dir = f'{output_dir}/mpi_tmp/'
        Path(my_output_dir).mkdir(parents=True, exist_ok=True)
        file_name = output_file.split('/')[-1]
        my_file_name = file_name.replace('.csv', f'_{mpi_rank}.csv')
        my_output_file = f"{my_output_dir}/{my_file_name}"
        my_results.to_csv(my_output_file, index=False)
    return


def enumeration_with_best_mol(output_file: str,
                              test: bool = False):
    best_ugi_mol = 'CC(C)Oc1ccc(cc1)N(C(C(=O)NCCc1ccns1)c1cccnc1)C(=O)c1cocn1'
    df_amines = load_enamine_amines(test)

    replacement_rxn = replace_isocyanide()

    df_isocyanide = replace_isocyanide_for_single_ugi(
        Chem.MolFromSmiles(best_ugi_mol), df_amines, replacement_rxn)

    df_isocyanide.drop(columns='mol').to_csv(output_file, index=False)


if __name__ == '__main__':
    Fire(enumeration_with_best_mol)
