import numpy as np
import pandas as pd

from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import MolFromSmiles, MolToSmiles

from useful_rdkit_utils import add_molecule_and_errors, mol2numpy_fp
from dock2hit.generate_mpnn_fps import generate_mpnn_fps_from_dataframe


def canon_tautomers(smiles_list):
    enumerator = rdMolStandardize.TautomerEnumerator()

    smiles_list = [MolToSmiles(enumerator.Canonicalize(
        MolFromSmiles(smi))) for smi in smiles_list]

    return smiles_list


def read_and_process_dataframe(path_to_dataframe, canonicalize_tautomers=True, IC50_threshold=None, pIC50_threshold=None):
    df = pd.read_csv(path_to_dataframe)

    if canonicalize_tautomers:
        df['SMILES'] = canon_tautomers(df['SMILES'])

    # assign IC50 of 100 to inactives
    df.loc[df['IC50'].astype(str).str.contains('>'), 'IC50'] = 100.0
    df.loc[df['IC50'].isnull(), 'IC50'] = 100.0

    df['IC50'] = df['IC50'].astype(float)
    df['pIC50'] = -np.log10(df['IC50'].to_numpy()) + 6

    indicies_of_actives = df[df['IC50'] != 100.0].index.to_numpy()

    if IC50_threshold is not None:
        df = df.query('IC50 < @IC50_threshold')
    if pIC50_threshold is not None:
        df = df.query('pIC50 >= @pIC50_threshold')

    pIC50_values = df['pIC50'].to_numpy()

    return df, pIC50_values, indicies_of_actives


def read_and_process_ugi_data():
    moonshot_data_dir = '/rds-d2/user/wjm41/hpc-work/datasets/Ugis/datasets/moonshot_evaluation_sets/'
    ugi_split_names = ['combined_acid.csv', 'combined_aldehyde.csv',
                       'combined_amine.csv', 'combined_isocyanide.csv']
    file_names = [f'{moonshot_data_dir}{file}' for file in ugi_split_names]

    new_activity_file = '/rds-d2/user/wjm41/hpc-work/datasets/Ugis/datasets/new_ugi_activity_2022_04_03.csv'
    file_names.append(new_activity_file)

    list_of_dataframes = []
    list_of_pIC50_values = []
    list_of_active_indices = []

    for file in file_names:

        df_for_this_file, y_train, indicies_of_actives = read_and_process_dataframe(
            file)

        list_of_dataframes.append(df_for_this_file)
        list_of_pIC50_values.append(y_train)
        list_of_active_indices.append(indicies_of_actives)

    taut_model = '/rds-d2/user/wjm41/hpc-work/models/dock2hit/ugi/ugi_taut/model_mol9039221.ckpt'

    names = ['acid', 'aldehyde', 'amine', 'isocyanide', 'new']
    for name, df in zip(names, list_of_dataframes):
        add_molecule_and_errors(df, mol_col_name='mol')
        df['morgan_fp'] = df['mol'].apply(mol2numpy_fp)
        df['mpnn_fp'] = [x for x in generate_mpnn_fps_from_dataframe(
            df, load_name=taut_model)]
        df['data_source'] = name

    df_all = pd.concat(list_of_dataframes).reset_index()
    df_all['IC50'] = df_all['IC50'].astype(float)
    return df_all
