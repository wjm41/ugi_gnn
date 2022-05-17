from fire import Fire

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

from dock2hit.library_generation.decompose_ugi import decompose_ugi_molecule_into_components
from dock2hit.library_generation.enumerate_ugi import generate_ugi_library


def generate_ugi_library_from_primary_amines(output_file):
    """
    Generate a library of UGI molecules from a SMILES file.

    :param smiles_file: The path to the SMILES file.
    :param output_file: The path to the output file.
    :return: None
    """

    best_ugi_mol = 'CC(C)Oc1ccc(cc1)N(C(C(=O)NCCc1ccns1)c1cccnc1)C(=O)c1cocn1'
    components = decompose_ugi_molecule_into_components(best_ugi_mol)
    components_as_mols = [Chem.MolFromSmiles(x) for x in components]

    best_acid = [components_as_mols[0]]
    best_amine = [components_as_mols[1]]
    best_aldehyde = [components_as_mols[2]]
    best_isocyanide = [components_as_mols[3]]

    enamine_dir = '/rds-d2/user/wjm41/hpc-work/datasets/Ugis/datasets/enamine_library_generation/'
    primary_sdf = enamine_dir + 'Enamine_Primary_Amines_37221cmpds_20220404.sdf'
    secondary_sdf = enamine_dir + 'Enamine_Secondary_Amines_25592cmpds_20220404.sdf'

    df_amines = PandasTools.LoadSDF(
        primary_sdf, idName='rdkit_ID', smilesName='SMILES', molColName='mol')[['ID', 'SMILES', 'mol']]
    # df_secondary = PandasTools.LoadSDF(
    #     secondary_sdf, smilesName='SMILES', molColName='mol')[['ID', 'SMILES', 'mol']]
    # df_amines = pd.concat([df_amines, df_secondary])

    amine_list = df_amines.mol.values
    library_enumerated_amines = generate_ugi_library(
        best_acid, amine_list, best_aldehyde, best_isocyanide)
    df_amines['ugi'] = library_enumerated_amines
    df_amines.dropna().drop(columns=['mol']).to_csv(output_file, index=False)
    print(
        f'Ugi library generated, original length {len(amine_list)}, final length: {len(df_amines)}')
    # f = open(output_file, "a")
    # f.writelines(library_enumerated_amines)
    # f.close()


if __name__ == "__main__":
    Fire(generate_ugi_library_from_primary_amines)
