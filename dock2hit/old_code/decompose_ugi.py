import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # removes annoying RDKit warnings

df = pd.read_csv('ugis_processed_new.csv')

ugi_decomp = '[*:1][C](=[O])[N]([*:2])[C]([*:3])([*:4])[C](=[O])[N]([#1])[*:5]>>[#6:1][C](=[O])[O][#1].[#6:2][N]([#1])[#1].[#6:3][C](=[O])[#1:4].[#6:5][N+]#[C-]'
rxn = rdChemReactions.ReactionFromSmarts(ugi_decomp)


patts = ['NCC(=O)@[Nr5](!@C)@C',
         'C(=O)@[Nr5]',
         '[CR](=O)@[NR]CC(=O)N',
         'C(=O)NC[CR](=O)[NR]'
        ]

patts = [Chem.MolFromSmarts(x) for x in patts]

#def decompose_ugi(row):
#    row['mask'] = True
#    mol = Chem.MolFromSmiles(row['smiles'])
#    
#    reactant = (Chem.AddHs(mol),)
#
#    parts = rxn.RunReactants(reactant)
#    c = []
#    for part in parts:
#        for j in part:
#            try:
#                x = Chem.MolToSmiles(Chem.RemoveHs(j))
#                if x not in c:
#                    c.append(x)
#            except:
#                continue
#                
#    if len(c)!=4:
#        row['mask'] = False
#    
#    else:    
#        row['acid'] = c[0]
#        row['amine'] = c[1]
#        row['aldehyde'] = c[2]
#        row['isocyanide'] = c[3]
#    return row
    
def decompose_array(smiles_list):
    comps = [None]*len(smiles_list)
    
    for i,smi in tqdm(enumerate(smiles_list), total=len(smiles_list)):
        mol = Chem.MolFromSmiles(smi)

        reactant = (Chem.AddHs(mol),)

        parts = rxn.RunReactants(reactant)
        c = []
        for part in parts:
            for j in part:
                try:
                    x = Chem.MolToSmiles(Chem.RemoveHs(j))
                    if x not in c:
                        c.append(x)
                except:
                    continue

        if len(c)==4:
            comps[i] = c
    
    return comps

tqdm.pandas()

## TEST
#df = df.iloc[:100]

#df = df.progress_apply(decompose_ugi, axis=1)
#df = df[df['mask']]
df['comps'] = decompose_array(df['smiles'])


df = df[df['comps'].notnull()]
df[['acid', 'amine', 'aldehyde', 'isocyanide']] = pd.DataFrame(df.comps.tolist(), index= df.index)

df.to_csv('ugis_decomposed_new.csv')
