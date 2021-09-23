import pandas as pd
from tqdm import tqdm
from rdkit import Chem

import sys

fname = str(sys.argv[1])

f = open(fname, 'r')
fnames = f.read().splitlines()

def remove_chirality(row):
    mol = Chem.MolFromSmiles(row['smiles'])
    Chem.RemoveStereochemistry(mol)
    row['smiles'] = Chem.MolToSmiles(mol)
    return row

for fname in tqdm(fnames):
    df = pd.read_csv(fname+'.can', delim_whitespace=True, 
                 header=None, names=['smiles','name','dock_score'], usecols=[0,2])


    df = df.apply(remove_chirality, axis=1)
    df.to_csv(fname+'.csv', index=False)
