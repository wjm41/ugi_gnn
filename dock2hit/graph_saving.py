
import pandas as pd
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from dgl.data.utils import save_graphs

df = pd.read_csv('../datasets/big_test.csv')

atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer()

e_feats = bond_featurizer.feat_size('e')
n_feats = atom_featurizer.feat_size('h')

graphs = []
log_every = 1000
for i, s in enumerate(df.smiles):
    if (i + 1) % log_every == 0:
        print('Processing molecule {:d}/{:d}'.format(i+1, len(df)))
    graphs.append(smiles_to_bigraph(s, node_featurizer=atom_featurizer,
                                    edge_featurizer=bond_featurizer))

save_graphs('../datasets/big_test_graphs.bin', graphs)
