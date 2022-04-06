from dgllife.model.model_zoo import MPNNPredictor
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph

atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer()

e_feats = bond_featurizer.feat_size('e')
n_feats = atom_featurizer.feat_size('h')

mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                         edge_in_feats=e_feats)
print('Default number of parameters: {}'.format(sum(p.numel() for p in mpnn_net.parameters() if p.requires_grad)))

mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                         edge_in_feats=e_feats,
                         num_layer_set2set=6)
print('2x set2set layers #parameters: {}'.format(sum(p.numel() for p in mpnn_net.parameters() if p.requires_grad)))

mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                         edge_in_feats=e_feats,
                         num_step_set2set=6)
print('2x set2set steps #parameters: {}'.format(sum(p.numel() for p in mpnn_net.parameters() if p.requires_grad)))

mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                         edge_in_feats=e_feats,
                         num_step_message_passing=12)
print('2 message passing #parameters: {}'.format(sum(p.numel() for p in mpnn_net.parameters() if p.requires_grad)))
