
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
import dgl
from tqdm import tqdm
from dgllife.model.model_zoo import MPNNPredictor
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph

from dock2hit.utils import get_device
from dock2hit.dataloader import SimpleSmilesCSVLoader


def generate_mpnn_fps_from_dataframe(df: pd.DataFrame,
                                     load_name: str = '/rds-d2/user/wjm41/hpc-work/models/dock2hit/ugi/ugi_taut/model_mol9039221.ckpt',
                                     smiles_col: str = 'SMILES',
                                     batch_size: int = 1):

    data_loader = SimpleSmilesCSVLoader(
        df, batch_size=batch_size, shuffle=False, smiles_col=smiles_col)

    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')

    mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                             edge_in_feats=e_feats,
                             node_out_feats=32,
                             edge_hidden_feats=32,
                             num_step_message_passing=4,
                             num_step_set2set=2,
                             num_layer_set2set=3)

    device = get_device()
    checkpoint = torch.load(load_name, map_location=device)
    mpnn_net.load_state_dict(checkpoint['mpnn_state_dict'])

    mpnn_net = mpnn_net.to(device)

    mpnn_net.eval()

    all_feats = []
    for smiles in tqdm(data_loader):
        try:
            graphs = [mol_to_bigraph(Chem.MolFromSmiles(smi), node_featurizer=atom_featurizer,
                                     edge_featurizer=bond_featurizer) for smi in smiles]  # generate and batch graphs
            bg = dgl.batch(graphs).to(device)
            bg.set_n_initializer(dgl.init.zero_initializer)
            bg.set_e_initializer(dgl.init.zero_initializer)

            atom_feats = bg.ndata.pop('h').to(device)
            bond_feats = bg.edata.pop('e').to(device)
            atom_feats, bond_feats = atom_feats.to(
                device), bond_feats.to(device)

            atom_feats = mpnn_net.gnn(bg, atom_feats, bond_feats)
            graph_feats = mpnn_net.readout(
                bg, atom_feats).cpu().detach().numpy()

            all_feats.append(graph_feats)
        except:
            print(smiles)
    all_feats = np.vstack(all_feats)
    return all_feats


def mpi_generate_mpnn_fps_from_dataframe(df, load_name):

    data_loader = SimpleSmilesCSVLoader(df, batch_size=1024, shuffle=False)

    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')

    mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                             edge_in_feats=e_feats,
                             node_out_feats=32,
                             edge_hidden_feats=32,
                             num_step_message_passing=4,
                             num_step_set2set=2,
                             num_layer_set2set=3)

    device = get_device()
    checkpoint = torch.load(load_name, map_location=device)
    mpnn_net.load_state_dict(checkpoint['mpnn_state_dict'])

    mpnn_net = mpnn_net.to(device)

    mpnn_net.eval()

    all_feats = []
    for smiles in data_loader:
        graphs = [mol_to_bigraph(Chem.MolFromSmiles(smi), node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer) for smi in smiles]  # generate and batch graphs
        bg = dgl.batch(graphs).to(device)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)

        atom_feats = bg.ndata.pop('h').to(device)
        bond_feats = bg.edata.pop('e').to(device)
        atom_feats, bond_feats = atom_feats.to(device), bond_feats.to(device)

        atom_feats = mpnn_net.gnn(bg, atom_feats, bond_feats)
        graph_feats = mpnn_net.readout(bg, atom_feats).cpu().detach().numpy()

        all_feats.append(graph_feats)

    all_feats = np.vstack(all_feats)
    return all_feats
