"""
Property prediction using a Message-Passing Neural Network.
"""
import subprocess
import argparse
import logging 

import dgl
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from dgllife.model.model_zoo import MPNNPredictor
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

def bash_command(cmd):
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    p.communicate()

class UgiLoaderFast:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, csv_path, y_scaler, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param: csv_path (string): path to .csv containing molecular data
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        self.df = pd.read_csv(csv_path)
        self.y_scaler = y_scaler
        self.df.reset_index(drop=True,inplace=True)
        self.dataset_len = len(self.df)

        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        df_batch = self.df.iloc[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        #return batch
        return df_batch['smiles'].values, self.y_scaler.transform(df_batch['dock_score'].to_numpy().reshape(-1,1))

    def __len__(self):
        return self.n_batches

    
def main(args):

    y_train = pd.read_csv(args.train_path)['dock_score'].to_numpy()

    y_scaler = StandardScaler()    
    y_scaler = y_scaler.fit(y_train.reshape(-1,1))

    data_loader = UgiLoaderFast(args.path, batch_size=1024, shuffle=False, y_scaler=y_scaler)

    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')
    print('Number of features: ', n_feats)

    mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                             edge_in_feats=e_feats
                             )

    if args.load_name is not None:
       mpnn_net.load_state_dict(torch.load(args.load_name, map_location=device))

    mpnn_net = mpnn_net.to(device)

    mpnn_net.eval()

    all_feats = []

    len_df = len(pd.read_csv(args.path))
    preds = np.array([None] * len_df).reshape(-1,1)
    labs = np.array([None] * len_df).reshape(-1,1)
    n = 0
    for i, (smiles, labels) in tqdm(enumerate(data_loader)):
        graphs = [mol_to_bigraph(Chem.MolFromSmiles(smi), node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for smi in smiles] # generate and batch graphs
        bg = dgl.batch(graphs).to(device)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)

        atom_feats = bg.ndata.pop('h').to(device)
        bond_feats = bg.edata.pop('e').to(device)
        atom_feats, bond_feats, labels = atom_feats.to(device), bond_feats.to(device), torch.tensor(labels, dtype=torch.float32).to(device)

        y_pred = mpnn_net(bg, atom_feats, bond_feats)
        labels = labels.unsqueeze(dim=1)

        batch_preds = y_scaler.inverse_transform(y_pred.cpu().detach().numpy())
        batch_labs = y_scaler.inverse_transform(labels.cpu().detach().numpy())

        preds[n: n + len(smiles) ] = batch_preds
        labs[n: n + len(smiles) ] = batch_labs.reshape(len(batch_labs),1)

        n += len(smiles)


    print(preds)
    print(labs)
    p = spearmanr(preds, labs)[0]
    rmse = np.sqrt(mean_squared_error(preds, labs))
    r2 = r2_score(preds, labs)

    print(f"RMSE: {rmse:.3f}, "
          f"RHO: {p:.3f}, "
          f"R2: {r2:.3f}")

if __name__ == '__main__':

    model_dir = '/rds-d2/user/wjm41/hpc-work/models/ugi_pretrained'

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='ugis-00000000.csv',
                        help='Path to the data .csv file.')
    parser.add_argument('-tp', '--train_path', type=str, default='ugis-00000000.csv',
                        help='Path to the training data .csv file.')
    parser.add_argument('-save_name', '--save_name', type=str, default='ugi_mpnn_fps.npy',
                        help='name for .npy file with saved ugi fingerprint')
    parser.add_argument('-load_name', '--load_name', default=model_dir + '/model_mol8517741.pt',
                        help='name for directory containing saved model params and tensorboard logs')

    args = parser.parse_args()

    main(args)
