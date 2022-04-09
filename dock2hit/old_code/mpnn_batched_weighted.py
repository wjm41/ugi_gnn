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
    def __init__(self, csv_path, inds, y_scaler, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param: csv_path (string): path to .csv containing molecular data
        :param: inds (array of ints): train/test indices to select subset of data
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        self.df = pd.read_csv(csv_path).iloc[inds]
        self.df.reset_index(drop=True,inplace=True)
        self.y_scaler = y_scaler
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
        return df_batch['smiles'].values, self.y_scaler.transform(df_batch['dock_score'].to_numpy().reshape(-1,1)), df_batch['weight'].values

    def __len__(self):
        return self.n_batches

#class UgiDataset(Dataset):
#    """Large molecular dataset."""
#
#    def __init__(self, csv_path, inds):
#        """
#        Args:
#            csv_path (string): path to .csv containing molecular data
#            inds (array of ints): train/test indices to select subset of data
#        """
#        self.df = pd.read_csv(csv_path).iloc[inds]
#        self.df.reset_index(drop=True,inplace=True)
#
#    def __len__(self):
#        return len(self.df)
#
#    def __getitem__(self, idx):
#        if torch.is_tensor(idx):
#            idx = idx.tolist()
#        df_batch = self.df.iloc[idx]
#
#        return df_batch['smiles'], df_batch['dock_score']

def main(args):
    y_train = pd.read_csv(args.path)['dock_score'].to_numpy()
    len_df = len(pd.read_csv(args.path))

    y_scaler = StandardScaler()

    # Initialise featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')
    print('Number of features: ', n_feats)

    r2_list = []
    rmse_list = []
    skipped_trials = 0

    for i in range(args.n_trials):
        writer = SummaryWriter('runs/'+args.save_name)

        if args.test:
            train_ind, test_ind, y_train, _  = train_test_split(range(len_df), y_train, test_size=args.test_set_size, random_state=i+5)
        else:
            train_ind = range(len_df)
 
        y_scaler = y_scaler.fit(y_train.reshape(-1,1)) 

        train_loader = UgiLoaderFast(args.path, inds=train_ind, batch_size=args.batch_size, shuffle=True, y_scaler=y_scaler)
        if args.test:
            test_loader = UgiLoaderFast(args.path, inds=test_ind, batch_size=args.batch_size, shuffle=False, y_scaler=y_scaler)

        mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                                 edge_in_feats=e_feats
                                 )
        if args.load_name is not None:
           mpnn_net.load_state_dict(torch.load(args.load_name))
        mpnn_net = mpnn_net.to(device)

        loss_fn = MSELoss()
        optimizer = torch.optim.Adam(mpnn_net.parameters(), lr=args.lr)

        print('beginning training... trial #{}'.format(i))
        epoch_losses = []
        epoch_rmses = []
        for epoch in range(1, args.n_epochs+1):
            mpnn_net.train()
            epoch_loss = 0
            preds = np.array([None] * len(train_ind)).reshape(-1,1)
            labs = np.array([None] * len(train_ind)).reshape(-1,1)
            n = 0
            for i, (smiles, labels, weights) in tqdm(enumerate(train_loader), total=int(len(train_ind)/args.batch_size)):
            #for i, df_batch in tqdm(enumerate(train_loader), total=int(len_df/args.batch_size)):
                graphs = [mol_to_bigraph(Chem.MolFromSmiles(smi), node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for smi in smiles] # generate and batch graphs
                bg = dgl.batch(graphs).to(device)
                bg.set_n_initializer(dgl.init.zero_initializer)
                bg.set_e_initializer(dgl.init.zero_initializer)

                atom_feats = bg.ndata.pop('h').to(device)
                bond_feats = bg.edata.pop('e').to(device)
                atom_feats, bond_feats = atom_feats.to(device), bond_feats.to(device)

                labels = torch.tensor(labels, dtype=torch.float32).to(device)
                weights = torch.tensor(weights, dtype=torch.float32).to(device)

                y_pred = mpnn_net(bg, atom_feats, bond_feats)
                print(y_pred.shape)
                print(labels.shape)
                #labels = labels.unsqueeze(dim=1)

                if args.debug:
                    print('label: {}'.format(labels))
                    print('y_pred: {}'.format(y_pred))

                loss = loss_fn(weights*y_pred, weights*labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()

                #preds.append(y_pred)
                #labs.append(labels)

                batch_preds = y_scaler.inverse_transform(y_pred.cpu().detach().numpy())
                batch_labs = y_scaler.inverse_transform(labels.cpu().detach().numpy())

                preds[n: n + len(smiles) ] = batch_preds
                labs[n: n + len(smiles) ] = batch_labs.reshape(len(batch_labs),1)
                n += len(smiles)

                if i%args.write_batch==0:
                    if args.debug:
                        print('label: {}'.format(labels))
                        print('y_pred: {}'.format(y_pred))
                    p = spearmanr(batch_preds, batch_labs)[0]
                    rmse = np.sqrt(mean_squared_error(batch_preds, batch_labs))
                    r2 = r2_score(batch_preds, batch_labs)

                    writer.add_scalar('loss/train', loss.detach().item(), i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.add_scalar('train/rmse', rmse, i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.add_scalar('train/rho', p, i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.add_scalar('train/R2', r2, i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.flush()
                if i%args.save_batch==0:
                    try:
                        torch.save(mpnn_net.state_dict(), '/rds-d2/user/wjm41/hpc-work/models/' + args.save_name +
                                   '/model_mol' + str(i*args.batch_size + (epoch-1)*len(train_ind))+'.pt')
                    except FileNotFoundError:
                        cmd = 'mkdir /rds-d2/user/wjm41/hpc-work/models/' + args.save_name
                        bash_command(cmd)
                        torch.save(mpnn_net.state_dict(), '/rds-d2/user/wjm41/hpc-work/models/' + args.save_name +
                                   '/model_mol' + str(i*args.batch_size + (epoch-1)*len(train_ind))+'.pt')

            #labs = np.concatenate(labs, axis=None)
            #preds = np.concatenate(preds, axis=None)

            p = spearmanr(preds, labs)[0]
            rmse = np.sqrt(mean_squared_error(preds, labs))
            r2 = r2_score(preds, labs)

            if args.debug:
                print(f"epoch: {epoch}, "
                      f"LOSS: {epoch_loss:.3f}, "
                      f"RMSE: {rmse:.3f}, "
                      f"RHO: {p:.3f}, "
                      f"R2: {r2:.3f}")
            #writer.add_scalar('loss/train', epoch_loss, epoch)
            #writer.flush()

            if epoch % 1 == 0:
                print(f"epoch: {epoch}, "
                      f"LOSS: {epoch_loss:.3f}, "
                      f"RMSE: {rmse:.3f}, "
                      f"RHO: {p:.3f}, "
                      f"R2: {r2:.3f}")
                logging.warning(f"epoch: {epoch}, "
                      f"LOSS: {epoch_loss:.3f}, "
                      f"RMSE: {rmse:.3f}, "
                      f"RHO: {p:.3f}, "
                      f"R2: {r2:.3f}")
                try:
                    torch.save(mpnn_net.state_dict(), '/rds-d2/user/wjm41/hpc-work/models/' + args.save_name +
                               '/model_epoch_' + str(epoch) + '.pt')
                except FileNotFoundError:
                    cmd = 'mkdir /rds-d2/user/wjm41/hpc-work/models/' + args.save_name
                    bash_command(cmd)
                    torch.save(mpnn_net.state_dict(), '/rds-d2/user/wjm41/hpc-work/models/' + args.save_name +
                               '/model_epoch_' + str(epoch) + '.pt')

            if args.test:
                # Evaluate
                mpnn_net.eval()
                #preds = []
                #labs = []

                preds = np.array([None] * len(test_ind)).reshape(-1,1)
                labs = np.array([None] * len(test_ind)).reshape(-1,1)

                n=0
                for i, (smiles, labels) in tqdm(enumerate(test_loader), total=int(len(test_ind)/args.batch_size)):
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
    
                    #if i%10==0:
                    #    pearson, p = pearsonr(batch_preds, batch_labs)
                    #    rmse = np.sqrt(mean_squared_error(batch_preds, batch_labs))
                    #    r2 = r2_score(batch_preds, batch_labs)

                    #    writer.add_scalar('test/rmse', rmse, i*args.batch_size + (epoch-1)*len(train_ind))
                    #    writer.add_scalar('test/rho', p, i*args.batch_size + (epoch-1)*len(train_ind))
                    #    writer.add_scalar('test/R2', r2, i*args.batch_size + (epoch-1)*len(train_ind))
                    #    writer.flush()

                p = spearmanr(preds, labs)[0]
                rmse = np.sqrt(mean_squared_error(preds, labs))
                r2 = r2_score(preds, labs)

                r2_list.append(r2)
                rmse_list.append(rmse)

                print(f'Test RMSE: {rmse:.3f}, RHO: {p:.3f}, R2: {r2:.3f}')
                logging.warning(f'Test RMSE: {rmse:.3f}, RHO: {p:.3f}, R2: {r2:.3f}')
    torch.save(mpnn_net.state_dict(), '/rds-d2/user/wjm41/hpc-work/models/' + args.save_name +
                    '/model_epoch_final.pt')
    if args.test:
        r2_list = np.array(r2_list)
        rmse_list = np.array(rmse_list)

        print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
        print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, default='ugis-00000000.csv',
                        help='Path to the data.csv file.')
    parser.add_argument('-n_trials', '--n_trials', type=int, default=3,
                        help='int specifying number of random train/test splits to use')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=64,
                        help='int specifying batch_size for training and evaluations')
    parser.add_argument('-write_batch', '--write_batch', type=int, default=100,
                        help='int specifying number of steps per tensorboard write')
    parser.add_argument('-save_batch', '--save_batch', type=int, default=500,
                        help='int specifying number of batches per model save')
    parser.add_argument('-n_epochs', '--n_epochs', type=int, default=10,
                        help='int specifying number of random train/test splits to use')
    parser.add_argument('-ts', '--test_set_size', type=float, default=0.1,
                        help='float in range [0, 1] specifying fraction of dataset to use as test set')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3,
                        help='float specifying learning rate used during training.')
    parser.add_argument('-test', action='store_true',
                        help='whether or not to do test/train split')
    parser.add_argument('-debug', action='store_true',
                        help='whether or not to print predictions and model weight gradients')
    parser.add_argument('-save_name', '--save_name', type=str, default='ugi-pretrained',
                        help='name for directory containing saved model params and tensorboard logs')
    parser.add_argument('-load_name', '--load_name', default=None,
                        help='name for directory containing saved model params and tensorboard logs')

    args = parser.parse_args()

    main(args)
