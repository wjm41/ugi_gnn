"""
Property prediction using a Message-Passing Neural Network.
"""

import argparse

import dgl
import pandas as pd
import numpy as np
import torch
from dgllife.model.model_zoo import MPNNPredictor
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
from rdkit import Chem
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'


# Collate Function for Dataloader
def collate(sample):
    graphs, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)

def main(args):
    df = pd.read_csv(args.path)
    
    X = [Chem.MolFromSmiles(m) for m in df['smiles'].values]
    y = df['dock_score'].values
 
    # Initialise featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')
    print('Number of features: ', n_feats)

    X = [mol_to_bigraph(m, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for m in X]

    r2_list = []
    rmse_list = []
    skipped_trials = 0

    for i in range(args.n_trials):
        writer = SummaryWriter('runs/'+args.savename)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_set_size, random_state=i+5)
         
        y_train = torch.Tensor(y_train)
        y_test = torch.Tensor(y_test)

        #y_train = y_train.reshape(-1, 1)
        #y_test = y_test.reshape(-1, 1)

        #  We standardise the outputs but leave the inputs unchanged

        #y_scaler = StandardScaler()
        #y_train_scaled = torch.Tensor(y_scaler.fit_transform(y_train))
        #y_test_scaled = torch.Tensor(y_scaler.transform(y_test))

        #train_data = list(zip(X_train, y_train_scaled))
        #test_data = list(zip(X_test, y_test_scaled))

        train_data = list(zip(X_train, y_train))
        test_data = list(zip(X_test, y_test))

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate, drop_last=False)

        mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                                 edge_in_feats=e_feats
                                 )
        mpnn_net.to(device)

        loss_fn = MSELoss()
        optimizer = torch.optim.Adam(mpnn_net.parameters(), lr=args.lr)

        mpnn_net.train()
        print('beginning training... trial #{}'.format(i))
        epoch_losses = []
        epoch_rmses = []
        for epoch in range(1, args.n_epochs+1):
            epoch_loss = 0
            preds = []
            labs = []
            for i, (bg, labels) in enumerate(train_loader):
                labels = labels.to(device)
                atom_feats = bg.ndata.pop('h').to(device)
                bond_feats = bg.edata.pop('e').to(device)
                atom_feats, bond_feats, labels = atom_feats.to(device), bond_feats.to(device), labels.to(device)
                y_pred = mpnn_net(bg, atom_feats, bond_feats)
                labels = labels.unsqueeze(dim=1)

                if args.debug:
                    print('label: {}'.format(labels))
                    print('y_pred: {}'.format(y_pred)
)
                loss = loss_fn(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()

                # Inverse transform to get RMSE
                #labels = y_scaler.inverse_transform(labels.reshape(-1, 1))
                #y_pred = y_scaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))

                # store labels and preds
                preds.append(y_pred)
                labs.append(labels)

            labs = np.concatenate(labs, axis=None)
            preds = np.concatenate(preds, axis=None)
            pearson, p = pearsonr(preds, labs)
            rmse = np.sqrt(mean_squared_error(preds, labs))
            r2 = r2_score(preds, labs)

            if args.debug:
                print(f"epoch: {epoch}, "
                      f"LOSS: {epoch_loss:.3f}, "
                      f"RMSE: {rmse:.3f}, "
                      f"R: {pearson:.3f}, "
                      f"R2: {r2:.3f}")
            writer.add_scalar('loss/train', epoch_loss, epoch)
            writer.add_scalar('train/rmse', rmse, epoch)
            writer.add_scalar('train/rho', p, epoch)
            writer.add_scalar('train/R2', r2, epoch)

            if epoch % 20 == 0:
                print(f"epoch: {epoch}, "
                      f"LOSS: {epoch_loss:.3f}, "
                      f"RMSE: {rmse:.3f}, "
                      f"R: {pearson:.3f}, "
                      f"R2: {r2:.3f}")
                try:
                    torch.save(mpnn_net.state_dict(), '/rds-d2/user/wjm41/hpc-work/models/' + args.savename +
                               '/model_epoch_' + str(epoch) + '.pt')
                except FileNotFoundError:
                    cmd = 'mkdir /rds-d2/user/wjm41/hpc-work/models/' + args.savename
                    os.system(cmd)
                    torch.save(mpnn_net.state_dict(), '/rds-d2/user/wjm41/hpc-work/models/' + args.savename +
                               '/model_epoch_' + str(epoch) + '.pt')

            if args.test:
                # Evaluate
                mpnn_net.eval()
                preds = []
                labs = []
                for i, (bg, labels) in enumerate(test_loader):
                    labels = labels.to(device)
                    atom_feats = bg.ndata.pop('h').to(device)
                    bond_feats = bg.edata.pop('e').to(device)
                    atom_feats, bond_feats, labels = atom_feats.to(device), bond_feats.to(device), labels.to(device)
                    y_pred = mpnn_net(bg, atom_feats, bond_feats)
                    labels = labels.unsqueeze(dim=1)

                    # Inverse transform to get RMSE
                    #labels = y_scaler.inverse_transform(labels.reshape(-1, 1))
                    #y_pred = y_scaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 1))

                    preds.append(y_pred)
                    labs.append(labels)

                labs = np.concatenate(labs, axis=None)
                preds = np.concatenate(preds, axis=None)

                pearson, p = pearsonr(preds, labs)
                rmse = np.sqrt(mean_squared_error(preds, labs))
                r2 = r2_score(preds, labs)

                r2_list.append(r2)
                rmse_list.append(rmse)

                print(f'Test RMSE: {rmse:.3f}, R: {pearson:.3f}, R2: {r2:.3f}')
    torch.save(mpnn_net.state_dict(), '/rds-d2/user/wjm41/hpc-work/models/' + args.savename +
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
    parser.add_argument('-savename', '--savename', type=str, default='ugi-pretrained',
                        help='name for directory containing saved model params and tensorboard logs')

    args = parser.parse_args()

    main(args)
