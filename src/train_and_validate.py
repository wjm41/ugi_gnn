"""
Property prediction using a Message-Passing Neural Network.
"""
import argparse
import logging

from datetime import datetime

import dgl
import pandas as pd
import numpy as np

from tqdm import tqdm
from dgllife.model.model_zoo import MPNNPredictor
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
from rdkit import Chem
from scipy.stats import spearmanr

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from dataloader import load_data
from optimizers import SGDHD, AdamHD, FelixHD, FelixExpHD
from utils import bash_command, human_len

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.dpi": 200})

if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'


class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError(
                'hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


def main(args):

    # Initialise featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')
    print('Number of features: ', n_feats)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(args.log_dir+current_time)
    writer.add_hparams({'optimizer': args.optimizer},
                       {'batch_size': args.batch_size, 'lr': args.lr, 'hypergrad': args.hypergrad_lr})

    data_loaders = load_data(args)

    # raise Exception
    # mpnn_net = MPNNPredictor(node_in_feats=n_feats,
    #                         edge_in_feats=e_feats,
    #                         num_layer_set2set=6)

    mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                             edge_in_feats=e_feats,
                             node_out_feats=32,
                             edge_hidden_feats=32,
                             num_step_message_passing=4,
                             num_step_set2set=2,
                             num_layer_set2set=3)
    mpnn_net = mpnn_net.to(device)

    # TODO warmup for adam
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(mpnn_net.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamHD':
        optimizer = AdamHD(mpnn_net.parameters(), lr=args.lr,
                           hypergrad_lr=args.hypergrad_lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(mpnn_net.parameters(), lr=args.lr)
    elif args.optimizer == 'SGDHD':
        optimizer = AdamHD(mpnn_net.parameters(), lr=args.lr,
                           hypergrad_lr=args.hypergrad_lr)
    elif args.optimizer == 'FelixHD':
        optimizer = FelixHD(mpnn_net.parameters(), lr=args.lr,
                            hypergrad_lr=args.hypergrad_lr)
    elif args.optimizer == 'FelixExpHD':
        optimizer = FelixExpHD(mpnn_net.parameters(), lr=args.lr,
                               hypergrad_lr=args.hypergrad_lr)
    else:
        print(args.optimizer)
        raise Exception('scrub')

    loss_fn = MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    start_epoch = 1
    start_batch = 0

    if args.load_name is not None:
        checkpoint = torch.load(args.load_name, map_location=device)
        mpnn_net.load_state_dict(checkpoint['mpnn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        start_batch = checkpoint['batch']

    print('Number of parameters: {}'.format(sum(p.numel()
          for p in mpnn_net.parameters() if p.requires_grad)))

    print('beginning training...')
    for epoch in range(start_epoch, args.n_epochs+1):
        mpnn_net.train()
        epoch_loss = 0
        preds = np.array([None] * len(train_ind)).reshape(-1, 1)
        labs = np.array([None] * len(train_ind)).reshape(-1, 1)
        n = 0
        start_batch = 0

        for i, (smiles, labels) in tqdm(enumerate(train_loader, start=start_batch),
                                        initial=start_batch,
                                        total=int(len(train_ind) /
                                                  args.batch_size),
                                        miniters=10000,
                                        unit='batch',
                                        unit_scale=True):

            # TODO multithread graph featurizer
            bg = [mol_to_bigraph(Chem.MolFromSmiles(smi), node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer) for smi in smiles]  # generate and batch graphs
            bg = dgl.batch(bg).to(device)
            bg.set_n_initializer(dgl.init.zero_initializer)
            bg.set_e_initializer(dgl.init.zero_initializer)

            atom_feats = bg.ndata.pop('h').to(device)
            bond_feats = bg.edata.pop('e').to(device)
            atom_feats, bond_feats, labels = atom_feats.to(device), bond_feats.to(
                device), torch.tensor(labels, dtype=torch.float32).to(device)
            y_pred = mpnn_net(bg, atom_feats, bond_feats)

            if args.debug:
                print('label: {}'.format(labels))
                print('y_pred: {}'.format(y_pred))

            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            if 'Felix' in args.optimizer:
                _, state_lrs = optimizer.step()
            else:
                optimizer.step()

            epoch_loss += loss.detach().item()

            batch_preds = y_scaler.inverse_transform(
                y_pred.cpu().detach().numpy())
            batch_labs = y_scaler.inverse_transform(
                labels.cpu().detach().numpy())

            preds[n: n + len(smiles)] = batch_preds
            labs[n: n + len(smiles)] = batch_labs.reshape(len(batch_labs), 1)
            n += len(smiles)

            if (i != 0 and i % args.write_batch == 0) or epoch % args.write_batch == 0:
                if args.debug:
                    print('label: {}'.format(labels))
                    print('y_pred: {}'.format(y_pred))

                p = spearmanr(batch_preds, batch_labs)[0]
                rmse = np.sqrt(mean_squared_error(batch_preds, batch_labs))
                r2 = r2_score(batch_preds, batch_labs)

                writer.add_scalar('loss/train', loss.detach().item(),
                                  i*args.batch_size + (epoch-1)*len(train_ind))
                writer.add_scalar('train/rmse', rmse, i *
                                  args.batch_size + (epoch-1)*len(train_ind))
                writer.add_scalar('train/rho', p, i *
                                  args.batch_size + (epoch-1)*len(train_ind))
                writer.add_scalar('train/R2', r2, i *
                                  args.batch_size + (epoch-1)*len(train_ind))
                if 'Felix' in args.optimizer:
                    writer.add_histogram('train/lrT', state_lrs, i *
                                         args.batch_size + (epoch-1)*len(train_ind))
                df = pd.DataFrame(
                    data={'dock_score': batch_labs.flatten(), 'preds': batch_preds.flatten()})
                plot = sns.jointplot(
                    data=df, x='dock_score', y='preds', kind='scatter')
                dot_line = [np.amin(df['dock_score']),
                            np.amax(df['dock_score'])]
                plot.ax_joint.plot(dot_line, dot_line, 'k:')
                plt.xlabel('Dock Scores')
                plt.ylabel('Predictions')
                writer.add_figure('Training batch', plot.fig, global_step=i *
                                  args.batch_size + (epoch-1)*len(train_ind))
                writer.flush()

                if val_loader is not None:
                    mpnn_net.eval()

                    val_preds = np.array([None] * len(val_ind)).reshape(-1, 1)
                    val_labs = np.array([None] * len(val_ind)).reshape(-1, 1)

                    m = 0
                    val_loss = 0
                    for j, (smiles, labels) in enumerate(val_loader):
                        bg = [mol_to_bigraph(Chem.MolFromSmiles(smi), node_featurizer=atom_featurizer,
                                             edge_featurizer=bond_featurizer) for smi in smiles]  # generate and batch graphs
                        bg = dgl.batch(bg).to(device)
                        bg.set_n_initializer(dgl.init.zero_initializer)
                        bg.set_e_initializer(dgl.init.zero_initializer)

                        atom_feats = bg.ndata.pop('h').to(device)
                        bond_feats = bg.edata.pop('e').to(device)
                        atom_feats, bond_feats, labels = atom_feats.to(device), bond_feats.to(
                            device), torch.tensor(labels, dtype=torch.float32).to(device)
                        y_pred = mpnn_net(bg, atom_feats, bond_feats)

                        loss = loss_fn(y_pred, labels)

                        val_loss += loss.detach().item()
                        batch_preds_val = y_scaler.inverse_transform(
                            y_pred.cpu().detach().numpy())
                        batch_labs_val = y_scaler.inverse_transform(
                            labels.cpu().detach().numpy())

                        val_preds[m: m + len(smiles)] = batch_preds_val
                        val_labs[m: m + len(smiles)
                                 ] = batch_labs_val.reshape(len(batch_labs_val), 1)
                        m += len(smiles)

                    scheduler.step(val_loss)
                    p = spearmanr(val_preds, val_labs)[0]
                    rmse = np.sqrt(mean_squared_error(val_preds, val_labs))
                    r2 = r2_score(val_preds, val_labs)
                    # TODO change logging
                    print(
                        f'Validation RMSE: {rmse:.3f}, RHO: {p:.3f}, R2: {r2:.3f}')
                    logging.warning(
                        f'Validation RMSE: {rmse:.3f}, RHO: {p:.3f}, R2: {r2:.3f}')

                    writer.add_scalar(
                        'loss/val', loss.detach().item(), i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.add_scalar(
                        'val/rmse', rmse, i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.add_scalar(
                        'val/rho', p, i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.add_scalar(
                        'val/R2', r2, i*args.batch_size + (epoch-1)*len(train_ind))

                    df = pd.DataFrame(
                        data={'dock_score': val_labs.flatten(), 'preds': val_preds.flatten()})
                    # df = df[df['dock_score'] < 5]

                    # TODO fix axes

                    plot = sns.jointplot(
                        data=df, x='dock_score', y='preds', kind='scatter')
                    dot_line = [np.amin(df['dock_score']),
                                np.amax(df['dock_score'])]
                    plot.ax_joint.plot(dot_line, dot_line, 'k:')
                    plt.xlabel('Dock Scores')
                    plt.ylabel('Predictions')
                    writer.add_figure(
                        'Validation Set', plot.fig, global_step=i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.flush()

                    mpnn_net.train()

            if (i != 0 and i % args.save_batch == 0) or epoch % args.write_batch == 0:
                try:
                    torch.save({
                        'epoch': epoch,
                        'mpnn_state_dict': mpnn_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'batch': i,
                    }, args.save_name +
                        '/model_mol' + str(i*args.batch_size + (epoch-1)*len(train_ind))+'.ckpt')
                except FileNotFoundError:
                    cmd = 'mkdir ' + args.save_name
                    bash_command(cmd)
                    torch.save({
                        'epoch': epoch,
                        'mpnn_state_dict': mpnn_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'batch': i,
                    }, args.save_name +
                        '/model_mol' + str(i*args.batch_size + (epoch-1)*len(train_ind))+'.ckpt')
        p = spearmanr(preds, labs)[0]
        rmse = np.sqrt(mean_squared_error(preds, labs))
        r2 = r2_score(preds, labs)

        if args.debug:
            print(f"epoch: {epoch}, "
                  f"LOSS: {epoch_loss:.3f}, "
                  f"RMSE: {rmse:.3f}, "
                  f"RHO: {p:.3f}, "
                  f"R2: {r2:.3f}")

        if epoch % 10 == 0:
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
                torch.save({
                    'epoch': epoch,
                    'mpnn_state_dict': mpnn_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'batch': 0,
                }, args.save_name +
                    '/model_epoch' + str(epoch)+'.ckpt')
            except FileNotFoundError:
                cmd = 'mkdir ' + args.save_name
                bash_command(cmd)
                torch.save({
                    'epoch': epoch,
                    'mpnn_state_dict': mpnn_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'batch': 0,
                }, args.save_name +
                    '/model_epoch' + str(epoch)+'.ckpt')

    torch.save({
        'epoch': epoch,
        'mpnn_state_dict': mpnn_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'batch': 0,
    }, args.save_name +
        '/model_epoch_final.ckpt')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    group_io = parser.add_argument_group("I/O")
    group_io.add_argument('-p', '--path', type=str, default='ugis-00000000.csv',
                          help='Path to the data.csv file.')
    group_io.add_argument('-log_dir', '--log_dir', type=str, default='.',
                          help='directory containing tensorboard logs')
    group_io.add_argument('-save_name', '--save_name', type=str, default='ugi-pretrained',
                          help='directory for saving model params')
    group_io.add_argument('-load_name', '--load_name', default=None,
                          help='name for directory containing saved model params checkpoint file for continued training.')

    group_data = parser.add_argument_group("Training - data")
    group_data.add_argument('-n_trials', '--n_trials', type=int, default=3,
                            help='int specifying number of random train/test splits to use')
    group_data.add_argument('-batch_size', '--batch_size', type=int, default=64,
                            help='int specifying batch_size for training and evaluations')
    group_data.add_argument('-write_batch', '--write_batch', type=int, default=1000,
                            help='int specifying number of steps per tensorboard write')
    group_data.add_argument('-save_batch', '--save_batch', type=int, default=5000,
                            help='int specifying number of batches per model save')
    group_data.add_argument('-n_epochs', '--n_epochs', type=int, default=10,
                            help='int specifying number of random train/test splits to use')
    group_data.add_argument('-ts', '--test_set_size', type=float, default=0.1,
                            help='float in range [0, 1] specifying fraction of dataset to use as test set')
    group_data.add_argument('-val', action='store_true',
                            help='whether or not to do random train/val split and log val_loss')
    group_data.add_argument('-val_size', type=int, default=1000,
                            help='Integer size of training set datapoints to use as random validation set.')
    group_data.add_argument('-val_path', type=str, default=None,
                            help='path to separate validation set ; if not None, overwrites -val options')

    group_optim = parser.add_argument_group("Training - optimizer")
    group_optim.add_argument('-optimizer', '--optimizer', type=str, default=None,
                             choices=['Adam', 'AdamHD', 'SGD',
                                      'SGDHD', 'FelixHD', 'FelixExpHD'],
                             help='name of optimizer to use during training.')
    group_optim.add_argument('-lr', '--lr', type=float, default=1e-3,
                             help='float specifying learning rate used during training.')
    group_optim.add_argument('-hypergrad_lr', '--hypergrad_lr', type=float, default=1e-3,
                             help='float specifying hypergradient learning rate used during training.')
    group_optim.add_argument('-hypergrad_lr_decay', '--hypergrad_lr_decay', type=float, default=1e-5,
                             help='float specifying hypergradient lr decay used during training.')
    group_optim.add_argument('-weight_decay', '--weight_decay', type=float, default=1e-4,
                             help='float specifying hypergradient weight decay used during training.')
    group_optim.add_argument('-hypergrad_warmup', '--hypergrad_warmup', type=int, default=100,
                             help='Number of steps warming up hypergrad before using for optimisation.')

    parser.add_argument('-debug', action='store_true',
                        help='whether or not to print predictions and model weight gradients')

    args = parser.parse_args()

    main(args)
