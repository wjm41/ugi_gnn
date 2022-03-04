"""
Property prediction using a Message-Passing Neural Network.
"""
import argparse
import logging

import dgl
import numpy as np

from tqdm import tqdm
from dgllife.model.model_zoo import MPNNPredictor
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from rdkit import Chem

import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader import load_data
from optimizers import load_optimizer

from tensorboard_logging import Logger
from utils import bash_command, human_len


if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'


def generate_batch(smiles, atom_featurizer, bond_featurizer):
    # TODO multithread graph featurizer
    bg = [smiles_to_bigraph(Chem.MolFromSmiles(smi), node_featurizer=atom_featurizer,
                            edge_featurizer=bond_featurizer) for smi in smiles]  # generate and batch graphs
    bg = dgl.batch(bg)
    return bg


def run_batch(model, bg):
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    atom_feats = bg.ndata.pop('h').to(device)
    bond_feats = bg.edata.pop('e').to(device)
    atom_feats, bond_feats = atom_feats.to(device), bond_feats.to(
        device),
    y_pred = model(bg, atom_feats, bond_feats)

    return y_pred


def main(args):
    logging.basicConfig(level=logging.INFO)

    # Initialise featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')
    print('Number of features: ', n_feats)

    if args.val or args.val_path is not None:
        train_loader, val_loader = load_data(args)
    else:
        train_loader = load_data(args)

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

    optimizer = load_optimizer(args, mpnn_net)

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
    logger = Logger(args)
    for epoch in range(start_epoch, args.n_epochs+1):
        mpnn_net.train()
        epoch_loss = 0

        n = 0
        start_batch = 0

        for i, (smiles, labels) in tqdm(enumerate(train_loader, start=start_batch),
                                        initial=start_batch,
                                        total=train_loader.n_batches,
                                        miniters=10000,
                                        unit='batch',
                                        unit_scale=True):

            bg = generate_batch(smiles, atom_featurizer,
                                bond_featurizer).to(device)
            y_pred = run_batch(mpnn_net, bg)

            labels = torch.tensor(labels, dtype=torch.float32).to(device)

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

            n += len(smiles)

            if i != 0 and i % args.write_batch == 0:

                batch_preds = train_loader.y_scaler.inverse_transform(
                    y_pred.cpu().detach().numpy())
                batch_labs = train_loader.y_scaler.inverse_transform(
                    labels.cpu().detach().numpy())

                # number of mols seen by model
                n_mols = i*args.batch_size + (epoch-1)*len(train_loader)
                if 'Felix' in args.optimizer:
                    logger.log(n_mols, loss, batch_preds, batch_labs,
                               split='train', state_lrs=state_lrs)
                else:
                    logger.log(n_mols, loss, batch_preds, batch_labs,
                               split='train')

                if args.debug:
                    print('label: {}'.format(labels))
                    print('y_pred: {}'.format(y_pred))

                if val_loader is not None:
                    mpnn_net.eval()

                    val_preds = np.array(
                        [None] * len(val_loader)).reshape(-1, 1)
                    val_labs = np.array(
                        [None] * len(val_loader)).reshape(-1, 1)

                    m = 0
                    val_loss = 0

                    for smiles, labels in val_loader:
                        bg = generate_batch(smiles, atom_featurizer,
                                            bond_featurizer).to(device)
                        with torch.no_grad():
                            y_pred = run_batch(mpnn_net, bg)

                        loss = loss_fn(y_pred, labels)

                        val_loss += loss.detach().item()
                        batch_preds_val = train_loader.y_scaler.inverse_transform(
                            y_pred.cpu().detach().numpy())
                        batch_labs_val = train_loader.y_scaler.inverse_transform(
                            labels.cpu().detach().numpy())

                        val_preds[m: m + len(smiles)] = batch_preds_val
                        val_labs[m: m + len(smiles)
                                 ] = batch_labs_val.reshape(len(batch_labs_val), 1)
                        m += len(smiles)

                    scheduler.step(val_loss)

                    logger.log(n_mols, val_loss, val_preds, val_labs,
                               split='val', log=True)

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
                        '/model_mol' + str(i*n_mols)+'.ckpt')
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
                        '/model_mol' + str(i*n_mols)+'.ckpt')

        if epoch % 10 == 0:
            logging.info(f"epoch: {epoch}, "
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
    group_io.add_argument('-p', '--train_path', type=str, default='ugis-00000000.csv',
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
