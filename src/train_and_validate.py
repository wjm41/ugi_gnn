"""
Property prediction using a Message-Passing Neural Network.
"""
import os
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

from .dataloader import load_data
from .optimizers import load_optimizer
from .tensorboard_logging import Logger
from .utils import bash_command, human_len, get_device
from . import parsing


def generate_batch(smiles, atom_featurizer, bond_featurizer, device):
    # TODO multithread graph featurizer
    bg = [smiles_to_bigraph(smi, node_featurizer=atom_featurizer,
                            edge_featurizer=bond_featurizer) for smi in smiles]  # generate and batch graphs
    bg = dgl.batch(bg)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    atom_feats = bg.ndata.pop('h')
    bond_feats = bg.edata.pop('e')
    return bg.to(device), atom_feats.to(device), bond_feats.to(device)


def validate(val_loader, model, atom_featurizer, bond_featurizer, loss_fn, device, y_scaler=None):
    model.eval()
    val_preds = np.empty((len(val_loader), 1))
    val_labs = np.empty((len(val_loader), 1))
    m = 0
    val_loss = 0

    for smiles, labels in val_loader:
        bg, atom_feats, bond_feats = generate_batch(
            smiles, atom_featurizer, bond_featurizer, device)
        with torch.no_grad():
            y_pred = model(bg, atom_feats, bond_feats)

        loss = loss_fn(y_pred, labels)
        val_loss += loss.detach().item()

        if y_scaler is not None:
            batch_preds_val = y_scaler.inverse_transform(
                y_pred.cpu().detach().numpy().reshape(-1, 1))
            batch_labs_val = y_scaler.inverse_transform(
                labels.cpu().detach().numpy().reshape(-1, 1))
        else:
            batch_preds_val = y_pred.cpu().detach().numpy()
            batch_labs_val = labels.cpu().detach().numpy()

        val_preds[m: m + len(smiles)] = batch_preds_val
        val_labs[m: m + len(smiles)] = batch_labs_val
        m += len(smiles)

    model.train()
    return val_loss, val_preds, val_labs


def main(args, device):

    # load data
    if args.val or args.val_path is not None:
        train_loader, val_loader = load_data(args)
    else:
        train_loader = load_data(args)

    # Initialise featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')

    # set up model
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

    logging.info('Number of model parameters: {}'.format(sum(p.numel()
                                                             for p in mpnn_net.parameters() if p.requires_grad)))
    logging.info(f'Number of node features: {n_feats}')
    logging.info(f'Number of edge features: {e_feats}')

    optimizer = load_optimizer(args, mpnn_net)

    scheduler = ReduceLROnPlateau(optimizer, 'min')

    if args.load_name is not None:
        checkpoint = torch.load(args.load_name, map_location=device)
        mpnn_net.load_state_dict(checkpoint['mpnn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']
    else:
        start_epoch = 0
        start_batch = 0

    loss_fn = MSELoss()

    logging.info('beginning training...')
    if args.log_dir is not None:
        logger = Logger(args)
    for epoch in range(start_epoch, args.n_epochs):
        mpnn_net.train()

        n = 0
        start_batch = 0

        for batch_num, (smiles, labels) in tqdm(enumerate(train_loader, start=start_batch),
                                                initial=start_batch,
                                                total=train_loader.n_batches,
                                                miniters=10000,
                                                unit='batch',
                                                unit_scale=True):

            bg, atom_feats, bond_feats = generate_batch(
                smiles, atom_featurizer, bond_featurizer, device)
            y_pred = mpnn_net(bg, atom_feats, bond_feats)

            labels = torch.tensor(labels, dtype=torch.float32).to(device)
            loss = loss_fn(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            if 'Felix' in args.optimizer:
                _, state_lrs = optimizer.step()
            else:
                optimizer.step()

            n += len(smiles)
            n_mols = batch_num*args.batch_size + epoch*len(train_loader)

            if batch_num % args.log_batch == 0 and args.log_dir is not None:

                batch_preds = train_loader.y_scaler.inverse_transform(
                    y_pred.cpu().detach().numpy().reshape(-1, 1))
                batch_labs = train_loader.y_scaler.inverse_transform(
                    labels.cpu().detach().numpy().reshape(-1, 1))

                # number of mols seen by model
                if 'Felix' in args.optimizer:
                    logger.log(n_mols, loss, batch_preds, batch_labs,
                               split='train', state_lrs=state_lrs)
                else:
                    logger.log(n_mols, loss, batch_preds, batch_labs,
                               split='train')

                if val_loader is not None:

                    val_loss, val_preds, val_labs = validate(val_loader=val_loader,
                                                             model=mpnn_net,
                                                             atom_featurizer=atom_featurizer,
                                                             bond_featurizer=bond_featurizer,
                                                             loss_fn=loss_fn,
                                                             device=device,
                                                             y_scaler=train_loader.y_scaler)
                    scheduler.step(val_loss)

                    logger.log(n_mols, val_loss, val_preds, val_labs,
                               split='val', log=True)

            if batch_num % args.save_batch == 0 and args.save_dir is not None:
                if not os.path.isdir(args.save_dir):
                    cmd = 'mkdir ' + args.save_dir
                    bash_command(cmd)
                torch.save({
                    'epoch': epoch,
                    'mpnn_state_dict': mpnn_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'batch': batch_num,
                }, args.save_dir +
                    f'/model_mol_{n_mols}.ckpt')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = parsing.add_io_args(parser)
    parser = parsing.add_data_args(parser)
    parser = parsing.add_optim_args(parser)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = get_device()
    main(args, device)
