import time
import os
import argparse
import logging
import concurrent

import dgl
import numpy as np

from tqdm import tqdm
from dgllife.model.model_zoo import MPNNPredictor
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from rdkit import Chem

import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dock2hit import parsing
from dock2hit.model import generate_batch, multi_featurize, multi_validate, collate
from dock2hit.dataloader import load_data
from dock2hit.optimizers import load_optimizer
from dock2hit.tensorboard_logging import Logger
from dock2hit.utils import bash_command, human_len, get_device


def train_and_validate(args, device):

    # load data (val_loader is None if no args.val and args.val_path is None)
    train_loader, val_loader = load_data(args)
    X_train, y_train = train_loader.df[args.smiles_col].values, train_loader.df[args.y_col].to_numpy(
    )

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

    if args.path_to_load_checkpoint is not None:
        checkpoint = torch.load(
            args.path_to_load_checkpoint, map_location=device)
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
    n_steps = 0
    for epoch in range(start_epoch, args.n_epochs):
        mpnn_net.train()

        start_batch = 0

        # start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            for batch_num, (smiles, labels) in tqdm(enumerate(train_loader, start=start_batch),
                                                    initial=start_batch,
                                                    total=len(train_loader),
                                                    miniters=20,
                                                    unit='batch',
                                                    unit_scale=True):

                if batch_num == 0:
                    graphs = multi_featurize(
                        smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer, n_jobs=4)
                    task = pool.submit(
                        multi_featurize, smiles, atom_featurizer, bond_featurizer, 4)
                else:
                    graphs = task.result()  # grab graphs from previous run/batch
                    task = pool.submit(
                        multi_featurize, smiles, atom_featurizer, bond_featurizer, 4)

                batch_data = list(zip(graphs, labels))
                batch_loader = DataLoader(
                    batch_data, batch_size=args.minibatch_size, shuffle=True, collate_fn=collate, drop_last=False, num_workers=4)
                n_mini = 0

                while not task.done():
                    # infinite looping
                    batch_iter = iter(batch_loader)
                    try:
                        bg, labels_mini = next(batch_iter)
                    except StopIteration:
                        batch_iter = iter(batch_loader)
                        bg, labels_mini = next(batch_iter)
                    bg = bg.to(device)
                    labels_mini = labels_mini.to(device)
                    atom_feats = bg.ndata.pop('h').to(device)
                    bond_feats = bg.edata.pop('e').to(device)
                    atom_feats, bond_feats, labels_mini = atom_feats.to(
                        device), bond_feats.to(device), labels_mini.to(device)

                    y_pred = mpnn_net(bg, atom_feats, bond_feats).squeeze()

                    loss = loss_fn(y_pred, labels_mini)

                    optimizer.zero_grad()
                    loss.backward()
                    if 'Felix' in args.optimizer:
                        _, state_lrs = optimizer.step()
                    else:
                        optimizer.step()

                    n_mini += 1
                # print(f'Number of minibatches = {n_mini}, \
                #         Number of molecules = {n_mini*args.minibatch_size} \
                #             ({100*n_mini*args.minibatch_size/args.batch_size:.1f}%)')
                n_steps += n_mini

                # TODO update batch num with minibatch num
                if n_steps % args.steps_per_log == 0 and args.log_dir is not None:

                    # TODO fix y_scaler when using dataloader
                    batch_preds = train_loader.y_scaler.inverse_transform(
                        y_pred.cpu().detach().numpy().reshape(-1, 1))
                    batch_labs = train_loader.y_scaler.inverse_transform(
                        labels_mini.cpu().detach().reshape(-1, 1))

                    # number of mols seen by model
                    if 'Felix' in args.optimizer:
                        logger.log(n_steps, loss.detach().item(), batch_preds, batch_labs,
                                   split='train', state_lrs=state_lrs)
                    else:
                        logger.log(n_steps, loss.detach().item(), batch_preds, batch_labs,
                                   split='train')

                    if val_loader is not None:

                        val_loss, val_preds, val_labs = multi_validate(val_loader=val_loader,
                                                                       model=mpnn_net,
                                                                       atom_featurizer=atom_featurizer,
                                                                       bond_featurizer=bond_featurizer,
                                                                       loss_fn=loss_fn,
                                                                       device=device,
                                                                       y_scaler=train_loader.y_scaler)
                        scheduler.step(val_loss)

                        logger.log(n_steps, val_loss, val_preds, val_labs,
                                   split='val')

                if batch_num % args.batches_per_save == 0 and args.save_dir is not None:
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
                        f'/model_step_{n_steps}.ckpt')
        print(f'Number of steps taken: {n_steps}')


def main():
    parser = argparse.ArgumentParser()
    parsing.add_io_args(parser)
    parsing.add_data_args(parser)
    parsing.add_optim_args(parser)
    parser.add_argument('-time_forward_pass', action='store_true',
                        help='if True, will log the time taken for a forward pass a batch.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = get_device()
    train_and_validate(args, device)


if __name__ == '__main__':
    main()
