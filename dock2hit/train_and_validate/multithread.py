from typing import Callable, Dict, Protocol
import time

import os
from argparse import ArgumentParser, Namespace
import logging
import concurrent
from dataclasses import dataclass

import dgl
import numpy as np

from tqdm import tqdm
from dgllife.model.model_zoo import MPNNPredictor
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from rdkit import Chem

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from dock2hit import parsing
from dock2hit.model import generate_batch, multi_featurize, multi_validate, collate
from dock2hit.dataloader import load_data
from dock2hit.optimizers import load_optimizer
from dock2hit.tensorboard_logging import Logger
from dock2hit.utils import bash_command, human_len, get_device


class IsDataclass(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: Dict


def log_training_minibatch(logger: Logger,
                           n_steps: int,
                           model_config: IsDataclass,
                           y_pred,
                           y_true,
                           loss,
                           rescale=False,
                           lr_histograms=None):

    if rescale:
        batch_preds = model_config.y_scaler.inverse_transform(
            y_pred.cpu().detach().numpy().reshape(-1, 1))
        batch_labs = model_config.y_scaler.inverse_transform(
            y_true.cpu().detach().reshape(-1, 1))
    else:
        batch_preds = y_pred.cpu().detach().numpy().reshape(-1, 1)
        batch_labs = y_true.cpu().detach().numpy().reshape(-1, 1)

    # number of mols seen by model
    if lr_histograms is not None:
        logger.log(n_steps, loss.detach().item(), batch_preds, batch_labs,
                   split='train', state_lrs=lr_histograms)
        logger.plot_predictions(n_steps, batch_preds,
                                batch_labs, split='train', rescale=rescale)
    else:
        logger.log(n_steps, loss.detach().item(), batch_preds, batch_labs,
                   split='train', rescale=False)
        logger.plot_predictions(n_steps, batch_preds,
                                batch_labs, split='train', rescale=rescale)
    return


def validate_and_log(logger: Logger, n_steps: int, model_config: IsDataclass, validation_function: Callable):

    val_loss, val_preds, val_labs = validation_function(val_loader=model_config.val_loader,
                                                        model=model_config.model,
                                                        atom_featurizer=model_config.atom_featurizer,
                                                        bond_featurizer=model_config.bond_featurizer,
                                                        loss_fn=model_config.loss_fn,
                                                        device=model_config.device,
                                                        y_scaler=model_config.y_scaler)
    model_config.scheduler.step(val_loss)

    logger.log(n_steps, val_loss, val_preds, val_labs,
               split='val')
    logger.plot_predictions(n_steps, val_preds,
                            val_labs, split='val', rescale=True)
    return


def run_minibatch(args: Namespace, batched_graphs, y_true, model_config: IsDataclass, logger: Logger, n_steps: int):
    model = model_config.model
    optimizer = model_config.optimizer
    device = model_config.device
    loss_fn = model_config.loss_fn

    batched_graphs = batched_graphs.to(device)
    y_true = y_true.to(device)
    atom_feats = batched_graphs.ndata.pop('h').to(device)
    bond_feats = batched_graphs.edata.pop('e').to(device)
    atom_feats, bond_feats, y_true = atom_feats.to(
        device), bond_feats.to(device), y_true.to(device)

    y_pred = model(
        batched_graphs, atom_feats, bond_feats).squeeze()

    loss = loss_fn(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    if 'Felix' in args.optimizer:
        _, state_lrs = optimizer.step()
    else:
        optimizer.step()

    model_config.model = model
    model_config.optimizer = optimizer

    if n_steps % args.steps_per_log == 0 and args.log_dir is not None:
        if 'Felix' in args.optimizer:
            log_training_minibatch(
                logger, n_steps, model_config, y_pred, y_true, loss, state_lrs)
        else:
            log_training_minibatch(
                logger, n_steps, model_config, y_pred, y_true, loss)
        # logger.log_gradients(n_steps, model_config)
        # logger.log_weights(n_steps, model_config)

        if model_config.val_loader is not None:
            validate_and_log(
                logger, n_steps, model_config, multi_featurize)
    return


@dataclass
class ModelConfig():
    """Class for storing training run configurations"""

    def __init__(self, args: Namespace):
        device = get_device()

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

        mpnn_net.to(device)
        print(f'mpnn_net id: {id(mpnn_net)}')

        optimizer, scheduler = load_optimizer(args, mpnn_net)

        logging.info('Number of model parameters: {}'.format(sum(p.numel()
                                                                 for p in mpnn_net.parameters() if p.requires_grad)))
        logging.info(f'Number of node features: {n_feats}')
        logging.info(f'Number of edge features: {e_feats}')

        # load loss function
        loss_fn = MSELoss()

        start_epoch = 0
        start_batch = 0

        self.model = mpnn_net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.loss_fn = loss_fn
        self.start_epoch = start_epoch
        self.start_batch = start_batch
        self.device = device

        self.create_data_loaders(args)
        if args.path_to_load_checkpoint is not None:
            self.load_checkpoint(args.path_to_load_checkpoint)

    def create_data_loaders(self, args: Namespace):
        # val_loader is None if no args.val and args.val_path is None
        train_loader, val_loader = load_data(args)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.y_scaler = train_loader.y_scaler

    def load_checkpoint(self, path_to_checkpoint: str):
        checkpoint = torch.load(
            path_to_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['mpnn_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.start_batch = checkpoint['batch']


def save_checkpoint(path_to_save_dir: str, epoch: int, n_steps: int, model_config: IsDataclass, batch_num: int):
    if not os.path.isdir(path_to_save_dir):
        os.mkdir(path_to_save_dir)

    torch.save({
        'epoch': epoch,
        'mpnn_state_dict': model_config.model.state_dict(),
        'optimizer_state_dict': model_config.optimizer.state_dict(),
        'scheduler_state_dict': model_config.scheduler.state_dict(),
        'batch': batch_num,
    }, path_to_save_dir +
        f'/model_step_{n_steps}.ckpt')
    return


def train_and_validate(args: Namespace):

    model_config = ModelConfig(args)

    logging.info('Beginning training...')
    if args.log_dir is not None:
        logger = Logger(args)
    n_steps = 0
    for epoch in range(model_config.start_epoch, args.n_epochs):
        model_config.model.train()

        start_batch = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            for batch_num, (smiles, labels) in tqdm(enumerate(model_config.train_loader, start=start_batch),
                                                    initial=start_batch,
                                                    total=len(
                                                        model_config.train_loader),
                                                    miniters=20,
                                                    unit='batch',
                                                    unit_scale=True):

                if batch_num == 0:
                    graphs = multi_featurize(
                        smiles, node_featurizer=model_config.atom_featurizer, edge_featurizer=model_config.bond_featurizer, n_jobs=4)
                    task = pool.submit(
                        multi_featurize, smiles, model_config.atom_featurizer, model_config.bond_featurizer, 4)
                else:
                    graphs = task.result()  # grab graphs from previous run/batch
                    task = pool.submit(
                        multi_featurize, smiles, model_config.atom_featurizer, model_config.bond_featurizer, 4)

                batch_data = list(zip(graphs, labels))
                batch_loader = DataLoader(
                    batch_data, batch_size=args.minibatch_size, shuffle=True, collate_fn=collate, drop_last=False, num_workers=4)

                # infinite looping
                batch_iter = iter(batch_loader)

                while not task.done():
                    try:
                        batched_graphs, labels_mini = next(batch_iter)
                    except StopIteration:
                        batch_iter = iter(batch_loader)
                        batched_graphs, labels_mini = next(batch_iter)
                    run_minibatch(args, batched_graphs, labels_mini,
                                  model_config, logger, n_steps)
                    n_steps += 1

                if batch_num % args.batches_per_save == 0 and args.save_dir is not None:
                    save_checkpoint(args.save_dir, epoch,
                                    n_steps, model_config, batch_num)
    print(f'Model training complete! Number of steps taken: {n_steps}')


def main():
    parser = ArgumentParser()
    parsing.add_io_args(parser)
    parsing.add_data_args(parser)
    parsing.add_optim_args(parser)
    parser.add_argument('-time_forward_pass', action='store_true',
                        help='if True, will log the time taken for a forward pass a batch.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train_and_validate(args)
    # TODO - append args to model config


if __name__ == '__main__':
    main()
