from joblib import Parallel, delayed, cpu_count

import numpy as np

import dgl
from dgllife.utils import smiles_to_bigraph

import torch
from torch.utils.data import DataLoader


def multi_featurize(smiles, node_featurizer, edge_featurizer, n_jobs=4):

    graphs = pmap(smiles_to_bigraph,
                  smiles,
                  node_featurizer=node_featurizer,
                  edge_featurizer=edge_featurizer,
                  n_jobs=n_jobs
                  )

    return graphs


def generate_batch(smiles, atom_featurizer, bond_featurizer, device):
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


def multi_validate(val_loader, model, atom_featurizer, bond_featurizer, loss_fn, device, y_scaler=None):
    model.eval()
    val_preds = []
    val_labs = []
    val_loss = 0

    for batch_num, (smiles, labels) in enumerate(val_loader):
        val_graphs = multi_featurize(
            smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer, n_jobs=4)
        val_batch = list(zip(val_graphs, labels))
        val_batch_loader = DataLoader(
            val_batch, batch_size=len(val_graphs), shuffle=True, collate_fn=collate, drop_last=False, num_workers=4)

        for bg, labels_mini in val_batch_loader:
            bg = bg.to(device)
            labels_mini = labels_mini.to(device)
            atom_feats = bg.ndata.pop('h').to(device)
            bond_feats = bg.edata.pop('e').to(device)
            atom_feats, bond_feats, labels_mini = atom_feats.to(
                device), bond_feats.to(device), labels_mini.to(device)
            with torch.no_grad():
                y_pred = model(bg, atom_feats, bond_feats).squeeze()

            loss = loss_fn(y_pred, labels_mini)
            val_loss += loss.detach().item()

            if y_scaler is not None:
                batch_preds_val = y_scaler.inverse_transform(
                    y_pred.cpu().detach().numpy().reshape(-1, 1))
                batch_labs_val = y_scaler.inverse_transform(
                    labels_mini.cpu().detach().numpy().reshape(-1, 1))
            else:
                batch_preds_val = y_pred.cpu().detach().numpy()
                batch_labs_val = labels_mini.cpu().detach().numpy()

            val_preds.append(batch_preds_val)
            val_labs.append(batch_labs_val)

    val_preds = np.vstack(val_preds)
    val_labs = np.vstack(val_labs)
    model.train()
    return val_loss, val_preds, val_labs


def pmap(pickleable_fn, data, n_jobs=None, verbose=0, **kwargs):
    """Parallel map using joblib.
    Parameters
    ----------
    pickleable_fn : callable
        Function to map over data.
    data : iterable
        Data over which we want to parallelize the function call.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. By default, it is one less than
        the number of CPUs.
    verbose: int, optional
        The verbosity level. If nonzero, the function prints the progress messages.
        The frequency of the messages increases with the verbosity level. If above 10,
        it reports all iterations. If above 50, it sends the output to stdout.
    kwargs
        Additional arguments for :attr:`pickleable_fn`.
    Returns
    -------
    list
        The i-th element of the list corresponds to the output of applying
        :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = cpu_count() - 1

    return Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(pickleable_fn)(d, **kwargs) for d in data
    )


# Collate Function for Dataloader
def collate(sample):
    graphs, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels, dtype=torch.float32)
