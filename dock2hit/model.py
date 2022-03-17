import dgl
import numpy as np

from dgllife.utils import smiles_to_bigraph


import torch


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
