"""
Property prediction using a Message-Passing Neural Network.
"""
import subprocess
import argparse
import logging 

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter

import dgl
from dgllife.model.model_zoo import MPNNPredictor
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.dpi":200})

from utils import UgiLoaderFast, bash_command, Optimizer, MultiProcessOptimizer, synchronize 

def main(rank, dev_id, args):
    if args.num_devices > 1:
        torch.set_num_threads(1)
    if dev_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(dev_id))
        # Set current device
        torch.cuda.set_device(device)

    y_train = pd.read_csv(args.path)['dock_score'].to_numpy()
    len_df = len(pd.read_csv(args.path))

    y_scaler = StandardScaler()

    # Initialise featurisers
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    e_feats = bond_featurizer.feat_size('e')
    n_feats = atom_featurizer.feat_size('h')

    if rank==0:
        print('Training MPNN on {} {}'.format(args.num_devices, device))
        print('Number of features: ', n_feats)

    r2_list = []
    rmse_list = []
    skipped_trials = 0

    writer = SummaryWriter(args.dir+'/runs/'+args.save_name)

    train_ind = range(len_df)
 
    y_scaler = y_scaler.fit(y_train.reshape(-1,1)) 

    train_loader = UgiLoaderFast(args.path, inds=train_ind, batch_size=args.batch_size, shuffle=True, y_scaler=y_scaler) # each device has a different shuffle

    if args.val_path is not None:
        val_ind = range(len(pd.read_csv(args.val_path)))
        val_loader = UgiLoaderFast(args.val_path, inds=val_ind, batch_size=args.batch_size, shuffle=False, y_scaler=y_scaler)

    #mpnn_net = MPNNPredictor(node_in_feats=n_feats,
    #                         edge_in_feats=e_feats,
    #                         num_layer_set2set=6)
    
    mpnn_net = MPNNPredictor(node_in_feats=n_feats,
                 edge_in_feats=e_feats,
                 node_out_feats=32,
                 edge_hidden_feats=32,
                 num_step_message_passing=4,
                 num_step_set2set=2,
                 num_layer_set2set=3)

    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(mpnn_net.parameters(), lr=args.lr)

    if args.num_devices <= 1:
        optimizer = Optimizer(mpnn_net, args.lr, optimizer)
    else:
        optimizer = MultiProcessOptimizer(args.num_devices, mpnn_net, args.lr,
                                          optimizer)

    start_epoch=1
    start_batch=0

    if args.load_name is not None:
       checkpoint = torch.load(args.load_name)
       mpnn_net.load_state_dict(checkpoint['mpnn_state_dict'])
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
       start_epoch = checkpoint['epoch']
       loss = checkpoint['loss']
       start_batch = checkpoint['batch']

    mpnn_net = mpnn_net.to(device)
     
    if rank==0:
        print('Number of parameters: {}'.format(sum(p.numel() for p in mpnn_net.parameters() if p.requires_grad)))

        print('beginning training...')
    for epoch in range(start_epoch, args.n_epochs+1):
        mpnn_net.train()
        epoch_loss = 0
        preds = np.array([None] * len(train_ind)).reshape(-1,1)
        labs = np.array([None] * len(train_ind)).reshape(-1,1)
        n = 0
        for i, (smiles, labels) in tqdm(enumerate(train_loader, start=start_batch), initial=start_batch, total=int(len(train_ind)/args.batch_size)):
            bg = [mol_to_bigraph(Chem.MolFromSmiles(smi), node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for smi in smiles] # generate and batch graphs
            bg = dgl.batch(bg).to(device)
            bg.set_n_initializer(dgl.init.zero_initializer)
            bg.set_e_initializer(dgl.init.zero_initializer)

            atom_feats = bg.ndata.pop('h').to(device)
            bond_feats = bg.edata.pop('e').to(device)
            atom_feats, bond_feats, labels = atom_feats.to(device), bond_feats.to(device), torch.tensor(labels, dtype=torch.float32).to(device)
            y_pred = mpnn_net(bg, atom_feats, bond_feats)

            if args.debug:
                print('label: {}'.format(labels))
                print('y_pred: {}'.format(y_pred))

            loss = loss_fn(y_pred, labels)
            optimizer.backward_and_step(loss)
            epoch_loss += loss.detach().item()

            batch_preds = y_scaler.inverse_transform(y_pred.cpu().detach().numpy())
            batch_labs = y_scaler.inverse_transform(labels.cpu().detach().numpy())

            preds[n: n + len(smiles) ] = batch_preds
            labs[n: n + len(smiles) ] = batch_labs.reshape(len(batch_labs),1)
            n += len(smiles)

            if (i!=0 and i%args.write_batch==0) or epoch%args.write_batch==0:
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

                df = pd.DataFrame(data={'dock_score': batch_labs.flatten(), 'preds': batch_preds.flatten()})
                plot = sns.jointplot(data=df, x='dock_score', y='preds', kind='scatter')
                dot_line = [np.amin(df['dock_score']), np.amax(df['dock_score'])]
                plot.ax_joint.plot(dot_line, dot_line, 'k:')
                plt.xlabel('Dock Scores')
                plt.ylabel('Predictions')
                writer.add_figure('Training batch', plot.fig, global_step = i*args.batch_size + (epoch-1)*len(train_ind))
                writer.flush()

                if args.val_path is not None:
                    mpnn_net.eval()

                    val_preds = np.array([None] * len(val_ind)).reshape(-1,1)
                    val_labs = np.array([None] * len(val_ind)).reshape(-1,1)

                    n=0
                    val_loss = 0
                    for j, (smiles, labels) in enumerate(val_loader):
                        bg = [mol_to_bigraph(Chem.MolFromSmiles(smi), node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer) for smi in smiles] # generate and batch graphs
                        bg = dgl.batch(bg).to(device)
                        bg.set_n_initializer(dgl.init.zero_initializer)
                        bg.set_e_initializer(dgl.init.zero_initializer)

                        atom_feats = bg.ndata.pop('h').to(device)
                        bond_feats = bg.edata.pop('e').to(device)
                        atom_feats, bond_feats, labels = atom_feats.to(device), bond_feats.to(device), torch.tensor(labels, dtype=torch.float32).to(device)
                        y_pred = mpnn_net(bg, atom_feats, bond_feats)

                        loss = loss_fn(y_pred, labels)

                        val_loss += loss.detach().item()
                        batch_preds_val = y_scaler.inverse_transform(y_pred.cpu().detach().numpy())
                        batch_labs_val = y_scaler.inverse_transform(labels.cpu().detach().numpy())
    
                        val_preds[n: n + len(smiles) ] = batch_preds_val
                        val_labs[n: n + len(smiles) ] = batch_labs_val.reshape(len(batch_labs_val),1)
                        n += len(smiles)

                    p = spearmanr(val_preds, val_labs)[0]
                    rmse = np.sqrt(mean_squared_error(val_preds, val_labs))
                    r2 = r2_score(val_preds, val_labs)

                    writer.add_scalar('loss/val', loss.detach().item(), i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.add_scalar('val/rmse', rmse, i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.add_scalar('val/rho', p, i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.add_scalar('val/R2', r2, i*args.batch_size + (epoch-1)*len(train_ind))

                    df = pd.DataFrame(data={'dock_score': val_labs.flatten(), 'preds': val_preds.flatten()})
                    df = df[df['dock_score']<5]

                    plot = sns.jointplot(data=df, x='dock_score', y='preds', kind='scatter')
                    dot_line = [np.amin(df['dock_score']), np.amax(df['dock_score'])]
                    plot.ax_joint.plot(dot_line, dot_line, 'k:')
                    plt.xlabel('Dock Scores')
                    plt.ylabel('Predictions')
                    writer.add_figure('Validation Set', plot.fig, global_step = i*args.batch_size + (epoch-1)*len(train_ind))
                    writer.flush()

                    mpnn_net.train()

            if (i!=0 and i%args.save_batch==0) or epoch%args.write_batch==0:
                try:
                    torch.save({
                                'epoch': epoch,
                                'mpnn_state_dict': mpnn_net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': loss,
                                'batch': i,
                                }, '/rds-d2/user/wjm41/hpc-work/models/'+ args.save_name +
                               '/model_mol' + str(i*args.batch_size + (epoch-1)*len(train_ind))+'.ckpt')
                except FileNotFoundError:
                    cmd = 'mkdir /rds-d2/user/wjm41/hpc-work/models/' + args.save_name
                    bash_command(cmd)
                    torch.save({
                                'epoch': epoch,
                                'mpnn_state_dict': mpnn_net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': loss,
                                'batch': i,
                                }, '/rds-d2/user/wjm41/hpc-work/models/'+ args.save_name +
                               '/model_mol' + str(i*args.batch_size + (epoch-1)*len(train_ind))+'.ckpt')

        synchronize(args.num_devices)

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
                            }, '/rds-d2/user/wjm41/hpc-work/models/'+ args.save_name +
                           '/model_epoch' + str(epoch)+'.ckpt')
            except FileNotFoundError:
                cmd = 'mkdir /rds-d2/user/wjm41/hpc-work/models/' + args.save_name
                bash_command(cmd)
                torch.save({
                            'epoch': epoch,
                            'mpnn_state_dict': mpnn_net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': loss,
                            'batch': 0,
                            }, '/rds-d2/user/wjm41/hpc-work/models/'+ args.save_name +
                           '/model_epoch' + str(epoch)+'.ckpt')
            print(f'Validation RMSE: {rmse:.3f}, RHO: {p:.3f}, R2: {r2:.3f}')
            logging.warning(f'Validation RMSE: {rmse:.3f}, RHO: {p:.3f}, R2: {r2:.3f}')
            
    
    torch.save({
                'epoch': epoch,
                'mpnn_state_dict': mpnn_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'batch': 0,
                }, '/rds-d2/user/wjm41/hpc-work/models/'+ args.save_name +
               '/model_epoch_final.ckpt')

def run(rank, dev_id, args):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=args.master_ip, master_port=args.master_port)
    torch.distributed.init_process_group(backend="nccl",
                                         init_method=dist_init_method,
                                         world_size=args.num_devices,
                                         rank=rank)
    assert torch.distributed.get_rank() == rank
    main(rank, dev_id, args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', default='0', type=str,
                        help='To use multi-gpu training, '
                             'pass multiple gpu ids with --gpus id1,id2,...')
    parser.add_argument('-dir', '--dir', type=str, default='/rds-d2/user/wjm41/hpc-work/datasets/Ugis/scripts',
                        help='Path to the data.csv file.')
    parser.add_argument('-p', '--path', type=str, default='ugis-00000000.csv',
                        help='Path to the data.csv file.')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=64,
                        help='int specifying batch_size for training and evaluations')
    parser.add_argument('-write_batch', '--write_batch', type=int, default=100,
                        help='int specifying number of steps per tensorboard write')
    parser.add_argument('-save_batch', '--save_batch', type=int, default=500,
                        help='int specifying number of batches per model save')
    parser.add_argument('-n_epochs', '--n_epochs', type=int, default=10,
                        help='int specifying number of random train/test splits to use')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3,
                        help='float specifying learning rate used during training.')
    parser.add_argument('-val_path', default=None,
                        help='if not None, logs validation loss')
    parser.add_argument('-debug', action='store_true',
                        help='whether or not to print predictions and model weight gradients')
    parser.add_argument('-save_name', '--save_name', type=str, default='ugi-pretrained',
                        help='name for directory containing saved model params and tensorboard logs')
    parser.add_argument('-load_name', '--load_name', default=None,
                        help='name for directory containing saved model params and tensorboard logs')
    parser.add_argument('--master-ip', type=str, default='127.0.0.1',
                        help='master ip address')
    parser.add_argument('--master-port', type=str, default='12345',
                        help='master port')

    args = parser.parse_args()

    devices = list(map(int, args.gpus.split(',')))
    args.num_devices = len(devices)

    if len(devices) == 1:
        device_id = devices[0] if torch.cuda.is_available() else -1
        main(0, device_id, args)
    else:
        # With multi-gpu training, the batch size increases and we need to
        # increase learning rate accordingly.
        args.lr = args.lr * args.num_devices
        mp = torch.multiprocessing.get_context('spawn')
        procs = []
        for id, device_id in enumerate(devices):
            print('Preparing for gpu {:d}/{:d}'.format(id + 1, args.num_devices))
            procs.append(mp.Process(target=run, args=(
                id, device_id, args), daemon=True))
            procs[-1].start()
        for p in procs:
            p.join()
