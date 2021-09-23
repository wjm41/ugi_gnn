import subprocess

import pandas as pd

from torch import nn
import torch.distributed as dist

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
            self.df = self.df.sample(frac=1).reset_index(drop=True) # non-fixed random state for multi-GPU training

        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        df_batch = self.df.iloc[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        #return batch
        return df_batch['smiles'].values, self.y_scaler.transform(df_batch['dock_score'].to_numpy().reshape(-1,1))


class Optimizer(nn.Module):
    """Wrapper for optimization
    Parameters
    ----------
    model : nn.Module
        Model being trained
    lr : float
        Initial learning rate
    optimizer : torch.optim.Optimizer
        model optimizer
    num_accum_times : int
        Number of times for accumulating gradients
    max_grad_norm : float or None
        If not None, gradient clipping will be performed
    """
    def __init__(self, model, lr, optimizer, num_accum_times=1, max_grad_norm=None):
        super(Optimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.step_count = 0
        self.num_accum_times = num_accum_times
        self.max_grad_norm = max_grad_norm
        self._reset()

    def _reset(self):
        self.optimizer.zero_grad()

    def _clip_grad_norm(self):
        grad_norm = None
        if self.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(self.model.parameters(),
                                        self.max_grad_norm)
        return grad_norm

    def backward_and_step(self, loss):
        """Backward and update model.
        Parameters
        ----------
        loss : torch.tensor consisting of a float only
        Returns
        -------
        grad_norm : float
            Gradient norm. If self.max_grad_norm is None, None will be returned.
        """
        self.step_count += 1
        loss.backward()
        if self.step_count % self.num_accum_times == 0:
            grad_norm = self._clip_grad_norm()
            self.optimizer.step()
            self._reset()

            return grad_norm
        else:
            return 0

    def decay_lr(self, decay_rate):
        """Decay learning rate.
        Parameters
        ----------
        decay_rate : float
            Multiply the current learning rate by the decay_rate
        """
        self.lr *= decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

class MultiProcessOptimizer(Optimizer):
    """Wrapper for optimization with multiprocess
    Parameters
    ----------
    n_processes : int
        Number of processes used
    model : nn.Module
        Model being trained
    lr : float
        Initial learning rate
    optimizer : torch.optim.Optimizer
        model optimizer
    max_grad_norm : float or None
        If not None, gradient clipping will be performed.
    """
    def __init__(self, n_processes, model, lr, optimizer, max_grad_norm=None):
        super(MultiProcessOptimizer, self).__init__(lr=lr, model=model, optimizer=optimizer,
                                                    max_grad_norm=max_grad_norm)
        self.n_processes = n_processes

    def _sync_gradient(self):
        """Average gradients across all subprocesses."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.requires_grad and p.grad is not None:
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= self.n_processes

    def backward_and_step(self, loss):
        """Backward and update model.
        Parameters
        ----------
        loss : torch.tensor consisting of a float only
        Returns
        -------
        grad_norm : float
            Gradient norm. If self.max_grad_norm is None, None will be returned.
        """
        loss.backward()
        self._sync_gradient()
        grad_norm = self._clip_grad_norm()
        self.optimizer.step()
        self._reset()

        return grad_norm

def synchronize(num_gpus):
    """Synchronize all processes for multi-gpu training.
    Parameters
    ----------
    num_gpus : int
        Number of gpus used
    """
    if num_gpus > 1:
        dist.barrier()
