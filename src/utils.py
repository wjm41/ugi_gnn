import logging
from math import log, floor
import subprocess

import dgl
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from joblib import Parallel, delayed, cpu_count

import torch
# import torch.distributed as dist


def human_len(input, byte=False):
    """Given a number or list like, returns the size/length of the object in human-readable form

    As an example, human_len(2048, byte=True) = 2KB ; human_len(np.ones(14000)) = 14K;

    Args:
        input: A number or an object with the __len__ method.
        byte: Whether or not to treat the length in byte form.

    Returns:
        length_string (string): Human-readable string of the object size.
    """
    try:
        input = input.__len__()
    except AttributeError:
        pass

    if byte:
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        k = 1024.0
    else:
        units = ['', 'K', 'M', 'G', 'T', 'P']
        k = 1000.0
    magnitude = int(floor(log(input, k)))
    length_string = '%.2f%s' % (input / k**magnitude, units[magnitude])
    return length_string


def bash_command(cmd: str):
    """Utility script for running a bash command from a python file.

    Args:
        cmd (str): bash command to-be-run.
    """
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    p.communicate()


def get_device() -> str:
    """Detects CUDA availability and returns the appropriate torch device.

    Returns:
        device (str): device to-be used in torch
    """
    if torch.cuda.is_available():
        logging.info(f'using GPU: {torch.cuda.get_device_name()}')
        device = 'cuda'
    else:
        logging.info('No GPU found, using CPU')
        device = 'cpu'
    return device


def pmap(pickleable_fn, data, n_jobs=None, verbose=1, **kwargs):
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


def multi_featurize(smiles, node_featurizer, edge_featurizer, n_jobs):

    # turn off logging
    graphs = pmap(smiles_to_bigraph,
                  smiles,
                  node_featurizer=node_featurizer,
                  edge_featurizer=edge_featurizer,
                  n_jobs=n_jobs
                  )

    return graphs
# class Optimizer(nn.Module):
#     """Wrapper for optimization
#     Parameters
#     ----------
#     model : nn.Module
#         Model being trained
#     lr : float
#         Initial learning rate
#     optimizer : torch.optim.Optimizer
#         model optimizer
#     num_accum_times : int
#         Number of times for accumulating gradients
#     max_grad_norm : float or None
#         If not None, gradient clipping will be performed
#     """

#     def __init__(self, model, lr, optimizer, num_accum_times=1, max_grad_norm=None):
#         super(Optimizer, self).__init__()
#         self.model = model
#         self.lr = lr
#         self.optimizer = optimizer
#         self.step_count = 0
#         self.num_accum_times = num_accum_times
#         self.max_grad_norm = max_grad_norm
#         self._reset()

#     def _reset(self):
#         self.optimizer.zero_grad()

#     def _clip_grad_norm(self):
#         grad_norm = None
#         if self.max_grad_norm is not None:
#             grad_norm = clip_grad_norm_(self.model.parameters(),
#                                         self.max_grad_norm)
#         return grad_norm

#     def backward_and_step(self, loss):
#         """Backward and update model.
#         Parameters
#         ----------
#         loss : torch.tensor consisting of a float only
#         Returns
#         -------
#         grad_norm : float
#             Gradient norm. If self.max_grad_norm is None, None will be returned.
#         """
#         self.step_count += 1
#         loss.backward()
#         if self.step_count % self.num_accum_times == 0:
#             grad_norm = self._clip_grad_norm()
#             self.optimizer.step()
#             self._reset()

#             return grad_norm
#         else:
#             return 0

#     def decay_lr(self, decay_rate):
#         """Decay learning rate.
#         Parameters
#         ----------
#         decay_rate : float
#             Multiply the current learning rate by the decay_rate
#         """
#         self.lr *= decay_rate
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = self.lr


# class MultiProcessOptimizer(Optimizer):
#     """Wrapper for optimization with multiprocess
#     Parameters
#     ----------
#     n_processes : int
#         Number of processes used
#     model : nn.Module
#         Model being trained
#     lr : float
#         Initial learning rate
#     optimizer : torch.optim.Optimizer
#         model optimizer
#     max_grad_norm : float or None
#         If not None, gradient clipping will be performed.
#     """

#     def __init__(self, n_processes, model, lr, optimizer, max_grad_norm=None):
#         super(MultiProcessOptimizer, self).__init__(lr=lr, model=model, optimizer=optimizer,
#                                                     max_grad_norm=max_grad_norm)
#         self.n_processes = n_processes

#     def _sync_gradient(self):
#         """Average gradients across all subprocesses."""
#         for param_group in self.optimizer.param_groups:
#             for p in param_group['params']:
#                 if p.requires_grad and p.grad is not None:
#                     dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
#                     p.grad.data /= self.n_processes

#     def backward_and_step(self, loss):
#         """Backward and update model.
#         Parameters
#         ----------
#         loss : torch.tensor consisting of a float only
#         Returns
#         -------
#         grad_norm : float
#             Gradient norm. If self.max_grad_norm is None, None will be returned.
#         """
#         loss.backward()
#         self._sync_gradient()
#         grad_norm = self._clip_grad_norm()
#         self.optimizer.step()
#         self._reset()

#         return grad_norm


# def synchronize(num_gpus):
#     """Synchronize all processes for multi-gpu training.
#     Parameters
#     ----------
#     num_gpus : int
#         Number of gpus used
#     """
#     if num_gpus > 1:
#         dist.barrier()
