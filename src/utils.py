import logging
from math import log, floor
import subprocess

import torch
# import torch.distributed as dist


def human_len(input):
    """Given a number or list like, returns the size/length of the object in human-readable form

    As an example, human_len(12500) = 12.5K ; human_len(np.ones(14000)) = 14K

    Args:
        input: A number or an object with the __len__ method.

    Returns:
        length_string (string): Human-readable string of the object size.
    """
    try:
        input = input.__len__()
    except AttributeError:
        pass

    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(input, k)))
    length_string = '%.2f%s' % (input / k**magnitude, units[magnitude])
    return length_string


def bash_command(cmd):
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    p.communicate()


def get_device():
    if torch.cuda.is_available():
        logging.info(f'using GPU: {torch.cuda.get_device_name()}')
        device = 'cuda'
    else:
        logging.info('No GPU found, using CPU')
        device = 'cpu'
    return device

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
