from __future__ import annotations
import argparse
from typing import Literal
import logging
from datetime import datetime


import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
sns.set(rc={"figure.dpi": 200})


class Logger:
    def __init__(self, args: argparse.Namespace):

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(f'{args.log_dir}/{current_time}')

        self.writer.add_text('optimizer', str(args.optimizer), 0)
        self.writer.add_text('batch_size', str(args.batch_size), 0)
        self.writer.add_text('lr', str(args.lr), 0)
        if 'Felix' in args.optimizer:
            self.writer.add_text('hypergrad', str(args.hypergrad), 0)

    def log(self,
            step: int,
            loss: float,
            batch_preds: np.ndarray,
            batch_labs: np.ndarray,
            split: Literal['train', 'val'] = 'train',
            rescale=False,
            state_lrs: np.ndarray = None,):

        p = spearmanr(batch_preds, batch_labs)[0]
        rmse = np.sqrt(mean_squared_error(batch_preds, batch_labs))
        r2 = r2_score(batch_preds, batch_labs)

        self.writer.add_scalar(f'loss/{split}', loss,
                               step)
        self.writer.add_scalar(f'{split}/rmse', rmse, step)
        self.writer.add_scalar(f'{split}/rho', p, step)
        self.writer.add_scalar(f'{split}/R2', r2, step)

        if state_lrs is not None:
            self.writer.add_histogram(f'{split}/lrT', state_lrs, step)

        self.plot_predictions(
            step, batch_preds, batch_labs, split, rescale=rescale)
        self.writer.flush()

        logging.info(
            f'{split} RMSE: {rmse:.3f}, RHO: {p:.3f}, R2: {r2:.3f}')

    def plot_predictions(self,
                         step: float,
                         y_pred,
                         y_true,
                         split: Literal['train', 'val'] = 'train',
                         title: str = None,
                         xlabel: str = 'True Values',
                         ylabel: str = 'Predicted Values',
                         rescale=False):

        plot = sns.jointplot(x=y_true.flatten(),
                             y=y_pred.flatten(),
                             kind='scatter',
                             xlim=(-60, 0),
                             ylim=(-60, 0))
        dot_line = [np.amin(y_true.flatten()),
                    np.amax(y_true.flatten())]
        plot.ax_joint.plot(dot_line, dot_line, 'k:')
        if title is None:
            title = f'Model predictions on {split} set'

        plot.ax_joint.set(xlabel=xlabel,
                          ylabel=ylabel)
        if rescale:
            plot.ax_joint.set(xlim=(-60, 0),
                              ylim=(-60, 0))
        else:
            plot.ax_joint.set(xlim=(-5, 5),
                              ylim=(-5, 5))
        plot.fig.suptitle(title)
        plot.fig.tight_layout()
        self.writer.add_figure(f'{split} minibatch',
                               plot.fig, global_step=step)
        return

    def log_gradients(self, step: int, model_config):
        for name, module in model_config.model.named_children():
            norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in module.parameters()]), 2)
            self.writer.add_scalar(f'gradient/{name}', norm, step)
        return

    def log_weights(self, step: int, model_config):
        for name, module in model_config.model.named_children():
            for pname, p in module.named_parameters():
                self.writer.add_histogram(
                    f'weight/{pname}', p, step)
        return
