from __future__ import annotations
import argparse
from typing import Literal
import logging
from datetime import datetime


import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
            n_mols: int,
            loss: float,
            batch_preds: np.ndarray,
            batch_labs: np.ndarray,
            split: Literal['train', 'val'] = 'train',
            state_lrs: np.ndarray = None,
            title: str = None,
            xlabel: str = 'True Values',
            ylabel: str = 'Predicted Values'):

        p = spearmanr(batch_preds, batch_labs)[0]
        rmse = np.sqrt(mean_squared_error(batch_preds, batch_labs))
        r2 = r2_score(batch_preds, batch_labs)

        self.writer.add_scalar(f'loss/{split}', loss,
                               n_mols)
        self.writer.add_scalar(f'{split}/rmse', rmse, n_mols)
        self.writer.add_scalar(f'{split}/rho', p, n_mols)
        self.writer.add_scalar(f'{split}/R2', r2, n_mols)

        if state_lrs is not None:
            self.writer.add_histogram(f'{split}/lrT', state_lrs, n_mols)

        plot = sns.jointplot(x=batch_labs.flatten(),
                             y=batch_preds.flatten(),
                             kind='scatter',
                             xlim=(-60, 60),
                             ylim=(-60, 60))
        dot_line = [np.amin(batch_labs.flatten()),
                    np.amax(batch_preds.flatten())]
        plot.ax_joint.plot(dot_line, dot_line, 'k:')
        if title is None:
            title = f'Model predictions on {split} set'

        plot.ax_joint.set(xlabel=xlabel, ylabel=ylabel)
        plot.fig.suptitle(title)
        plot.fig.tight_layout()
        self.writer.add_figure(f'{split} minibatch',
                               plot.fig, global_step=n_mols)
        self.writer.flush()

        logging.info(
            f'{split} RMSE: {rmse:.3f}, RHO: {p:.3f}, R2: {r2:.3f}')
    
# TODO TESTS! No need for df
