import logging
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.dpi": 200})


class Logger:
    def __init__(self, args):

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(args.log_dir+current_time)

        self.writer.add_text('optimizer', str(args.optimizer), 0)
        self.writer.add_text('batch_size', str(args.batch_size), 0)
        self.writer.add_text('lr', str(args.lr), 0)
        if 'Felix' in args.optimizer:
            self.writer.add_text('hypergrad', str(args.hypergrad), 0)

    def log(self, n_mols, loss, batch_preds, batch_labs, split='train', state_lrs=None):
        p = spearmanr(batch_preds, batch_labs)[0]
        rmse = np.sqrt(mean_squared_error(batch_preds, batch_labs))
        r2 = r2_score(batch_preds, batch_labs)

        self.writer.add_scalar(f'loss/{split}', loss.detach().item(),
                               n_mols)
        self.writer.add_scalar(f'{split}/rmse', rmse, n_mols)
        self.writer.add_scalar(f'{split}/rho', p, n_mols)
        self.writer.add_scalar(f'{split}/R2', r2, n_mols)

        if state_lrs is not None:
            self.writer.add_histogram(f'{split}/lrT', state_lrs, n_mols)

        df = pd.DataFrame(
            data={'dock_score': batch_labs.flatten(), 'preds': batch_preds.flatten()})
        plot = sns.jointplot(
            data=df, x='dock_score', y='preds', kind='scatter')
        dot_line = [np.amin(df['dock_score']),
                    np.amax(df['dock_score'])]
        plot.ax_joint.plot(dot_line, dot_line, 'k:')
        plt.xlabel('Dock Scores')
        plt.ylabel('Predictions')
        self.writer.add_figure(f'{split} batch', plot.fig, global_step=n_mols)
        self.writer.flush()

        logging.info(
            f'{split} RMSE: {rmse:.3f}, RHO: {p:.3f}, R2: {r2:.3f}')
