import pytest
from unittest.mock import MagicMock
import argparse

import numpy as np

from dock2hit.tensorboard_logging import Logger
from dock2hit.parsing import add_io_args, add_data_args, add_optim_args


def test_logging():

    mock_args = MagicMock()
    mock_args.path_to_train_data = 'test/test_data/HIV.csv',
    mock_args.log_dir = 'test/test_runs/test_logger/'
    mock_args.y_col = 'activity'
    mock_args.n_epochs = 10
    mock_args.lr = 1e-3
    mock_args.optimizer = 'Adam'
    mock_args.batch_size = 32

    logger = Logger(mock_args)
    mock_mols = 0
    mock_loss = 1.5
    mock_preds = np.arange(100) + np.random.normal(size=100)
    mock_labs = np.arange(100)
    mock_split = 'train'
    mock_title = 'Testing tensorboard logger on random numbers'
    mock_xlabel = 'np.arange'
    mock_ylabel = 'np.arange + np.normal()'
    logger.log(n_mols=mock_mols,
               loss=mock_loss,
               batch_preds=mock_preds,
               batch_labs=mock_labs,
               split=mock_split,
               title=mock_title,
               xlabel=mock_xlabel,
               ylabel=mock_ylabel)


def test_log_minibatch():
    pass
