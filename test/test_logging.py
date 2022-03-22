import pytest
from unittest.mock import MagicMock
import argparse

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
