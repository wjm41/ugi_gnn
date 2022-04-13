import pytest
from unittest.mock import MagicMock
import argparse

import numpy as np

from dock2hit.tensorboard_logging import Logger
from dock2hit.parsing import add_io_args, add_data_args, add_optim_args
from dock2hit.train_and_validate.multithread import ModelConfig


@pytest.fixture()
def mock_args():
    mock_args = MagicMock()
    mock_args.batch_size = 32
    mock_args.n_epochs = 1
    mock_args.lr = 1e-3
    mock_args.optimizer = 'Adam'
    mock_args.path_to_train_data = 'test/test_data/HIV.csv'
    mock_args.log_dir = 'test/test_runs/test_logger/'
    mock_args.path_to_external_val = None
    mock_args.random_train_val_split = False
    mock_args.path_to_load_checkpoint = None
    mock_args.y_col = 'activity'
    return mock_args


@pytest.fixture()
def model_configs(mock_args):
    model_configs = ModelConfig(mock_args)
    return model_configs


def test_logging(mock_args):

    logger = Logger(mock_args)
    mock_mols = 0
    mock_loss = 1.5
    mock_preds = np.arange(100) + np.random.normal(size=100)
    mock_labs = np.arange(100)
    mock_split = 'train'

    logger.log(step=mock_mols,
               loss=mock_loss,
               y_pred=mock_preds,
               y_true=mock_labs,
               split=mock_split,
               )


def test_plot_prediction(mock_args):
    logger = Logger(mock_args)
    mock_split = 'train'
    mock_mols = 0
    mock_loss = 1.5
    mock_preds = np.arange(100) + np.random.normal(size=100)
    mock_labs = np.arange(100)
    mock_split = 'train'
    mock_title = 'Testing tensorboard logger on random numbers'
    mock_xlabel = 'np.arange'
    mock_ylabel = 'np.arange + np.normal()'
    logger.plot_predictions(step=mock_mols,
                            y_pred=mock_preds,
                            y_true=mock_labs,
                            split=mock_split,
                            title=mock_title,
                            xlabel=mock_xlabel,
                            ylabel=mock_ylabel)


def test_log_minibatch():
    pass


def test_log_weights(mock_args, model_configs):
    logger = Logger(mock_args)
    logger.log_weights(step=0, model_config=model_configs)
