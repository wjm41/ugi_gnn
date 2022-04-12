import argparse
import pytest
import logging
import time
from unittest.mock import MagicMock

from dock2hit.train_and_validate.multithread import ModelConfig


@pytest.fixture()
def model_configs():
    mock_args = MagicMock()
    mock_args.batch_size = 32
    mock_args.n_epochs = 1
    mock_args.lr = 1e-3
    mock_args.optimizer = 'Adam'
    mock_args.path_to_train_data = 'test/test_data/HIV.csv'
    mock_args.path_to_external_val = None
    mock_args.random_train_val_split = False
    mock_args.path_to_load_checkpoint = None
    mock_args.y_col = 'activity'
    model_configs = ModelConfig(mock_args)
    return model_configs


def test_load_model_weights(model_configs):
    pass


def test_run_minbatch():
    pass


def test_run_validation():
    pass


def test_save_checkpoint():
    pass
