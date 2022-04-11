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
