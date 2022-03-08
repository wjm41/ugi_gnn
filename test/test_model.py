import argparse
import pytest
import logging
import time
from unittest.mock import MagicMock

from src.parsing import add_io_args, add_data_args, add_optim_args
from src.train_and_validate import main as featurize_on_the_fly
from src.train_and_validate_collate import main as pre_featurize
from src.utils import get_device


# def mock_arguments():
#     mock_arguments = MagicMock()
#     mock_arguments.batch_size = 32

#     return mock_arguments

def test_model_speed(model):
    parser = argparse.ArgumentParser()
    parser = add_io_args(parser)
    parser = add_data_args(parser)
    parser = add_optim_args(parser)

    parser.add_argument('-time_forward_pass', action='store_true',
                        help='if True, will log the time taken for a forward pass a batch.')
    args = parser.parse_args(['-p', 'test/test_data/HIV.csv',
                              '-y_col', 'activity',
                              #   '-batch_size', '32',
                              '-batch_size', '41127',
                              #   '-time_forward_pass'
                              ])
    device = get_device()
    start_time = time.perf_counter()
    model(args, device)
    end_time = time.perf_counter()
    assert end_time - start_time < 30


logging.basicConfig(level=logging.INFO)
logging.info('Testing pre-featurised model.')
test_model_speed(pre_featurize)
# logging.info('Testing featurize-on-the-fly model.')
# test_model_speed(featurize_on_the_fly)
