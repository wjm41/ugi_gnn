import argparse
import pytest
import logging
import time
from unittest.mock import MagicMock

from src.parsing import add_io_args, add_data_args, add_optim_args
from src.train_and_validate import main
from src.utils import get_device


# def mock_arguments():
#     mock_arguments = MagicMock()
#     mock_arguments.batch_size = 32

#     return mock_arguments

def test_model_speed():
    parser = argparse.ArgumentParser()
    parser = add_io_args(parser)
    parser = add_data_args(parser)
    parser = add_optim_args(parser)

    args = parser.parse_args(['-p', 'test/test_data/HIV.csv',
                              '-log_dir', 'test/test_runs/HIV',
                              '-y_col', 'activity',
                              '-batch_size', '41128'])
    device = get_device()
    start_time = time.perf_counter()
    main(args, device)
    end_time = time.perf_counter()
    assert end_time - start_time < 30


logging.basicConfig(level=logging.INFO)
test_model_speed()
