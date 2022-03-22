import pytest
import argparse

from dock2hit.tensorboard_logging import Logger
from dock2hit.parsing import add_io_args, add_data_args, add_optim_args


def test_logging():

    parser = argparse.ArgumentParser()
    parser = add_io_args(parser)
    parser = add_data_args(parser)
    parser = add_optim_args(parser)

    args = parser.parse_args(['-p', 'test/test_data/HIV.csv',
                              '-log_dir', 'test_runs/test_logger/'
                              '-y_col', 'activity',
                              '-n_epochs', '10',
                              '-batch_size', '32',
                              ])

    logger = Logger(args)
