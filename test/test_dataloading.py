from argparse import ArgumentParser
import pytest
import unittest
from unittest.mock import MagicMock

import pandas as pd

from dock2hit import dataloader, parsing
from dock2hit.utils import read_csv_or_pkl


@pytest.fixture()
def setup_args():
    mock_args = MagicMock()
    mock_args.batch_size = 32
    mock_args.n_epochs = 1
    return mock_args


@pytest.fixture(params=[('test/test_data/HIV.csv', 'activity'),
                        ('test/test_data/HIV.pkl', 'activity'),
                        ('test/test_data/D4_test.csv', 'dockscore'),
                        ('test/test_data/D4_test.pkl', 'dockscore')],
                ids=['HIV.csv',
                     'HIV.pkl',
                     'D4_test.csv',
                     'D4_test.pkl', ])
def data_args(request):
    test_csv, y_col = request.param
    return test_csv, y_col


@pytest.fixture()
def mock_args(setup_args, data_args):
    test_csv, y_col = data_args
    setup_args.path_to_train_data = test_csv
    setup_args.y_col = y_col
    return setup_args


def test_just_train_set(mock_args):
    mock_args.path_to_external_val = None
    mock_args.random_train_val_split = None

    length_of_train_data = len(read_csv_or_pkl(mock_args.path_to_train_data))
    train_loader, val_loader = dataloader.load_data(mock_args)
    assert train_loader.dataset_len == length_of_train_data
    assert val_loader is None


def test_random_train_val_split(mock_args):

    mock_args.path_to_external_val = None
    mock_args.random_train_val_split = True
    mock_args.size_of_val_set = 100

    length_of_train_data = len(read_csv_or_pkl(mock_args.path_to_train_data))

    train_loader, val_loader = dataloader.load_data(mock_args)

    assert train_loader.dataset_len == length_of_train_data - mock_args.size_of_val_set
    assert val_loader.dataset_len == mock_args.size_of_val_set


def test_external_val_set(mock_args):
    mock_args.path_to_external_val = mock_args.path_to_train_data
    mock_args.random_train_val_split = None

    length_of_train_data = len(read_csv_or_pkl(mock_args.path_to_train_data))

    length_of_val_data = len(read_csv_or_pkl(mock_args.path_to_train_data))

    train_loader, val_loader = dataloader.load_data(mock_args)
    assert train_loader.dataset_len == length_of_train_data
    assert val_loader.dataset_len == length_of_val_data


def test_integration_with_parsing(data_args):
    test_csv, y_col = data_args
    parser = ArgumentParser()
    parsing.add_data_args(parser)
    parsing.add_io_args(parser)
    parsing.add_optim_args(parser)

    integration_args = parser.parse_args(
        ['-path_to_train_data', f'{test_csv}',
            '-y_col', f'{y_col}'])

    length_of_train_data = len(read_csv_or_pkl(
        integration_args.path_to_train_data))
    train_loader, val_loader = dataloader.load_data(integration_args)
    assert train_loader.dataset_len == length_of_train_data
