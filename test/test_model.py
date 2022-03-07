from unittest.mock import MagicMock


def mock_arguments():
    mock_arguments = MagicMock()
    mock_arguments.batch_size = 32

    return mock_arguments


print(mock_arguments())
