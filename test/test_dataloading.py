import pytest

from dock2hit import dataloader, parsing


# @pytest.mark.parametrize(
#     # fmt : off
#     "test_input,expected_output",
#     [
#         pytest.param('test_data.pkl', hello,
#                      id="one_pair"),
#         pytest.param([0, 1, 2], None, marks=pytest.mark.xfail(raises=ValueError, reason='Requires even number of particles'),
#                      id='odd_pair')
#     ],
# )
# # fmt : on
# def test_ultraloader(test_input, expected_output):
#     test_loader = dataloader.UltraLoader(test_input)
#     # TODO test length, n_batches, random_state, inds, y_transform
#     assert len(test_loader) == expected_output

@pytest.mark.parametrize(
    "test_csv",
    ["test_data/HIV.csv",
     ],
)
def test_just_train_set(test_csv):
    pass


def test_random_train_val_split():
    pass


def test_external_val_set():
    pass


def test_integration_with_parsing():
    pass
