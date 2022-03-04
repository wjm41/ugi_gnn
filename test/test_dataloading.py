import pytest

from src import dataloader


@pytest.mark.parametrize(
    # fmt : off
    "test_input,expected_output",
    [
        pytest.param([0, 1], [tuple((tuple((0, 1)),))],
                     id="one_pair"),
        pytest.param([0, 1, 2], None, marks=pytest.mark.xfail(raises=ValueError, reason='Requires even number of particles'),
                     id='odd_pair')
    ],
)
# fmt : on
def test_ultraloader(test_input, expected_output):
    test_loader = dataloader.UltraLoader(test_input)
    assert test_loader == expected_output
