import pytest

from app.predictor import predict


@pytest.mark.parametrize(
    "values,horizon,expected",
    [
        ([1.0, 2.0, 3.0, 4.0], 1, 5.0),
        ([1.0, 2.0, 3.0, 4.0], 3, 7.0),
        ([4.0, 3.0, 2.0, 1.0], 2, -1.0),
        ([2.0, 2.0, 2.0, 2.0], 30, 2.0),
    ],
)
def test_prediction(values, horizon, expected):
    assert predict(values, horizon) == pytest.approx(expected)


@pytest.mark.parametrize(
    "values,horizon",
    [
        ([1.0] * 3, 1),
        ([1.0] * 129, 1),
        ([1.0] * 4, 0),
        ([1.0] * 4, 31),
        ([1.0, 2.0, 3.0, float("nan")], 1),
    ],
)
def test_invalid(values, horizon):
    with pytest.raises(ValueError):
        predict(values, horizon)
