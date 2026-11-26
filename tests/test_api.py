import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_valid_request():
    response = client.post("/predict", json={"values": [1, 2, 3, 4], "horizon": 2})
    assert response.status_code == 200
    assert response.json() == {"prediction": 6.0, "model_version": "baseline-1"}


@pytest.mark.parametrize(
    "n,horizon,status",
    [
        (3, 1, 422),
        (4, 1, 200),
        (128, 30, 200),
        (129, 1, 422),
        (4, 0, 422),
        (4, 31, 422),
    ],
)
def test_boundaries(n, horizon, status):
    assert (
        client.post(
            "/predict", json={"values": [1] * n, "horizon": horizon}
        ).status_code
        == status
    )


def test_wrong_type():
    assert client.post("/predict", json={"values": [1, 2, 3, "abc"]}).status_code == 422
