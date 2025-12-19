import httpx
import pytest

# Define the base URL for the running API.
# This should match the host and port where the FastAPI app is served.
BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="module")
def api_client():
    """
    A pytest fixture that provides an httpx client and checks if the server is running.
    """
    try:
        with httpx.Client(base_url=BASE_URL) as client:
            # Check if the server is alive
            response = client.get("/")
            response.raise_for_status()  # Raise an exception for bad status codes
            yield client
    except (httpx.ConnectError, httpx.HTTPStatusError) as e:
        pytest.fail(
            f"Could not connect to the API at {BASE_URL}. "
            f"Please ensure the server is running with 'uv run python -m src.GemAI.main serve'.\n"
            f"Error: {e}"
        )


def test_read_root(api_client):
    """
    Tests if the root endpoint ('/') is accessible and returns a 200 OK status
    with the correct content type.
    """
    response = api_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_docs(api_client):
    """
    Tests if the auto-generated API documentation is available.
    """
    response = api_client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_valid_input(api_client):
    """
    Tests the /predict endpoint with a valid payload to ensure it returns a
    200 OK status and the expected response format.
    """
    valid_payload = {
        "carat": 0.75,
        "cut": "Ideal",
        "color": "D",
        "clarity": "IF",
        "depth": 62.1,
        "table": 57,
        "x": 5.71,
        "y": 5.73,
        "z": 3.55,
    }
    response = api_client.post("/predict", json=valid_payload)
    assert response.status_code == 200

    response_data = response.json()
    assert "price_bwp" in response_data
    assert isinstance(response_data["price_bwp"], float)


def test_predict_invalid_input(api_client):
    """
    Tests the /predict endpoint with an invalid payload (wrong data type) to ensure
    it returns a 422 Unprocessable Entity error.
    """
    invalid_payload = {
        "carat": "not-a-float",  # Invalid type
        "cut": "Ideal",
        "color": "D",
        "clarity": "IF",
        "depth": 62.1,
        "table": 57,
        "x": 5.71,
        "y": 5.73,
        "z": 3.55,
    }
    response = api_client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
