# test_app.py
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"].startswith("Welcome")


def test_post_prediction_valid_low_age():
    response = client.post(
        "/predict",
        json={
            "age": 25,
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
    )
    assert response.status_code == 200
    # As the exact output depends on the model, we cannot hardcode the expected prediction here.
    # Instead, we check that the response contains a prediction key
    assert "prediction" in response.json()


def test_post_prediction_valid_high_age():
    response = client.post(
        "/predict",
        json={
            "age": 35,
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
    )
    assert response.status_code == 200
    # As the exact output depends on the model, we cannot hardcode the expected prediction here.
    # Instead, we check that the response contains a prediction key
    assert "prediction" in response.json()


def test_post_prediction_invalid():
    response = client.post(
        "/predict",
        json={
            "age": "invalid",
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
    )
    assert response.status_code == 422
