import pytest
from fastapi.testclient import TestClient
from app.main import app, artifacts

@pytest.fixture(scope="session")
def client():
    return TestClient(app)

@pytest.fixture(scope="session")
def payload_valido():
    return {
        "CreditScore": 501,
        "Geography": "Spain",
        "Gender": "Female",
        "Age": 62,
        "Tenure": 0,
        "Balance": 38000,
        "EstimatedSalary": 132351
    }
