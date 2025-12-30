import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import app, artifacts


class DummyModel:
    def predict_proba(self, X):
        # Probabilidade depende fortemente da feature 0
        score = X[0, 0] * 0.6 + X[0, 1] * 0.2
        proba = 1 / (1 + np.exp(-score))
        return np.array([[1 - proba, proba]])


class DummyScaler:
    def transform(self, X):
        return X.values


@pytest.fixture(scope="session")
def client():
    artifacts["model"] = DummyModel()
    artifacts["scaler"] = DummyScaler()
    artifacts["columns"] = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "EstimatedSalary",
        "Balance_Salary_Ratio",
        "Age_Tenure",
        "High_Value_Customer",
        "Geography_Germany",
        "Geography_Spain",
        "Gender_Male",
    ]
    artifacts["threshold"] = 0.5
    artifacts["balance_median"] = 1000
    artifacts["salary_median"] = 1000

    return TestClient(app)
