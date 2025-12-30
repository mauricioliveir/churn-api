def test_explicabilidade(client):

    cliente_1 = {
        "CreditScore": 300,
        "Geography": "Spain",
        "Gender": "Male",
        "Age": 65,
        "Tenure": 1,
        "Balance": 200000,
        "EstimatedSalary": 20000
    }

    cliente_2 = {
        "CreditScore": 800,
        "Geography": "France",
        "Gender": "Female",
        "Age": 30,
        "Tenure": 8,
        "Balance": 0,
        "EstimatedSalary": 90000
    }

    r1 = client.post("/previsao", json=cliente_1).json()
    r2 = client.post("/previsao", json=cliente_2).json()

    assert r1["previsao"] == "Vai cancelar"
    assert r2["previsao"] == "Vai cancelar"

    assert r1["explicabilidade"] != r2["explicabilidade"]
