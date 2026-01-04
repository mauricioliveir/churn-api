import requests
import time

URL = "https://churn-hackathon.onrender.com/previsao"

clientes_alto_risco = [
    {"CreditScore": 380, "Geography": "Germany", "Gender": "Female", "Age": 65, "Tenure": 1, "Balance": 150000, "EstimatedSalary": 120000},
    {"CreditScore": 420, "Geography": "Spain", "Gender": "Male", "Age": 58, "Tenure": 0, "Balance": 210000, "EstimatedSalary": 95000},
    {"CreditScore": 450, "Geography": "France", "Gender": "Female", "Age": 72, "Tenure": 2, "Balance": 180000, "EstimatedSalary": 150000},
    {"CreditScore": 400, "Geography": "Germany", "Gender": "Male", "Age": 48, "Tenure": 0, "Balance": 130000, "EstimatedSalary": 80000},
    {"CreditScore": 350, "Geography": "France", "Gender": "Female", "Age": 55, "Tenure": 3, "Balance": 95000, "EstimatedSalary": 190000},
    {"CreditScore": 500, "Geography": "Germany", "Gender": "Female", "Age": 60, "Tenure": 1, "Balance": 250000, "EstimatedSalary": 110000},
    {"CreditScore": 410, "Geography": "Spain", "Gender": "Male", "Age": 52, "Tenure": 1, "Balance": 160000, "EstimatedSalary": 140000},
    {"CreditScore": 430, "Geography": "France", "Gender": "Female", "Age": 59, "Tenure": 2, "Balance": 125000, "EstimatedSalary": 75000},
    {"CreditScore": 480, "Geography": "Germany", "Gender": "Male", "Age": 50, "Tenure": 0, "Balance": 140000, "EstimatedSalary": 105000},
    {"CreditScore": 460, "Geography": "Spain", "Gender": "Female", "Age": 56, "Tenure": 1, "Balance": 195000, "EstimatedSalary": 185000},
]

print(f"🚀 Iniciando teste de estresse em: {URL}\n")

for i, cliente in enumerate(clientes_alto_risco, 1):
    try:
        response = requests.post(URL, json=cliente)
        if response.status_code == 200:
            data = response.json()
            print(f"Teste {i}: Previsão: {data['previsao']} | Risco: {data['nivel_risco']} | Proba: {data['probabilidade']}")
            print(f"Motivos: {data['explicabilidade']}\n")
        else:
            print(f"Erro no teste {i}: Status {response.status_code}")
    except Exception as e:
        print(f"Falha de conexão no teste {i}: {e}")
    
    time.sleep(0.5) 

print("🏁 Teste de stress concluído com sucesso!")
