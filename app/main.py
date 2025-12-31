import os
import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional, List

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =========================================================
# LOGGING
# =========================================================
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")

artifacts: dict = {}

# =========================================================
# LIFESPAN
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")
        
    loaded = joblib.load(MODEL_PATH)

    artifacts.update(loaded)
    print("✅ Pipeline carregado com sucesso")
    yield

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="ChurnInsight API", 
    version="1.2.1", 
    lifespan=lifespan
)

# =========================================================
# UTILS
# =========================================================
def classificar_faixa_score(score: int) -> str:
    if score >= 701: return "Excelente"
    if score >= 501: return "Bom"
    if score >= 301: return "Regular"
    return "Baixo"
    
def gerar_recomendacao(nivel_risco: str) -> str:
    recomendas = {
        "ALTO": "Ação imediata recomendada: contato ativo e oferta personalizada",
        "MÉDIO": "Monitoramento recomendado e campanhas de retenção",
        "BAIXO": "Cliente estável - manutenção padrão"
    }
    return recomendas.get(nivel_risco, "Manutenção padrão")

def calcular_explicabilidade_local(
    model,
    X: np.ndarray,
    feature_names: List[str],
    baseline_proba: float, 
    input_data: dict 
) -> List[str]:
    
    mapeamento_contrato = {
        "CreditScore": "CreditScore",
        "Age": "Age",
        "Tenure": "Tenure",
        "Balance": "Balance",
        "EstimatedSalary": "EstimatedSalary",
        "Geography_Germany": input_data.get("Geography"), 
        "Geography_Spain": input_data.get("Geography"),
        "Gender_Male": input_data.get("Gender"),         
        "Balance_Salary_Ratio": "Balance",      
        "Age_Tenure": "Age",                    
        "High_Value_Customer": "Balance"        
    }

    impactos = []
    for i, feature in enumerate(feature_names):
        X_mod = X.copy()
        X_mod[0, i] = 0 # Perturbação local
        proba_mod = model.predict_proba(X_mod)[0, 1]
        impactos.append((feature, baseline_proba - proba_mod))

    impactos_ordenados = sorted(impactos, key=lambda x: x[1], reverse=True)
   
    features_finais = []
    for feat_interna, _ in impactos_ordenados:
        nome_amigavel = mapeamento_contrato.get(feat_interna)
        if nome_amigavel and nome_amigavel not in features_finais:
            features_finais.append(nome_amigavel)
        if len(features_finais) >= 3:
            break

    return features_finais

# =========================================================
# SCHEMAS
# =========================================================
class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=0, le=1000)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=92)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    EstimatedSalary: float = Field(..., ge=0)

class PredictionOutput(BaseModel):
    previsao: str
    probabilidade: float
    nivel_risco: str
    recomendacao: str
    explicabilidade: Optional[List[str]] = None

# =========================================================
# ENDPOINT
# =========================================================
@app.post("/previsao", response_model=PredictionOutput)
def predict_churn(data: CustomerInput):
    if not artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    # 1. One-hot encoding (Sync com Notebook)
    df["Geography_Germany"] = 1 if data.Geography == "Germany" else 0
    df["Geography_Spain"] = 1 if data.Geography == "Spain" else 0
    df["Gender_Male"] = 1 if data.Gender == "Male" else 0

    # 2. Feature Engineering
    df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Age_Tenure"] = df["Age"] * df["Tenure"]
    df["High_Value_Customer"] = (
        (df["Balance"] > artifacts.get("balance_median", 0)) &
        (df["EstimatedSalary"] > artifacts.get("salary_median", 0))
    ).astype(int)

    # 3. Escalonamento e Predição
    df_final = df[artifacts["columns"]]
    X_scaled = artifacts["scaler"].transform(df_final)

    proba = float(artifacts["model"].predict_proba(X_scaled)[0, 1])
    threshold = artifacts["threshold"]

    # 4. Definição de Risco (Conforme Notebook)
    if proba >= 0.5:
        risco = "ALTO"
    elif proba >= 0.3:
        risco = "MÉDIO"
    else:
        risco = "BAIXO"

    previsao = "Vai cancelar" if proba >= threshold else "Vai continuar"

    # 5. Explicabilidade
    explicabilidade_output = None
    if previsao == "Vai cancelar":
        explicabilidade_output = calcular_explicabilidade_local(
            artifacts["model"], X_scaled, artifacts["columns"], proba, input_dict
        )

    return PredictionOutput(
        previsao=previsao,
        probabilidade=round(proba, 4),
        nivel_risco=risco,
        recomendacao=gerar_recomendacao(risco),
        explicabilidade=explicabilidade_output
    )

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health_check():
    return {
        "status": "UP",
        "model_loaded": "model" in artifacts,
        "scaler_loaded": "scaler" in artifacts
    }
