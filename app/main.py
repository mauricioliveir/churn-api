from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

import uuid
import joblib
import shutil
import tempfile

import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
TMP_DIR = Path(tempfile.gettempdir())

APP_VERSION = "12.1.0"

# =========================================================
# METADADOS / SWAGGER
# =========================================================
description = """

Equipe H12-25-B-Equipo 25-Data Science


A **ChurnInsight API** disponibiliza previsões de churn (cancelamento) para clientes de serviços recorrentes.

### Contexto de negócio
Setores típicos: Fintech, Telecom, Streaming, E-commerce e SaaS.

### Funcionalidades (MVP)
- Previsão unitária de churn com probabilidade
- Processamento assíncrono de previsões em lote (CSV)
- Consulta de status e download do resultado do batch
- Healthcheck do serviço e do carregamento do modelo

### Observações
- O modelo é carregado no startup a partir de `models/model.joblib`.
- O batch processa o CSV em chunks para reduzir consumo de memória.
"""

tags_metadata = [
    {
        "name": "root",
        "description": "Endpoint raiz e HEAD (útil para checar status rápido).",
    },
    {
        "name": "health",
        "description": "Healthcheck e status do carregamento do modelo.",
    },
    {
        "name": "previsao",
        "description": "Predição unitária de churn (JSON -> JSON).",
    },
    {
        "name": "batch",
        "description": "Predição em lote (CSV) + status + download do resultado.",
    },
]

app = FastAPI(
    title="ChurnInsight API — Plataforma de Previsão de Churn (Data Science)",
    description=description,
    version=APP_VERSION,
    contact={
        "name": "Repositório do projeto",
        "url": "https://github.com/LeticiaPaesano/Churn_Hackathon",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=tags_metadata,
)

# =========================================================
# SCHEMAS (Pydantic) - Request/Response bem documentados
# =========================================================
class ChurnPayload(BaseModel):
    """Dados de entrada para predição unitária de churn."""

    CreditScore: int = Field(
        ...,
        ge=350,
        le=850,
        description="Score financeiro do cliente (faixa típica 350–850).",
        examples=[520],
    )
    Geography: Literal["France", "Germany", "Spain"] = Field(
        ...,
        description="País do cliente (categorias suportadas pelo modelo).",
        examples=["Germany"],
    )
    Gender: Literal["Male", "Female"] = Field(
        ...,
        description="Gênero do cliente (categorias suportadas pelo modelo).",
        examples=["Female"],
    )
    Age: int = Field(
        ...,
        ge=18,
        le=92,
        description="Idade do cliente.",
        examples=[55],
    )
    Tenure: int = Field(
        ...,
        ge=0,
        le=10,
        description="Tempo de relacionamento/contrato em anos (proxy do dataset).",
        examples=[1],
    )
    Balance: float = Field(
        ...,
        ge=0,
        le=500000,
        description="Saldo/Balance do cliente.",
        examples=[152000.50],
    )
    EstimatedSalary: float = Field(
        ...,
        ge=0,
        le=200000,
        description="Salário estimado (proxy do dataset).",
        examples=[110000.00],
    )


class ChurnPredictionResponse(BaseModel):
    previsao: Literal["Vai cancelar", "Vai continuar"] = Field(
        ...,
        description="Classe prevista.",
        examples=["Vai cancelar"],
    )
    probabilidade: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probabilidade associada ao churn (classe positiva).",
        examples=[0.55],
    )
    nivel_risco: Literal["ALTO", "BAIXO"] = Field(
        ...,
        description="Nível de risco derivado do threshold de custo do modelo.",
        examples=["ALTO"],
    )
    explicabilidade: List[str] = Field(
        default_factory=list,
        description="Top-3 fatores (campos do payload) que mais contribuíram para o risco ALTO.",
        examples=[["Age", "CreditScore", "Balance"]],
    )


class BatchStartResponse(BaseModel):
    job_id: str = Field(..., description="Identificador do job de batch.", examples=["b7e7f6a7-9f5a-4c1d-a0f5-8a6dfe51a2c0"])
    status: Literal["PROCESSANDO"] = Field(..., examples=["PROCESSANDO"])


class BatchStatusResponse(BaseModel):
    status: Literal["PROCESSANDO", "FINALIZADO", "ERRO"] = Field(..., examples=["FINALIZADO"])


# =========================================================
# ARTEFATOS / MODELO
# =========================================================
artifacts: Dict = {}
model_loaded = False

@app.on_event("startup")
def load_artifacts():
    global model_loaded

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)

    # compat (se existir chave antiga)
    if "threshold" in loaded:
        loaded["threshold_cost"] = loaded["threshold"]

    required = {
        "model",
        "scaler",
        "columns",
        "threshold_cost",
        "balance_median",
        "salary_median",
    }
    missing = required - set(loaded.keys())
    if missing:
        raise RuntimeError(f"Erro: Faltam chaves no model.joblib: {missing}")

    artifacts.update(loaded)
    model_loaded = True
    print("✅ Artefatos carregados com sucesso!")

# =========================================================
# ENDPOINTS ROOT / HEALTH
# =========================================================
@app.get(
    "/",
    tags=["root"],
    summary="Página raiz (status e links)",
    description="Retorna uma página HTML simples com links úteis.",
    include_in_schema=True,
    response_class=HTMLResponse,
)
@app.head(
    "/",
    tags=["root"],
    summary="HEAD da raiz",
    description="Retorna apenas headers (bom para checks rápidos).",
)
def root():
    # Para HEAD, o FastAPI usa a mesma função; o cliente ignora body.
    return f"""
    <html>
      <head><title>ChurnInsight API</title></head>
      <body>
        <h2>ChurnInsight API (Data Science)</h2>
        <ul>
          <li>Status: {"online" if True else "offline"}</li>
          <li>Model loaded: {str(model_loaded).lower()}</li>
          <li>Versão: {APP_VERSION}</li>
          <li><a href="/docs">Swagger UI</a></li>
          <li><a href="/openapi.json">OpenAPI JSON</a></li>
          <li><a href="/health">Health</a></li>
        </ul>
      </body>
    </html>
    """

@app.get(
    "/health",
    tags=["health"],
    summary="Healthcheck",
    description="Retorna status do serviço e se o modelo foi carregado no startup.",
)
def health():
    return {"status": "ok", "model_loaded": model_loaded}

@app.get(
    "/favicon.ico",
    include_in_schema=False,
)
def favicon():
    return Response(status_code=204)

# =========================================================
# PREPARAÇÃO DE DADOS
# =========================================================
def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    for col in artifacts["columns"]:
        if col not in df_proc.columns:
            df_proc[col] = 0
    return df_proc[artifacts["columns"]]

# =========================================================
# MAPA DE FEATURES (explicabilidade)
# =========================================================
FEATURE_MAP = {
    "CreditScore": "CreditScore",
    "Age": "Age",
    "Tenure": "Tenure",
    "Balance": "Balance",
    "EstimatedSalary": "EstimatedSalary",
    "Age_Tenure": "Age",
    "Balance_Salary": "Balance",
    "Geography_France": "Geography",
    "Geography_Spain": "Geography",
    "Geography_Germany": "Geography",
    "Gender_Male": "Gender",
    "Gender_Female": "Gender",
}

def calcular_explicabilidade_local(X_scaled: np.ndarray, payload_dict: Dict) -> List[str]:
    model = artifacts["model"]
    features = artifacts["columns"]

    importances = model.feature_importances_
    impactos = importances * np.abs(X_scaled[0])

    impacto_por_contrato: Dict[str, float] = {}
    for feature, impacto in zip(features, impactos):
        campo = FEATURE_MAP.get(feature)
        if campo:
            impacto_por_contrato[campo] = impacto_por_contrato.get(campo, 0) + float(impacto)

    ranking = sorted(impacto_por_contrato.items(), key=lambda x: x[1], reverse=True)

    explicabilidade: List[str] = []
    seen = set()

    for campo, _ in ranking:
        if campo in seen:
            continue
        seen.add(campo)

        # Mantém seu comportamento: para Geography/Gender mostra o valor; demais mostra o nome do campo
        val = str(payload_dict[campo]) if campo in ("Geography", "Gender") else campo
        explicabilidade.append(val)

        if len(explicabilidade) == 3:
            break

    return explicabilidade

# =========================================================
# ENDPOINT /PREVISAO
# =========================================================
@app.post(
    "/previsao",
    tags=["previsao"],
    response_model=ChurnPredictionResponse,
    summary="Predição unitária de churn",
    description=(
        "Recebe dados do cliente e retorna previsão + probabilidade + nível de risco. "
        "Quando risco = ALTO, retorna também top-3 fatores de explicabilidade."
    ),
)
def previsao(payload: ChurnPayload):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    data = payload.model_dump()
    df = pd.DataFrame([data])

    df_proc = preparar_dataframe(df)
    X_scaled = artifacts["scaler"].transform(df_proc)

    proba = float(artifacts["model"].predict_proba(X_scaled)[0, 1])
    risco = "ALTO" if proba >= artifacts["threshold_cost"] else "BAIXO"
    previsao_txt = "Vai cancelar" if risco == "ALTO" else "Vai continuar"

    explicabilidade = calcular_explicabilidade_local(X_scaled, data) if risco == "ALTO" else []

    return {
        "previsao": previsao_txt,
        "probabilidade": round(proba, 4),
        "nivel_risco": risco,
        "explicabilidade": explicabilidade,
    }

# =========================================================
# BATCH - processamento em background
# =========================================================
def obter_explicabilidade_lote(
    X_scaled: np.ndarray,
    chunk_df: pd.DataFrame,
    mask_cancelar: np.ndarray,
) -> List[str]:
    model = artifacts["model"]
    features = artifacts["columns"]
    importances = model.feature_importances_

    impactos_matriz = np.abs(X_scaled) * importances
    nomes_campos = np.array([FEATURE_MAP.get(f, f) for f in features])

    resultados: List[str] = []
    for i in range(impactos_matriz.shape[0]):
        if not mask_cancelar[i]:
            resultados.append("")
            continue

        indices = np.argsort(impactos_matriz[i])[::-1]
        final_names, seen = [], set()

        for idx in indices:
            nome = nomes_campos[idx]
            if nome in seen:
                continue
            seen.add(nome)
            final_names.append(str(chunk_df.iloc[i][nome]) if nome in ["Geography", "Gender"] else nome)
            if len(final_names) == 3:
                break

        resultados.append(", ".join(final_names))

    return resultados

def processar_csv(job_id: str, input_path: Path):
    try:
        output_path = TMP_DIR / f"{job_id}_resultado.csv"
        is_first = True

        for chunk in pd.read_csv(input_path, chunksize=5000):
            mask_valid = (
                (chunk["CreditScore"].between(350, 850)) &
                (chunk["Geography"].isin(["France", "Germany", "Spain"])) &
                (chunk["Age"].between(18, 92)) &
                (chunk["Tenure"].between(0, 10)) &
                (chunk["Balance"].between(0, 500000))
            )

            chunk_valid = chunk[mask_valid].copy()
            if chunk_valid.empty:
                continue

            df_proc = preparar_dataframe(chunk_valid)
            X_scaled = artifacts["scaler"].transform(df_proc)

            probs = artifacts["model"].predict_proba(X_scaled)[:, 1]
            chunk_valid["probabilidade"] = probs.round(4)

            mask_alto = probs >= artifacts["threshold_cost"]
            chunk_valid["nivel_risco"] = np.where(mask_alto, "ALTO", "BAIXO")
            chunk_valid["previsao"] = np.where(mask_alto, "Vai cancelar", "Vai continuar")
            chunk_valid["explicabilidade"] = obter_explicabilidade_lote(X_scaled, chunk_valid, mask_alto)

            chunk_valid.to_csv(output_path, mode="a", index=False, header=is_first)
            is_first = False

        if input_path.exists():
            input_path.unlink()

    except Exception as e:
        (TMP_DIR / f"{job_id}.error").write_text(str(e))

@app.post(
    "/previsao-lote",
    tags=["batch"],
    response_model=BatchStartResponse,
    summary="Inicia predição em lote (CSV)",
    description=(
        "Envia um arquivo CSV e inicia o processamento em background. "
        "Retorna um job_id para consulta de status e download."
    ),
)
def previsao_lote(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Arquivo CSV com colunas compatíveis com o modelo."),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser CSV")

    job_id = str(uuid.uuid4())
    input_path = TMP_DIR / f"{job_id}.csv"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(processar_csv, job_id, input_path)
    return {"job_id": job_id, "status": "PROCESSANDO"}

@app.get(
    "/previsao-lote/status/{job_id}",
    tags=["batch"],
    response_model=BatchStatusResponse,
    summary="Consulta status do batch",
    description="Retorna PROCESSANDO, FINALIZADO ou ERRO.",
)
def status_lote(job_id: str):
    if (TMP_DIR / f"{job_id}.error").exists():
        return {"status": "ERRO"}
    if (TMP_DIR / f"{job_id}_resultado.csv").exists():
        return {"status": "FINALIZADO"}
    return {"status": "PROCESSANDO"}

@app.get(
    "/previsao-lote/download/{job_id}",
    tags=["batch"],
    summary="Download do resultado do batch",
    description="Faz download do CSV gerado ao final do processamento.",
    responses={
        200: {"description": "Arquivo CSV com resultados."},
        404: {"description": "Resultado não encontrado (job_id inválido ou ainda processando)."},
    },
)
def download(job_id: str):
    path = TMP_DIR / f"{job_id}_resultado.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Resultado não encontrado")
    return FileResponse(path, filename="resultado.csv")
