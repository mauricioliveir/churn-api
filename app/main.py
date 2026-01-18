from pathlib import Path
from typing import Dict, List, Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import uuid
import joblib
import shutil
import tempfile
import os
import threading

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
# .env (ADICIONADO)
# =========================================================
# Carrega variáveis do arquivo ~/churn-api/.env (raiz do projeto).
# No Docker Compose com env_file, isso não é obrigatório, mas ajuda se rodar fora do Docker.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=BASE_DIR.parent / ".env")
except Exception:
    pass

# =========================================================
# PERSISTÊNCIA (MySQL) - ADICIONADO
# =========================================================
# Driver: mysql-connector-python + pool pequeno. [web:426]
# Se falhar, a API não quebra.
try:
    from mysql.connector.pooling import MySQLConnectionPool  # type: ignore

    _MYSQL_AVAILABLE = True
except Exception:
    MySQLConnectionPool = None
    _MYSQL_AVAILABLE = False

_DB_POOL = None
_DB_POOL_LOCK = threading.Lock()


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _init_db_pool():
    global _DB_POOL
    if not _MYSQL_AVAILABLE:
        return None

    host = os.getenv("MYSQL_HOST", "").strip()
    user = os.getenv("MYSQL_USER", "").strip()
    password = os.getenv("MYSQL_PASSWORD", "").strip()
    database = os.getenv("MYSQL_DB", "").strip()
    port = _get_env_int("MYSQL_PORT", 3306)

    if not host or not user or not password or not database:
        return None

    pool_size = _get_env_int("MYSQL_POOL_SIZE", 3)

    try:
        _DB_POOL = MySQLConnectionPool(
            pool_name="churn_pool",
            pool_size=pool_size,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
    except Exception:
        _DB_POOL = None

    return _DB_POOL


def _get_conn():
    global _DB_POOL
    if _DB_POOL is None:
        with _DB_POOL_LOCK:
            if _DB_POOL is None:
                _init_db_pool()

    if _DB_POOL is None:
        return None

    try:
        return _DB_POOL.get_connection()
    except Exception:
        return None


def persist_previsao_individual(payload: Dict, resp: Dict):
    conn = _get_conn()
    if conn is None:
        return

    cur = None
    try:
        cur = conn.cursor()
        sql = """
            INSERT INTO previsoes_individual
            (id, credit_score, geography, gender, age, tenure, balance, estimated_salary,
             probabilidade, nivel_risco, previsao, explicabilidade)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        explic = resp.get("explicabilidade", [])
        if isinstance(explic, list):
            explic = ",".join([str(x) for x in explic])

        cur.execute(
            sql,
            (
                str(uuid.uuid4()),
                int(payload.get("CreditScore")) if payload.get("CreditScore") is not None else None,
                payload.get("Geography"),
                payload.get("Gender"),
                int(payload.get("Age")) if payload.get("Age") is not None else None,
                int(payload.get("Tenure")) if payload.get("Tenure") is not None else None,
                float(payload.get("Balance")) if payload.get("Balance") is not None else None,
                float(payload.get("EstimatedSalary")) if payload.get("EstimatedSalary") is not None else None,
                float(resp.get("probabilidade")) if resp.get("probabilidade") is not None else None,
                resp.get("nivel_risco"),
                resp.get("previsao"),
                str(explic) if explic is not None else "",
            ),
        )
        conn.commit()  # commit explícito no Connector/Python. [web:418]
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        try:
            if cur is not None:
                cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def persist_previsoes_lote(jobid: str, df_chunk: pd.DataFrame):
    if df_chunk is None or df_chunk.empty:
        return

    conn = _get_conn()
    if conn is None:
        return

    cur = None
    try:
        cur = conn.cursor()
        sql = """
            INSERT INTO previsoes_lote
            (jobid, credit_score, geography, gender, age, tenure, balance, estimated_salary,
             probabilidade, nivel_risco, previsao, explicabilidade)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        rows = []
        for _, r in df_chunk.iterrows():
            rows.append(
                (
                    jobid,
                    int(r["CreditScore"]) if pd.notna(r.get("CreditScore")) else None,
                    r.get("Geography"),
                    r.get("Gender"),
                    int(r["Age"]) if pd.notna(r.get("Age")) else None,
                    int(r["Tenure"]) if pd.notna(r.get("Tenure")) else None,
                    float(r["Balance"]) if pd.notna(r.get("Balance")) else None,
                    float(r["EstimatedSalary"]) if pd.notna(r.get("EstimatedSalary")) else None,
                    float(r["probabilidade"]) if pd.notna(r.get("probabilidade")) else None,
                    r.get("nivel_risco"),
                    r.get("previsao"),
                    str(r.get("explicabilidade")) if r.get("explicabilidade") is not None else "",
                )
            )

        cur.executemany(sql, rows)  # insert em lote. [web:398]
        conn.commit()  # [web:418]
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        try:
            if cur is not None:
                cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


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
"""

tags_metadata = [
    {"name": "root", "description": "Endpoint raiz e HEAD (útil para checar status rápido)."},
    {"name": "health", "description": "Healthcheck e status do carregamento do modelo."},
    {"name": "previsao", "description": "Predição unitária de churn (JSON -> JSON)."},
    {"name": "batch", "description": "Predição em lote (CSV) + status + download do resultado."},
]

app = FastAPI(
    title="ChurnInsight API — Plataforma de Previsão de Churn (Data Science)",
    description=description,
    version=APP_VERSION,
    contact={"name": "Repositório do projeto", "url": "https://github.com/LeticiaPaesano/Churn_Hackathon"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    openapi_tags=tags_metadata,
)

# =========================================================
# SCHEMAS
# =========================================================
class ChurnPayload(BaseModel):
    CreditScore: int = Field(..., ge=350, le=850, examples=[520])
    Geography: Literal["France", "Germany", "Spain"] = Field(..., examples=["Germany"])
    Gender: Literal["Male", "Female"] = Field(..., examples=["Female"])
    Age: int = Field(..., ge=18, le=92, examples=[55])
    Tenure: int = Field(..., ge=0, le=10, examples=[1])
    Balance: float = Field(..., ge=0, le=500000, examples=[152000.50])
    EstimatedSalary: float = Field(..., ge=0, le=200000, examples=[110000.00])


class ChurnPredictionResponse(BaseModel):
    previsao: Literal["Vai cancelar", "Vai continuar"]
    probabilidade: float = Field(..., ge=0, le=1)
    nivel_risco: Literal["ALTO", "BAIXO"]
    explicabilidade: List[str] = Field(default_factory=list)


class BatchStartResponse(BaseModel):
    job_id: str
    status: Literal["PROCESSANDO"]


class BatchStatusResponse(BaseModel):
    status: Literal["PROCESSANDO", "FINALIZADO", "ERRO"]


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

    if "threshold" in loaded:
        loaded["threshold_cost"] = loaded["threshold"]

    required = {"model", "scaler", "columns", "threshold_cost", "balance_median", "salary_median"}
    missing = required - set(loaded.keys())
    if missing:
        raise RuntimeError(f"Erro: Faltam chaves no model.joblib: {missing}")

    artifacts.update(loaded)
    model_loaded = True
    print("✅ Artefatos carregados com sucesso!")

    _init_db_pool()


# =========================================================
# ENDPOINTS ROOT / HEALTH (AGORA JSON)
# =========================================================
@app.get(
    "/",
    tags=["root"],
    summary="Status (JSON)",
    description="Retorna JSON com versão e status de carregamento do modelo.",
)
@app.head(
    "/",
    tags=["root"],
    summary="HEAD da raiz",
    description="Retorna apenas headers (bom para checks rápidos).",
)
def root():
    return {"version": APP_VERSION, "model_loaded": model_loaded}


@app.get(
    "/health",
    tags=["health"],
    summary="Healthcheck",
    description="Retorna status do serviço e se o modelo foi carregado no startup.",
)
def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


# =========================================================
# FUNÇÕES AUXILIARES DO MODELO
# =========================================================
def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    for col in artifacts["columns"]:
        if col not in df_proc.columns:
            df_proc[col] = 0
    return df_proc[artifacts["columns"]]


FEATURE_MAP = {
    "CreditScore": "CreditScore",
    "Age": "Age",
    "Tenure": "Tenure",
    "Balance": "Balance",
    "EstimatedSalary": "EstimatedSalary",
    "AgeTenure": "Age",
    "BalanceSalary": "Balance",
    "GeographyFrance": "Geography",
    "GeographySpain": "Geography",
    "GeographyGermany": "Geography",
    "GenderMale": "Gender",
    "GenderFemale": "Gender",
}


def calcular_explicabilidade_local(X_scaled: np.ndarray, payload_dict: Dict) -> List[str]:
    model = artifacts["model"]
    features = artifacts["columns"]
    importances = model.feature_importances_
    impactos = importances * np.abs(X_scaled[0])

    impacto_por_contrato: Dict[str, float] = {}
    for feature, impacto in zip(features, impactos):
        campo = FEATURE_MAP.get(feature, feature)
        if campo:
            impacto_por_contrato[campo] = impacto_por_contrato.get(campo, 0) + float(impacto)

    ranking = sorted(impacto_por_contrato.items(), key=lambda x: x[1], reverse=True)

    explicabilidade: List[str] = []
    seen = set()
    for campo, _ in ranking:
        if campo in seen:
            continue
        seen.add(campo)
        val = str(payload_dict[campo]) if campo in ["Geography", "Gender"] else campo
        explicabilidade.append(val)
        if len(explicabilidade) == 3:
            break

    return explicabilidade


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

        indices = np.argsort(impactos_matriz[i])[-3:][::-1]
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


def processar_csv(jobid: str, inputpath: Path):
    try:
        outputpath = TMP_DIR / f"{jobid}_resultado.csv"
        isfirst = True

        for chunk in pd.read_csv(inputpath, chunksize=5000):
            maskvalid = (
                chunk["CreditScore"].between(350, 850)
                & chunk["Geography"].isin(["France", "Germany", "Spain"])
                & chunk["Age"].between(18, 92)
                & chunk["Tenure"].between(0, 10)
                & chunk["Balance"].between(0, 500000)
            )
            chunkvalid = chunk[maskvalid].copy()

            if chunkvalid.empty:
                continue

            dfproc = preparar_dataframe(chunkvalid)
            Xscaled = artifacts["scaler"].transform(dfproc)
            probs = artifacts["model"].predict_proba(Xscaled)[:, 1]
            chunkvalid["probabilidade"] = probs.round(4)

            maskalto = probs >= artifacts["threshold_cost"]
            chunkvalid["nivel_risco"] = np.where(maskalto, "ALTO", "BAIXO")
            chunkvalid["previsao"] = np.where(maskalto, "Vai cancelar", "Vai continuar")
            chunkvalid["explicabilidade"] = obter_explicabilidade_lote(Xscaled, chunkvalid, maskalto)

            try:
                persist_previsoes_lote(jobid, chunkvalid)
            except Exception:
                pass

            chunkvalid.to_csv(outputpath, mode="a", index=False, header=isfirst)
            isfirst = False

        if inputpath.exists():
            inputpath.unlink()

    except Exception as e:
        (TMP_DIR / f"{jobid}.error").write_text(str(e))


# =========================================================
# ENDPOINTS DE PREVISÃO
# =========================================================
@app.post(
    "/previsao",
    tags=["previsao"],
    response_model=ChurnPredictionResponse,
    summary="Predição unitária de churn",
)
def previsao(payload: ChurnPayload):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    data = payload.model_dump()
    df = pd.DataFrame([data])
    dfproc = preparar_dataframe(df)

    Xscaled = artifacts["scaler"].transform(dfproc)
    proba = float(artifacts["model"].predict_proba(Xscaled)[0, 1])

    risco = "ALTO" if proba >= artifacts["threshold_cost"] else "BAIXO"
    previsaotxt = "Vai cancelar" if risco == "ALTO" else "Vai continuar"
    explicabilidade = calcular_explicabilidade_local(Xscaled, data) if risco == "ALTO" else []

    resp = {
        "previsao": previsaotxt,
        "probabilidade": round(proba, 4),
        "nivel_risco": risco,
        "explicabilidade": explicabilidade,
    }

    try:
        persist_previsao_individual(data, resp)
    except Exception:
        pass

    return resp


# =========================================================
# ENDPOINTS BATCH
# =========================================================
@app.post(
    "/previsao-lote",
    tags=["batch"],
    response_model=BatchStartResponse,
    summary="Inicia predição em lote (CSV)",
)
def previsao_lote(backgroundtasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser CSV")

    jobid = str(uuid.uuid4())
    inputpath = TMP_DIR / f"{jobid}.csv"

    with open(inputpath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    backgroundtasks.add_task(processar_csv, jobid, inputpath)
    return {"job_id": jobid, "status": "PROCESSANDO"}


@app.get(
    "/previsao-lote/status/{jobid}",
    tags=["batch"],
    response_model=BatchStatusResponse,
    summary="Consulta status do batch",
)
def status_lote(jobid: str):
    if (TMP_DIR / f"{jobid}.error").exists():
        return {"status": "ERRO"}
    if (TMP_DIR / f"{jobid}_resultado.csv").exists():
        return {"status": "FINALIZADO"}
    return {"status": "PROCESSANDO"}


@app.get(
    "/previsao-lote/download/{jobid}",
    tags=["batch"],
    summary="Download do resultado do batch",
)
def download(jobid: str):
    path = TMP_DIR / f"{jobid}_resultado.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Resultado não encontrado")
    return FileResponse(path, filename="resultado.csv")
