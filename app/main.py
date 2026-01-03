from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
import logging
from contextlib import asynccontextmanager

# =========================================================
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

# =========================================================

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


# =========================
# STARTUP
# =========================
@app.on_event("startup")
def load_artifacts():
    global artifacts

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "model" / "model.joblib"

    if not model_path.exists():
        print(f"❌ Modelo não encontrado em {model_path}")
        artifacts = {}
        return

    artifacts = joblib.load(model_path)
    print("✅ Modelo carregado com sucesso")


# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {
        "status": "UP",
        "model_loaded": bool(artifacts)
    }


# =========================
# AUXILIARES
# =========================
def classificar_risco(proba: float, threshold: float):
    if proba >= threshold:
        return "ALTO"
    elif proba >= 0.6 * threshold:
        return "MÉDIO"
    return "BAIXO"


def gerar_previsao(risco: str):
    return "Vai Sair" if risco == "ALTO" else "Vai Ficar"


def gerar_recomendacao(risco: str):
    if risco == "ALTO":
        return "Ação imediata recomendada: contato ativo e oferta personalizada"
    if risco == "MÉDIO":
        return "Monitorar cliente e avaliar incentivos"
    return "Manter relacionamento padrão"


def calcular_explicabilidade_local(model, X, columns, proba, raw_input):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    impactos = np.abs(shap_values[1][0])
    top_idx = np.argsort(impactos)[-3:][::-1]
    return [columns[i] for i in top_idx]


def preparar_features(row: dict):
    df = pd.DataFrame([row])

    df["Geography_Germany"] = int(row["Geography"] == "Germany")
    df["Geography_Spain"] = int(row["Geography"] == "Spain")
    df["Gender_Male"] = int(row["Gender"] == "Male")

    df["Balance_Salary_Ratio"] = row["Balance"] / (row["EstimatedSalary"] + 1)
    df["Age_Tenure"] = row["Age"] * row["Tenure"]

    df["High_Value_Customer"] = int(
        row["Balance"] > artifacts["balance_median"]
        and row["EstimatedSalary"] > artifacts["salary_median"]
    )

    for col in artifacts["columns"]:
        if col not in df:
            df[col] = 0

    df = df[artifacts["columns"]]
    X_scaled = artifacts["scaler"].transform(df)
    return df, X_scaled


# =========================
# PREVISÃO LOTE (CSV)
# =========================
@app.post("/previsao-lote")
def previsao_lote(file: UploadFile = File(...)):

    if not artifacts:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser CSV")

    df = pd.read_csv(file.file)

    required_cols = artifacts["raw_columns"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Colunas ausentes: {list(missing)}"
        )

    resultados = []
    threshold = artifacts["threshold"]

    for _, row in df.iterrows():
        problemas = []
        explicabilidade = None

        for col in required_cols:
            if pd.isna(row[col]):
                problemas.append(col)

        for col, stats in artifacts["numeric_stats"].items():
            if col in row and not pd.isna(row[col]):
                z = abs((row[col] - stats["mean"]) / stats["std"])
                if z > 3:
                    problemas.append(col)

        try:
            input_dict = row.to_dict()
            _, X_scaled = preparar_features(input_dict)

            proba = artifacts["model"].predict_proba(X_scaled)[0, 1]
            risco = classificar_risco(proba, threshold)
            previsao = gerar_previsao(risco)

            if risco == "ALTO":
                explicabilidade = calcular_explicabilidade_local(
                    artifacts["model"],
                    X_scaled,
                    artifacts["columns"],
                    proba,
                    input_dict
                )

        except Exception:
            proba = np.nan
            risco = "ERRO"
            previsao = "Erro"
            problemas.append("Falha processamento")

        resultados.append({
            **row.to_dict(),
            "previsao": previsao,
            "probabilidade": float(f"{proba:.2f}") if not np.isnan(proba) else None,
            "nivel_risco": risco,
            "recomendacao": gerar_recomendacao(risco),
            "explicabilidade": "|".join(explicabilidade) if explicabilidade else None,
            "erro_linha": len(problemas) > 0,
            "colunas_problema": ",".join(set(problemas))
        })

    df_out = pd.DataFrame(resultados)

    output_name = f"{Path(file.filename).stem}_previsionado.csv"
    output_path = Path(tempfile.gettempdir()) / output_name
    df_out.to_csv(output_path, index=False)

    return FileResponse(
        output_path,
        media_type="text/csv",
        filename=output_name
    )
