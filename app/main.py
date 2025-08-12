# app/main.py

import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow

# --- Configuração da Aplicação ---
app = FastAPI(
    title="Loan Approval API V2",
    description="API para prever aprovação de empréstimo, carregando o modelo do MLflow Model Registry.",
    version="2.0.0"
)

# --- Carregamento do Modelo do MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

MODEL_NAME = "loan_approval_model"
MODEL_ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = None

@app.on_event("startup")
def load_model():
    """Carrega o modelo na inicialização da API."""
    global model
    try:
        # Usa a variável para configurar o cliente MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"Conectado ao MLflow em: {MLFLOW_TRACKING_URI}")
        print(f"Modelo '{MODEL_NAME}' (alias: {MODEL_ALIAS}) carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar modelo do MLflow: {e}")
        model = None

# --- Modelo de Dados de Entrada (Pydantic) ---
class LoanApplication(BaseModel):
    Age: float
    Annual_Income: float
    Credit_Score: float
    Loan_Amount: float
    Loan_Duration_Years: int
    Number_of_Open_Accounts: float
    Had_Past_Default: int

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 45.0,
                "Annual_Income": 85000.0,
                "Credit_Score": 650.0,
                "Loan_Amount": 25000.0,
                "Loan_Duration_Years": 10,
                "Number_of_Open_Accounts": 5.0,
                "Had_Past_Default": 0
            }
        }

@app.get("/", summary="Verificação de Saúde")
def read_root():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME if model else None,
        "model_alias": MODEL_ALIAS if model else None,
    }

@app.post("/api/v1/predict", summary="Realiza uma Predição de Aprovação de Empréstimo")
def predict(application: LoanApplication):
    if model is None:
        return {"error": "Modelo não está disponível. Verifique os logs e se um modelo possui o alias 'production'."}

    input_data = pd.DataFrame([application.dict()])
    prediction_result = model.predict(input_data)[0]
    approval_status = "Aprovado" if prediction_result == 1 else "Não Aprovado"

    return {
        "prediction": approval_status,
        "prediction_code": int(prediction_result)
    }