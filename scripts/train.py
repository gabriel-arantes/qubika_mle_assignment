# scripts/train.py

import os
import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# CORREÇÃO: As importações devem funcionar corretamente com o comando 'python -m'
from scripts.model_factory import get_model_factory
from scripts.preprocessing_factory import get_preprocessor_factory

# --- Configuração do Experimento MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "loan_approval_v2"
mlflow.set_experiment(EXPERIMENT_NAME)

def train(model_name: str, preprocessor_name: str):
    """
    Orquestra o processo de treinamento:
    1. Obtém um pré-processador da sua factory.
    2. Obtém um modelo da sua factory.
    3. Monta o pipeline final.
    4. Treina e registra tudo com MLflow.
    """
    print("="*40)
    print(f"Iniciando treinamento com modelo '{model_name}' e pré-processador '{preprocessor_name}'")

    with mlflow.start_run():
        # --- Log de Parâmetros Iniciais ---
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("preprocessor_name", preprocessor_name)

        # --- Carregamento e Preparação dos Dados ---
        # CORREÇÃO: O caminho deve ser relativo à raiz do projeto, onde o comando será executado.
        df = pd.read_csv("data/dataset.csv")

        # Remove a coluna de índice se presente
        if df.columns[0].strip() in (',', 'Unnamed: 0'):
            df = df.iloc[:, 1:]

        X = df.drop('Loan_Approval', axis=1)
        y = df['Loan_Approval']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # --- Montagem do Pipeline ---
        preprocessor_factory = get_preprocessor_factory(preprocessor_name)
        model_factory = get_model_factory(model_name)

        preprocessor = preprocessor_factory.create_preprocessor()
        model = model_factory.create_model()

        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # --- Treinamento e Avaliação ---
        final_pipeline.fit(X_train, y_train)
        y_pred = final_pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Acurácia: {accuracy:.4f} | F1-Score: {f1:.4f}")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # --- Registro do Modelo no MLflow ---
        signature = mlflow.models.infer_signature(X_train, final_pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="model",
            signature=signature,
            registered_model_name="loan_approval_model"
        )
        print("Modelo registrado com sucesso no MLflow com o nome 'loan_approval_model'")
        print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Treinamento e Registro de Modelos")
    parser.add_argument(
        "--model-name",
        type=str,
        default="random_forest",
        help=f"Nome do modelo a ser treinado."
    )
    parser.add_argument(
        "--preprocessor-name",
        type=str,
        default="median_imputer",
        help=f"Nome do pré-processador a ser usado."
    )
    args = parser.parse_args()

    train(model_name=args.model_name, preprocessor_name=args.preprocessor_name)