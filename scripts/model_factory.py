# scripts/model_factory.py

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class ModelFactory(ABC):
    """Interface para criar um estimador de modelo."""
    @abstractmethod
    def create_model(self) -> BaseEstimator:
        pass

class LogisticRegressionFactory(ModelFactory):
    """Factory para criar um modelo de Regressão Logística."""
    def create_model(self) -> BaseEstimator:
        return LogisticRegression(random_state=42, max_iter=1000)

class RandomForestFactory(ModelFactory):
    """Factory para criar um modelo de Random Forest."""
    def create_model(self) -> BaseEstimator:
        # Podemos facilmente expor hiperparâmetros aqui no futuro
        return RandomForestClassifier(n_estimators=100, random_state=42)

# Mapeamento de modelos disponíveis
AVAILABLE_MODELS = {
    "logistic_regression": LogisticRegressionFactory(),
    "random_forest": RandomForestFactory(),
}

def get_model_factory(name: str) -> ModelFactory:
    """Retorna a instância da factory de modelo."""
    factory = AVAILABLE_MODELS.get(name)
    if not factory:
        raise ValueError(f"Modelo '{name}' não reconhecido.")
    return factory