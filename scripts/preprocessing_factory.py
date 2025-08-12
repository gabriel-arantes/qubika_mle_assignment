# scripts/preprocessing_factory.py

from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class PreprocessingFactory(ABC):
    """Interface para criar um pipeline de pré-processamento."""
    @abstractmethod
    def create_preprocessor(self) -> Pipeline:
        pass

class MedianImputerFactory(PreprocessingFactory):
    """Factory que cria um pipeline com imputação de mediana."""
    def create_preprocessor(self) -> Pipeline:
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

class MedianImputerWithScalerFactory(PreprocessingFactory):
    """Factory que cria um pipeline com imputação e escalonamento."""
    def create_preprocessor(self) -> Pipeline:
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

# Mapeamento de pré-processadores disponíveis
AVAILABLE_PREPROCESSORS = {
    "median_imputer": MedianImputerFactory(),
    "median_imputer_with_scaler": MedianImputerWithScalerFactory()
}

def get_preprocessor_factory(name: str) -> PreprocessingFactory:
    """Retorna a instância da factory de pré-processamento."""
    factory = AVAILABLE_PREPROCESSORS.get(name)
    if not factory:
        raise ValueError(f"Pré-processador '{name}' não reconhecido.")
    return factory