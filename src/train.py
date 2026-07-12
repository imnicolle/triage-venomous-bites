"""
Treinamento e comparação de modelos de classificação multiclasse.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models() -> dict:
    """Retorna um dicionário com os modelos a serem comparados."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, multi_class="multinomial"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),
    }


def train_model(model, X_train, y_train):
    """Treina um modelo e retorna ele já ajustado."""
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str) -> None:
    """Salva o modelo treinado em disco."""
    joblib.dump(model, path)


def load_model(path: str):
    """Carrega um modelo salvo em disco."""
    return joblib.load(path)
