"""
Funções de carregamento e pré-processamento do dataset de triagem.

IMPORTANTE: ajuste TARGET_COLUMN e as listas de colunas depois de inspecionar
o CSV real (rode notebooks/01_eda.ipynb primeiro para ver os nomes exatos
das colunas do dataset).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# TODO: confirmar o nome real da coluna alvo (animal/espécie) após a EDA
TARGET_COLUMN = "animal"  # placeholder - ajustar conforme o dataset real


def load_data(path: str) -> pd.DataFrame:
    """Carrega o CSV bruto do dataset."""
    return pd.read_csv(path)


def basic_info(df: pd.DataFrame) -> None:
    """Imprime um resumo rápido do dataframe: shape, tipos, nulos."""
    print("Shape:", df.shape)
    print("\nTipos de dados:\n", df.dtypes)
    print("\nValores nulos por coluna:\n", df.isnull().sum())
    print("\nDistribuição da variável alvo:\n", df[TARGET_COLUMN].value_counts())


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN):
    """Separa features (X) e alvo (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def encode_categorical(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """Aplica Label Encoding nas colunas categóricas informadas."""
    df = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def scale_numeric(X_train, X_test, numeric_cols: list):
    """Padroniza colunas numéricas com StandardScaler (fit no treino)."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, scaler


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split treino/teste com estratificação pela classe alvo."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
