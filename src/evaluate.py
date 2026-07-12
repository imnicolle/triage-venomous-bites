"""
Funções de avaliação e visualização dos resultados dos modelos.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)


def evaluate_model(model, X_test, y_test, class_names=None) -> dict:
    """Calcula métricas de avaliação e imprime o relatório de classificação."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"Acurácia: {acc:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}\n")
    print("Relatório de classificação:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return {"accuracy": acc, "f1_macro": f1_macro, "y_pred": y_pred}


def plot_confusion_matrix(y_test, y_pred, class_names=None, save_path=None):
    """Plota (e opcionalmente salva) a matriz de confusão."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()


def compare_models(results: dict) -> None:
    """Imprime uma tabela comparando acurácia e F1 dos modelos treinados.

    `results` deve ser um dict no formato:
        {"nome_modelo": {"accuracy": ..., "f1_macro": ...}, ...}
    """
    print(f"{'Modelo':<25}{'Acurácia':<12}{'F1 (macro)':<12}")
    print("-" * 49)
    for name, metrics in results.items():
        print(f"{name:<25}{metrics['accuracy']:<12.4f}{metrics['f1_macro']:<12.4f}")
