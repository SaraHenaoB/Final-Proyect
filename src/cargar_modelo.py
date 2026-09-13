"""
cargar_modelo.py
Proyecto Final - Sistema de recomendacion estacional
Equipo MetricEdge

Una sola responsabilidad: cargar el artefacto .joblib que genera
train_model_recomendacion_estacional.py, y devolverlo en un formato
comodo de usar. Si el dia de manana cambia como se guarda el modelo
por dentro, solo hay que tocar este archivo, no la demo ni el drift.
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "seasonal_recommendation_model.joblib")


def cargar_modelo_estacional(model_path: str = MODEL_PATH) -> dict:
    """
    Carga el artefacto guardado por train_model_recomendacion_estacional.py
    y devuelve sus piezas ya listas para usar (Series de pandas en vez de
    diccionarios crudos), para que quien lo use no tenga que saber el
    formato interno del .joblib.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontro el modelo entrenado en: {model_path}\n"
            f"Corre primero: python src/train_model_recomendacion_estacional.py"
        )

    artifact = joblib.load(model_path)

    monthly_means = pd.Series(artifact["monthly_means"])
    product_ranking = pd.Series(artifact["product_ranking"]).sort_values(ascending=False)
    category_ranking = pd.Series(artifact["category_ranking"]).sort_values(ascending=False)
    forecast = pd.DataFrame(artifact["forecast"])

    return {
        "nombre_modelo": artifact["model_name"],
        "monthly_means": monthly_means,
        "product_ranking": product_ranking,
        "category_ranking": category_ranking,
        "metrics": artifact["metrics"],
        "forecast": forecast,
    }


if __name__ == "__main__":
    modelo = cargar_modelo_estacional()
    print("Modelo cargado:", modelo["nombre_modelo"])
    print("\nTop 5 categorias:")
    print(modelo["category_ranking"].head())
    print("\nMetricas guardadas:")
    for k, v in modelo["metrics"].items():
        print(f"  {k}: {v}")
