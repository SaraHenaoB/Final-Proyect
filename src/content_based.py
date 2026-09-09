"""
Este archivo contiene la clase ContentBasedEngine optimizada con métodos para recomendaciones
Item-to-Item y User-to-Item (vector de perfil de usuario) 
Facilitando la integración con el sistema híbrido
"""

# Librerías necesarias
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Para localizar los archivos por sus nombres sin tener que especificar ruta
BASE_DIR = Path(__file__).resolve().parent.parent

class ContentBasedEngine:
    def __init__(
            self,
            matrix_path=BASE_DIR / "models" / "content_cosine_sim.joblib",
            product_path=BASE_DIR / "data" / "processed" / "product_clean.csv",
    ):

        """Inicializa el motor de contenido cargando la matriz y el catálogo de productos."""
        self.cosine_sim = joblib.load(matrix_path)
        self.product_df = pd.read_csv(product_path)
        self.indices = pd.Series(
            self.product_df.index, index=self.product_df["product_id"]
        ).drop_duplicates()

    def product_recomendation (self, product_id, k_top=5):
        """Genera recomendaciones Item-to-Item para un producto objetivo."""
        if product_id not in self.indices:
            return pd.DataFrame()

        idx = self.indices[product_id]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
            1: k_top + 1
        ]

        product_indices = [i[0]for i in sim_scores]
        res = self.product_df.iloc[product_indices].copy()
        res["similarity_score"] = [round(s[1], 4) for s in sim_scores]
        return res

    def customer_recomendation_product(self, user_purchases, k_top=5):
        """Genera recomendaciones User-to-Item basadas en el historial de compras del usuario."""
        valid_purchases = [pid for pid in user_purchases if pid in self.indices]
        if not valid_purchases:
            return pd.DataFrame()

        indices_comprados = [self.indices[pid] for pid in valid_purchases]

        # Calcular perfil promedio de similitud del usuario contra todo el catálogo
        sim_scores = self.cosine_sim[indices_comprados].mean(axis=0)

        # Filtrar productos previamente comprados
        sim_series = pd.Series(sim_scores, index=self.product_df["product_id"])
        sim_series = sim_series.drop(labels=valid_purchases, errors="ignore")

        top_products = sim_series.nlargest(k_top)

        res = self.product_df[
            self.product_df["product_id"].isin(top_products.index)
        ].copy()
        res["similarity_score"] = res["product_id"].map(top_products).round(4)
        return res.sort_values(by="similarity_score", ascending=False)

    def get_user_scores(self, user_purchases):
        """Retorna la serie completa de puntuaciones de contenido para ensamblaje en Modelo Híbrido."""
        valid_purchases = [pid for pid in user_purchases if pid in self.indices]
        if not valid_purchases:
            return pd.Series(0.0, index=self.product_df["product_id"])

        indices_comprados = [self.indices[pid] for pid in valid_purchases]
        sim_scores = self.cosine_sim[indices_comprados].mean(axis=0)

        return pd.Series(sim_scores, index=self.product_df["product_id"])