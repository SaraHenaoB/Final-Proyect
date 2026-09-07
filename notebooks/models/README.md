#### 📦 Módulo Basado en Contenido (Modelo B) — Sistema de Recomendación

Este módulo contiene la implementación completa, evaluación y empaquetamiento del **Modelo Basado en Contenido** para el catálogo de productos de comercio electrónico.

#### 📋 Resumen del Modelo

El modelo genera recomendaciones **Item-to-Item** y **User-to-Item** mediante el cálculo de **Similitud Coseno** sobre las características vectorizadas del catálogo:
* **Texto (TF-IDF):** Categorías, subcategorías, marcas y descripciones.
* **Variables Numéricas (MinMaxScaler):** Precios y calificaciones (*ratings*).

Sin embargo, al no tener una relación colaborativa que permita identificar mejor el comportamiento de los usuarios, el modelo se encarga de predecir categorias/productos/marcas que el usuario más adquirio, teniendo un porcentaje de precisión del *0.16%*, mientras que el porcentaje de aciertos (recalls - proporción de compras futuras reales que fueron capturadas en las recomendaciones) es del *0.33%*
> **💡 Justificación de Negocio:**
> Los diveros valores de las métricas es el esperado en un filtrado por contenido sobre un catálogo extenso (1,175 ítems). Este modelo es clave para resolver el problema de **Cold Start de producto** y alimentar la arquitectura del **Modelo Híbrido (Modelo C)**.

#### 📊 Resultados de Evaluación Formal (Top-5)

La evaluación se realizó mediante una partición temporal (80% Train / 20% Test) para prevenir *data leakage*, garantizando reproducibilidad determinista mediante ordenamiento explícito de clientes y semilla aleatoria fija (`seed=42`):

| Métrica | Valor | Descripción |
| :--- | :--- | :--- |
| **Precision@5** | **0.16%** | Proporción de productos recomendados que el usuario compró en el periodo de prueba. |
| **Recall@5** | **0.33%** | Proporción de las compras futuras reales que fueron capturadas en el Top-5. |
| **MAP@5** | **0.0009** | *Mean Average Precision*, penaliza las coincidencias en posiciones inferiores. |
| **Catalog Coverage** | **~25.19%** | Porcentaje del catálogo total recomendado activamente a través de los usuarios con una muestra de 500 usuarios. |

#### 📂 Estructura de Artefactos Exportados

Paso a paso para ejecutar las celdas finales del notebook para generar los artefactos serializados en la carpeta `models/`:

* `models/content_cosine_sim.joblib` — Matriz de similitud coseno (*1175 \times 1175*).
* `models/content_tfidf_vectorizer.joblib` — Modelo TF-IDF ajustado.
* `models/content_scaler.joblib` — Escalador MinMax de variables numéricas.
* `src/content_based.py` — Script ejecutable con la clase `ContentBasedEngine`.

#### 🚀 Instrucciones de Uso para el Equipo (Handoff)

Para utilizar el motor de recomendación en otros módulos o en el **Modelo Híbrido**, importa la clase `ContentBasedEngine` desde `src.content_based`:

Descargar dependencias:
```pip install -r requirements.txt
```

```python
try:
    from src.content_based import ContentBasedEngine
except ImportError:
    from content_based import ContentBasedEngine

# 1. Instanciar el motor (carga automáticamente artefactos y catálogo)
engine = ContentBasedEngine()

# 2. Obtener recomendaciones por ID de producto (Item-to-Item)
sample_item = product_model["product_id"].iloc[0]
print("-"*100)
print(f"Recomendaciones para producto: {sample_item}")
display(engine.product_recomendation(sample_item, k_top=3))

# 3. Obtener recomendaciones por historial de compras del usuario (User-to-Item)
sample_user_purchases = train_df["product_id"].head(3).tolist()
print("-"*100)
print(f"Recomendaciones para historial de usuario")
display(
    engine.customer_recomendation_product(
        sample_user_purchases, k_top=3
    )
)

# 4. Obtener serie de scores para ensamble en Modelo Híbrido
scores_continuos = engine.get_user_scores(user_purchases=sample_user_purchases)