# Modelo_Hibrido.ipynb

Modelo híbrido de recomendación, construido sobre los modelos ya desarrollados por el equipo — el
colaborativo (ALS) y el basado en contenido (TF-IDF + similitud coseno, con impulso de recompra) — sin
modificar la lógica de ninguno de los dos. El de contenido genera los candidatos, el score de ALS se
suma como una feature más, y un clasificador de re-ranking decide qué candidatos ofrecerle a cada cliente.

## 📁 Archivos necesarios

Tienen que estar **en la misma carpeta** que el notebook:

| Archivo | Qué es |
|---|---|
| `hybrid_candidates_dataset.csv` | Dataset de entrenamiento del clasificador (candidatos + features + etiqueta) |
| `hybrid_val_full_candidates.csv` | Candidatos completos (sin submuestrear) de los clientes de validación, necesarios para medir Precision@10/Recall@10 correctamente |

Si no los tenés a mano, se regeneran desde cero con `generar_candidatos_hibrido.py` (tarda ~15 minutos,
porque recalcula la similitud de contenido para cada cliente).

## 📦 Librerías

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`. Ninguna requiere instalación
especial más allá de `pip install`.

## ⚠️ Por qué existe este notebook

La primera versión del re-ranker (armada por la compañera que hizo el modelo de contenido) reportaba
**ROC-AUC = 0.72**, pero evaluado sobre las mismas filas con las que se entrenó — sin un conjunto de
validación separado. Este notebook corrige eso: separa **clientes** (no filas) en train/validación, para
que la métrica mida generalización real y no memorización.

## 📊 Resultados

| Métrica | Valor |
|---|---|
| ROC-AUC en-muestra (como estaba, inválido) | 0.72 |
| ROC-AUC con validación honesta (clientes nunca vistos) | **~0.51** |
| Cobertura del generador de candidatos (compra real dentro de los 20 candidatos) | 4.7% de los clientes |
| Recall@10 dentro de ese pool de candidatos | 48%-53% |

**Conclusión del notebook:** el cuello de botella real no es el clasificador de re-ranking, es la
generación de candidatos — solo en el 4.7% de los clientes la compra futura está entre los 20 candidatos
que arma el modelo de contenido. Agregar el score de ALS como feature no mejora de forma sustancial sobre
usar solo el ranking de contenido.

## ▶️ Cómo correrlo

Con los 2 CSV en la misma carpeta, ejecutar el notebook de punta a punta — no requiere ningún ajuste de
rutas ni parámetros.
