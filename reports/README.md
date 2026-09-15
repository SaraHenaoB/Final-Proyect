# Informe Técnico — MetricEdge

**Sistema de Recomendación Estacional de Stock — documentación completa del proceso**

[⬅ Volver al README principal](../README.md)

---

## 📖 Resumen

MetricEdge es un proyecto de ciencia de datos end-to-end construido para una empresa ficticia de e-commerce. El objetivo original era un recomendador de productos personalizado, por cliente. Tras un análisis exploratorio riguroso, los datos mostraron que las interacciones cliente-producto en este dataset **no tienen señal individual explotable** — hallazgo validado con tres enfoques de clasificación independientes (ver [Comparación de modelos](#-comparación-de-modelos) más abajo).

En vez de forzar una solución débil, el equipo redirigió el esfuerzo hacia el problema donde los datos **sí** mostraban un patrón fuerte y validado: la **demanda estacional**. El resultado es un **sistema de recomendación de stock estacional** que le indica a un retailer qué categorías de producto priorizar — y cuánta demanda esperar — antes de los meses de alta temporada (noviembre-diciembre).

Este documento cubre el proceso completo: calidad de datos, limpieza (ETL), análisis exploratorio de negocio, comparación de los 3 modelos candidatos con evaluación honesta (incluyendo dos fugas de datos encontradas y corregidas en el camino), y el pipeline construido sobre el modelo elegido.

## 📊 El dataset

[E-commerce Sales & Customer Analytics (150k)](https://www.kaggle.com/datasets/datascikhan/e-commerce-sales-and-customer-analytics) — Kaggle. Datos simulados de e-commerce (2021-2025), 4 tablas originales:

| Tabla | Granularidad | Filas | Contenido |
|---|---|---|---|
| `customer_master.csv` | 1 fila por cliente | 25.000 | Demografía y costo de adquisición |
| `product_catalog.csv` | 1 fila por producto | 1.175 | Categoría, subcategoría, marca, precio, rating |
| `order_items.csv` | 1 fila por línea de producto | 397.569 | Producto, cantidad, descuento, profit |
| `ecommerce_sales_customer_analytics_150k.csv` | 1 fila por orden | 138.116 | Cliente, fecha, estado, canal, pago, devoluciones, reseña |

## 🧹 Limpieza de datos (EDA + ETL)

Antes de limpiar cualquier dato hay que entenderlo. El trabajo se dividió en dos etapas: el **diagnóstico exploratorio** (`notebooks/01_eda_primario.ipynb`) decide, con evidencia, qué necesita cada problema de calidad — y el **ETL** (`src/etl.py`) ejecuta esas decisiones en un pipeline reproducible.

| Problema detectado | Decisión |
|---|---|
| Nulos en `return_status`, `campaign_name`, `coupon_code`, `delivery_days`, etc. | No se imputan — son respuestas de negocio estructurales ("sin devolución", "sin cupón"), no datos faltantes |
| 87 líneas duplicadas en `order_items` (mismo producto agregado dos veces a una orden) | Se combinan en una sola fila, recalculando `discount_percentage` desde los montos ya sumados |
| Órdenes `Cancelled` / `Pending` | Excluidas del análisis de comportamiento — no representan una compra confirmada |
| `customer_order_count` filtraba el total futuro en cada fila | Recalculada desde cero, de forma progresiva, según la fecha de cada orden |
| 89 clientes sin ninguna orden | Se mantienen en la tabla de clientes, excluidos del análisis de comportamiento de compra |

Detalle completo en `docs/01_readme_eda_primario.md`.

## 🔍 Análisis de negocio (EDA)

| Métrica | Valor |
|---|---|
| Ticket promedio (AOV) | **$1.282,50** |
| Categoría líder en ventas | **Electronics** (~$41,1M) |
| Tasa de devolución | 6.5%–7.2%, uniforme en las 15 categorías |
| Retraso logístico | **14.9%** de las órdenes entregadas |
| Coeficiente de variación de popularidad de producto | **0.057** — las ventas se reparten casi uniformemente entre los 1.175 productos |

Ese último número fue la primera señal de alerta: con casi ninguna variación de popularidad a nivel de producto, cualquier modelo que dependiera de la preferencia individual del cliente iba a tener poco de dónde aprender. Esa hipótesis se puso a prueba directamente con tres modelos candidatos.

---

## 🤖 Comparación de modelos

*(Desarrollo completo en `notebooks/04_evaluacion_metricas_estacional.ipynb`)*

Se plantearon y evaluaron tres problemas de negocio bajo el mismo rigor: **split temporal 80/20** (nunca aleatorio, para no filtrar información del futuro), identificación y eliminación explícita de variables con fuga de información antes de entrenar, y varios algoritmos comparados por problema.

> ⚠️ Las métricas **no son comparables entre sí** — los Modelos 1 y 2 son clasificación binaria (importan ROC-AUC, PR-AUC, Recall, F1); el Modelo 3 es ranking + pronóstico (importan Precision@K, MAPE, R²). La tabla funciona como matriz de decisión, no como un único ranking.

### Modelo 1 — Predicción de retrasos en la entrega

**Target:** `delivery_status == 'Delayed'` · Tasa real de retraso: 14.87%

| Modelo | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.5069 | 0.1498 | 0.153 | 0.140 | 0.146 |
| **LightGBM** | **0.5038** | **0.1494** | **0.147** | **0.199** | **0.169** |
| Random Forest | 0.5025 | 0.1488 | 0.144 | 0.016 | 0.029 |

**Veredicto: sin señal predictiva útil.** Ni siquiera con tasas históricas de retraso por cliente/depósito/transportista/región (calculadas sin fuga), ningún modelo logra distinguir de forma confiable un pedido que se va a retrasar de uno que no.

### Modelo 2 — Predicción de devoluciones

**Target:** `return_status == 'Returned'`

| Modelo | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **HistGradientBoosting** | **0.6044** | **0.1164** | **0.090** | **0.617** | **0.157** |
| Logistic Regression | 0.5901 | 0.0966 | 0.089 | 0.583 | 0.154 |
| LightGBM | 0.5800 | 0.1051 | 0.092 | 0.323 | 0.143 |
| Random Forest | 0.5796 | 0.1004 | 0.095 | 0.346 | 0.149 |

**Veredicto: señal débil, pero presente.** El mejor modelo detecta el 61.7% de las devoluciones reales (Recall), con precisión baja. Es el único candidato con algo de poder de discriminación por encima del azar — no suficiente para ser la base de un producto por sí solo.

### Modelo 3 — Recomendación estacional de stock ✅ *Elegido*

**Pregunta de negocio:** ¿qué categorías conviene reforzar y cuánto volumen de demanda esperar antes de la próxima temporada alta? Dos componentes — ranking estacional y pronóstico de demanda — validados con **backtesting rolling** sobre varios años.

| Métrica | Resultado |
|---|---|
| Precision@5 (categoría) | **100%** — 4 de 4 años de backtesting (2022-2025) |
| Precision@10 (producto) | 50% |
| Precision@20 (producto) | 56.25% |
| MAE del pronóstico | **$71.815** |
| MAPE del pronóstico | **2.47%** |
| R² del pronóstico | **0.984** |

**Top-5 categorías a priorizar:** Electronics, Jewelry, Home Appliances, Automotive, Sports & Outdoors — con un pronóstico en vivo de ≈ **$4,3M** (noviembre) y **$4,66M** (diciembre) de demanda esperada para la próxima temporada alta.

### Resumen ejecutivo

| Problema | Mejor resultado (sin fuga) | ¿Hay señal real? |
|---|---|---|
| Retrasos en la entrega | ROC-AUC ≈ 0.50 | No |
| Devoluciones | ROC-AUC ≈ 0.60 | Muy débil |
| **Recomendación estacional de stock** | **Precision@5 = 100%, R² = 0.984** | **Sí — fuerte y estable** |

Los tres resultados son consistentes entre sí y con el resto del análisis del dataset: este dataset simulado no codifica una relación causal genuina entre las variables disponibles y estos resultados de negocio a nivel individual (retrasos, devoluciones), salvo por un par de variables que resultaron ser fuga de información. Donde sí hay señal real y aprovechable es en la **demanda estacional agregada por categoría** — un patrón consistente y genuino a lo largo de 5 años de historia.

**Decisión:** el equipo avanzó con el **Modelo 3 — Recomendación estacional de stock** como el candidato más defendible para una implementación de negocio real. Los modelos de retrasos y devoluciones quedan documentados como hallazgos exploratorios, a revisitar si se dispusiera de datos operativos más ricos.

---

## ⚙️ Pipeline

1. **Feature engineering** (`src/ft_engineering_model_recomendacion_estacional.py`) — agrega las ventas a nivel mensual (global, por categoría, por producto) y construye variables estacionales (`is_peak_season`, codificación cíclica seno/coseno del mes).
2. **Entrenamiento** (`src/train_model_recomendacion_estacional.py`) — backtesting rolling del pronóstico estacional (MAE/MAPE/R²), Precision@K del ranking, ranking final de categorías/productos y pronóstico de la próxima temporada. Guarda el artefacto entrenado en `src/models/seasonal_recommendation_model.joblib`.
3. **Demo** (`app.py`, Streamlit) — elegí una fecha futura y obtené la recomendación de stock, el pronóstico de demanda asociado, una vista de validación histórica y una pestaña de monitoreo de drift (`src/drift_utils.py`).
4. **Dashboard** (`dashboard/e_commerce_dashboard.pbix`, Power BI) — visualización de negocio de las métricas y recomendaciones para stakeholders no técnicos.

---

Para instrucciones de instalación, demo en vivo y stack tecnológico, ver el [README principal](../README.md).

