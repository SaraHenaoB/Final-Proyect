# MetricEdge — Sistema de Recomendación para E-commerce

**Proyecto Final — Data Science (Henry)**
**Equipo MetricEdge S.A.S** · *Predictive analytics for smart business decisions.*

Product Owner: SoyHenry · Data Scientists: Christian Tamayo, Verónica Iacono, Simón Bedoya, Deiberlyn Nin, Juan Martín Rossi · Scrum Master: Sara Henao

---

## 📌 Introducción

MetricEdge es el equipo consultor a cargo del Proyecto Final de la carrera de Data Science. El proyecto consiste en construir, para una empresa ficticia de e-commerce, un **sistema de recomendación de productos** que aprenda del historial de compras de sus clientes.

La información original proviene de un dataset público de Kaggle con datos simulados de e-commerce que presenta problemas de calidad reales: nulos con distinto significado según la columna, líneas duplicadas, una columna de negocio no confiable y órdenes que no representan una compra real. Para resolverlo se implementó un proceso completo de **ETL (Extracción, Transformación y Carga)** documentado en `EDA Y ETL CORREGIDOS/etl.py`, seguido de un **EDA profundo** de viabilidad de negocio y dos **modelos de recomendación** entrenados y comparados bajo la misma prueba.

Este README documenta cada paso realizado, las decisiones tomadas y los criterios aplicados durante la limpieza, la preparación y el modelado, con el fin de garantizar la reproducibilidad del proceso y servir de base para el Sprint 2.

## 🎯 Objetivos del proyecto

1. **Entender el negocio**: ¿cómo recomendar a cada cliente los productos con mayor probabilidad de resultarle relevantes o de ser adquiridos en compras futuras?
2. **Diagnosticar y limpiar** el dataset original mediante un EDA primario y un ETL reproducible.
3. **Evaluar la viabilidad de negocio** del sistema de recomendación (sparsity, distribución de popularidad, riesgos operativos) mediante un EDA profundo.
4. **Construir la matriz de interacciones** cliente-producto y los vectores de características de producto.
5. **Entrenar y comparar al menos dos modelos** de recomendación bajo la misma regla de evaluación.
6. **Dejar el proyecto listo** para Sprint 2: modelo híbrido, demo funcional y dashboard.

## 📊 El dataset

[E-commerce Sales & Customer Analytics (150k)](https://www.kaggle.com/datasets/datascikhan/e-commerce-sales-and-customer-analytics) — Kaggle. Datos simulados de e-commerce (2021-2025), 4 tablas originales:

| Tabla | Granularidad | Filas | Contenido |
|---|---|---|---|
| `customer_master.csv` | 1 fila por cliente | 25.000 | Demografía y costo de adquisición (CAC) |
| `product_catalog.csv` | 1 fila por producto | 1.175 | Categoría, subcategoría, marca, precio, rating |
| `order_items.csv` | 1 fila por línea de producto dentro de una orden | 397.569 | Qué se compró, cantidad, descuento, profit |
| `ecommerce_sales_customer_analytics_150k.csv` | 1 fila por orden | 138.116 | Cliente, fecha, estado, canal, pagos, devoluciones, reseña |

## 🗂️ Estructura del repositorio

```
PF-Henry/
├── Documentos/                              # Consigna, lineamientos y propuesta aprobada
├── data/                                     # CSV crudos (input del ETL)
├── EDA Y ETL CORREGIDOS/
│   ├── PF_01_EDA_Primario.ipynb              # 1. Diagnóstico de calidad de datos
│   ├── etl.py                                # 2. Aplica las 8 reglas de limpieza confirmadas
│   └── README_ETL_EDA.md
├── EDA profundo/
│   ├── PF_01_EDA_Profundo.ipynb              # 3. Viabilidad de negocio + feature engineering
│   └── README.md
├── Models/
│   ├── Filtrado colaborativo/
│   │   └── Modelo1_Filtrado_Colaborativo.ipynb   # 4. ALS (implicit)
│   └── Filtrado basado en contenido/
│       ├── content_based.ipynb / content_based.py  # 5. TF-IDF + similitud coseno
│       ├── README.md
│       └── Artefactos_Model_Content_Based/       # .joblib serializados
└── Versiones anteriores/                     # Iteraciones superadas, se conservan por versionamiento
```

`data/processed/` (generado por `etl.py`) contiene los 4 archivos limpios (`*_clean.csv`) más `interacciones_clean.csv`, ya filtrado a órdenes válidas — evita repetir el cruce `order_items` ⋈ `sales` en cada notebook de modelado.

---

## 🧹 Proceso de limpieza y transformación (EDA Primario + ETL)

Antes de limpiar cualquier dato hay que entenderlo. Por eso el trabajo se dividió en dos etapas con propósitos distintos: el **EDA primario** es un diagnóstico (revisa los datos tal como llegan y decide, con evidencia, qué hacer con cada problema) y el **ETL** es la ejecución (aplica esas decisiones de forma ordenada y repetible). Separar ambos pasos evita escribir reglas de limpieza sin haber comprobado antes que son necesarias o correctas.

### 1. Verificación de integridad referencial

Se confirmó que las 4 tablas se pueden combinar sin registros huérfanos:

| Verificación | Resultado |
|---|---|
| `product_id` en `order_items` sin match en `product_catalog` | ✅ 0 |
| `order_id` en `order_items` sin match en `sales` | ✅ 0 |
| `order_id` en `sales` sin match en `order_items` | ✅ 0 |
| `customer_id` en `sales` sin match en `customer_master` | ✅ 0 |

### 2. Valores faltantes: nulos con significado de negocio, no datos perdidos

| Columna | % nulo | Qué significa el vacío |
|---|---|---|
| `return_status`, `return_reason` | 93.15% | La orden nunca fue devuelta |
| `campaign_name` | 60.26% | La venta no vino de ninguna campaña |
| `coupon_code` | 80.01% | No se usó ningún cupón |
| `delivery_days`, `estimated_delivery_days`, `customer_rating`, `review_sentiment`, `customer_review` | 17.78% | La orden no llegó a estado `Completed` (se resuelve solo al filtrar por estado) |

**Decisión:** ninguno de estos nulos se completó con un promedio o un valor inventado. Se dejó constancia explícita de que la ausencia es en sí misma una respuesta de negocio ("sin devolución", "sin cupón", "sin campaña"), para no perder esa información.

### 3. Registros duplicados

| Tabla | Duplicados encontrados |
|---|---|
| `customer_master` (`customer_id`) | 0 |
| `product_catalog` (`product_id`) | 0 |
| `sales` (`order_id`) | 0 |
| `order_items` (`order_id` + `product_id`) | **87** |

**Decisión:** las 87 líneas de `order_items` (mismo producto agregado al carrito más de una vez dentro de la misma orden) se combinan en una sola fila, conservando las 12 columnas originales:

| Columna | Criterio de combinación |
|---|---|
| `unit_price` | Promedio (se confirmó que nunca varía entre duplicados — es el precio de catálogo) |
| `quantity`, `discount_amount`, `gross_sales`, `tax_amount`, `shipping_cost`, `net_sales`, `product_cost`, `profit` | Suma (montos/cantidades totales de la línea combinada) |
| `discount_percentage` | **Recalculado** desde `discount_amount / gross_sales` ya sumados — nunca se promedia un porcentaje directamente, daría un número sin sentido matemático |

### 4. Qué órdenes representan una compra real

| Estado de la orden | ¿Cuenta como preferencia confirmada? |
|---|---|
| `Completed` | ✅ Sí |
| `Returned` | ✅ Sí (la compra sí ocurrió) |
| `Cancelled` | ❌ No (nunca se concretó) |
| `Pending` | ❌ No (todavía no se resuelve) |

**Decisión:** cualquier análisis de comportamiento de compra —incluida la matriz usuario-producto para los modelos— usa únicamente órdenes `Completed` y `Returned`.

### 5. Una columna no confiable, corregida

El dataset trae `customer_order_count` (cuántas órdenes en total hizo cada cliente). Se investigó cómo estaba construida y se encontraron dos problemas:

- Es el **mismo valor en todas las órdenes de un cliente**, sin importar la fecha — no cuenta de forma progresiva, ya trae de antemano el total final (usarla tal cual filtraría información del futuro hacia atrás).
- Solo coincide con el conteo real de órdenes de ese cliente en **~60% de los casos**, lo que sugiere que el archivo es una porción de una base más grande.

**Decisión:** no se usa la columna original. Se recalcula desde cero, de forma progresiva según la fecha real de cada orden, usando solo los datos disponibles en este archivo.

### 6. Clientes sin ninguna compra

Se encontraron **89 clientes** (0.36%) que existen en `customer_master` pero nunca hicieron ninguna orden, ni siquiera cancelada.

**Decisión:** se conservan en la base de clientes (sirven para análisis general de registro vs. conversión), pero se excluyen de cualquier análisis de comportamiento de compra — no hay conducta que aprender de ellos.

### 7. Tipos de dato

`order_date` se convierte de texto a tipo fecha real, necesaria para poder ordenar y calcular con ella en las etapas siguientes.

### 8. Archivos generados por el ETL

| Archivo | Ubicación | Descripción |
|---|---|---|
| `customer_clean.csv` | `data/processed/` | Clientes, sin cambios de contenido |
| `product_clean.csv` | `data/processed/` | Catálogo, sin cambios de contenido |
| `order_items_clean.csv` | `data/processed/` | Deduplicado (regla 3) |
| `sales_clean.csv` | `data/processed/` | `customer_order_count` recalculado (regla 5), tipos corregidos (regla 7) |
| `interacciones_clean.csv` | `data/processed/` | `order_items` ⋈ `sales`, ya filtrado a `Completed`/`Returned` (regla 4) — evita repetir el cruce en cada notebook |

---

## 🔍 EDA Profundo: viabilidad de negocio y feature engineering

A diferencia del EDA primario (calidad estructural), el EDA profundo evalúa si el sistema de recomendación es viable con estos datos y prepara las variables para los modelos.

### Radiografía financiera y operativa

| Métrica | Valor |
|---|---|
| Ticket promedio (AOV) | **$1.282,50** |
| Segmento dominante | Consumer (B2C), ~100% de los ingresos |
| Canales dominantes | Mobile App + Website (canales propios) |
| Categoría líder en ventas netas | **Electronics** (~$41,1M) |
| Categoría líder en ganancia | **Electronics** (~$14,1M) |
| Tasa de devolución | 6.5%–7.2%, uniforme en las 15 categorías (causa sistémica, no de una categoría puntual) |
| Retraso logístico | **14.9%** de las órdenes con dato de entrega llegó fuera de la fecha estimada *(se calcula sobre las órdenes que sí tienen `delivery_days`, no sobre el total — una orden `Cancelled`/`Pending` nunca tiene fecha de entrega, incluirla en el denominador subestimaría el problema)* |

### El hallazgo que definió la arquitectura de modelado: ausencia de Pareto

| Métrica | Valor |
|---|---|
| Sparsity de la matriz usuario-producto | ~98.6%–98.8% (en línea con datasets de referencia como MovieLens) |
| Coeficiente de variación de popularidad de producto | **0.057** |
| % de ventas que concentra el Top 20% del catálogo | **21.5%** (vs. ~80% en un e-commerce típico) |
| Clientes con 4+ compras | ~80% |
| Clientes de una sola compra (cold start) | ~2% |

**Por qué importa:** un catálogo con un coeficiente de variación de solo 0.057 significa que las ventas se reparten de forma casi uniforme entre los 1.175 productos — no hay un "Top 10" que sostenga el negocio. Esto **descarta un baseline de popularidad global** como estrategia viable y exige modelos que capturen afinidad específica cliente-producto (colaborativo y/o basado en contenido).

**Limitación conocida a documentar:** al ser un dataset sintético, es probable que esta ausencia de Pareto y la "fidelidad extrema" de los clientes sean un artefacto de cómo se generaron los datos, no un patrón de negocio real a explotar tal cual. Esto se refleja directamente en las métricas de los modelos (ver siguiente sección).


---

## 🤖 Modelos de recomendación entrenados (Sprint 1)

Ambos modelos se evalúan bajo la misma regla de juego: **split 80% train / 20% test**, midiendo si el Top-K recomendado contiene productos que el cliente efectivamente compró en el conjunto de test.

### Modelo 1 — Filtrado colaborativo (ALS, librería `implicit`)

| Aspecto | Detalle |
|---|---|
| Entrada | Matriz dispersa cliente-producto (24.838 × 1.175), densidad 1.21% |
| Por qué ALS | El feedback es implícito (compras/cantidad), no hay rating explícito por producto a nivel individual |
| Calibración | Factor de confianza `α=40` sobre la matriz de train (la mayoría de los clientes compra 1 sola unidad; sin este ajuste ALS interpreta esa señal como casi nula), `factors=64`, `regularization=0.1`, `iterations=20` |
| **Resultado** | **Precision@10 = 0.86%** |

### Modelo 2 — Filtrado basado en contenido (TF-IDF + similitud coseno)

| Aspecto | Detalle |
|---|---|
| Entrada | Vectores de producto: texto (categoría, subcategoría, marca, descripción vía TF-IDF) + numéricas normalizadas (precio, rating) |
| Uso | Recomendaciones item-to-item y user-to-item (perfil de usuario = promedio de los vectores de lo comprado en train) |
| **Resultado (Top-10)** | **Precision@10 = 0.24%, Recall@10 = 0.58%** |
| Cobertura de catálogo (Top-5) | 25.19% (296 de 1.175 productos distintos recomendados) |

### Comparación (K=10)

| Modelo | Precision@10 | Recall@10 |
|---|---|---|
| Filtrado colaborativo (ALS) | **0.86%** | — |
| Filtrado basado en contenido (TF-IDF + coseno) | 0.24% | 0.58% |

El colaborativo rinde **~3.5x mejor** que el basado en contenido en Precision@10 — señal de que el comportamiento colectivo de compra (aunque débil) aporta más que la similitud de atributos del catálogo en este dataset.

### Por qué las métricas son bajas (y no es un error)

1. **Techo matemático de la métrica:** cada cliente compra en promedio ~14 productos; al ocultar el 20% para test quedan solo 2-3 productos objetivo por usuario, muy por debajo de lo que Precision@10 exige acertar.
2. **Catálogo grande relativo al Top-K:** acertar el ítem exacto entre 1.175 productos es estadísticamente difícil incluso con una señal fuerte.
3. **Ausencia de Pareto:** sin productos "héroe" ni afinidad de categoría fuerte, hay poca señal real de preferencia individual para que cualquier modelo aprenda — confirmado de forma directa: solo ~1.1% de las compras de un período de test son recompra de un producto ya adquirido, y el acierto de categoría es equivalente al azar.

Esto está alineado con la consigna del Sprint 1: el foco está en la correcta aplicación de la metodología (split sin fuga de información, métricas justificadas, comparación justa entre modelos) y no en perseguir una métrica alta a costa de sobreajustar sobre un dataset sintético.

## 🚀 Roadmap de modelado — Modelo híbrido (Sprint 2)

Documentado en `Documentos/Modelos propuestos para MVP.docx`: la propuesta principal es un **enfoque híbrido por conmutación (switching)**:

- **Clientes/productos con historial suficiente** → Filtrado colaborativo (ALS), maximiza personalización basada en comportamiento grupal.
- **Cold start de cliente o producto nuevo** → Filtrado basado en contenido, usando metadatos de categoría/marca del producto y perfil demográfico del cliente.

---

## ✅ Estado frente al Done de Sprint 1

| Criterio (consigna) | Estado |
|---|---|
| Análisis exploratorio enfocado, con visualizaciones clave | ✅ EDA primario + EDA profundo |
| Identificación y tratamiento de valores faltantes y problemas de calidad | ✅ 8 reglas de limpieza documentadas y aplicadas en `etl.py` |
| Ingeniería de características básica | ✅ Matriz de interacciones + vectores de producto (TF-IDF/MinMax) |
| Modelo de recomendación entrenado y baseline implementado | ✅ 2 modelos entrenados y comparados (ALS, Content-Based) |
| Notebook con análisis y pipeline básico reproducible | ✅ |
| Notebook ejecutable sin errores | ✅ |
| Enfoque técnico validado por el PO | ⏳ Pendiente de revisión formal |

## 📋 Pendientes antes del cierre de Sprint 1

- [ ] Corregir en `EDA profundo/PF_01_EDA_Profundo.ipynb` la conclusión de categorías financieras (Electronics es la categoría líder en ventas y ganancia, no Books & Media/Home Appliances/Jewelry).
- [ ] Ajustar el cálculo de % de retraso logístico para dividir solo por órdenes con dato de entrega (14.9%, no 12.23%).
- [ ] Agregar la salvedad de "posible artefacto de datos sintéticos" a las conclusiones de fidelidad extrema y distribución de ventas por edad.
- [ ] Renumerar la sección de outliers (IQR) para que quede en el orden correcto respecto a la sección de Sparsity.

---

## 🔁 Cómo reproducir este proceso

1. Clona este repositorio:

   ```bash
   git clone <url-del-repositorio>
   ```

2. Instala las dependencias principales:

   ```bash
   pip install pandas numpy scipy scikit-learn implicit joblib matplotlib seaborn jupyter
   ```

3. Corre el ETL desde la raíz del repo (lee `data/raw/*.csv`, escribe `data/processed/*_clean.csv`):

   ```bash
   python "EDA Y ETL CORREGIDOS/etl.py"
   ```

4. (Opcional) Revisa los hallazgos del EDA:

   ```bash
   jupyter nbconvert --to notebook --execute "EDA Y ETL CORREGIDOS/PF_01_EDA_Primario.ipynb"
   jupyter nbconvert --to notebook --execute "EDA profundo/PF_01_EDA_Profundo.ipynb"
   ```

5. Entrena y evalúa los modelos:

   ```bash
   jupyter nbconvert --to notebook --execute "Models/Filtrado colaborativo/Modelo1_Filtrado_Colaborativo.ipynb"
   jupyter nbconvert --to notebook --execute "Models/Filtrado basado en contenido/content_based.ipynb"
   ```

## 📄 Licencia

Este proyecto es de uso académico, desarrollado como Proyecto Final de la carrera de Data Science (Henry).

## 📬 Contacto

Para dudas, sugerencias o reportar inconsistencias, abrir un *issue* en el repositorio o contactar al equipo MetricEdge S.A.S.
