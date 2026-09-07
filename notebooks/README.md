# EDA Profundo — Proyecto Final: Sistema de Recomendación E-commerce

**Equipo:** MetricEdge

**Dataset:** E-commerce Sales & Customer Analytics (150k), Kaggle

## El dataset de partida

El proyecto trabaja con cuatro tablas originales:

- **Clientes**: un registro por cada cliente registrado.
- **Catálogo de productos**: un registro por cada producto disponible.
- **Detalle de órdenes**: un registro por cada producto específico dentro de cada compra.
- **Ventas**: un registro por cada orden completa, con su estado, fecha, canal y cliente.


**¿Qué se hizo en el EDA Profundo?**
A diferencia del EDA primario (enfocado en la limpieza estructural de los datos), el EDA Profundo se diseñó para entender la viabilidad del negocio y preparar las variables (Feature Engineering) para los motores de recomendación. Las acciones principales fueron:

Evaluación Estructural de la Matriz de Interacciones: Se calculó la Sparsity (dispersión) cruzando usuarios vs. productos para definir la viabilidad de modelos colaborativos.

Análisis de Distribución de Catálogo: Se graficó la curva acumulativa de ventas (Long Tail) para comprobar la existencia (o ausencia) de productos "héroe" bajo la regla de Pareto.

Medición de Retención y Cold Start: Se segmentó a los clientes según su frecuencia histórica de compra para cuantificar el volumen real de usuarios nuevos frente a los fidelizados.

Construcción del Score de Relevancia (Target): Mediante reglas de negocio, se sintetizó el feedback implícito y explícito (compras repetidas, calificaciones y devoluciones) en un score numérico que servirá como variable objetivo para los modelos de clasificación.

Radiografía Financiera y Operativa: Se analizaron los tickets promedios, la rentabilidad real por categoría, la adopción de canales de venta (App vs. Web) y se detectaron riesgos operativos (retrasos logísticos).

**Hallazgos del EDA Profundo (Business Insights)**
Durante la fase exploratoria, descubrimos patrones atípicos que definieron por completo la arquitectura de nuestros modelos:

Lealtad Extrema (Ausencia de Cold Start Severo): El 80.6% de los clientes han comprado 4 o más veces. Solo el 2.1% son compradores de una sola vez.

Catálogo Democratizado (Falso Pareto): Las ventas están distribuidas equitativamente en los 1,175 productos. No hay un "Top 10" que sostenga el negocio de manera aislada.

Ticket Premium y B2C Dominante: El ticket promedio es de $1,282.50 USD, impulsado casi en su totalidad por el segmento "Consumer" a través de canales nativos (Mobile App y Website).

Alta Dispersión (Sparsity del 98.6%): Los clientes compran de manera muy aislada y no descubren orgánicamente productos fuera de su nicho de interés inicial.



***