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

## Transición Estratégica: Del Sistema de Recomendación al Pronóstico de Demanda

La fase de investigación inicial contempló 6 arquitecturas de recomendación personalizada (filtrado colaborativo, híbrido, canasta de compra, entre otros). Sin embargo, la alta dispersión de los datos evidenció que el comportamiento de compra a nivel individual carece de una señal predictiva consistente. 

Frente a esto, el **Análisis Exploratorio de Datos (EDA) profundo** dictó un cambio de rumbo hacia un modelo predictivo a nivel macro (Pronóstico de Demanda). A continuación, se detallan las razones técnicas de este pivote y las recomendaciones de negocio derivadas.

### Razones Estructurales para el Cambio de Modelo (Hallazgos del EDA)

1. **La Estacionalidad es la Única Variable Confiable:** 
   El análisis de los años 2021-2025 demostró un patrón inquebrantable donde las ventas de noviembre y diciembre superan entre un 55% y 70% a las de un mes típico. Categorías específicas tienen ciclos predecibles (ej. *Office Supplies* trepando sistemáticamente al Top 5 cada verano).
2. **El Crecimiento Depende del Tráfico, No del Ticket Promedio:** 
   A pesar de las fluctuaciones masivas en los ingresos totales, el comportamiento de carrito es estático. El Ticket Promedio (AOV) se ha mantenido congelado en ~$1,250-$1,300, y las unidades por orden entre 5.5 y 6.4. El volumen de ventas estacional responde a la entrada masiva de nuevos compradores, haciendo ineficiente un recomendador enfocado en *cross-selling* a usuarios existentes.
3. **Homogeneidad Demográfica (Ausencia de Sesgo):** 
   El mapa de calor generacional confirmó que las temporadas altas afectan a todos los rangos de edad por igual (varianza < 0.5% entre grupos). Las variables demográficas estáticas no aportan ganancia de información predictiva, cediendo el protagonismo absoluto a la dimensión temporal.

### Recomendaciones de Negocio a partir del Pronóstico

El despliegue de este modelo de pronóstico de demanda permitirá a la empresa accionar sobre tres frentes críticos descubiertos durante el EDA:

* **Recomendación 1 - Mitigación del Riesgo Logístico Crónico:**
  El EDA detectó que el **14.9% de los pedidos que logran entregarse sufren retrasos**. Al pronosticar con precisión los picos de volumen de transacciones, la empresa debe renegociar las capacidades con sus proveedores de última milla y ajustar la dotación de personal de almacén anticipadamente para proteger la experiencia del cliente.
* **Recomendación 2 - Reestructuración de la Estrategia de Descuentos (Márgenes):**
  Se descubrió el "Síndrome del Black Friday": la temporada de Otoño/Invierno duplica las ventas brutas, pero desploma el margen de beneficio operativo (del ~48% en Primavera al ~40%). El pronóstico de demanda debe usarse para planificar compras por volumen con proveedores, reduciendo el costo de adquisición de inventario en lugar de sacrificar todo el margen mediante descuentos agresivos.
* **Recomendación 3 - Abastecimiento Predictivo por Categoría:**
  Al integrar *Target Encoding* estacional, el modelo predecirá qué categorías específicas liderarán cada mes. Se recomienda alinear el presupuesto de marketing y el flujo de caja para sobre-stockear *Sports & Outdoors* en la recta final del año y *Office Supplies* previo al periodo estival, minimizando el costo de oportunidad por quiebre de stock.


***