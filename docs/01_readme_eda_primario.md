# EDA Primario y ETL — Proyecto Final: Sistema de Recomendación E-commerce

**Equipo:** MetricEdge

**Dataset:** E-commerce Sales & Customer Analytics (150k), Kaggle

## El dataset de partida

El proyecto trabaja con cuatro tablas originales:

- **Clientes**: un registro por cada cliente registrado.
- **Catálogo de productos**: un registro por cada producto disponible.
- **Detalle de órdenes**: un registro por cada producto específico dentro de cada compra.
- **Ventas**: un registro por cada orden completa, con su estado, fecha, canal y cliente.

## ¿Por qué el proyecto se organiza en dos pasos separados: EDA primario y ETL?

Antes de limpiar cualquier dato hay que entenderlo. Por eso el trabajo se dividió en dos etapas con propósitos distintos:

El **EDA primario** es un diagnóstico. Su trabajo es revisar los datos tal como llegan, encontrar problemas de calidad (información faltante, registros repetidos, inconsistencias) y decidir, con argumento, qué hacer con cada uno.

El **ETL** es la ejecución. Toma las decisiones ya tomadas en el diagnóstico y las aplica de forma ordenada y repetible, dejando un conjunto de datos limpio y confiable, listo para que el resto del equipo lo use.

Separar estos dos pasos evita un error común: escribir reglas de limpieza sin haber comprobado antes que son necesarias o correctas.

## Qué se hizo en el EDA primario

### 1. Entender qué representa cada tabla

Antes de intervenir los datos, se identificó el nivel de detalle de cada tabla. La tabla de clientes y la de productos tienen un registro por cliente y por producto. La tabla de ventas tiene un registro por cada orden completa. La tabla de detalle de órdenes, en cambio, tiene un registro por cada producto específico dentro de cada orden, es la única que permite saber exactamente qué se compró en cada ocasión.

### 2. Confirmar que las tablas se pueden combinar sin errores

Se revisó que todo producto vendido exista en el catálogo, que todo cliente que aparece en una venta exista en la base de clientes, y así con el resto de las combinaciones posibles entre las cuatro tablas. El resultado fue limpio en los cuatro casos: no se encontró ningún registro suelto que no calzara con las demás tablas.

### 3. Revisar la información faltante

Se revisaron las cuatro tablas en busca de datos vacíos, no solo la que resultó tenerlos. La base de clientes, el catálogo de productos y el detalle de órdenes no presentaron ningún valor faltante. El problema está concentrado únicamente en la tabla de ventas, donde se encontraron varias columnas con datos vacíos, y se investigó cada una por separado antes de decidir qué hacer. Se confirmó que ciertos vacíos no son en realidad información perdida, sino la forma en que el dato representa una condición que no aplica. Por ejemplo, la razón de una devolución aparece vacía únicamente cuando esa orden nunca fue devuelta, no porque falte el dato. Lo mismo ocurre con el código de cupón (vacío cuando no se usó ninguno) y con el nombre de campaña de marketing (vacío cuando la venta no vino de ninguna campaña).

También se encontraron vacíos en los datos posteriores a la compra, como la calificación del cliente o los tiempos de entrega, y se confirmó que estos coinciden con órdenes que nunca llegaron a completarse, así que no representan un problema aparte, se resuelven solos más adelante al filtrar por el estado de la orden.

**Decisión:** ninguno de estos datos faltantes se completó con un promedio o con un valor inventado. En su lugar, se dejó constancia explícita de que su ausencia es en sí misma una respuesta de negocio ("sin devolución", "sin cupón", "sin campaña"), para no perder esa información.

### 4. Confirmar que no haya registros repetidos donde no deberían existir

Antes de revisar el caso conocido del detalle de órdenes, se confirmó que la base de clientes, el catálogo de productos y la tabla de ventas no tuvieran más de un registro para el mismo cliente, producto u orden. El resultado fue limpio en las tres: cada identificador aparece exactamente una vez donde se espera que así sea.

Luego se revisó el detalle de órdenes, y ahí sí se encontraron 87 casos donde el mismo producto aparece más de una vez dentro de la misma orden, con cantidades y montos distintos cada vez. La explicación más razonable es que el cliente agregó el mismo producto al carrito en más de un momento durante la misma compra.

**Decisión:** estos casos no se eliminan, se combinan en una sola fila por cada orden y producto, conservando las doce columnas originales de la tabla (precio, cantidad, descuento, impuesto, envío, costo y utilidad), cada una sumada o recalculada según corresponda a su naturaleza. El precio unitario, por ejemplo, se mantiene igual porque no cambia entre las repeticiones, mientras que los montos totales sí se suman, y el porcentaje de descuento se vuelve a calcular a partir de los montos ya combinados, en vez de promediar los porcentajes originales, para que el resultado siga siendo matemáticamente correcto.

### 5. Decidir qué órdenes representan una compra real

No todas las órdenes reflejan una decisión de compra confirmada. Se identificaron cuatro estados posibles: órdenes completadas, devueltas, canceladas y pendientes. Se determinó que solo las completadas y las devueltas representan una elección real del cliente, porque en ambos casos la compra sí llegó a ocurrir. Las canceladas nunca se concretaron, y las pendientes todavía no se resuelven, así que ninguna de las dos se puede tratar como una preferencia confirmada.

**Decisión:** cualquier análisis de comportamiento de compra debe considerar únicamente las órdenes completadas o devueltas.

### 6. Investigar una columna que resultó no ser confiable

El dataset trae una columna que indica cuántas órdenes en total ha hecho cada cliente. Antes de usarla, se investigó cómo estaba construida, y se encontraron dos problemas serios. El primero es que el valor es el mismo en todas las órdenes de un cliente, sin importar la fecha de cada una, lo cual significa que no cuenta de forma progresiva, sino que ya trae de antemano el total final. Usar esta columna tal cual le daría a cualquier análisis información del futuro del cliente que en la realidad todavía no existiría en el momento de cada compra.

El segundo problema es que ese número no siempre coincide con la cantidad real de órdenes que aparecen en este archivo para ese cliente, coincide en poco más del sesenta por ciento de los casos. Esto sugiere que el archivo que se está usando es apenas una parte de una base de datos más grande.

**Decisión:** no usar esta columna tal como viene. En su lugar, se calcula de nuevo el número de órdenes de cada cliente, pero de forma progresiva según la fecha real de cada una, usando solamente la información que sí está disponible en este archivo.

### 7. Identificar clientes sin ninguna compra

Se encontraron 89 clientes que existen en la base de clientes pero que nunca realizaron ninguna orden, ni siquiera una cancelada. Representan menos del uno por ciento del total.

**Decisión:** estos clientes no se eliminan de la base de clientes, porque pueden servir para un análisis general de quién se registra pero no compra. Sin embargo, se excluyen de cualquier análisis de comportamiento de compra, porque no existe ningún dato de conducta que se pueda aprender de ellos.

### El resultado del EDA primario

Todas estas decisiones se consolidaron en una lista final de ocho reglas de limpieza, cada una respaldada por la evidencia encontrada. Esta lista es la que el ETL convierte en acción.

## Qué se hizo en el ETL

El ETL tomó las ocho reglas del EDA primario y las aplicó en tres momentos:

**Primero**, se extrajeron las cuatro tablas originales y se corrigió el tipo de dato de la fecha de las órdenes, que originalmente llega como texto y se convierte en una fecha real, necesaria para poder ordenar y calcular con ella más adelante.

**Segundo**, se aplicaron las ocho reglas de limpieza: se combinaron los productos repetidos dentro de una misma orden, se dejó marcado con claridad cuáles vacíos representan una condición real de negocio en vez de completarlos con datos inventados, se recalculó de forma correcta el número de órdenes por cliente, y se identificó y marcó a los clientes que nunca compraron nada.

**Tercero**, se guardó el resultado limpio en cinco archivos separados: los cuatro archivos originales ya corregidos (clientes, productos, detalle de órdenes y ventas), y un quinto archivo adicional que combina el detalle de órdenes con la información de la venta, ya filtrado únicamente por las órdenes que representan una compra real. Este quinto archivo no reemplaza a los cuatro originales, existe como una ayuda adicional para quien vaya a construir el análisis de comportamiento de compra más adelante, evitándole repetir ese cruce de información desde cero.
