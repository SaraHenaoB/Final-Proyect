"""
etl.py
Proyecto Final - Sistema de recomendacion e-commerce
Equipo MetricEdge


Este script aplica las reglas de limpieza confirmadas en el EDA primario
(ver notebooks/PF_01_EDA_Primario.ipynb) y deja guardado en data/processed/
un dataset limpio de proposito general, listo para el EDA profundo, para
consultas SQL, o para feature engineering. No construye la matriz de
interacciones ni calcula metricas de negocio (eso es EDA profundo), solo
deja los datos correctos y confiables.

"""

import os
import pandas as pd

ESTADOS_VALIDOS = ["Completed", "Returned"]


# ---------------------------------------------------------------------
# EXTRACCION
# ---------------------------------------------------------------------
def extraer_datos():
    """
    Extrae las 4 tablas crudas desde CSV, sin ninguna transformacion.
    Este script se debe correr desde la raiz del repositorio, asi:
        python src/etl.py
    """
    customer = pd.read_csv("data/raw/customer_master.csv")
    product = pd.read_csv("data/raw/product_catalog.csv")
    order_items = pd.read_csv("data/raw/order_items.csv")
    sales = pd.read_csv("data/raw/ecommerce_sales_customer_analytics_150k.csv")

    return customer, product, order_items, sales


def cargar_tipos(customer, product, order_items, sales):
    """
    Aplica los tipos de dato correctos. No cambia contenido de negocio.
    """
    sales["order_date"] = pd.to_datetime(sales["order_date"])
    return customer, product, order_items, sales


def validar_integridad_referencial(customer, product, order_items, sales):
    """
    Verifica que las llaves entre tablas calcen sin huerfanos.
    """
    reporte = {
        "order_id_en_order_items_sin_match_en_sales": len(
            set(order_items["order_id"]) - set(sales["order_id"])
        ),
        "order_id_en_sales_sin_match_en_order_items": len(
            set(sales["order_id"]) - set(order_items["order_id"])
        ),
        "product_id_en_order_items_sin_match_en_catalog": len(
            set(order_items["product_id"]) - set(product["product_id"])
        ),
        "customer_id_en_sales_sin_match_en_customer_master": len(
            set(sales["customer_id"]) - set(customer["customer_id"])
        ),
    }
    return reporte


# ---------------------------------------------------------------------
# TRANSFORMACION (las 8 reglas confirmadas en el EDA primario)
# ---------------------------------------------------------------------
def deduplicar_order_items(order_items):
    """
    Regla 1: agrupa por order_id + product_id, combinando las 12 columnas
    originales sin perder ninguna. El criterio de agregacion depende de
    la naturaleza de cada columna, verificado contra los datos reales:

    - unit_price: se confirmo que nunca varia entre duplicados del mismo
      par (es el precio de catalogo del producto), se toma el promedio,
      que en la practica es igual al valor original.
    - quantity, discount_amount, gross_sales, tax_amount, shipping_cost,
      net_sales, product_cost, profit: son montos o cantidades totales de
      esa linea, se suman para reflejar el total real de la combinacion
      order_id + product_id.
    - discount_percentage: NO se promedia. Se confirmo que si varia entre
      duplicados (distinto descuento en cada adicion al carrito), asi que
      promediar los porcentajes originales daria un numero sin sentido
      matematico. Se recalcula desde los montos ya sumados
      (discount_amount / gross_sales), que es la unica forma correcta de
      obtener el porcentaje efectivo de la linea combinada.
    """
    agregado = order_items.groupby(["order_id", "product_id"], as_index=False).agg(
        unit_price=("unit_price", "mean"),
        quantity=("quantity", "sum"),
        discount_amount=("discount_amount", "sum"),
        gross_sales=("gross_sales", "sum"),
        tax_amount=("tax_amount", "sum"),
        shipping_cost=("shipping_cost", "sum"),
        net_sales=("net_sales", "sum"),
        product_cost=("product_cost", "sum"),
        profit=("profit", "sum"),
    )
    agregado["discount_percentage"] = agregado["discount_amount"] / agregado["gross_sales"]

    orden_columnas = ["order_id", "product_id", "quantity", "unit_price",
                       "discount_percentage", "discount_amount", "gross_sales",
                       "tax_amount", "shipping_cost", "net_sales", "product_cost", "profit"]
    return agregado[orden_columnas]


def recategorizar_nulos_estructurales(sales):
    """
    Reglas 3 y 4: return_reason, return_status, coupon_code y campaign_name
    tienen nulos estructurales (ausencia real de la condicion, no dato
    faltante). Se recategorizan en vez de imputarse con un valor generico.
    Los nulos de posventa (regla 5) no se tocan aca: se documentan pero no
    se imputan, porque desaparecen naturalmente al aplicar el filtro de
    order_status en construir_interacciones_limpias().
    """
    sales = sales.copy()
    sales["return_status"] = sales["return_status"].fillna("Sin devolucion")
    sales["return_reason"] = sales["return_reason"].fillna("Sin devolucion")
    sales["coupon_code"] = sales["coupon_code"].fillna("Sin cupon")
    sales["campaign_name"] = sales["campaign_name"].fillna("Sin campana")
    return sales


def recalcular_customer_order_count(sales):
    """
    Regla 7: customer_order_count e is_repeat_customer originales no se usan.
    Se recalculan de forma acumulada por fecha, usando solo las ordenes
    presentes en este archivo, para evitar fuga de informacion (el valor
    original es fijo por cliente sin importar la fecha de cada orden) y la
    inconsistencia detectada contra el conteo real (coincidia en poco mas
    del 60% de los casos).
    """
    sales = sales.copy()
    sales = sales.sort_values(["customer_id", "order_date"])
    sales["customer_order_count_acumulado"] = sales.groupby("customer_id").cumcount() + 1
    sales["is_repeat_customer_recalculado"] = sales["customer_order_count_acumulado"] > 1
    sales = sales.drop(columns=["customer_order_count", "is_repeat_customer"])
    return sales


def excluir_clientes_sin_ordenes(customer, sales):
    """
    Regla 6: se identifican los customer_id que existen en customer_master
    pero no tienen ninguna fila en sales. No se eliminan de customer_master
    (se conservan para analisis demografico general), pero se marca
    explicitamente cuales son, para que cualquier analisis de comportamiento
    los excluya con criterio, no por omision accidental.
    """
    customer = customer.copy()
    clientes_con_orden = set(sales["customer_id"])
    customer["tiene_ordenes"] = customer["customer_id"].isin(clientes_con_orden)
    return customer


def construir_interacciones_limpias(order_items_dedup, sales_transformado):
    """
    Regla 2: aplica el filtro de order_status (solo Completed y Returned
    cuentan como interaccion real) y deja una tabla de interacciones ya
    deduplicada y filtrada, lista para que el EDA profundo o el feature
    engineering la usen sin tener que repetir esta logica.
    """
    interacciones = order_items_dedup.merge(
        sales_transformado[["order_id", "customer_id", "order_date", "order_status"]],
        on="order_id", how="left"
    )
    interacciones = interacciones[interacciones["order_status"].isin(ESTADOS_VALIDOS)].copy()
    return interacciones


def transformar_datos(customer, product, order_items, sales):
    """
    Orquesta las 8 reglas de limpieza sobre las tablas ya extraidas y
    tipificadas. Retorna las tablas limpias mas la tabla de interacciones
    ya lista.
    """
    order_items_dedup = deduplicar_order_items(order_items)
    sales_transformado = recategorizar_nulos_estructurales(sales)
    sales_transformado = recalcular_customer_order_count(sales_transformado)
    customer_transformado = excluir_clientes_sin_ordenes(customer, sales_transformado)
    interacciones = construir_interacciones_limpias(order_items_dedup, sales_transformado)

    return customer_transformado, product, order_items_dedup, sales_transformado, interacciones


# ---------------------------------------------------------------------
# CARGA (guardar el dataset limpio)
# ---------------------------------------------------------------------
def guardar_datos_limpios(customer, product, order_items, sales, interacciones):
    """
    Guarda las tablas ya transformadas en data/processed/, en formato CSV,
    listas para el EDA profundo, para SQL, o para feature engineering.
    """
    os.makedirs("data/processed", exist_ok=True)

    customer.to_csv("data/processed/customer_clean.csv", index=False)
    product.to_csv("data/processed/product_clean.csv", index=False)
    order_items.to_csv("data/processed/order_items_clean.csv", index=False)
    sales.to_csv("data/processed/sales_clean.csv", index=False)
    interacciones.to_csv("data/processed/interacciones_clean.csv", index=False)


# ---------------------------------------------------------------------
# EJECUCION COMPLETA
# ---------------------------------------------------------------------
def ejecutar_etl():
    print("=== ETL: Extraccion ===")
    customer, product, order_items, sales = extraer_datos()
    customer, product, order_items, sales = cargar_tipos(customer, product, order_items, sales)
    print(f"customer    : {customer.shape[0]} filas")
    print(f"product     : {product.shape[0]} filas")
    print(f"order_items : {order_items.shape[0]} filas")
    print(f"sales       : {sales.shape[0]} filas")

    print("\n=== Validacion de integridad referencial ===")
    reporte = validar_integridad_referencial(customer, product, order_items, sales)
    for chequeo, resultado in reporte.items():
        estado = "OK" if resultado == 0 else "REVISAR"
        print(f"[{estado}] {chequeo}: {resultado}")

    print("\n=== ETL: Transformacion (8 reglas confirmadas en el EDA primario) ===")
    customer, product, order_items, sales, interacciones = transformar_datos(
        customer, product, order_items, sales
    )
    print(f"order_items tras deduplicar        : {order_items.shape[0]} filas")
    print(f"interacciones tras filtrar status  : {interacciones.shape[0]} filas")
    print(f"clientes sin ninguna orden marcados: {(~customer['tiene_ordenes']).sum()}")

    print("\n=== ETL: Carga (guardando en data/processed/) ===")
    guardar_datos_limpios(customer, product, order_items, sales, interacciones)
    print("Archivos guardados en data/processed/:")
    print("  customer_clean.csv, product_clean.csv, order_items_clean.csv,")
    print("  sales_clean.csv, interacciones_clean.csv")

    return customer, product, order_items, sales, interacciones


if __name__ == "__main__":
    ejecutar_etl()
