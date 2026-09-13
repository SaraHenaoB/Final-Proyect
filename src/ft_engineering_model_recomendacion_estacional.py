# Archivo para transformaciones de columnas y preparación de datos para entrenamientos
# Librerias a usar
import os
import pandas as pd
import numpy as np


# Carga de datos 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLAS = os.path.join(BASE_DIR, "data", "processed")


def load_data(
        sales_filepath: str = os.path.join(TABLAS, "sales_clean.csv"),
        items_filepath: str = os.path.join(TABLAS, "order_items_clean.csv"),
        product_filepath: str = os.path.join(TABLAS, "product_clean.csv")
    ):
    
    """Carga de las tablas a utilizar con script dinamico para que el codigo corra sin importar desde que carpteta 
    se este localizado"""

    filepaths = {
        "sales": sales_filepath,
        "items": items_filepath,
        "products": product_filepath
    }

    # Validar que los archivos existan
    for name, path in filepaths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: No se encontró el archivo '{name}' en la ruta: {path}")

    # Cargar DataFrames
    sales = pd.read_csv(sales_filepath)
    items = pd.read_csv(items_filepath)
    products = pd.read_csv(product_filepath)

    # Validar fecha
    if "order_date" not in sales.columns:
        raise KeyError(
            "La columna 'order_date' no está presente en sales_clean.csv."
        )

    sales["order_date"] = pd.to_datetime(
        sales["order_date"],
        errors="coerce"
    )

    # Eliminar fechas inválidas
    sales = sales.dropna(subset=["order_date"]).copy()

    print("Tablas cargadas exitosamente desde data/processed/...\n")
    print("-"*50)
    print("\nValidación correcta de los estados de las columnas!")
    return sales, items, products

# Agregación mensual global
def aggregate_monthly_global(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Genera ventas mensuales a nivel global.
    """

    df = sales.copy()

    # Para el modelo usamos únicamente órdenes completadas
    if "order_status" in df.columns:
        df = df[
            df["order_status"] == "Completed"
        ].copy()

    # Año-mes
    df["year_month"] = df["order_date"].dt.to_period("M")

    monthly = (
        df.groupby("year_month")
        .agg(
            net_sales=("net_sales", "sum"),
            quantity=("quantity", "sum"),
            total_orders=("order_id", "nunique")
        )
        .reset_index()
    )

    # Variables temporales
    monthly["year"] = monthly["year_month"].dt.year
    monthly["month"] = monthly["year_month"].dt.month
    monthly["quarter"] = monthly["year_month"].dt.quarter

    # Orden cronológico
    monthly = (
        monthly
        .sort_values("year_month")
        .reset_index(drop=True)
    )

    return monthly

# Se preparan los datos de productos utilizando las 3 tablas
def prepare_product_sales(
        sales: pd.DataFrame,
        items: pd.DataFrame,
        products: pd.DataFrame
)-> pd.DataFrame:
    """ Se creará ordenes +  productos + catalogos. Dando como resultado cada fila que representa un producto vendido
    dentro de unaa orden"""

    completed_order= sales[sales["order_status"]=="Completed"][["order_id","order_date"]].copy()

    # Fecha de ordenes con Items
    product_sales = items.merge(
        completed_order,
        on="order_id",
        how="inner"
    )

    # Extrayendo información del catalogo
    product_colums=[
    "product_id",
    "product_name",
    "product_category",
    "product_subcategory",
    "brand",
    "supplier",
    "product_rating"
    ]

    # Solo columnas existentes:
    product_colums = [col for col in product_colums
                      if col in products.columns
                      ]

    product_sales = product_sales.merge(
        products[product_colums],
        on="product_id",
        how="left"
    )

    return product_sales

# Agregación mensual por categoria de productos
def aggregate_monthy_category(product_sales:pd.DataFrame) -> pd.DataFrame:
    """ Generando ventas mensuales de categorias"""

    df = product_sales.copy()
    df["year_month"] = (df["order_date"].dt.to_period("M"))
    monthly = (
        df.groupby(["year_month", "product_category"]).agg(
            net_sales=("net_sales", "sum"),
            quantity=("quantity", "sum"),
            total_orders=("order_id", "nunique")
        )
        .reset_index()
    )

    monthly["year"] = monthly["year_month"].dt.year
    monthly["month"] = monthly["year_month"].dt.month
    monthly["quarter"] = monthly["year_month"].dt.quarter

    monthly = (monthly.sort_values(["year_month", "product_category"]).reset_index(drop=True))

    return monthly

# Agregación mensual por producto
def aggregate_monthly_product(product_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Extrayendo ventas mensuales por producto.
    """

    df = product_sales.copy()

    df["year_month"] = (df["order_date"].dt.to_period("M"))

    monthly = (df.groupby(["year_month", "product_id"]).agg(
            net_sales=("net_sales", "sum"),
            quantity=("quantity", "sum"),
            total_orders=("order_id", "nunique")
        )
        .reset_index()
    )

    monthly["year"] = monthly["year_month"].dt.year
    monthly["month"] = monthly["year_month"].dt.month
    monthly["quarter"] = monthly["year_month"].dt.quarter

    monthly = (monthly.sort_values(["year_month", "product_id"]).reset_index(drop=True))

    return monthly


# Variables estacionales
def add_seasonal_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Añade características cíclicas y marcas de alta temporada (Q4 / Navidad)."""
    df = monthly_df.copy()

    # Alta temporada (Noviembre y Diciembre)
    df["is_peak_season"] = (df["month"].isin([11, 12])).astype(int)

    # Transformación ciclica del Mes (Seno y Coseno)
    # Permite a modelos basados en regresión entender que Diciembre (12) y Enero (1) están juntos
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    return df

# Pipeline completo 
def run_feature_engineering(
        sales_filepath: str = os.path.join(TABLAS, "sales_clean.csv"),
        items_filepath: str = os.path.join(TABLAS, "order_items_clean.csv"),
        product_filepath: str = os.path.join(TABLAS, "product_clean.csv"),
        output_dir: str = os.path.join(TABLAS, "models")
    ):

    """Ejecutor del pipeline Feature Engineering y guarda los datasets procesados."""
    print("Pipeline: Feature Engineering del Modelo Estacional de recomendación operando correctamente...")
    
    # Cargar datos
    sales, items, products = load_data(
        sales_filepath,
        items_filepath,
        product_filepath
        )
    print(f"Ventas: {sales.shape}")
    print(f"Items: {items.shape}")
    print(f"Productos: {products.shape}")

    # Agregación Global
    monthly_global = aggregate_monthly_global(sales)
    monthly_global = add_seasonal_features(monthly_global)

    print("Agregaciones globales gestionadas correctamente!")

    # Gestión de información de productos
    product_sales = prepare_product_sales(
        sales,
        items,
        products
    )

    print(f"Filas despues del merge: \n{product_sales.shape}")

    # Categorias de productos
    monthly_category = aggregate_monthy_category(product_sales)
    monthly_category = add_seasonal_features(monthly_category)

    print("Ventas mensuales por categorias finalizadas con exito!")

    # Agregación información productos
    monthly_product = aggregate_monthly_product(product_sales)
    monthly_product = add_seasonal_features(monthly_product)

    print("Ventas mensuales por productos finalizadas con exito!")

    # Guardando los datos/tablas procesados para el modelo sobre nueva carpeta
    os.makedirs(
        output_dir,
        exist_ok=True)

    global_path = os.path.join(
        output_dir,
        "monthly_sales_global.csv"
    )

    category_path = os.path.join(
        output_dir,
        "monthly_sales_category.csv"
    )

    product_path = os.path.join(
        output_dir,
        "monthly_sales_product.csv"
    )

    monthly_global.to_csv(
        global_path,
        index=False
    )

    monthly_category.to_csv(
        category_path,
        index=False
    )

    monthly_product.to_csv(
        product_path,
        index=False
    )

    # Resumen de todo el pipeline
    print("-" * 50)
    print("\nFEATURE ENGINEERING COMPLETADO\n")
    print(f"Global:    {monthly_global.shape}")
    print(f"Categoria: {monthly_category.shape}")
    print(f"Producto:  {monthly_product.shape}")

    print("-"*50)
    print("\nArchivos generados:\n")
    print(f"{global_path}")
    print(f"{category_path}")
    print(f"{product_path}")

    print("-"*50)
    print("\nVariables estacionales creadas:\n")

    print("1. year")
    print("2. month")
    print("3. quarter")
    print("4. is_peak_season")
    print("5. sin_month")
    print("6. cos_month")

# Ejecución
if __name__ == "__main__":

    run_feature_engineering(
        sales_filepath=os.path.join(TABLAS, "sales_clean.csv"),
        items_filepath=os.path.join(TABLAS, "order_items_clean.csv"),
        product_filepath=os.path.join(TABLAS, "product_clean.csv"),
        output_dir=os.path.join(TABLAS, "models")
    )