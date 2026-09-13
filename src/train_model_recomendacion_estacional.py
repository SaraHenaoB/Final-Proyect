# Librerias
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 1. IMPORTAR FUNCIONES DEL FEATURE ENGINEERING
try:
    from ft_engineering_model_recomendacion_estacional import (
        load_data,
        aggregate_monthly_global,
        prepare_product_sales,
        aggregate_monthy_category,
        aggregate_monthly_product,
        add_seasonal_features,
    )
except (ModuleNotFoundError, ImportError):
    from src.ft_engineering_model_recomendacion_estacional import (
        load_data,
        aggregate_monthly_global,
        prepare_product_sales,
        aggregate_monthy_category,
        aggregate_monthly_product,
        add_seasonal_features,
    )

# 2. RUTAS
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "src", "models")
os.makedirs(MODEL_DIR,exist_ok=True)

# 3. CARGAR DATOS Y APLICAR EL MISMO FT_ENGINEERING
def build_training_data():

    print("CARGANDO DATOS PARA TRAINING")
    sales_path = os.path.join(
        DATA_DIR,
        "sales_clean.csv"
    )

    items_path = os.path.join(
        DATA_DIR,
        "order_items_clean.csv"
    )

    product_path = os.path.join(
        DATA_DIR,
        "product_clean.csv"
    )

    sales, items, products = load_data(
        sales_filepath=sales_path,
        items_filepath=items_path,
        product_filepath=product_path
    )

    print("\nDatos originales:")
    print(f"Ventas:    {sales.shape}")
    print(f"Items:     {items.shape}")
    print(f"Productos: {products.shape}")

    # FILTRAR SOLO ÓRDENES COMPLETADAS
    sales = sales[sales["order_status"] == "Completed"].copy()

    print("\n" + "-" * 50)
    print("FILTRADO DE ÓRDENES")
    
    print(f"Órdenes completadas: {len(sales):,}")
    print(f"Estados utilizados: {sales['order_status'].unique()}")

    # AGREGACIÓN GLOBAL
    monthly_global = aggregate_monthly_global(
        sales
    )

    monthly_global = add_seasonal_features(
        monthly_global
    )

    print("\nAgregación global:")
    print(f"Filas: {len(monthly_global):,}")

    # PREPARAR VENTAS DE PRODUCTOS
    product_sales = prepare_product_sales(
        sales,
        items,
        products
    )

    print("\nVentas de productos:")
    print(f"Filas después del merge: "f"\n{len(product_sales):,}")

    # AGREGACIÓN POR CATEGORÍA
    monthly_category = aggregate_monthy_category(
        product_sales
    )

    monthly_category = add_seasonal_features(
        monthly_category
    )

    print("\nAgregación por categoría:")
    print(f"Filas: {len(monthly_category):,}")

    # AGREGACIÓN POR PRODUCTO
    monthly_product = aggregate_monthly_product(
        product_sales
    )

    monthly_product = add_seasonal_features(
        monthly_product
    )

    print("\nAgregación por producto:")
    print(f"Filas: {len(monthly_product):,}")

    return (
        sales,
        monthly_global,
        monthly_category,
        monthly_product
    )

# 4. SEASONAL NAIVE
def seasonal_naive_forecast(train_df,test_df):

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["month"] = (
        train_df["year_month"]
        .dt.month
    )

    test_df["month"] = (
        test_df["year_month"]
        .dt.month
    )

    # Media histórica por mes
    monthly_means = (train_df.groupby("month")["net_sales"].mean())

    # Predicción según el mes
    predictions = (test_df["month"].map(monthly_means).values)

    return predictions

# 5. BACKTESTING
def evaluate_seasonal_model(monthly_global):

    print("\n")
    print("-" * 50)
    print("EVALUACIÓN DEL MODELO ESTACIONAL")
    print("-" * 50)


    df = monthly_global.copy()

    df["year_month"] = pd.to_datetime(
        df["year_month"].astype(str)
    )

    df = df.sort_values("year_month"
    ).reset_index(
        drop=True
    )

    # Años disponibles
    available_years = sorted(df["year_month"]
        .dt.year
        .unique())

    print(f"\nAños disponibles: {available_years}")

    # Backtesting: se evaluan los ultimos 3 años, dejando los 2 primeros como
    # historia minima de entrenamiento. Criterio acordado con el equipo para que
    # todos los documentos del proyecto reporten la misma metrica: predecir un año
    # con un solo año de historia previa no es representativo del uso real del
    # modelo, que siempre contara con varios años acumulados.
    test_years = available_years[2:]

    print(f"Años usados como test: {test_years}")

    # Backtesting
    yearly_results = []

    all_true = []
    all_pred = []

    for test_year in test_years:

        train_df = df[df["year_month"].dt.year < test_year].copy()

        test_df = df[df["year_month"].dt.year == test_year].copy()

        if train_df.empty:
            continue

        if test_df.empty:
            continue

        predictions = seasonal_naive_forecast(
            train_df,
            test_df
        )

        y_true = (
            test_df["net_sales"].values)

        mae = mean_absolute_error(y_true, predictions)

        mape = mean_absolute_percentage_error(y_true, predictions) * 100

        r2 = r2_score(y_true, predictions)

        yearly_results.append({
            "Año": test_year,
            "MAE": mae,
            "MAPE (%)": mape,
            "R²": r2
        })

        all_true.extend(y_true)

        all_pred.extend(predictions)

    results_df = pd.DataFrame(yearly_results)

    # MÉTRICAS GLOBALES
    mae_global = mean_absolute_error(
        all_true,
        all_pred
    )

    mape_global = (
        mean_absolute_percentage_error(
            all_true,
            all_pred) * 100
    )

    r2_global = r2_score(
        all_true,
        all_pred
    )

    print("\n" + "-" * 50)
    print("MÉTRICAS POR AÑO")
    print(results_df.round(4))
    print("\n" + "-" * 50)
    print("MÉTRICAS GLOBALES")
    print(f"MAE  : ${mae_global:,.2f}")
    print(f"MAPE : {mape_global:.2f}%")
    print(f"R²   : {r2_global:.4f}")

    return (
        results_df,
        mae_global,
        mape_global,
        r2_global
    )

# 6. PRECISION@K — RANKING ESTACIONAL
def precision_at_k(monthly_product, k=10):

    df = monthly_product.copy()

    df["year_month"] = pd.to_datetime(df["year_month"].astype(str))

    years = sorted(df["year_month"].dt.year.unique())

    yearly_precision = []

    for test_year in years[1:]:

        # HISTÓRICO
        train = df[df["year_month"].dt.year < test_year].copy()
       
        # TEMPORADA ALTA DEL AÑO TEST
        test_peak = df[
            (df["year_month"].dt.year== test_year)&(df["year_month"].dt.month.isin([11, 12]))].copy()

        if train.empty or test_peak.empty:
            continue
        
        # TOP K HISTÓRICO
        top_train = (train.groupby("product_id")["net_sales"].sum().nlargest(k).index)

        # TOP K REAL DE TEMPORADA
        top_test = (test_peak.groupby("product_id")["net_sales"].sum().nlargest(k).index)

        # INTERSECCIÓN
        hits = len(set(top_train)&set(top_test))

        precision = hits / k

        yearly_precision.append({"Año": test_year,"Hits": hits,"Precision@{}".format(k):precision})

    precision_df = pd.DataFrame(yearly_precision)

    if not precision_df.empty:

        precision_mean = (precision_df["Precision@{}".format(k)].mean())

    else:
        precision_mean = np.nan

    return (
        precision_df,
        precision_mean)

# 6.1 PRECISION@K — NIVEL CATEGORIA (metrica principal del modelo)
def precision_at_k_category(monthly_category, k=5):
    """
    Mismo criterio que precision_at_k, pero a nivel de categoria de producto.
    Esta es la metrica principal del modelo: el ranking de categorias a
    reforzar es lo que se le entrega al negocio, no el de productos sueltos.
    """
    df = monthly_category.copy()

    df["year_month"] = pd.to_datetime(df["year_month"].astype(str))

    years = sorted(df["year_month"].dt.year.unique())

    yearly_precision = []

    for test_year in years[1:]:

        train = df[df["year_month"].dt.year < test_year].copy()

        test_peak = df[
            (df["year_month"].dt.year == test_year) &
            (df["year_month"].dt.month.isin([11, 12]))].copy()

        if train.empty or test_peak.empty:
            continue

        top_train = (train.groupby("product_category")["net_sales"].sum().nlargest(k).index)

        top_test = (test_peak.groupby("product_category")["net_sales"].sum().nlargest(k).index)

        hits = len(set(top_train) & set(top_test))

        yearly_precision.append({
            "Año": test_year,
            "Hits": hits,
            "Precision@{}".format(k): hits / k})

    precision_df = pd.DataFrame(yearly_precision)

    if not precision_df.empty:
        precision_mean = (precision_df["Precision@{}".format(k)].mean())
    else:
        precision_mean = np.nan

    return (precision_df, precision_mean)

# 7. RANKING FINAL DE PRODUCTOS
def train_final_ranking(monthly_product,monthly_category):

    print("\n")
    print("-" * 50)
    print("GENERANDO RANKING FINAL")

    # Ranking de productos 
    product_ranking = (monthly_product.groupby("product_id")["net_sales"].sum().sort_values(ascending=False))

    # Ranking de categorías
    category_ranking = (monthly_category.groupby("product_category")["net_sales"].sum().sort_values(ascending=False))

    print("\nTop 10 productos:")
    print(product_ranking.head(10))
    print("-" * 50)
    print("\nTop categorías:")
    print(category_ranking.head())

    return (
        product_ranking,
        category_ranking)

# 8. PREDICCIÓN DE LA PRÓXIMA TEMPORADA
def generate_future_forecast(monthly_global):

    df = monthly_global.copy()

    df["year_month"] = pd.to_datetime(
        df["year_month"].astype(str)
    )

    df = df.sort_values("year_month")

    last_date = df["year_month"].max()

    future_year = (last_date.year + 1)

    future_dates = pd.date_range(start=f"{future_year}-11-01", periods=2, freq="MS")

    historical_means = (df.assign(month=df["year_month"].dt.month).groupby("month")["net_sales"].mean())

    forecast_values = [historical_means.get(date.month,np.nan)
        for date in future_dates]

    forecast_df = pd.DataFrame({

        "periodo":future_dates.strftime("%Y-%m"),

        "prediccion_net_sales": forecast_values})

    print("\n")
    print("-" * 50)
    print("PREDICCIÓN FUTURA — TEMPORADA ALTA")
    print(forecast_df.round(2))
    return forecast_df

# 9. GUARDAR MODELO
def save_model(product_ranking,category_ranking,monthly_means,metrics,forecast):

    artifact = {
        "model_name":"Seasonal Naive Recommendation Model",
        "model_type":"Seasonal Naive",
        "monthly_means":monthly_means.to_dict(),
        "product_ranking":product_ranking.to_dict(),
        "category_ranking":category_ranking.to_dict(),
        "metrics":metrics,
        "forecast":forecast.to_dict(orient="records")
    }

    model_path = os.path.join(MODEL_DIR,"seasonal_recommendation_model.joblib")

    joblib.dump(artifact, model_path)

    print("-" * 50)
    print("MODELO GUARDADO")
    print(f"\n✓ {model_path}")
    return model_path

# 10. MAIN
def main():
    print("-" * 50)
    print("TRAINING PIPELINE — MODELO ESTACIONAL")

    # Feature Engineering
    (sales, monthly_global, monthly_category, monthly_product) = build_training_data()

    # Backtesting
    (yearly_results, mae, mape, r2) = evaluate_seasonal_model(monthly_global)

    # Precision@10
    (precision10_df,precision10) = precision_at_k(monthly_product,k=10)

    # Precision@20
    (precision20_df, precision20) = precision_at_k(monthly_product,k=20)

    # Precision@5 a nivel CATEGORIA (metrica principal del modelo)
    (precision5_cat_df, precision5_cat) = precision_at_k_category(monthly_category, k=5)

    print("-" * 50)
    print("PRECISION DEL RANKING ESTACIONAL")
    print("\nPrecision@10 por año:")
    print(precision10_df.round(4))
    print(f"\nPrecision@10 promedio:\n{precision10:.2%}")
    print("\nPrecision@20 por año:")
    print(precision20_df.round(4))
    print(f"\nPrecision@20 promedio: \n{precision20:.2%}")
    print("\nPrecision@5 por año (CATEGORIAS - metrica principal):")
    print(precision5_cat_df.round(4))
    print(f"\nPrecision@5 categoria promedio: \n{precision5_cat:.2%}")

    # Ranking final
    (product_ranking,category_ranking) = train_final_ranking(monthly_product,monthly_category)

    # Medias históricas mensuales
    monthly_means = (monthly_global.assign(month=monthly_global["year_month"].dt.month).groupby("month")[
            "net_sales"].mean())

    # Predicción futura
    forecast = generate_future_forecast(monthly_global)

    # Resumen de métricas
    metrics = {
        "Precision@5_categoria":float(precision5_cat),
        "Precision@10":float(precision10),
        "Precision@20":float(precision20),
        "MAE":float(mae),
        "MAPE (%)":float(mape),
        "R2":float(r2)
    }

    metrics_df = pd.DataFrame({
        "Métrica": [
            "Precision@5_categoria",
            "Precision@10",
            "Precision@20",
            "MAE",
            "MAPE (%)",
            "R²"
        ],

        "Valor": [
            precision5_cat,
            precision10,
            precision20,
            mae,
            mape,
            r2
        ]
    })

    print("\n")
    print("-" * 50)
    print("RESUMEN FINAL DE MÉTRICAS")
    print(metrics_df.round(4))

    # Guardar modelo
    save_model(
        product_ranking,
        category_ranking,
        monthly_means,
        metrics,
        forecast
    )

    print("\n")
    print("-" * 50)
    print("TRAINING COMPLETADO CORRECTAMENTE")

# EJECUCIÓN
if __name__ == "__main__":
    main()