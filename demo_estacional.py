"""
demo_estacional.py
Proyecto Final - Sistema de Recomendacion Estacional por Categoria
Equipo MetricEdge

Punto de entrada unico de la demo. No calcula nada por su cuenta:
importa las funciones de feature engineering ya construidas por el
equipo, carga el modelo ya entrenado, y usa drift_utils para el
monitoreo. Este archivo solo se encarga de la interfaz.

Para correr:
    streamlit run demo_estacional.py
"""

import sys
import os
import numpy as np
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ft_engineering_model_recomendacion_estacional import (
    load_data,
    aggregate_monthly_global,
    prepare_product_sales,
    aggregate_monthy_category,
)
from cargar_modelo import cargar_modelo_estacional, predecir_fecha
from drift_utils import calcular_psi, calcular_ks, calcular_js, calcular_chi2, clasificar_alerta_psi

st.set_page_config(page_title="MetricEdge - Recomendacion Estacional", layout="wide")

COLOR_ALERTA = {"verde": "#2ECC71", "amarillo": "#F1C40F", "rojo": "#E74C3C", "gris": "#95A5A6"}

# ---------------------------------------------------------------------
# Carga de datos y del modelo (cacheado, no se repite en cada clic)
# ---------------------------------------------------------------------
@st.cache_data
def cargar_todo():
    sales, items, products = load_data()
    sales = sales[sales["order_status"] == "Completed"].copy()

    monthly_global = aggregate_monthly_global(sales)
    product_sales = prepare_product_sales(sales, items, products)
    monthly_category = aggregate_monthy_category(product_sales)

    return sales, product_sales, monthly_global, monthly_category


@st.cache_resource
def cargar_modelo():
    return cargar_modelo_estacional()


sales, product_sales, monthly_global, monthly_category = cargar_todo()
modelo = cargar_modelo()

st.title("Sistema de Recomendación Estacional por Categoría")
st.caption("Equipo MetricEdge — Proyecto Final Data Science")

tab_recomendacion, tab_validacion, tab_drift = st.tabs(
    ["📦 Recomendación", "📊 Validación histórica", "🔎 Monitoreo de Data Drift"]
)

# =======================================================================
# TAB 1: Recomendacion (usa el modelo YA entrenado, no recalcula nada)
# =======================================================================
with tab_recomendacion:
    st.header("Recomendación de stock para la próxima temporada alta")

    # Agregando bloque en streamlit para que se pueda ingresar fecha y que realice predicción
    MESES_ES = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    ultimo_mes_entrenamiento = pd.to_datetime(
        monthly_global["year_month"].astype(str)
    ).max()
    primer_mes_futuro = (
        ultimo_mes_entrenamiento + pd.offsets.MonthBegin(1)
    ).date()

    anio_default = primer_mes_futuro.year
    if primer_mes_futuro.month > 11:
        anio_default += 1

    fecha_default = date(anio_default, 11, 1)

    fecha_prediccion = st.date_input(
        "📅 Seleccione la fecha que desea estimar",
        value=fecha_default,
        min_value=primer_mes_futuro,
        help=(
            "El modelo es mensual y estacional. La predicción depende del mes "
            "seleccionado, no del día específico."
        ),
    )

    prediccion = predecir_fecha(modelo, fecha_prediccion)

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Fecha seleccionada", fecha_prediccion.strftime("%d/%m/%Y"))
    pc2.metric("Mes", MESES_ES[fecha_prediccion.month])
    pc3.metric("Ventas netas estimadas", f"${prediccion:,.0f}")


    if fecha_prediccion.month in [11, 12]:
        st.success(
            "📈 La fecha seleccionada pertenece a la temporada alta "
            "(noviembre-diciembre)."
        )
    else:
        st.info(
            "La fecha seleccionada está fuera de la temporada alta; "
            "se muestra la estimación mensual basada en el patrón histórico."
        )

    st.caption(
        "Importante: Seasonal Naive predice a nivel mensual. "
        "Por ejemplo, 05/11 y 25/11 reciben la misma predicción porque "
        "ambas pertenecen a noviembre."
    )

    st.divider()

    mes_seleccionado = fecha_prediccion.month
    nombre_mes = MESES_ES[mes_seleccionado]

    product_sales_mes = product_sales[
        product_sales["order_date"].dt.month == mes_seleccionado
    ].copy()

    if product_sales_mes.empty:
        st.warning(
            f"No hay ventas históricas disponibles para {nombre_mes}. "
            "No se pueden generar recomendaciones estacionales para ese mes."
        )
    else:
        # Top-5 categorías para el mes seleccionado
        top5_cat = (
            product_sales_mes
            .groupby("product_category")["net_sales"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Top-5 stock de categorías — {nombre_mes}")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(top5_cat.index[::-1], top5_cat.values[::-1], color="#2E86AB")
            ax.set_xlabel(f"Ventas netas históricas de {nombre_mes} ($)")
            ax.set_title(f"Comportamiento histórico de {nombre_mes}")
            st.pyplot(fig)

        with col2:
            st.subheader("Pronóstico de demanda")
            forecast_seleccionado = pd.DataFrame({
                "periodo": [fecha_prediccion.strftime("%Y-%m")],
                "prediccion_net_sales": [prediccion],
            })
            st.dataframe(forecast_seleccionado, hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"${modelo['metrics']['MAE']:,.0f}")
            c2.metric("MAPE", f"{modelo['metrics']['MAPE (%)']:.2f}%")
            c3.metric("R²", f"{modelo['metrics']['R2']:.3f}")

            st.caption(
                "Las métricas MAE, MAPE y R² corresponden a la validación global "
                "del modelo y no cambian al seleccionar una fecha."
            )

        col_resumen, col_productos = st.columns(2)

        # Obtener la principal categoría, subcategoría y producto del mes
        categoria_principal = top5_cat.index[0] if len(top5_cat) > 0 else "N/D"
        datos_principal = product_sales_mes[
            product_sales_mes["product_category"] == categoria_principal
        ].copy()

        if not datos_principal.empty and "product_subcategory" in datos_principal.columns:
            datos_principal["product_subcategory"] = (
                datos_principal["product_subcategory"].fillna("Sin subcategoría")
            )
            top_subcat_resumen = (
                datos_principal
                .groupby("product_subcategory")["net_sales"]
                .sum()
                .sort_values(ascending=False)
            )
            subcategoria_principal = (
                top_subcat_resumen.index[0] if len(top_subcat_resumen) > 0 else "N/D"
            )
        else:
            subcategoria_principal = "N/D"

        if not datos_principal.empty:
            top_prod_resumen = (
                datos_principal
                .groupby("product_name")["net_sales"]
                .sum()
                .sort_values(ascending=False)
            )
            producto_principal = (
                top_prod_resumen.index[0] if len(top_prod_resumen) > 0 else "N/D"
            )
        else:
            producto_principal = "N/D"

        with col_resumen:
            st.subheader("📋 Resumen de la recomendación")
            st.caption(
                f"Resumen para {nombre_mes} basado en las ventas históricas "
                "de los años disponibles."
            )

            st.metric("💰 Ventas netas estimadas", f"${prediccion:,.0f}")
            st.write(f"**🏆 Categoría principal:** {categoria_principal}")
            st.write(f"**📂 Subcategoría principal:** {subcategoria_principal}")
            st.write(f"**📦 Producto principal:** {producto_principal}")

            if mes_seleccionado in [11, 12]:
                st.success(
                    "📈 Temporada alta: se recomienda priorizar inventario "
                    "en las categorías y productos destacados."
                )
            else:
                st.info(
                    "📊 Temporada regular: las recomendaciones se basan en "
                    "el comportamiento histórico de este mes."
                )

        with col_productos:
            st.subheader(
                f"📦 Productos y subcategorías recomendados para {nombre_mes}"
            )
            st.caption(
                f"Las recomendaciones se obtienen a partir de las ventas históricas "
                f"de {nombre_mes} en los años disponibles."
            )

            for categoria in top5_cat.index:
                with st.expander(f"📦 {categoria}"):
                    datos_categoria = product_sales_mes[
                        product_sales_mes["product_category"] == categoria
                    ].copy()

                    # Top-3 subcategorías dentro de la categoría para el mes seleccionado
                    if "product_subcategory" in datos_categoria.columns:
                        datos_categoria["product_subcategory"] = (
                            datos_categoria["product_subcategory"]
                            .fillna("Sin subcategoría")
                        )

                        top_subcat = (
                            datos_categoria
                            .groupby("product_subcategory")["net_sales"]
                            .sum()
                            .sort_values(ascending=False)
                            .head(3)
                        )

                        st.write("**Subcategorías a priorizar:**")
                        for subcat, valor_subcat in top_subcat.items():
                            st.write(
                                f"• {subcat} — ventas históricas de {nombre_mes}: "
                                f"${valor_subcat:,.0f}"
                            )

                    # Top-3 productos dentro de la categoría para el mes seleccionado
                    top_prod = (
                        datos_categoria
                        .groupby("product_name")["net_sales"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(3)
                    )

                    st.write("**Productos a priorizar:**")
                    for nombre, valor in top_prod.items():
                        st.write(
                            f"• {nombre} — ventas históricas de {nombre_mes}: "
                            f"${valor:,.0f}"
                        )

# =======================================================================
# TAB 2: Validacion historica en vivo (Precision@5 categoria, walk-forward)
# =======================================================================
with tab_validacion:
    st.header("Validación temporal: ¿el modelo hubiera acertado en años anteriores?")
    st.write(
        "Se entrena únicamente con años **anteriores** al seleccionado, y se compara "
        "contra la temporada alta (nov-dic) real de ese año. Esto es una validación en "
        "vivo, no un número guardado de antemano."
    )

    anio_test = st.selectbox("Año a validar", [2022, 2023, 2024, 2025], index=3)

    df_cat = monthly_category.copy()
    train = df_cat[df_cat.year < anio_test]
    test_alta = df_cat[(df_cat.year == anio_test) & (df_cat.month.isin([11, 12]))]

    top_train = train.groupby("product_category")["net_sales"].sum().sort_values(ascending=False).head(5)
    top_test = test_alta.groupby("product_category")["net_sales"].sum().sort_values(ascending=False).head(5)
    aciertos = len(set(top_train.index) & set(top_test.index))

    col1, col2 = st.columns(2)
    col1.metric("Precision@5 (categorías)", f"{aciertos/5:.0%}")
    col2.metric("Categorías acertadas", f"{aciertos} de 5")

    col_izq, col_der = st.columns(2)
    with col_izq:
        st.write(f"**Predicho (datos < {anio_test})**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(top_train.index[::-1], top_train.values[::-1], color="#2E86AB")
        st.pyplot(fig)
    with col_der:
        st.write(f"**Real (temporada alta {anio_test})**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(top_test.index[::-1], top_test.values[::-1], color="#2ECC71")
        st.pyplot(fig)

# =======================================================================
# TAB 3: Monitoreo de Data Drift
# =======================================================================
with tab_drift:
    st.header("Monitoreo de Data Drift")
    st.write(
        "Compara la distribución de los datos de **referencia** (con los que se entrenó "
        "el modelo) contra datos **nuevos**, simulados o subidos en vivo, usando las "
        "mismas 4 métricas aplicadas en el proyecto de riesgo crediticio del equipo."
    )

    origen = st.radio(
        "¿Con qué datos nuevos quieres probar el drift?",
        ["Simular con 2025 (mes a mes)", "Subir mi propio archivo CSV"],
    )

    referencia = sales[sales.order_date.dt.year < 2025]

    if origen == "Simular con 2025 (mes a mes)":
        mes_simulado = st.slider("Mes de 2025 simulado como 'ya llegado'", 1, 12, 6)
        produccion = sales[
            (sales.order_date.dt.year == 2025) & (sales.order_date.dt.month <= mes_simulado)
        ]
        st.caption(f"Datos de producción simulados: enero a mes {mes_simulado} de 2025 ({len(produccion)} órdenes).")
    else:
        archivo = st.file_uploader("Sube un CSV con las mismas columnas que sales_clean.csv", type="csv")
        if archivo is None:
            st.info("Sube un archivo para calcular el drift, o cambia a la simulación automática.")
            st.stop()
        produccion = pd.read_csv(archivo, parse_dates=["order_date"])
        columnas_necesarias = {"order_id", "order_date", "net_sales", "order_status"}
        faltantes = columnas_necesarias - set(produccion.columns)
        if faltantes:
            st.error(
                f"Al archivo le faltan estas columnas obligatorias: {', '.join(sorted(faltantes))}. "
                f"El archivo debe tener al menos: order_id, order_date, net_sales y order_status."
            )
            st.stop()
        produccion = produccion[produccion.order_status == "Completed"].copy()
        if len(produccion) == 0:
            st.warning(
                "El archivo no contiene ninguna orden en estado 'Completed', que es el único "
                "estado que este modelo considera como demanda real. No hay nada que comparar."
            )
            st.stop()

    if len(produccion) == 0:
        st.warning("No hay datos de producción para este mes todavía, elige un mes más adelante.")
        st.stop()

    st.subheader("Resultado de las 4 métricas de drift")

    psi_val = calcular_psi(referencia.net_sales, produccion.net_sales)
    ks_stat, ks_p = calcular_ks(referencia.net_sales, produccion.net_sales)
    js_val = calcular_js(referencia.net_sales, produccion.net_sales)

    prod_full = produccion.merge(product_sales[["order_id", "product_category"]].drop_duplicates("order_id"), on="order_id", how="left")
    chi2_stat, chi2_p = calcular_chi2(
        referencia.merge(product_sales[["order_id", "product_category"]].drop_duplicates("order_id"), on="order_id", how="left")["product_category"].dropna(),
        prod_full["product_category"].dropna(),
    )

    color, mensaje = clasificar_alerta_psi(psi_val)

    st.markdown(
        f"<div style='background-color:{COLOR_ALERTA[color]};padding:15px;border-radius:8px;color:white;font-weight:bold;'>"
        f"PSI = {psi_val:.4f} — {mensaje}</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("KS (estadístico)", f"{ks_stat:.4f}", help=f"p-valor: {ks_p:.4f}")
    c2.metric("Jensen-Shannon", f"{js_val:.4f}")
    if np.isnan(chi2_stat):
        c3.metric("Chi² (categorías)", "N/D",
                  help="No se pudo calcular: los order_id del archivo no cruzan con el catálogo de productos.")
    else:
        c3.metric("Chi² (categorías)", f"{chi2_stat:.2f}", help=f"p-valor: {chi2_p:.4f}")

    st.subheader("Comparación visual de las distribuciones de venta neta")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(referencia.net_sales, bins=40, alpha=0.5, label="Referencia (< 2025)", density=True, color="#2E86AB")
    ax.hist(produccion.net_sales, bins=40, alpha=0.5, label="Producción (nuevo)", density=True, color="#E74C3C")
    ax.legend()
    ax.set_xlabel("Venta neta por orden ($)")
    st.pyplot(fig)

    st.caption(
        "Umbrales de alerta (PSI): menor a 0.1 sin cambio, entre 0.1 y 0.25 moderado, "
        "mayor o igual a 0.25 se recomienda reentrenar."
    )
