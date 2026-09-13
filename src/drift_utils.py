"""
drift_utils.py
Proyecto Final - Sistema de recomendacion estacional
Equipo MetricEdge

Funciones puras para medir data drift: reciben dos distribuciones
(referencia y produccion) y devuelven un numero. No dependen de
Streamlit ni de ninguna interfaz, para poder probarse por separado
antes de conectarlas a la demo.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon


def calcular_psi(referencia, produccion, bins=10):
    """
    Population Stability Index. Mide que tanto se corrio la distribucion
    de una variable NUMERICA entre dos periodos.

    Interpretacion (umbral estandar de la industria):
        PSI < 0.1              -> sin cambio significativo
        0.1 <= PSI < 0.25       -> cambio moderado, monitorear
        PSI >= 0.25             -> cambio significativo, considerar reentrenar
    """
    referencia = np.asarray(referencia, dtype=float)
    produccion = np.asarray(produccion, dtype=float)

    referencia = referencia[~np.isnan(referencia)]
    produccion = produccion[~np.isnan(produccion)]

    if len(referencia) == 0 or len(produccion) == 0:
        return np.nan

    # Los limites de los bins se definen SIEMPRE sobre la referencia,
    # nunca sobre produccion, para que la comparacion sea justa.
    limites = np.percentile(referencia, np.linspace(0, 100, bins + 1))
    limites[0] = -np.inf
    limites[-1] = np.inf
    limites = np.unique(limites)

    freq_ref, _ = np.histogram(referencia, bins=limites)
    freq_prod, _ = np.histogram(produccion, bins=limites)

    prop_ref = freq_ref / freq_ref.sum()
    prop_prod = freq_prod / freq_prod.sum()

    # Evitar log(0) o division por 0 rellenando con un valor minimo
    epsilon = 1e-4
    prop_ref = np.where(prop_ref == 0, epsilon, prop_ref)
    prop_prod = np.where(prop_prod == 0, epsilon, prop_prod)

    psi = np.sum((prop_prod - prop_ref) * np.log(prop_prod / prop_ref))
    return float(psi)


def calcular_ks(referencia, produccion):
    """
    Prueba de Kolmogorov-Smirnov. Compara si dos muestras vienen de la
    misma distribucion. Devuelve el estadistico KS y el p-valor.
    Un p-valor menor a 0.05 sugiere que las distribuciones son distintas.
    """
    referencia = np.asarray(referencia, dtype=float)
    produccion = np.asarray(produccion, dtype=float)
    referencia = referencia[~np.isnan(referencia)]
    produccion = produccion[~np.isnan(produccion)]

    if len(referencia) == 0 or len(produccion) == 0:
        return np.nan, np.nan

    estadistico, p_valor = ks_2samp(referencia, produccion)
    return float(estadistico), float(p_valor)


def calcular_js(referencia, produccion, bins=10):
    """
    Divergencia de Jensen-Shannon. Compara dos distribuciones completas,
    de forma mas estable que KS cuando hay valores raros o poco comunes.
    Devuelve un numero entre 0 (identicas) y 1 (completamente distintas).
    """
    referencia = np.asarray(referencia, dtype=float)
    produccion = np.asarray(produccion, dtype=float)
    referencia = referencia[~np.isnan(referencia)]
    produccion = produccion[~np.isnan(produccion)]

    if len(referencia) == 0 or len(produccion) == 0:
        return np.nan

    limites = np.percentile(referencia, np.linspace(0, 100, bins + 1))
    limites[0] = -np.inf
    limites[-1] = np.inf
    limites = np.unique(limites)

    freq_ref, _ = np.histogram(referencia, bins=limites)
    freq_prod, _ = np.histogram(produccion, bins=limites)

    prop_ref = freq_ref / freq_ref.sum()
    prop_prod = freq_prod / freq_prod.sum()

    return float(jensenshannon(prop_ref, prop_prod))


def calcular_chi2(referencia_categorica, produccion_categorica):
    """
    Prueba Chi-cuadrado. Mide si cambio la PROPORCION entre categorias
    (por ejemplo, si la participacion de cada categoria de producto en
    las ventas totales se mantiene o cambio). Recibe dos series de texto
    (nombres de categoria), no numeros.
    Devuelve el estadistico chi2 y el p-valor. Un p-valor menor a 0.05
    sugiere que la proporcion entre categorias cambio de forma real.
    """
    conteo_ref = pd.Series(referencia_categorica).dropna().value_counts()
    conteo_prod = pd.Series(produccion_categorica).dropna().value_counts()

    # Si alguno de los dos lados viene vacio no hay nada que comparar.
    # Devolver NaN en vez de dejar que chi2_contingency lance una excepcion,
    # para que la demo pueda mostrar "sin datos" en vez de romperse.
    if conteo_ref.sum() == 0 or conteo_prod.sum() == 0:
        return np.nan, np.nan

    categorias = sorted(set(conteo_ref.index) | set(conteo_prod.index))
    tabla = pd.DataFrame({
        "referencia": [conteo_ref.get(c, 0) for c in categorias],
        "produccion": [conteo_prod.get(c, 0) for c in categorias],
    }, index=categorias)

    # Chi2 exige que ninguna fila sume 0
    tabla = tabla[(tabla.sum(axis=1) > 0)]

    # Con una sola categoria no hay grados de libertad para la prueba
    if tabla.shape[0] < 2:
        return np.nan, np.nan

    estadistico, p_valor, _, _ = chi2_contingency(tabla.T)
    return float(estadistico), float(p_valor)


def clasificar_alerta_psi(psi):
    """
    Traduce un valor de PSI al color y mensaje de alerta acordados con
    el equipo: verde sin cambio, amarillo moderado, rojo reentrenar.
    """
    if np.isnan(psi):
        return "gris", "Sin datos suficientes para calcular PSI"
    if psi < 0.1:
        return "verde", "Sin cambio significativo"
    if psi < 0.25:
        return "amarillo", "Cambio moderado, monitorear de cerca"
    return "rojo", "Cambio significativo, se recomienda reentrenar el modelo"
