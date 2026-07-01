import pandas as pd
import numpy as np

# =====================================================
# VARIABLES PREDICTORAS
# =====================================================
FEATURES = [
    'dia_semana','mes','dia',
    'mes_sin','mes_cos',
    'dia_sem_sin','dia_sem_cos',
    'lag_1','lag_2','lag_3','lag_7','lag_14','lag_21',
    'media_movil_7','media_movil_14',
    'std_7','max_7','min_7',
    'diff_1','diff_7',
    'trend'
]

# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================
def preparar_dataset(ruta_excel):

    df = pd.read_excel(ruta_excel)

    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha")

    # -------------------------
    # calendario
    # -------------------------
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.month
    df["dia"] = df["fecha"].dt.day

    # estacionalidad cíclica
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"]/12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"]/12)

    df["dia_sem_sin"] = np.sin(2 * np.pi * df["dia_semana"]/7)
    df["dia_sem_cos"] = np.cos(2 * np.pi * df["dia_semana"]/7)

    # -------------------------
    # lags
    # -------------------------
    for lag in [1, 2, 3, 7, 14, 21]:
        df[f"lag_{lag}"] = df["demanda_real"].shift(lag)

    # -------------------------
    # rolling
    # -------------------------
    df["media_movil_7"] = df["demanda_real"].rolling(7).mean()
    df["media_movil_14"] = df["demanda_real"].rolling(14).mean()

    df["std_7"] = df["demanda_real"].rolling(7).std()
    df["max_7"] = df["demanda_real"].rolling(7).max()
    df["min_7"] = df["demanda_real"].rolling(7).min()

    # -------------------------
    # diferencias
    # -------------------------
    df["diff_1"] = df["demanda_real"].diff(1)
    df["diff_7"] = df["demanda_real"].diff(7)

    # -------------------------
    # tendencia
    # -------------------------
    df["trend"] = df["lag_1"] - df["lag_7"]

    # -------------------------
    # limpieza final
    # -------------------------
    df = df.dropna().reset_index(drop=True)

    X = df[FEATURES]
    y = df["demanda_real"]

    return df, X, y