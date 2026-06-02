import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path='data/dataset_sintetico_demanda_lima.xlsx'):
    """Carga el dataset sintético"""
   
    df = pd.read_excel(file_path)
    df['fecha'] = pd.to_datetime(df['fecha'])
   
    return df


def feature_engineering(df):
    """Crea variables exógenas y transforma variables categóricas"""
   
    df = df.copy()

    # Convertir día de semana en variables numéricas
    df = pd.get_dummies(df, columns=['dia_semana'], prefix='dia')

    return df


def prepare_data(df, target='demanda_real'):
    """Prepara datos para entrenamiento"""

    exclude_cols = [
        target,
        'fecha',
        'desperdicio_sin_modelo',
        'desperdicio_con_modelo',
        'ahorro_economico_diario',
        'co2_evitable_diario_kg'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[target]

    # Escalado
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # División temporal
    train_size = int(len(X) * 0.8)

    X_train = X_scaled[:train_size]
    X_test = X_scaled[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test, scaler, feature_cols