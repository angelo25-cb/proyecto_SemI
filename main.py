"""
Framework de Previsión de Demanda para PYMES
main.py - Ejecución principal (entrenamiento y guardado de resultados)
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print(" EJECUTANDO main.py - ENTRENAMIENTO DE MODELOS")
print("=" * 60)

# 1. Cargar datos
print("\n Cargando datos...")
excel_paths = ['data/raw/dataset_sintetico_demanda_lima.xlsx', '../data/raw/dataset_sintetico_demanda_lima.xlsx']
df = None
for path in excel_paths:
    if os.path.exists(path):
        df = pd.read_excel(path)
        print(f"    Datos cargados desde: {path}")
        break

if df is None:
    print("    Generando datos sintéticos...")
    fechas = pd.date_range('2023-01-01', periods=500)
    df = pd.DataFrame({'fecha': fechas, 'demanda_real': np.random.randint(50, 200, 500)})

df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values('fecha').reset_index(drop=True)

# 2. Preparar features
df['dia_semana'] = df['fecha'].dt.dayofweek
df['mes'] = df['fecha'].dt.month
df['dia'] = df['fecha'].dt.day
for lag in [1, 2, 3, 7]:
    df[f'lag_{lag}'] = df['demanda_real'].shift(lag)
df['media_movil_7'] = df['demanda_real'].rolling(7).mean()
df = df.dropna().reset_index(drop=True)

feature_cols = ['dia_semana', 'mes', 'dia', 'lag_1', 'lag_2', 'lag_3', 'lag_7', 'media_movil_7']
X = df[feature_cols].values
y = df['demanda_real'].values

# 3. Dividir datos
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 4. Función de métricas
def calcular_metricas(y_true, y_pred, X_test):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = X_test.shape[1]
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'R2_Ajustado': r2_adj}

# 5. Entrenar modelos
print("\n Entrenando modelos...")
results = []
modelos = {}

for name, model in [
    ('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)),
    ('XGBoost', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0))
]:
    print(f"   {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calcular_metricas(y_test, y_pred, X_test)
    metrics['Modelo'] = name
    results.append(metrics)
    modelos[name] = {'modelo': model, 'y_pred': y_pred, 'y_test': y_test, 'X_test': X_test}

# 6. Guardar resultados
results_df = pd.DataFrame(results)
results_df = results_df[['Modelo', 'MAE', 'RMSE', 'MAPE', 'R2', 'R2_Ajustado']].sort_values('MAE')

os.makedirs('reports/tables', exist_ok=True)
results_df.to_csv('reports/tables/model_comparison_full.csv', index=False)
print(f"\n Tabla guardada en reports/tables/model_comparison_full.csv")
print(results_df.to_string(index=False))


# Guardar predicciones y datos para notebook 03
import joblib
os.makedirs('data/processed', exist_ok=True)
for name, data in modelos.items():
    joblib.dump({
        'modelo': data['modelo'],
        'y_pred': data['y_pred'],
        'y_test': data['y_test'],
        'X_train': X_train,      # ← DEBE ESTAR
        'X_test': data['X_test'],
        'y_train': y_train       # ← DEBE ESTAR
    }, f'data/processed/{name.lower().replace(" ", "_")}_data.pkl')
print("\n Datos guardados para notebook 03")