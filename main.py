

"""
Framework de Previsión de Demanda para PYMES
Ejecución principal del proyecto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print(" Framework de Previsión de Demanda - PYMES")
print("=" * 60)

# 1. Cargar datos
print("\n Cargando datos...")
df = pd.read_excel('data/raw/dataset_sintetico_demanda_lima.xlsx')
df['fecha'] = pd.to_datetime(df['fecha'])
df = df.sort_values('fecha').reset_index(drop=True)
print(f"  Datos cargados: {len(df)} días")

# 2. Preparar features
print("\n Preparando features...")
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

print(f"  Features: {len(feature_cols)}")
print(f"  Muestras: {len(X)}")

# 3. Dividir datos
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"  Train: {len(X_train)} días, Test: {len(X_test)} días")

# 4. Función para calcular métricas
def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# 5. Entrenar modelos
print("\n" + "=" * 60)
print(" ENTRENANDO MODELOS")
print("=" * 60)

results = []

# Random Forest
print("\n Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
metrics_rf = calcular_metricas(y_test, y_pred_rf)
metrics_rf['Modelo'] = 'Random Forest'
results.append(metrics_rf)
print(f"  MAE: {metrics_rf['MAE']:.2f}, MAPE: {metrics_rf['MAPE']:.1f}%")

# Gradient Boosting
print("\n Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
metrics_gb = calcular_metricas(y_test, y_pred_gb)
metrics_gb['Modelo'] = 'Gradient Boosting'
results.append(metrics_gb)
print(f"  MAE: {metrics_gb['MAE']:.2f}, MAPE: {metrics_gb['MAPE']:.1f}%")

# XGBoost
print("\n XGBoost...")
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
metrics_xgb = calcular_metricas(y_test, y_pred_xgb)
metrics_xgb['Modelo'] = 'XGBoost'
results.append(metrics_xgb)
print(f"  MAE: {metrics_xgb['MAE']:.2f}, MAPE: {metrics_xgb['MAPE']:.1f}%")

# 6. Resultados finales
print("\n" + "=" * 60)
print(" RESULTADOS FINALES")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('MAE')
print(results_df.to_string(index=False))

# Mejor modelo
best = results_df.iloc[0]
print(f"\n MEJOR MODELO: {best['Modelo']}")
print(f"   MAE: {best['MAE']:.2f}")
print(f"   MAPE: {best['MAPE']:.1f}%")

# Gráfico comparativo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(results_df['Modelo'], results_df['MAE'], color='steelblue')
axes[0].set_title('MAE por Modelo (menor es mejor)')
axes[1].barh(results_df['Modelo'], results_df['MAPE'], color='coral')
axes[1].set_title('MAPE % por Modelo (menor es mejor)')
plt.tight_layout()
plt.show()

print("\n Ejecución completada!")

# Guardar resultados
import os
os.makedirs('data/results', exist_ok=True)
results_df.to_csv('data/results/validation_results.csv', index=False)
print(" Resultados guardados en data/results/validation_results.csv")