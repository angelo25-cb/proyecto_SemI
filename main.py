"""
Framework de Previsión de Demanda para PYMES
Carga los resultados generados por Notebook03
y los prepara para el Dashboard Streamlit.
"""

import pandas as pd
import os
import shutil

print("=" * 60)
print(" PREPARANDO RESULTADOS PARA STREAMLIT")
print("=" * 60)

# =====================================================
# Rutas
# =====================================================

source_file = "reports/tables/model_comparison_full.csv"

target_folder = "data/results"
target_file = os.path.join(
    target_folder,
    "validation_results.csv"
)

# =====================================================
# Verificar existencia
# =====================================================

if not os.path.exists(source_file):
    print("\n ERROR")
    print(f"No existe: {source_file}")
    print("Ejecute primero Notebook03.")
    exit()

# =====================================================
# Crear carpeta destino
# =====================================================

os.makedirs(target_folder, exist_ok=True)

# =====================================================
# Copiar resultados
# =====================================================

shutil.copy(source_file, target_file)

print("\n Archivo generado:")
print(target_file)

# =====================================================
# Mostrar resumen
# =====================================================

df = pd.read_csv(target_file)

print("\n RESULTADOS FINALES")
print("-" * 60)

print(df.to_string(index=False))

# =====================================================
# Mejor modelo
# =====================================================

if "MAE" in df.columns:

    best_row = df.loc[df["MAE"].idxmin()]

    print("\n MEJOR MODELO")

    print(f"Modelo : {best_row['Modelo']}")
    print(f"MAE    : {best_row['MAE']:.4f}")

    if "MAPE" in df.columns:
        print(f"MAPE   : {best_row['MAPE']:.2f}%")

print("\n Resultados listos para Streamlit.")