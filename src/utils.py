import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_config(config_path: str = "config.yaml") -> dict:
    """Cargar configuración desde YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_results(results_df: pd.DataFrame, output_path: str = "data/results/"):
    """Guardar resultados de validación"""
    os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"validation_results_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    results_df.to_csv(filepath, index=False)
    print(f"✅ Resultados guardados en {filepath}")
    
    return filepath

def plot_comprehensive_results(results_df: pd.DataFrame):
    """Gráficos completos de comparación de modelos"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE comparativo
    results_sorted = results_df.sort_values('MAE')
    axes[0, 0].barh(results_sorted['Model'], results_sorted['MAE'], color='steelblue')
    axes[0, 0].set_title('MAE (menor es mejor)')
    axes[0, 0].set_xlabel('MAE')
    
    # MAPE comparativo
    axes[0, 1].barh(results_sorted['Model'], results_sorted['MAPE'], color='coral')
    axes[0, 1].set_title('MAPE % (menor es mejor)')
    axes[0, 1].set_xlabel('MAPE (%)')
    
    # Reducción de desperdicio
    axes[1, 0].barh(results_sorted['Model'], results_sorted['Reduccion_Desperdicio'], color='green')
    axes[1, 0].set_title('Reducción de Desperdicio (%)')
    axes[1, 0].set_xlabel('Reducción (%)')
    
    # Ahorro económico
    axes[1, 1].barh(results_sorted['Model'], results_sorted['Ahorro_Economico_Num'], color='gold')
    axes[1, 1].set_title('Ahorro Económico ($)')
    axes[1, 1].set_xlabel('Ahorro (USD)')
    
    plt.tight_layout()
    plt.show()