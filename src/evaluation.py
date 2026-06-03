import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Evaluador de modelos predictivos"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics = {}
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcular MAPE (Mean Absolute Percentage Error)"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mape
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calcular todas las métricas"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = self.calculate_mape(y_true, y_pred)
        
        # Porcentaje de mejora sobre baseline ingenuo
        naive_pred = np.roll(y_true, 1)
        naive_pred[0] = y_true[0]
        naive_mae = mean_absolute_error(y_true, naive_pred)
        improvement = ((naive_mae - mae) / naive_mae) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Improvement_vs_Naive': improvement
        }
    
    def time_series_cv(self, model, X: np.ndarray, y: np.ndarray, 
                       n_splits: int = 5) -> Dict:
        """Validación cruzada temporal"""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics_list = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Clonar modelo para cada fold
            from copy import deepcopy
            model_fold = deepcopy(model)
            
            model_fold.fit(X_train, y_train)
            y_pred = model_fold.predict(X_test)
            
            metrics = self.calculate_metrics(y_test, y_pred)
            metrics['Fold'] = fold + 1
            metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list)
    
    def plot_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     title: str = "Predicción vs Real"):
        """Graficar resultados"""
        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label='Real', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predicho', color='red', alpha=0.7)
        plt.title(f'{title} - {self.model_name}')
        plt.xlabel('Tiempo')
        plt.ylabel('Demanda')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, results_df: pd.DataFrame):
        """Comparar múltiples modelos"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['MAE', 'RMSE', 'MAPE']
        for i, metric in enumerate(metrics):
            results_df_sorted = results_df.sort_values(metric)
            axes[i].barh(results_df_sorted['Model'], results_df_sorted[metric], 
                         color='steelblue')
            axes[i].set_title(f'{metric} por Modelo')
            axes[i].set_xlabel(metric)
        
        plt.tight_layout()
        plt.show()