import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from typing import Dict, Any, Optional

class ModelFactory:
    """Fábrica de modelos predictivos"""
    
    @staticmethod
    def get_model(model_name: str, params: Dict[str, Any]):
        """Retorna una instancia del modelo solicitado"""
        
        if model_name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 10),
                random_state=42,
                n_jobs=-1
            )
        
        elif model_name == "Gradient Boosting":
            return GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 200),
                learning_rate=params.get('learning_rate', 0.05),
                max_depth=params.get('max_depth', 5),
                random_state=42
            )
        
        elif model_name == "XGBoost":
            return XGBRegressor(
                n_estimators=params.get('n_estimators', 300),
                learning_rate=params.get('learning_rate', 0.05),
                max_depth=params.get('max_depth', 6),
                random_state=42,
                verbosity=0
            )
        
        elif model_name == "Prophet":
            return Prophet(
                yearly_seasonality=params.get('yearly_seasonality', True),
                weekly_seasonality=params.get('weekly_seasonality', True),
                daily_seasonality=params.get('daily_seasonality', False)
            )
        
        else:
            raise ValueError(f"Modelo {model_name} no reconocido")
    
    @staticmethod
    def prepare_prophet_data(df, date_col='fecha', target_col='demanda_real'):
        """Preparar datos específicos para Prophet"""
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        return prophet_df