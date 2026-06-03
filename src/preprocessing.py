import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Tuple, Optional

class DataPreprocessor:
    """Clase para preprocesar datos de demanda"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.encoders = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Cargar datos desde CSV o Excel"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        print(f"✅ Datos cargados: {df.shape}")
        return df
    
    def clean_data(self, df: pd.DataFrame, date_col: str = 'fecha', 
                   target_col: str = 'demanda_real') -> pd.DataFrame:
        """Limpiar datos: nulos, outliers, tipos"""
        df = df.copy()
        
        # Convertir fecha
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Ordenar por fecha
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Eliminar nulos en target
        df = df.dropna(subset=[target_col])
        
        # Detectar outliers con IQR
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)].shape[0]
        print(f"  Outliers detectados: {outliers}")
        
        # Capar outliers
        df[target_col] = df[target_col].clip(lower_bound, upper_bound)
        
        return df
    
    def create_features(self, df: pd.DataFrame, date_col: str = 'fecha',
                        target_col: str = 'demanda_real') -> pd.DataFrame:
        """Crear características temporales"""
        df = df.copy()
        
        # Features temporales básicas
        df['año'] = df[date_col].dt.year
        df['mes'] = df[date_col].dt.month
        df['dia'] = df[date_col].dt.day
        df['dia_semana'] = df[date_col].dt.dayofweek
        df['semana'] = df[date_col].dt.isocalendar().week
        
        # Features cíclicas (seno/coseno)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        
        # Features de rezagos (lags)
        for lag in [1, 2, 3, 7, 14, 28]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Media móvil
        df['media_movil_7'] = df[target_col].rolling(window=7).mean()
        df['media_movil_30'] = df[target_col].rolling(window=30).mean()
        
        # Eliminar filas con NaN generados por lags
        df = df.dropna().reset_index(drop=True)
        
        print(f"  Features creadas: {df.shape[1]} columnas")
        return df
    
    def prepare_for_model(self, df: pd.DataFrame, target_col: str = 'demanda_real',
                          exclude_cols: list = ['fecha']) -> Tuple[np.ndarray, np.ndarray, list]:
        """Preparar X e y para modelos ML"""
        # Seleccionar columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col and col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        return X, y, feature_cols