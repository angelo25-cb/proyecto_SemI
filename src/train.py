import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import joblib

class ModelTrainer:
    """Entrenador de modelos"""
    
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.history = {}
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, 
                         test_size: float = 0.2) -> Tuple:
        """Dividir datos respetando orden temporal"""
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_prophet(self, df: pd.DataFrame, date_col: str = 'fecha', 
                      target_col: str = 'demanda_real') -> Any:
        """Entrenar modelo Prophet"""
        from src.models import ModelFactory
        
        prophet_df = ModelFactory.prepare_prophet_data(df, date_col, target_col)
        
        # Dividir en tiempo
        split_idx = int(len(prophet_df) * 0.8)
        train_df = prophet_df[:split_idx]
        
        self.model.fit(train_df)
        
        return self.model
    
    def train_sklearn(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Entrenar modelo sklearn/xgboost"""
        self.model.fit(X_train, y_train)
        return self.model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   epochs: int = 100, batch_size: int = 16) -> Any:
        """Entrenar modelo LSTM (requiere TensorFlow)"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            from tensorflow.keras.callbacks import EarlyStopping
            
            # Reformatear para LSTM
            X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            
            # Construir modelo
            lstm_model = Sequential([
                LSTM(50, input_shape=(X_train.shape[1], 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            
            # Early stopping
            early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
            
            # Entrenar
            history = lstm_model.fit(
                X_train_lstm, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            self.history['loss'] = history.history['loss']
            return lstm_model
            
        except ImportError:
            print("⚠️ TensorFlow no instalado. LSTM no disponible.")
            return None
    
    def save_model(self, filepath: str):
        """Guardar modelo entrenado"""
        joblib.dump(self.model, filepath)
        print(f"✅ Modelo guardado en {filepath}")