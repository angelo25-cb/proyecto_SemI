import numpy as np
import pandas as pd

class DemandForecaster:
    """Módulo para hacer predicciones futuras"""
    
    def __init__(self, model, model_name: str, preprocessor=None):
        self.model = model
        self.model_name = model_name
        self.preprocessor = preprocessor
    
    def predict_future(self, last_data: np.ndarray, periods: int = 30,
                       window: int = 7) -> np.ndarray:
        """Predecir períodos futuros (para modelos secuenciales)"""
        predictions = []
        current_window = last_data[-window:].copy()
        
        for _ in range(periods):
            # Predecir siguiente punto
            pred = self.model.predict(current_window.reshape(1, -1))[0]
            predictions.append(pred)
            
            # Actualizar ventana
            current_window = np.append(current_window[1:], pred)
        
        return np.array(predictions)
    
    def predict_prophet(self, periods: int = 30) -> pd.DataFrame:
        """Predecir con Prophet"""
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def predict_future_prophet(self, periods: int = 30) -> pd.DataFrame:
        """Predecir futuros períodos con Prophet"""
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast.tail(periods)[['ds', 'yhat']]