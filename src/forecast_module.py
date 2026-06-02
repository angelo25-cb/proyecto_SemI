import pandas as pd
import joblib
from datetime import datetime, timedelta
import os


class ForecastModule:

    def __init__(self):

        self.model = joblib.load('models/xgboost_model.pkl')

        os.makedirs('results', exist_ok=True)

    def predict_next_days(self, days=7):

        today = datetime.now()

        predictions = []

        for i in range(days):

            future_date = today + timedelta(days=i)

            input_data = pd.DataFrame({

                'dia_semana_num': [future_date.weekday()],
                'es_fin_semana': [1 if future_date.weekday() >= 5 else 0],
                'temperatura_promedio': [19],
                'precipitacion_mm': [1.5],
                'mes': [future_date.month],
            })

            dias = [
                'Monday',
                'Tuesday',
                'Wednesday',
                'Thursday',
                'Friday',
                'Saturday',
                'Sunday'
            ]

            for dia in dias:

                input_data[f'dia_{dia}'] = 1 if future_date.strftime('%A') == dia else 0

            pred = self.model.predict(input_data)[0]

            predictions.append({

                'fecha': future_date.strftime('%Y-%m-%d'),
                'demanda_pronosticada': round(pred)
            })

        forecast_df = pd.DataFrame(predictions)

        # Guardar CSV
        forecast_df.to_csv(
            'results/pronosticos_7dias.csv',
            index=False
        )

        print("\nPronóstico guardado en results/pronosticos_7dias.csv")

        return forecast_df