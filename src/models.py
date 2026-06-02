from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np


class DemandForecastModels:

    def __init__(self):
        self.models = {}
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)

    def evaluate_model(self, y_test, pred, model_name):

        mae = mean_absolute_error(y_test, pred)

        rmse = np.sqrt(mean_squared_error(y_test, pred))

        mape = mean_absolute_percentage_error(y_test, pred) * 100

        print(f"\n *{model_name}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

        # Gráfico
        plt.figure(figsize=(10,5))
        plt.plot(y_test.values, label='Demanda Real')
        plt.plot(pred, label='Pronóstico')
        plt.title(f'{model_name} - Real vs Pronosticado')
        plt.xlabel('Observaciones')
        plt.ylabel('Demanda')
        plt.legend()

        plt.savefig(f'results/{model_name}_forecast.png')
        plt.close()

        return mae, rmse, mape

    def train_xgboost(self, X_train, y_train, X_test, y_test):

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae, rmse, mape = self.evaluate_model(
            y_test,
            pred,
            "XGBoost"
        )

        joblib.dump(model, 'models/xgboost_model.pkl')

        print("XGBoost guardado correctamente")

        return model

    def train_randomforest(self, X_train, y_train, X_test, y_test):

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=42
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae, rmse, mape = self.evaluate_model(
            y_test,
            pred,
            "RandomForest"
        )

        joblib.dump(model, 'models/randomforest_model.pkl')

        print("Random Forest guardado correctamente")

        return model