# ================================================
# Entrenamiento de Modelos de Pronóstico
# ================================================
from src.preprocessing import load_data, feature_engineering, prepare_data
from src.models import DemandForecastModels


def main():
    print("Iniciando entrenamiento de modelos...\n")
   
    # Cargar y preparar datos
    df = load_data()
    print(f"Dataset cargado: {df.shape[0]} registros")
   
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)
   
    # Entrenar modelos
    trainer = DemandForecastModels()
    trainer.train_xgboost(X_train, y_train, X_test, y_test)
    trainer.train_randomforest(X_train, y_train, X_test, y_test)
   
    print("\n Entrenamiento completado exitosamente!")
    print("Modelos guardados en la carpeta 'models/'")


if __name__ == "__main__":
    main()