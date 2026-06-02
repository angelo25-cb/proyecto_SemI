from src.forecast_module import ForecastModule
from src.excess_management import generate_recommendations

def main():

    print("=== SISTEMA DE PRONÓSTICO DE DEMANDA ===\n")

    forecast = ForecastModule()

    result = forecast.predict_next_days(days=7)

    result = generate_recommendations(result)

    print(result)

    print("\n Pronóstico y recomendaciones generadas correctamente!")


if __name__ == "__main__":
    main()