def generate_recommendations(forecast_df):

    recommendations = []

    for _, row in forecast_df.iterrows():

        demanda = row['demanda_pronosticada']

        if demanda > 310:

            alerta = " Alta demanda esperada - aumentar producción"

        elif demanda < 290:

            alerta = " Baja demanda esperada - reducir producción"

        else:

            alerta = "Producción balanceada"

        recommendations.append(alerta)

    forecast_df['recomendacion_operativa'] = recommendations

    return forecast_df