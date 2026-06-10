import streamlit as st

def show_recommendation(
    best_model,
    mae,
    rmse,
    mape,
    r2,
    r2_adj
):

    st.markdown(
        f"""
        <div class="recommendation-box">

        <h2>Recomendación Final</h2>

        <p>
        Se recomienda implementar el modelo
        <strong>{best_model}</strong>,
        ya que presentó el mejor desempeño global
        durante el proceso de evaluación.
        </p>

        <h4>Indicadores obtenidos</h4>

        <ul>
            <li><strong>MAE:</strong> {mae:.2f}</li>
            <li><strong>RMSE:</strong> {rmse:.2f}</li>
            <li><strong>MAPE:</strong> {mape:.2f}%</li>
            <li><strong>R²:</strong> {r2:.4f}</li>
            <li><strong>R² Ajustado:</strong> {r2_adj:.4f}</li>
        </ul>

        <p>
        Estos resultados indican una adecuada capacidad
        predictiva para apoyar la planificación de la producción,
        reducir desperdicios y mejorar la toma de decisiones
        operativas en pymes de alimentos preparados.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )