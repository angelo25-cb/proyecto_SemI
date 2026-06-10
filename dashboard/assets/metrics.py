import streamlit as st

def show_metrics(
    best_model,
    best_mae,
    best_rmse,
    best_mape,
    best_r2,
    best_r2_adj
):

    st.subheader("Indicadores Clave de Desempeño")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            " Modelo Recomendado",
            best_model
        )

    with col2:
        st.metric(
            " MAE",
            f"{best_mae:.2f}"
        )

    with col3:
        st.metric(
            " RMSE",
            f"{best_rmse:.2f}"
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            " MAPE",
            f"{best_mape:.2f}%"
        )

    with col5:
        st.metric(
            " R²",
            f"{best_r2:.4f}"
        )

    with col6:
        st.metric(
            " R² Ajustado",
            f"{best_r2_adj:.4f}"
        )