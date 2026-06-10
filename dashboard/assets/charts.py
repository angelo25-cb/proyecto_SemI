import plotly.express as px
import streamlit as st

# ==========================================
# MAE
# ==========================================
def mae_chart(df, model_col):

    fig = px.bar(
        df,
        x=model_col,
        y="MAE",
        color="MAE",
        text_auto=".2f",
        title=" Error Absoluto Medio (MAE)"
    )

    fig.update_layout(
        height=450,
        xaxis_title="Modelo",
        yaxis_title="MAE",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# RMSE
# ==========================================
def rmse_chart(df, model_col):

    fig = px.bar(
        df,
        x=model_col,
        y="RMSE",
        color="RMSE",
        text_auto=".2f",
        title=" Raíz del Error Cuadrático Medio (RMSE)"
    )

    fig.update_layout(
        height=450,
        xaxis_title="Modelo",
        yaxis_title="RMSE",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# MAPE
# ==========================================
def mape_chart(df, model_col):

    fig = px.bar(
        df,
        x=model_col,
        y="MAPE",
        color="MAPE",
        text_auto=".2f",
        title=" Error Porcentual Absoluto Medio (MAPE)"
    )

    fig.update_layout(
        height=450,
        xaxis_title="Modelo",
        yaxis_title="MAPE (%)",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# R²
# ==========================================
def r2_chart(df, model_col):

    fig = px.bar(
        df,
        x=model_col,
        y="R2",
        color="R2",
        text_auto=".4f",
        title="📈 Coeficiente de Determinación (R²)"
    )

    fig.update_layout(
        height=450,
        xaxis_title="Modelo",
        yaxis_title="R²",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# R² AJUSTADO
# ==========================================
def r2_adjusted_chart(df, model_col):

    fig = px.bar(
        df,
        x=model_col,
        y="R2_Ajustado",
        color="R2_Ajustado",
        text_auto=".4f",
        title=" Coeficiente de Determinación Ajustado"
    )

    fig.update_layout(
        height=450,
        xaxis_title="Modelo",
        yaxis_title="R² Ajustado",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)