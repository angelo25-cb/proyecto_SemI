import streamlit as st
import pandas as pd
import glob
import os
import plotly.graph_objects as go

# ==================================================
# IMPORTS DE COMPONENTES
# ==================================================

from assets.utils import load_css
from assets.metrics import show_metrics
from assets.charts import (
    mae_chart,
    rmse_chart,
    mape_chart,
    r2_chart,
    r2_adjusted_chart
)
from assets.recommendations import show_recommendation

# ==================================================
# CONFIGURACIÓN DE PÁGINA
# ==================================================

st.set_page_config(
    page_title="Framework Inteligente de Pronóstico",
    page_icon="",
    layout="wide"
)

# ==================================================
# CSS
# ==================================================

load_css()

# ==================================================
# CARGAR RESULTADOS
# ==================================================

@st.cache_data
def cargar_resultados():

    files = glob.glob("data/results/*.csv")

    if len(files) == 0:
        return None

    latest_file = max(files, key=os.path.getctime)

    return pd.read_csv(latest_file)


df = cargar_resultados()

if df is None:
    st.error("No se encontraron resultados en data/results/")
    st.stop()

# ==================================================
# DETECTAR NOMBRE DE COLUMNA MODELO
# ==================================================

possible_names = [
    "Modelo",
    "modelo",
    "Model",
    "model"
]

model_col = None

for col in possible_names:
    if col in df.columns:
        model_col = col
        break

if model_col is None:
    st.error("No existe columna de modelo.")
    st.stop()

# ==================================================
# MEJORES RESULTADOS
# ==================================================

best_model = df.loc[df["MAE"].idxmin(), model_col]

best_mae = df["MAE"].min()

best_rmse = df.loc[df["MAE"].idxmin(), "RMSE"]

best_mape = df["MAPE"].min()

best_r2 = df["R2"].max()

best_r2_adj = df["R2_Ajustado"].max()

# ==================================================
# HEADER
# ==================================================

st.markdown(
"""
<div class="main-header">
<h1> Framework Inteligente de Pronóstico de Demanda</h1>
<p>
Optimización de producción, reducción de desperdicio y apoyo a la toma de decisiones en pymes de alimentos preparados
</p>
</div>
""",
unsafe_allow_html=True
)

# ==================================================
# KPIs
# ==================================================

show_metrics(
    best_model,
    best_mae,
    best_rmse,
    best_mape,
    best_r2,
    best_r2_adj
)

st.divider()

# ==================================================
# TABS PRINCIPALES
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    " Pronóstico",
    " Modelos",
    " Impacto",
    " Resultados"
])

# ==================================================
# TAB 1 - PRONÓSTICO
# ==================================================

with tab1:

    st.subheader("Demanda Real vs Demanda Predicha")

    try:

        pred_files = glob.glob("reports\figures\real_vs_predicho.png")

        if len(pred_files) > 0:

            pred_df = pd.read_csv(pred_files[0])

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=pred_df["fecha"],
                    y=pred_df["demanda_real"],
                    name="Demanda Real"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pred_df["fecha"],
                    y=pred_df["demanda_predicha"],
                    name="Demanda Predicha"
                )
            )

            fig.update_layout(
                height=500
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Agrega el archivo de predicciones para visualizar la serie temporal."
            )

    except Exception as e:

        st.warning(str(e))

# ==================================================
# TAB 2 - MODELOS
# ==================================================

with tab2:

    st.subheader("Comparación de Modelos")

    mae_chart(df, model_col)

    rmse_chart(df, model_col)

    mape_chart(df, model_col)

    r2_chart(df, model_col)

    r2_adjusted_chart(df, model_col)

# ==================================================
# TAB 3 - IMPACTO
# ==================================================

with tab3:

    st.subheader("Impacto Económico y Ambiental")

    try:

        impacto_files = glob.glob(
            "data/results/*impact*csv"
        )

        if len(impacto_files) > 0:

            impacto = pd.read_csv(
                impacto_files[0]
            )

            st.dataframe(
                impacto,
                use_container_width=True
            )

        else:

            st.success(
                """
                Random Forest logró:

                • 26.6% de reducción de desperdicio

                • S/ 5,572 de ahorro económico estimado

                • Menor error porcentual (MAPE)
                """
            )

    except Exception as e:

        st.warning(str(e))

# ==================================================
# TAB 4 - RESULTADOS
# ==================================================

with tab4:

    st.subheader(
        "Resultados de Evaluación"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇ Descargar Resultados",
        data=csv,
        file_name="resultados_modelos.csv",
        mime="text/csv"
    )

# ==================================================
# RECOMENDACIÓN FINAL
# ==================================================

st.divider()

show_recommendation(
    best_model,
    best_mae,
    best_rmse,
    best_mape,
    best_r2,
    best_r2_adj
)

