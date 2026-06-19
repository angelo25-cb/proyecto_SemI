import streamlit as st
import pandas as pd
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
st.set_page_config(
    page_title="Framework Inteligente de Pronóstico",
    layout="wide"
)
# ==========================================
# CARGAR CSS
# ==========================================
css_file = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "style.css"
)
if os.path.exists(css_file):
    with open(css_file, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
# ==========================================
# CARGAR RESULTADOS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

results_file = os.path.join(
    BASE_DIR,
    "reports",
    "tables",
    "model_comparison_full.csv"
)
if not os.path.exists(results_file):
    st.error("No existe model_comparison_full.csv")
    st.stop()

df = pd.read_csv(results_file)
# ==========================================
# MEJOR MODELO
# ==========================================

best_row = df.loc[df["MAE"].idxmin()]

best_model = best_row["Modelo"]
best_mae = best_row["MAE"]
best_rmse = best_row["RMSE"]
best_mape = best_row["MAPE"]
best_r2 = best_row["R2"]
# ==========================================
# HEADER
# ==========================================
st.title("Framework Inteligente de Pronóstico de Demanda")
st.markdown(
"""
Predicción de demanda para optimizar producción,
reducir desperdicios y mejorar la toma de decisiones.
"""
)
# ==========================================
# MÉTRICAS
# ==========================================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Mejor Modelo", best_model)
c2.metric("MAE", round(best_mae,2))
c3.metric("RMSE", round(best_rmse,2))
c4.metric("MAPE", round(best_mape,2))
c5.metric("R2", round(best_r2,4))

st.divider()
# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "Pronóstico",
    "Modelos",
    "Impacto"
])
# ==========================================
# TAB 1
# ==========================================
with tab1:
    st.subheader("Demanda Real vs Predicha")
    img_path = os.path.join(
        BASE_DIR,
        "reports",
        "figures",
        "real_vs_predicho.png"
    )
    if os.path.exists(img_path):
        st.image(
            img_path,
            use_container_width=True
        )
    else:
        st.error(
            f"No existe:\n{img_path}"
        )
# ==========================================
# TAB 2
# ==========================================
with tab2:
    st.subheader("Comparación de Modelos")
    st.dataframe(
        df,
        use_container_width=True
    )
# ==========================================
# TAB 3
# ==========================================
with tab3:
    st.subheader("Impacto Económico")
    impacto_img = os.path.join(
        BASE_DIR,
        "reports",
        "figures",
        "impacto_economico.png"
    )
    if os.path.exists(impacto_img):
        st.image(
            impacto_img,
            use_container_width=True
        )
    else:
        st.error(
            f"No existe:\n{impacto_img}"
        )
# ==========================================
# RESULTADOS
# ==========================================
