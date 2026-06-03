import streamlit as st
import pandas as pd
import os
import glob

st.set_page_config(page_title="Pronóstico PYMES", layout="wide")

st.title(" Dashboard de Previsión de Demanda")
st.markdown("### Reducción de desperdicio en PYMES de alimentos preparados")

# Cargar resultados
@st.cache_data
def load_results():
    results_files = glob.glob("data/results/*.csv")
    if results_files:
        latest = max(results_files, key=os.path.getctime)
        return pd.read_csv(latest)
    return None

results_df = load_results()

if results_df is not None:
    st.success(f" Resultados cargados correctamente")
    
    # Mostrar columnas disponibles (para depuración)
    with st.expander(" Ver columnas disponibles"):
        st.write(list(results_df.columns))
    
    # Identificar el nombre de la columna del modelo
    model_col = None
    for col in results_df.columns:
        if col.lower() in ['modelo', 'model', 'Model', 'Modelo']:
            model_col = col
            break
    
    if model_col is None:
        st.error(f"No se encontró columna de modelo. Columnas disponibles: {list(results_df.columns)}")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_model = results_df.loc[results_df['MAE'].idxmin(), model_col]
            st.metric("Mejor Modelo (MAE)", best_model)
        
        with col2:
            best_mape = results_df['MAPE'].min()
            st.metric("MAPE Mínimo", f"{best_mape:.1f}%")
        
        with col3:
            if 'Reduccion_Desperdicio' in results_df.columns:
                best_waste = results_df['Reduccion_Desperdicio'].max()
                st.metric("♻️ Máx Reducción Desperdicio", f"{best_waste:.1f}%")
        
        # Tabla de resultados
        st.subheader("Comparación de Modelos")
        st.dataframe(results_df, use_container_width=True)
        
        # Gráfico
        st.subheader("Visualización de Métricas")
        chart_df = results_df.set_index(model_col)[['MAE', 'MAPE']]
        st.bar_chart(chart_df)
        
        # Recomendación
        st.subheader("Recomendación")
        st.info(f"**Modelo recomendado: {best_model}**")
        
else:
    st.warning(" No se encontraron resultados en data/results/")
    st.info("Ejecuta `python main.py` primero para generar los resultados")