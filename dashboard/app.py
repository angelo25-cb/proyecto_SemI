import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de página
st.set_page_config(
    page_title="Pronóstico de Demanda - PYME",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilo CSS personalizado (sin emoticones)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a5276 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        color: #d4e6f1;
        margin: 0.5rem 0 0 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a5276;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4e6f1 100%);
        border-left: 5px solid #1a5276;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado personalizado
st.markdown("""
<div class="main-header">
    <h1>Panel de Control de Previsión de Demanda</h1>
    <p>Análisis comparativo de modelos predictivos para reducción de desperdicio</p>
</div>
""", unsafe_allow_html=True)

# Cargar datos
@st.cache_data
def load_data():
    # Buscar archivo de resultados
    results_files = glob.glob("data/results/*.csv")
    if not results_files:
        results_files = glob.glob("reports/tables/*.csv")
    
    if results_files:
        latest = max(results_files, key=os.path.getctime)
        df = pd.read_csv(latest)
        return df, os.path.basename(latest)
    return None, None

df_results, filename = load_data()

if df_results is not None:
    # Verificar columnas necesarias
    required_cols = ['MAE', 'RMSE', 'MAPE']
    metric_cols = [col for col in required_cols if col in df_results.columns]
    
    # Identificar columna de modelo
    model_col = None
    for col in df_results.columns:
        if col.lower() in ['modelo', 'model', 'Model', 'Modelo']:
            model_col = col
            break
    
    if model_col is None:
        st.error("No se encontró la columna de modelo en los datos")
        st.stop()
    
    # ============================================
    # SECCIÓN 1: MÉTRICAS PRINCIPALES (TARJETAS)
    # ============================================
    st.markdown("### Indicadores Clave de Rendimiento")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_model = df_results.loc[df_results['MAE'].idxmin(), model_col]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_model}</div>
            <div class="metric-label">Mejor Modelo (MAE)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_mae = df_results['MAE'].min()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_mae:.2f}</div>
            <div class="metric-label">Error Absoluto Medio (MAE) mínimo</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        best_mape = df_results['MAPE'].min()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_mape:.1f}%</div>
            <div class="metric-label">Error Porcentual (MAPE) mínimo</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'R2' in df_results.columns:
            best_r2 = df_results['R2'].max()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{best_r2:.4f}</div>
                <div class="metric-label">R² máximo (poder explicativo)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">N/A</div>
                <div class="metric-label">R² no disponible</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # SECCIÓN 2: TABLA COMPARATIVA COMPLETA
    # ============================================
    st.markdown("### Comparación de Modelos")
    
    # Seleccionar columnas para mostrar
    display_cols = [model_col] + ['MAE', 'RMSE', 'MAPE']
    if 'R2' in df_results.columns:
        display_cols.append('R2')
    if 'R2_Ajustado' in df_results.columns:
        display_cols.append('R2_Ajustado')
    
    # Formatear la tabla
    df_display = df_results[display_cols].copy()
    df_display = df_display.sort_values('MAE')
    
    # Renombrar columnas para mejor presentación
    rename_map = {
        model_col: 'Modelo',
        'MAE': 'MAE (unidades)',
        'RMSE': 'RMSE (unidades)',
        'MAPE': 'MAPE (%)',
        'R2': 'R²',
        'R2_Ajustado': 'R² Ajustado'
    }
    df_display = df_display.rename(columns=rename_map)
    
    # Redondear valores
    for col in df_display.columns:
        if col != 'Modelo':
            df_display[col] = df_display[col].round(4)
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # ============================================
    # SECCIÓN 3: GRÁFICOS COMPARATIVOS
    # ============================================
    st.markdown("### Visualización de Métricas")
    
    # Crear pestañas para diferentes gráficos
    tab1, tab2 = st.tabs(["Comparación de MAE y MAPE", "Comparación de R² y R² Ajustado"])
    
    with tab1:
        # Gráfico de barras para MAE y MAPE
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # MAE (eje izquierdo)
        fig.add_trace(
            go.Bar(x=df_results[model_col], y=df_results['MAE'], 
                   name='MAE (unidades)', marker_color='#3498db'),
            secondary_y=False
        )
        
        # MAPE (eje derecho)
        fig.add_trace(
            go.Bar(x=df_results[model_col], y=df_results['MAPE'], 
                   name='MAPE (%)', marker_color='#e74c3c'),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Modelo")
        fig.update_yaxes(title_text="MAE (unidades)", secondary_y=False)
        fig.update_yaxes(title_text="MAPE (%)", secondary_y=True)
        fig.update_layout(title="Comparación de MAE y MAPE por Modelo", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'R2' in df_results.columns:
            # Gráfico de barras para R²
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df_results[model_col], y=df_results['R2'], 
                                   name='R²', marker_color='#27ae60'))
            if 'R2_Ajustado' in df_results.columns:
                fig2.add_trace(go.Bar(x=df_results[model_col], y=df_results['R2_Ajustado'], 
                                       name='R² Ajustado', marker_color='#f39c12'))
            fig2.update_layout(title="Poder Explicativo (R² y R² Ajustado)", height=500)
            fig2.update_yaxes(title_text="Coeficiente (0-1)", range=[0, 1])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Las métricas R² no están disponibles en los datos cargados")
    
    # ============================================
    # SECCIÓN 4: RECOMENDACIÓN FINAL
    # ============================================
    st.markdown("### Recomendación")
    
    # Obtener el mejor modelo según MAE
    best_idx = df_results['MAE'].idxmin()
    best_model_name = df_results.loc[best_idx, model_col]
    best_mae = df_results.loc[best_idx, 'MAE']
    best_mape = df_results.loc[best_idx, 'MAPE']
    
    # Obtener segundo mejor para comparación
    second_best_idx = df_results['MAE'].nsmallest(2).index[-1]
    second_best_name = df_results.loc[second_best_idx, model_col]
    improvement = ((df_results.loc[second_best_idx, 'MAE'] - best_mae) / df_results.loc[second_best_idx, 'MAE']) * 100
    
    st.markdown(f"""
    <div class="recommendation-box">
        <h4>Modelo Seleccionado: {best_model_name}</h4>
        <p>Basado en el análisis de precisión y estabilidad, se recomienda implementar el modelo <strong>{best_model_name}</strong> para la previsión de demanda.</p>
        <ul>
            <li>Error Absoluto Medio (MAE): <strong>{best_mae:.2f}</strong> unidades</li>
            <li>Error Porcentual (MAPE): <strong>{best_mape:.1f}%</strong></li>
            <li>Mejora respecto al segundo mejor modelo: <strong>{improvement:.1f}%</strong> en MAE</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # PIE DE PÁGINA
    # ============================================
    st.markdown("---")
    st.caption(f"Datos cargados desde: {filename} | Actualizado automáticamente con nuevas ejecuciones")

else:
    # Mensaje cuando no hay datos
    st.warning("No se encontraron resultados de validación")
    st.info("""
    **Para generar resultados:**
    1. Ejecute `python main.py` en la terminal
    2. Los resultados se guardarán automáticamente en `data/results/`
    3. Refresque esta página después de la ejecución
    """)
    
    # Mostrar estructura esperada
    with st.expander("Ver estructura de archivos esperada"):
        st.code("""
        proyecto_SemI/
        ├── data/
        │   └── results/
        │       └── validation_results.csv
        ├── reports/
        │   └── tables/
        │       └── model_comparison_full.csv
        └── dashboard/
            └── app.py
        """)