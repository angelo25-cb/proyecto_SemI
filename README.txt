#  Pronóstico-PYMES: Framework de Previsión de Demanda

[![Python Version](https://img.shields.io/badge/python-3.14.3-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

##  Descripción del Proyecto

Framework para la previsión de demanda y reducción de desperdicio en PYMES de alimentos preparados. 
El sistema implementa múltiples modelos predictivos para optimizar la producción y minimizar pérdidas económicas.

###  Modelos Implementados

| Modelo | Tipo | Estado |
|--------|------|--------|
| Random Forest | Machine Learning |  Activo |
| Gradient Boosting | Machine Learning |  Activo |
| XGBoost | Machine Learning |  Activo |
| Prophet | Series Temporales |  En desarrollo |
| LSTM | Deep Learning |  En desarrollo |

##  Tecnologías Utilizadas

- **Python 3.14.3** - Versión principal del proyecto
- **pandas 2.0+** - Manipulación de datos
- **numpy 1.24+** - Cálculos numéricos
- **scikit-learn 1.3+** - Modelos ML
- **XGBoost 2.0+** - Gradient boosting optimizado
- **Streamlit 1.28+** - Dashboard interactivo
- **matplotlib/seaborn** - Visualizaciones
- **plotly** - Gráficos interactivos

##  Estructura del Proyecto

proyecto_SemI/
├── main.py                          # Punto de entrada principal
├── requirements.txt                 # Dependencias
├── config.yaml                      # Configuración global
├── data/
│   ├── raw/                         # Datos crudos
│   ├── processed/                   # Datos procesados
│   └── results/                     # Resultados de validación
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # Limpieza y preparación
│   ├── models.py                    # Definición de modelos
│   ├── train.py                     # Entrenamiento
│   ├── forecast_module.py           # Predicciones futuras
│   ├── evaluation.py                # Métricas y validación
│   ├── utils.py                     # Utilidades generales
│   └── excess_management.py         # Cálculo de desperdicio
├── notebooks/
│   ├── 01_eda.ipynb                 # Análisis exploratorio
│   ├── 02_model_comparison.ipynb    # Comparación de modelos
│   └── 03_results_analysis.ipynb    # Análisis de resultados
├── reports/
│   ├── figures/                     # Gráficos
│   └── tables/                      # Tablas de resultados
└── dashboard/                       # Interfaz Streamlit
    └── app.py


##  Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/angelo25-cb/pronostico-pymes.git
cd pronostico-pymes

# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Colocar dataset: 
    data/raw/dataset_sintetico_demanda_lima.xlsx

# 3. Ejecutar main
python main.py

# 4. Ver dashboard
python -m streamlit run dashboard/app.py    

Este framework te permite:

Comparar 5 modelos (RF, GB, XGB, Prophet, LSTM)

Calcular MAE, RMSE, MAPE

Estimar reducción de desperdicio (%)

Calcular ahorro económico ($)

Visualizar resultados en dashboard interactivo