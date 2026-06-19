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

##  Tecnologías Utilizadas

- **Python 3.14.3** - Versión principal del proyecto
- **pandas 2.0+** - Manipulación de datos
  **Python Versión 3.13.9===> mi laptop
- **numpy 1.24+** - Cálculos numéricos
- **scikit-learn 1.3+** - Modelos ML
- **XGBoost 2.0+** - Gradient boosting optimizado
- **Streamlit 1.28+** - Dashboard interactivo
- **matplotlib/seaborn** - Visualizaciones
- **plotly** - Gráficos interactivos

##  Estructura del Proyecto

proyecto_SemI/

├── data/
│   ├── raw/
│   │   └── dataset_demanda_lima.xlsx
│   │
│   ├── processed/
│   │   ├── random_forest.pkl
│   │   ├── gradient_boosting.pkl
│   │   └── xgboost.pkl
│   │
│   └── results/
│       ├── resultados_modelos.csv
│       ├── metricas_modelos.csv
│       └── demanda_vs_prediccion.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_results_analysis.ipynb
│
reports/
│
├── figures/
│   ├── error_distribution.png
│   ├── feature_importance.png
│   ├── impacto_economico.png
│   ├── real_vs_predicho.png
│   └── validation_cruzada_temporal_comparative.png
│
└── tables/
    ├── metricas_modelos.csv
    ├── resultados_modelos.csv
    └── resumen_ejecutivo.csv
│
├── dashboard/
│   ├── app.py
│   └── assets/
│       └── style.css
│
├── requirements.txt
│── config.yaml
└── README.md

##  Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/angelo25-cb/proyecto_SemI.git
cd proyecto_SemI

# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Colocar dataset: 
    data/raw/dataset_sintetico_demanda_lima.xlsx

# 3. Ejecutar main
python main.py
C:\ProgramData\anaconda3\python.exe main.py ==> para mi laptop
# 4. Ver dashboard
python -m streamlit run dashboard/app.py  

C:\ProgramData\anaconda3\python.exe -m streamlit run dashboard/app.py ==> para mi laptop
Este framework te permite:

Comparar 3 modelos (RF, GB, XGB)

Calcular MAE, RMSE, MAPE, R2, R2_ajustado

Estimar reducción de desperdicio (%)

Calcular ahorro económico ($)

Visualizar resultados en dashboard interactivo