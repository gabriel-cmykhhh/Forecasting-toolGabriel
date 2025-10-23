# --------------------------------------------------------------------------
# |                   IMPORTAR LIBRERÍAS                                   |
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import requests
import glob  # <-- NUEVO: Para encontrar archivos en carpetas

# --------------------------------------------------------------------------
# |                   CONFIGURACIÓN DE LA PÁGINA                             |
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="WELCOME TO MY WEBSITE, HARDIK GUAPO",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------
# |                   FUNCIONES DE CARGA DE DATOS (MODIFICADAS)            |
# --------------------------------------------------------------------------

# --- MODIFICADO: Ahora acepta 'file_path' en lugar de 'uploaded_file' ---
@st.cache_data
def load_asepeyo_energy_data(file_path):
    """Carga y procesa el archivo de consumo energético desde una ruta."""
    try:
        df = pd.read_csv(file_path, sep=',', decimal='.')
        if 'Fecha' not in df.columns or 'Energía activa (kWh)' not in df.columns:
            st.error(f"El archivo {file_path} debe contener 'Fecha' y 'Energía activa (kWh)'.")
            return pd.DataFrame()
            
        df.rename(columns={'Fecha': 'fecha', 'Energía activa (kWh)': 'consumo_kwh'}, inplace=True)
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo de consumo {file_path}: {e}")
        return pd.DataFrame()

# --- MODIFICADO: Ahora acepta 'file_path' y lee desde disco ---
@st.cache_data
def load_nasa_weather_data(file_path):
    """Carga y procesa el archivo de clima histórico de NASA POWER desde una ruta."""
    try:
        # --- Lógica de lectura de archivo modificada ---
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        start_row = 0
        for i, line in enumerate(lines):
            if line.strip() == "YEAR,MO,DY,HR,RH2M,T2M":
                start_row = i
                break
        
        df = pd.read_csv(file_path, skiprows=start_row)
        # --- Fin de la modificación de lectura ---
        
        expected_cols = ['YEAR', 'MO', 'DY', 'HR', 'T2M']
        if not all(col in df.columns for col in expected_cols):
            st.error(f"El archivo {file_path} debe contener 'YEAR', 'MO', 'DY', 'HR', 'T2M'.")
            return pd.DataFrame()

        df['fecha'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}))
        df.rename(columns={'T2M': 'temperatura_c'}, inplace=True)
        df['temperatura_c'] = df['temperatura_c'].replace(-999, np.nan).ffill()
        
        return df[['fecha', 'temperatura_c']]
    except Exception as e:
        st.error(f"Error al procesar el archivo de clima de la NASA {file_path}: {e}")
        return pd.DataFrame()

# --- Esta función no cambia ---
@st.cache_data
def get_weather_forecast(api_key, lat, lon):
    """Obtiene el pronóstico del tiempo diario desde la API de Meteosource."""
    BASE_URL = "https://www.meteosource.com/api/v1/free/point"
    params = {
        "lat": lat,
        "lon": lon,
        "sections": "daily",
        "units": "metric",
        "key": api_key
    }
    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            daily_data = data.get('daily', {}).get('data', [])
            if not daily_data:
                st.error("La API no devolvió datos de pronóstico diario.")
                return pd.DataFrame()
            processed_data = []
            for day in daily_data:
                processed_data.append({
                    'fecha': day.get('day'),
                    'temp_max_c': day.get('all_day', {}).get('temperature_max'),
                    'temp_min_c': day.get('all_day', {}).get('temperature_min')
                })
            df_clima_futuro = pd.DataFrame(processed_data)
            df_clima_futuro['fecha'] = pd.to_datetime(df_clima_futuro['fecha'])
            df_clima_futuro.dropna(inplace=True)
            df_clima_futuro['temp_avg_c'] = (df_clima_futuro['temp_max_c'] + df_clima_futuro['temp_min_c']) / 2
            return df_clima_futuro
        else:
            st.error(f"Error en la API de Meteosource (Código {response.status_code}): {response.json().get('detail', 'Error desconocido')}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al conectar con la API del clima: {e}")
        return pd.DataFrame()

# --- Esta función no cambia ---
def crear_features_temporales(df):
    """Crea columnas de features diarias basadas en la fecha."""
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['dia_mes'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['es_finde'] = (df['dia_semana'] >= 5).astype(int)
    return df
# [ ... (todo el código anterior) ... ]

# --------------------------------------------------------------------------
# |                   BARRA LATERAL (SIDEBAR) (MODIFICADA)                 |
# --------------------------------------------------------------------------

logo_path = "Logo_ASEPEYO.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=200)
else:
    st.sidebar.warning("Logo_ASEPEYO.png no encontrado.")

st.sidebar.title("Configuración de la Predicción")
st.sidebar.markdown("---")

# --- MODIFICADO: Carga de archivos con Selectbox ---
st.sidebar.header("1. Carga de Datos Históricos")

# --- NUEVO: Camino robusto a la carpeta 'data' ---
# Esto encuentra el directorio donde se está ejecutando el script (app.py)
# y luego lo une a la carpeta 'data'
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ no está definido si se ejecuta en un entorno como un notebook
    SCRIPT_DIR = os.path.abspath('.')

DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# --- NUEVO: Línea de depuración para ver dónde está buscando ---
st.sidebar.info(f"Buscando datos en: {DATA_DIR}")

selected_energy_file = None
selected_weather_file = None

try:
    # Asumir convención de nombres: "energy_*.csv" y "weather_*.csv"
    energy_pattern = os.path.join(DATA_DIR, "energy_*.csv")
    weather_pattern = os.path.join(DATA_DIR, "weather_*.csv")
    
    # Obtener solo el nombre del archivo (basename) para mostrar en el dropdown
    energy_files = [os.path.basename(f) for f in glob.glob(energy_pattern)]
    weather_files = [os.path.basename(f) for f in glob.glob(weather_pattern)]

    if not energy_files:
        st.sidebar.error(f"No se encontraron archivos 'energy_*.csv' en la carpeta '{DATA_DIR}'.")
    else:
        selected_energy_file = st.sidebar.selectbox("Selecciona el archivo de Consumo", energy_files)

    if not weather_files:
        st.sidebar.error(f"No se encontraron archivos 'weather_*.csv' en la carpeta '{DATA_DIR}'.")
    else:
        selected_weather_file = st.sidebar.selectbox("Selecciona el archivo de Clima Histórico", weather_files)
        
except Exception as e:
    st.sidebar.error(f"Error al leer la carpeta '{DATA_DIR}': {e}")
# --- FIN DE LA MODIFICACIÓN ---

st.sidebar.markdown("---")

st.sidebar.header("2. Configuración del Pronóstico (API)")
api_key = st.sidebar.text_input("API Key de Meteosource", type="password")
lat = st.sidebar.text_input("Latitud", "40.4168")
lon = st.sidebar.text_input("Longitud", "-3.7038")

st.sidebar.markdown("---")

st.sidebar.header("3. Variables Adicionales")
ocupacion_media = st.sidebar.slider("Ocupación Media (%) del Centro", 0, 100, 80)

# --------------------------------------------------------------------------
# |                   CUERPO DE LA APLICACIÓN (MODIFICADO)                 |
# --------------------------------------------------------------------------

st.title("Sistema de Inteligencia Energética con IA")
st.subheader("Herramienta de Predicción de Consumo para Instalaciones de Asepeyo")
st.markdown("---")

# --- MODIFICADO: Comprobar si los archivos fueron seleccionados ---
if selected_energy_file and selected_weather_file and api_key and lat and lon:
    with st.spinner('Procesando datos, contactando API y entrenando el modelo de IA...'):
        
        # --- MODIFICADO: Construir la ruta completa al archivo ---
        energy_path = os.path.join(DATA_DIR, selected_energy_file)
        weather_path = os.path.join(DATA_DIR, selected_weather_file)
        
        # --- MODIFICADO: Llamar a las funciones con las rutas ---
        df_energia = load_asepeyo_energy_data(energy_path)
        df_clima_pasado = load_nasa_weather_data(weather_path)
        
        df_clima_futuro = get_weather_forecast(api_key, lat, lon)
        
        if df_clima_futuro.empty:
            st.error("No se pudo obtener el pronóstico del clima. Revisa tu API Key o las coordenadas e inténtalo de nuevo.")
            st.stop()
            
        if not df_energia.empty and not df_clima_pasado.empty:
            
            # --- Agregación de datos históricos (sin cambios) ---
            df_historico_horario = pd.merge(df_energia, df_clima_pasado, on='fecha', how='inner')
            df_historico_horario.dropna(inplace=True)
            df_historico_horario.set_index('fecha', inplace=True)
            consumo_diario = df_historico_horario['consumo_kwh'].resample('D').sum()
            clima_diario = df_historico_horario['temperatura_c'].resample('D').agg(['min', 'max', 'mean'])
            df_historico_daily = pd.concat([consumo_diario, clima_diario], axis=1)
            df_historico_daily.rename(columns={
                'min': 'temp_min_c',
                'max': 'temp_max_c',
                'mean': 'temp_avg_c'
            }, inplace=True)
            df_historico_daily.reset_index(inplace=True)
            df_historico_daily.dropna(inplace=True)
            
            # --- Ingeniería de Features (sin cambios) ---
            df_historico_daily = crear_features_temporales(df_historico_daily)
            df_historico_daily['ocupacion'] = ocupacion_media
            df_futuro = df_clima_futuro.copy()
            df_futuro = crear_features_temporales(df_futuro)
            df_futuro['ocupacion'] = ocupacion_media
            
            # --- Entrenamiento del Modelo (sin cambios) ---
            features = [
                'temp_min_c', 
                'temp_max_c', 
                'temp_avg_c', 
                'dia_semana', 
                'dia_mes', 
                'mes', 
                'es_finde', 
                'ocupacion'
            ]
            target = 'consumo_kwh'
            
            for col in features:
                 if col not in df_futuro.columns:
                      st.error(f"Falta la columna de feature '{col}' en los datos futuros.")
                      st.stop()

            X = df_historico_daily[features]
            y = df_historico_daily[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            modelo.fit(X_train, y_train)

            # --- Resultados (sin cambios) ---
            X_futuro = df_futuro[features]
            df_futuro['consumo_predicho_kwh'] = modelo.predict(X_futuro)
            pred_test = modelo.predict(X_test)
            r2 = r2_score(y_test, pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred_test))

            st.success("✅ ¡Modelo diario entrenado y predicción completada!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Precisión del Modelo (R²)", f"{r2:.2f}")
            col2.metric("Error Medio (RMSE)", f"{rmse:.2f} kWh/día")
            col3.metric("Consumo Total Predicho", f"{df_futuro['consumo_predicho_kwh'].sum():,.0f} kWh")
            st.markdown("---")

            st.subheader("Gráfico de la Predicción de Consumo Energético (Diario)")
            df_historico_plot = df_historico_daily[['fecha', 'consumo_kwh']].rename(columns={'consumo_kwh': 'Consumo'})
            df_historico_plot['Tipo'] = 'Histórico'
            df_futuro_plot = df_futuro[['fecha', 'consumo_predicho_kwh']].rename(columns={'consumo_predicho_kwh': 'Consumo'})
            df_futuro_plot['Tipo'] = 'Predicción'
            df_plot = pd.concat([df_historico_plot, df_futuro_plot])
            fig = px.line(df_plot, x='fecha', y='Consumo', color='Tipo', 
                          title='Consumo Histórico vs. Predicción Futura (Diario)',
                          labels={'fecha': 'Fecha', 'Consumo': 'Consumo Total (kWh/día)'},
                          color_discrete_map={'Histórico': 'blue', 'Predicción': 'orange'})
            fig.add_vline(x=df_historico_daily['fecha'].max(), line_width=2, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Datos Detallados de la Predicción (Diaria)")
            st.dataframe(df_futuro[['fecha', 'temp_min_c', 'temp_max_c', 'temp_avg_c', 'consumo_predicho_kwh']].round(2))

else:
    # --- MODIFICADO: Mensaje de bienvenida actualizado ---
    st.info("ℹ️ **Para comenzar**, por favor selecciona los archivos de datos históricos y completa la configuración de la API en la barra lateral izquierda.")
    st.markdown("""
    Esta herramienta utilizará los datos que proporcionas para:
    1.  **Cargar** los datos históricos de consumo y clima desde la carpeta `data`.
    2.  **Obtener** el pronóstico del tiempo diario más reciente usando la API de Meteosource.
    3.  **Entrenar** un modelo de Inteligencia Artificial (Machine Learning).
    4.  **Predecir** el **consumo total diario** futuro basándose en ese pronóstico.
    """)
