import streamlit as st
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai

# Configuración de la página Web
st.set_page_config(page_title="Agri-Fintech Credit Scoring", page_icon="🌾", layout="wide")

# Título Principal
st.title("🌾 Agri-Fintech: Evaluador de Riesgo Crediticio con IoT & IA")
st.markdown("Plataforma de Scoring Crediticio Alternativo basado en sensores agrícolas y modelos de lenguaje.")

# Cargar Modelo de Machine Learning y Datos
@st.cache_resource
def cargar_recursos():
    with open('modelo_agrotech.pkl', 'rb') as f:
        datos_modelo = pickle.load(f)
    df_50 = pd.read_csv('50_ejemplares_kaggle.csv')
    return datos_modelo['modelo'], datos_modelo['columnas'], df_50

try:
    modelo_rf, columnas_ml, df_50 = cargar_recursos()
    st.sidebar.success("✅ Modelo y Datasets Cargados")
except Exception as e:
    st.error("⚠️ Sube los archivos 'modelo_agrotech.pkl' y '50_ejemplares_kaggle.csv' al directorio principal.")
    st.stop()

# Configurar API de Gemini
st.sidebar.header("🔑 Configuración del Agente IA")
api_key = st.sidebar.text_input("Ingresa tu Gemini API Key:", type="password")

if api_key:
    genai.configure(api_key=api_key)

# Pestañas del Dashboard
tab1, tab2 = st.tabs(["📊 Catálogo de 50 Granjas (Kaggle)", "📝 Carga / Simulación Manual"])

def calcular_credit_score(yield_pred, loan_amount, ndvi, disease_status, price_per_kg=0.35):
    # Ecuación de Capacidad de Pago
    estimated_revenue = yield_pred * price_per_kg
    # Estimated DSCR (Debt Service Coverage Ratio)
    dscr = estimated_revenue / max(loan_amount, 1)
    
    # Base Credit Score (300 to 850)
    score = 600 + (dscr * 100) + (ndvi * 100)
    
    if disease_status in ['Severe', 'Moderate']:
        score -= 80
    
    score = int(np.clip(score, 300, 850))
    
    if score >= 720:
        decision, color = "APROBADO", "green"
    elif score >= 600:
        decision, color = "RIESGO MODERADO (Aprobación Condicionada)", "orange"
    else:
        decision, color = "RECHAZADO", "red"
        
    return score, decision, color, dscr, estimated_revenue

# TAB 1: SELECCIÓN DE 50 CASOS
with tab1:
    st.subheader("Selecciona una Farm ID para Evaluación Instantánea")
    
    farm_id = st.selectbox("Granjas disponibles:", df_50['farm_id'].tolist())
    farm_data = df_50[df_50['farm_id'] == farm_id].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Cultivo:** {farm_data['crop_type']}")
        st.write(f"**Región:** {farm_data['region']}")
        st.write(f"**NDVI Index:** {farm_data['NDVI_index']}")
    with col2:
        st.write(f"**Humedad Suelo:** {farm_data['soil_moisture_%']}%")
        st.write(f"**Sistema Riego:** {farm_data['irrigation_type']}")
        st.write(f"**Enfermedades:** {farm_data['crop_disease_status']}")
    with col3:
        loan_requested = st.number_input("Monto del Microcrédito Solicitado ($ USD):", min_value=100, max_value=5000, value=800, step=50, key="loan_tab1")

    if st.button("Evaluar Crédito (Tab 1)", type="primary"):
        # Preparar vector de características para el modelo ML
        input_data = pd.DataFrame([farm_data])
        input_ml = pd.get_dummies(input_data, columns=['irrigation_type', 'crop_disease_status'], drop_first=False)
        input_ml = input_ml.reindex(columns=columnas_ml, fill_value=0)
        
        # Predicción ML
        yield_pred = modelo_rf.predict(input_ml)[0]
        score, decision, color, dscr, revenue = calcular_credit_score(yield_pred, loan_requested, farm_data['NDVI_index'], farm_data['crop_disease_status'])
        
        # Despliegue de Resultados
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cosecha Estimada (kg/ha)", f"{yield_pred:,.1f} kg")
        m2.metric("Ingreso Neto Estimado", f"${revenue:,.2f} USD")
        m3.metric("Credit Score Syntético", f"{score} / 850")
        m4.metric("Ratio DSCR", f"{dscr:.2f}x")
        
        st.markdown(f"### Dictamen del Modelo: <span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)
        
        # Evaluación por Agente IA (Gemini)
        if api_key:
            st.divider()
            st.subheader("🤖 Reporte Financiero del Agente de IA (Gemini)")
            with st.spinner("El Agente está redactando el análisis de riesgo crediticio..."):
                prompt = f"""
                Actúa como un Oficial Senior de Riesgo Crediticio en Microfinanzas Agrícolas.
                Analiza el siguiente caso y emite un informe estructurado en 3 párrafos:
                1. Análisis Técnico Agrónomo (salud del cultivo, riego y clima).
                2. Viabilidad Financiera (Capacidad de pago, DSCR y riesgo de mora).
                3. Recomendación Final y Mitigantes de Riesgo.

                Datos del Agricultor:
                - ID: {farm_data['farm_id']} | Cultivo: {farm_data['crop_type']} | Región: {farm_data['region']}
                - NDVI Index: {farm_data['NDVI_index']} | Estado de Enfermedad: {farm_data['crop_disease_status']}
                - Sistema de Riego: {farm_data['irrigation_type']} | Precipitación: {farm_data['rainfall_mm']} mm
                - Préstamo Solicitado: ${loan_requested} USD
                - Rendimiento Estimado por ML: {yield_pred:.1f} kg/ha
                - Ingresos Estimados: ${revenue:.2f} USD | DSCR: {dscr:.2f}
                - Score Crediticio Asignado: {score} ({decision})
                """
                try:
                    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    response = gemini_model.generate_content(prompt)
                    st.info(response.text)
                except Exception as err:
                    st.error(f"Error al conectar con la API de Gemini: {err}")
        else:
            st.warning("⚠️ Ingresa tu Gemini API Key en la barra lateral para generar el dictamen del Agente de IA.")

# TAB 2: SIMULADOR MANUAL
with tab2:
    st.subheader("Ingreso Manual de Parámetros de Sensores")
    c1, c2, c3 = st.columns(3)
    with c1:
        s_moisture = st.slider("Humedad del Suelo (%)", 0.0, 100.0, 45.0)
        s_ph = st.slider("pH del Suelo", 4.0, 9.0, 6.5)
        temp = st.slider("Temperatura Promedio (°C)", 10.0, 45.0, 26.0)
    with c2:
        rain = st.slider("Lluvia Total (mm)", 50.0, 1500.0, 600.0)
        ndvi = st.slider("Índice NDVI (Salud)", 0.2, 0.95, 0.65)
        days = st.number_input("Días Totales de Cultivo", 60, 200, 120)
    with c3:
        irrigation = st.selectbox("Tipo de Riego", ["Drip", "Sprinkler", "Manual", "None"])
        disease = st.selectbox("Estado Fitosanitario", ["None", "Mild", "Moderate", "Severe"])
        loan_manual = st.number_input("Monto Crédito Solicitado ($ USD):", 100, 10000, 1200, step=100)

    if st.button("Evaluar Caso Manual", type="primary"):
        data_manual = {
            'soil_moisture_%': s_moisture,
            'soil_pH': s_ph,
            'temperature_C': temp,
            'rainfall_mm': rain,
            'humidity_%': 50.0,
            'sunlight_hours': 8.0,
            'NDVI_index': ndvi,
            'total_days': days,
            'irrigation_type': irrigation,
            'crop_disease_status': disease
        }
        input_df = pd.DataFrame([data_manual])
        input_ml = pd.get_dummies(input_df, columns=['irrigation_type', 'crop_disease_status'], drop_first=False)
        input_ml = input_ml.reindex(columns=columnas_ml, fill_value=0)
        
        yield_pred = modelo_rf.predict(input_ml)[0]
        score, decision, color, dscr, revenue = calcular_credit_score(yield_pred, loan_manual, ndvi, disease)
        
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cosecha Estimada (kg/ha)", f"{yield_pred:,.1f} kg")
        m2.metric("Ingreso Neto Estimado", f"${revenue:,.2f} USD")
        m3.metric("Credit Score Syntético", f"{score} / 850")
        m4.metric("Ratio DSCR", f"{dscr:.2f}x")
        
        st.markdown(f"### Dictamen del Modelo: <span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)
        
        if api_key:
            st.divider()
            st.subheader("🤖 Reporte Financiero del Agente de IA (Gemini)")
            with st.spinner("El Agente está redactando el análisis de riesgo crediticio..."):
                prompt_manual = f"""
                Actúa como un Oficial Senior de Riesgo Crediticio en Microfinanzas Agrícolas.
                Analiza la siguiente simulación manual:
                1. Análisis Técnico Agrónomo.
                2. Viabilidad Financiera.
                3. Recomendación Final.

                Datos Simulados:
                - NDVI: {ndvi} | Enfermedad: {disease} | Riego: {irrigation} | Lluvia: {rain} mm
                - Préstamo Solicitado: ${loan_manual} USD
                - Rendimiento Estimado: {yield_pred:.1f} kg/ha
                - Ingresos Estimados: ${revenue:.2f} USD | DSCR: {dscr:.2f}
                - Score Crediticio: {score} ({decision})
                """
                try:
                    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    response = gemini_model.generate_content(prompt_manual)
                    st.info(response.text)
                except Exception as err:
                    st.error(f"Error al conectar con la API de Gemini: {err}")
        else:
            st.warning("⚠️ Ingresa tu Gemini API Key en la barra lateral para generar el dictamen del Agente de IA.")
