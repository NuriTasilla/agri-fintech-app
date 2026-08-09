import streamlit as st
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai
import plotly.graph_objects as go

# 1. Configuración de Página (Wide & Light)
st.set_page_config(
    page_title="Agri-Fintech Credit Cockpit", 
    page_icon="🌾", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 2. Inyección de CSS (Diseño Minimalista Light / Clean Corporate Fintech)
st.markdown("""
<style>
    /* Fondo General en Blanco/Gris Neutro Suave */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Tarjetas Blancas con Bordes Suaves y Sombra Sutil */
    .card-light {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
    }
    
    /* Titulares Estilo Fintech */
    .main-title {
        color: #0F172A;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    
    .sub-title {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 24px;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 12px;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 6px;
    }
    
    /* Badges / Etiquetas de Estado */
    .badge-approved { background-color: #DCFCE7; color: #166534; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid #BBF7D0; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid #FDE68A; }
    .badge-rejected { background-color: #FEE2E2; color: #991B1B; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid #FECACA; }
    .badge-info { background-color: #EFF6FF; color: #1E40AF; padding: 4px 10px; border-radius: 8px; font-weight: 600; font-size: 0.8rem; }

    /* Ajuste de inputs de Streamlit */
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado
st.markdown('<div class="main-title">🌾 Agri-Fintech AI: Credit Cockpit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Plataforma de evaluación de riesgo crediticio para pequeños agricultores basada en IoT satelital y modelos LLM.</div>', unsafe_allow_html=True)

# 3. Cargar Modelo de Machine Learning y Dataset
@st.cache_resource
def cargar_recursos():
    with open('modelo_agrotech.pkl', 'rb') as f:
        datos_modelo = pickle.load(f)
    df_50 = pd.read_csv('50_ejemplares_kaggle.csv')
    return datos_modelo['modelo'], datos_modelo['columnas'], df_50

try:
    modelo_rf, columnas_ml, df_50 = cargar_recursos()
except Exception as e:
    st.error("⚠️ Error: Sube los archivos 'modelo_agrotech.pkl' y '50_ejemplares_kaggle.csv' al repositorio.")
    st.stop()

# 4. Configurar API Gemini con soporte multi-modelo ultra-robusto
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"].strip()
else:
    st.sidebar.markdown("### 🔑 API Key")
    user_key = st.sidebar.text_input("Gemini API Key:", type="password")
    if user_key:
        api_key = user_key.strip()

if api_key:
    genai.configure(api_key=api_key)

def consultar_agente_gemini(prompt_texto):
    # Intentar obtener modelos dinámicamente asignados a la API Key
    modelos_a_probar = []
    try:
        models_list = genai.list_models()
        for m in models_list:
            if 'generateContent' in m.supported_generation_methods:
                modelos_a_probar.append(m.name)
    except Exception:
        pass
    
    # Modelos de respaldo estándar
    modelos_a_probar.extend(['gemini-1.5-flash', 'models/gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro'])
    
    ultimo_error = None
    for nombre_modelo in modelos_a_probar:
        try:
            model = genai.GenerativeModel(nombre_modelo)
            response = model.generate_content(prompt_texto)
            return response.text
        except Exception as e:
            ultimo_error = e
            continue
            
    raise Exception(f"No se pudo establecer comunicación con la API de Gemini. Detalle: {ultimo_error}")

# 5. Funciones Financieras y Gráficos Light Mode
def calcular_credit_score(yield_pred, loan_amount, ndvi, disease_status, price_per_kg=0.35):
    estimated_revenue = yield_pred * price_per_kg
    dscr = estimated_revenue / max(loan_amount, 1)
    score = 600 + (dscr * 100) + (ndvi * 100)
    
    if disease_status in ['Severe', 'Moderate']:
        score -= 80
        
    score = int(np.clip(score, 300, 850))
    
    if score >= 720:
        return score, "APROBADO", "badge-approved", dscr, estimated_revenue
    elif score >= 600:
        return score, "RIESGO MODERADO", "badge-warning", dscr, estimated_revenue
    else:
        return score, "RECHAZADO", "badge-rejected", dscr, estimated_revenue

def plot_gauge_score(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 50, 'color': '#0F172A', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#2563EB", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [300, 599], 'color': "rgba(239, 68, 68, 0.15)"},
                {'range': [600, 719], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [720, 850], 'color': "rgba(16, 185, 129, 0.15)"}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(t=20, b=10, l=20, r=20))
    return fig

def plot_radar_agronomo(ndvi, moisture, disease, irrigation):
    val_ndvi = min(ndvi * 100, 100)
    val_moist = min(moisture, 100)
    
    dict_disease = {"None": 100, "Mild": 75, "Moderate": 40, "Severe": 10}
    val_dis = dict_disease.get(disease, 50)
    
    dict_irrig = {"Drip": 100, "Sprinkler": 80, "Manual": 40, "None": 10}
    val_irrig = dict_irrig.get(irrigation, 50)
    
    categories = ['Vigor (NDVI)', 'Humedad Suelo', 'Salud Cultivo', 'Eficiencia Riego']
    values = [val_ndvi, val_moist, val_dis, val_irrig]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.15)',
        line=dict(color='#2563EB', width=2)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#E2E8F0", tickfont=dict(color="#64748B", size=9)),
            angularaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#334155", size=11, family="Inter")),
            bgcolor="white"
        ),
        paper_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=30, b=30, l=30, r=30)
    )
    return fig

# 6. Pestañas de Navegación
tab1, tab2 = st.tabs(["📊 Catálogo de Clientes (Kaggle)", "📝 Evaluación Manual de Parcelas"])

# --- TAB 1: EVALUACIÓN DE PARCELA ---
with tab1:
    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        farm_id = st.selectbox("Seleccionar ID de Parcela / Agricultor:", df_50['farm_id'].tolist())
        farm_data = df_50[df_50['farm_id'] == farm_id].iloc[0]
    with c_sel2:
        loan_requested = st.number_input("Monto del Crédito Solicitado ($ USD):", 100, 10000, 800, step=50)

    # FICHA TÉCNICA RESTRUCTURADA DE LA PARCELA (Visible siempre al seleccionar)
    st.markdown('<div class="card-light">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📍 Ficha Técnica & Datos Agrónomos de la Parcela</div>', unsafe_allow_html=True)
    
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Cultivo", str(farm_data['crop_type']))
    f2.metric("Ubicación / Región", str(farm_data['region']))
    f3.metric("Índice NDVI", f"{farm_data['NDVI_index']:.2f}")
    f4.metric("Sistema de Riego", str(farm_data['irrigation_type']))
    f5.metric("Estado Fitosanitario", str(farm_data['crop_disease_status']))

    with st.expander("🔍 Ver parámetros detallados de sensores (Suelo, Clima y Ciclo)"):
        s1, s2, s3, s4 = st.columns(4)
        s1.write(f"**Humedad del Suelo:** {farm_data['soil_moisture_%']}%")
        s2.write(f"**pH del Suelo:** {farm_data['soil_pH']}")
        s3.write(f"**Precipitación:** {farm_data['rainfall_mm']} mm")
        s4.write(f"**Temperatura Prom:** {farm_data['temperature_C']} °C")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡ Evaluar Riesgo y Generar Dictamen FinTech", type="primary", use_container_width=True):
        
        # Predicción Machine Learning
        input_data = pd.DataFrame([farm_data])
        input_ml = pd.get_dummies(input_data, columns=['irrigation_type', 'crop_disease_status'], drop_first=False)
        input_ml = input_ml.reindex(columns=columnas_ml, fill_value=0)
        
        yield_pred = modelo_rf.predict(input_ml)[0]
        score, decision, badge_style, dscr, revenue = calcular_credit_score(
            yield_pred, loan_requested, farm_data['NDVI_index'], farm_data['crop_disease_status']
        )
        
        # RESULTADOS: BENTO GRID LIGHT MODE
        st.write("")
        col_res1, col_res2, col_res3 = st.columns([1.2, 1, 1.1])
        
        # Card 1: Score & Dictamen
        with col_res1:
            st.markdown('<div class="card-light" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Score Crediticio Alternativo</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_gauge_score(score), use_container_width=True)
            st.markdown(f'<span class="{badge_style}">{decision}</span>', unsafe_allow_html=True)
            st.write("")
            st.markdown('</div>', unsafe_allow_html=True)

        # Card 2: Capacidad Financiera
        with col_res2:
            st.markdown('<div class="card-light">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Capacidad Financiera</div>', unsafe_allow_html=True)
            st.metric("Rendimiento Estimado (ML)", f"{yield_pred:,.1f} kg/ha")
            st.metric("Ingreso Neto Estimado", f"${revenue:,.2f} USD")
            st.metric("Ratio de Cobertura (DSCR)", f"{dscr:.2f}x", "Apto" if dscr >= 1.2 else "Riesgo de Mora")
            st.markdown('</div>', unsafe_allow_html=True)

        # Card 3: Balance Agro-Satelital
        with col_res3:
            st.markdown('<div class="card-light">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Balance Agrónomo 360°</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_radar_agronomo(
                farm_data['NDVI_index'], farm_data['soil_moisture_%'], 
                farm_data['crop_disease_status'], farm_data['irrigation_type']
            ), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # REPORTES DEL AGENTE IA (Estructurado en Tarjetas Blancas)
        if api_key:
            st.markdown("### 🤖 Dictamen del Agente de Inteligencia Artificial")
            with st.spinner("Analizando viabilidad de microcrédito agrícola..."):
                prompt = f"""
                Actúa como un Oficial Senior de Riesgo Crediticio en Microfinanzas Agrícolas.
                Analiza el siguiente caso y responde EXCLUSIVAMENTE en 3 bloques claramente delimitados:

                ### 🟢 Fortalezas del Cliente
                - (Enumera 2 o 3 puntos fuertes técnicos o de rendimiento)

                ### ⚠️ Alertas de Riesgo
                - (Enumera 2 o 3 riesgos climáticos, sanitarios o de capacidad de pago)

                ### 📜 Condiciones para Desembolso
                - (Propon 2 condiciones preventivas específicas para otorgar el crédito)

                Datos:
                ID: {farm_data['farm_id']} | Cultivo: {farm_data['crop_type']} | Región: {farm_data['region']}
                NDVI: {farm_data['NDVI_index']} | Enfermedad: {farm_data['crop_disease_status']} | Riego: {farm_data['irrigation_type']}
                Monto Solicitado: ${loan_requested} USD | Rendimiento Estimado: {yield_pred:.1f} kg/ha
                Ingresos Estimados: ${revenue:.2f} USD | DSCR: {dscr:.2f}x | Score: {score}/850 ({decision})
                """
                try:
                    respuesta_ia = consultar_agente_gemini(prompt)
                    
                    st.markdown('<div class="card-light">', unsafe_allow_html=True)
                    ai_col1, ai_col2 = st.columns(2)
                    
                    # Separar secciones
                    partes = respuesta_ia.split('###')
                    fortalezas = "🟢 Fortalezas"
                    alertas = "⚠️ Alertas"
                    condiciones = "📜 Condiciones"
                    
                    for p in partes:
                        if "Fortalezas" in p: fortalezas = "###" + p
                        elif "Alertas" in p: alertas = "###" + p
                        elif "Condiciones" in p: condiciones = "###" + p

                    with ai_col1:
                        st.markdown(fortalezas)
                        st.markdown(alertas)
                    with ai_col2:
                        st.markdown(condiciones)
                        st.success("✅ Análisis procesado correctamente por el Agente de IA.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as err:
                    st.error(f"Error al consultar el Agente de IA: {err}")
        else:
            st.warning("🔑 Agrega tu Gemini API Key para generar el análisis cualitativo con IA.")

# --- TAB 2: EVALUACIÓN MANUAL ---
with tab2:
    st.markdown('<div class="card-light">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Ingreso de Parámetros de Sensores & Parcela</div>', unsafe_allow_html=True)
    
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        s_moisture = st.slider("Humedad del Suelo (%)", 0.0, 100.0, 45.0)
        s_ph = st.slider("pH del Suelo", 4.0, 9.0, 6.5)
        temp = st.slider("Temperatura Promedio (°C)", 10.0, 45.0, 26.0)
    with mc2:
        rain = st.slider("Precipitación Total (mm)", 50.0, 1500.0, 600.0)
        ndvi = st.slider("Índice NDVI (Vigor)", 0.2, 0.95, 0.65)
        days = st.number_input("Días de Cultivo", 60, 200, 120)
    with mc3:
        irrigation = st.selectbox("Sistema de Riego", ["Drip", "Sprinkler", "Manual", "None"])
        disease = st.selectbox("Estado Fitosanitario", ["None", "Mild", "Moderate", "Severe"])
        loan_manual = st.number_input("Monto Crédito ($ USD):", 100, 10000, 1200, step=100)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🧪 Evaluar Simulación Manual", type="primary", use_container_width=True):
        data_manual = {
            'soil_moisture_%': s_moisture, 'soil_pH': s_ph, 'temperature_C': temp,
            'rainfall_mm': rain, 'humidity_%': 50.0, 'sunlight_hours': 8.0,
            'NDVI_index': ndvi, 'total_days': days, 'irrigation_type': irrigation, 'crop_disease_status': disease
        }
        input_df = pd.DataFrame([data_manual])
        input_ml = pd.get_dummies(input_df, columns=['irrigation_type', 'crop_disease_status'], drop_first=False)
        input_ml = input_ml.reindex(columns=columnas_ml, fill_value=0)
        
        yield_pred = modelo_rf.predict(input_ml)[0]
        score, decision, badge_style, dscr, revenue = calcular_credit_score(yield_pred, loan_manual, ndvi, disease)
        
        st.write("")
        col_res1, col_res2, col_res3 = st.columns([1.2, 1, 1.1])
        with col_res1:
            st.markdown('<div class="card-light" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Score Crediticio</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_gauge_score(score), use_container_width=True)
            st.markdown(f'<span class="{badge_style}">{decision}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res2:
            st.markdown('<div class="card-light">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Proyección Financiera</div>', unsafe_allow_html=True)
            st.metric("Rendimiento Estimado", f"{yield_pred:,.1f} kg/ha")
            st.metric("Ingreso Neto Estimado", f"${revenue:,.2f} USD")
            st.metric("Cobertura (DSCR)", f"{dscr:.2f}x")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res3:
            st.markdown('<div class="card-light">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Radar Agrónomo</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_radar_agronomo(ndvi, s_moisture, disease, irrigation), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
