import streamlit as st
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai
import plotly.graph_objects as go

# 1. Configuración de la página (Wide & Dark)
st.set_page_config(page_title="Agri-Fintech Copilot", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

# 2. Inyección de CSS (Diseño Premium Dark Mode / UI Neumorfismo)
st.markdown("""
<style>
    /* Fondo principal y tipografía */
    .stApp {
        background-color: #0D0E15;
        color: #FFFFFF;
    }
    
    /* Tarjetas estilo Bento Grid (Glassmorphism) */
    .bento-card {
        background: #181C25;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #2A2F3D;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* Números Hero para Métricas */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #9CA3AF !important;
    }
    
    /* Badges / Píldoras de Estado */
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #10B981; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid rgba(16, 185, 129, 0.3); display: inline-block;}
    .badge-red { background: rgba(239, 68, 68, 0.15); color: #EF4444; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid rgba(239, 68, 68, 0.3); display: inline-block;}
    .badge-yellow { background: rgba(245, 158, 11, 0.15); color: #F59E0B; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid rgba(245, 158, 11, 0.3); display: inline-block;}
    .badge-blue { background: rgba(59, 130, 246, 0.15); color: #3B82F6; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; border: 1px solid rgba(59, 130, 246, 0.3); display: inline-block;}
</style>
""", unsafe_allow_html=True)

# Título y Header
st.markdown("## 🌾 Agri-Fintech AI: Credit Cockpit")
st.markdown("<span style='color:#9CA3AF; font-size:1.1rem;'>Plataforma de evaluación crediticia alternativa impulsada por IoT satelital e Inteligencia Artificial.</span>", unsafe_allow_html=True)
st.write("")

# 3. Cargar Modelo y Datos
@st.cache_resource
def cargar_recursos():
    with open('modelo_agrotech.pkl', 'rb') as f:
        datos_modelo = pickle.load(f)
    df_50 = pd.read_csv('50_ejemplares_kaggle.csv')
    return datos_modelo['modelo'], datos_modelo['columnas'], df_50

try:
    modelo_rf, columnas_ml, df_50 = cargar_recursos()
except Exception as e:
    st.error("⚠️ Error: Sube los archivos 'modelo_agrotech.pkl' y '50_ejemplares_kaggle.csv'.")
    st.stop()

# 4. Configurar API Gemini (Auto-Secrets)
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"].strip()
else:
    st.sidebar.markdown("### 🔑 Acceso API")
    user_key = st.sidebar.text_input("Gemini API Key:", type="password")
    if user_key:
        api_key = user_key.strip()

if api_key:
    genai.configure(api_key=api_key)

def consultar_agente_gemini(prompt_texto):
    modelos_candidatos = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro', 'models/gemini-1.5-flash']
    ultimo_error = None
    for nombre_modelo in modelos_candidatos:
        try:
            model = genai.GenerativeModel(nombre_modelo)
            response = model.generate_content(prompt_texto)
            return response.text
        except Exception as e:
            ultimo_error = e
            continue
    raise Exception(f"Falla de conexión LLM: {ultimo_error}")

# 5. Funciones Financieras y de UI
def calcular_credit_score(yield_pred, loan_amount, ndvi, disease_status, price_per_kg=0.35):
    estimated_revenue = yield_pred * price_per_kg
    dscr = estimated_revenue / max(loan_amount, 1)
    score = 600 + (dscr * 100) + (ndvi * 100)
    
    if disease_status in ['Severe', 'Moderate']:
        score -= 80
        
    score = int(np.clip(score, 300, 850))
    
    if score >= 720:
        return score, "APROBADO", "#10B981", dscr, estimated_revenue
    elif score >= 600:
        return score, "RIESGO MODERADO", "#F59E0B", dscr, estimated_revenue
    else:
        return score, "RECHAZADO", "#EF4444", dscr, estimated_revenue

# --- GRÁFICOS VISUALES AVANZADOS (PLOTLY) ---
def plot_gauge_score(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 60, 'color': 'white', 'weight': 'bold'}},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#3B82F6", 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [300, 599], 'color': "rgba(239, 68, 68, 0.4)"},   # Red
                {'range': [600, 719], 'color': "rgba(245, 158, 11, 0.4)"},  # Yellow
                {'range': [720, 850], 'color': "rgba(16, 185, 129, 0.4)"}   # Green
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=20, b=20, l=20, r=20))
    return fig

def plot_radar_agronomo(ndvi, moisture, disease, irrigation):
    # Normalizar valores a escala 0-100 para el radar
    val_ndvi = min(ndvi * 100, 100)
    val_moist = min(moisture, 100)
    
    dict_disease = {"None": 100, "Mild": 75, "Moderate": 40, "Severe": 10}
    val_dis = dict_disease.get(disease, 50)
    
    dict_irrig = {"Drip": 100, "Sprinkler": 80, "Manual": 40, "None": 10}
    val_irrig = dict_irrig.get(irrigation, 50)
    
    categories = ['Vigor (NDVI)', 'Humedad Suelo', 'Salud (Fitosanitario)', 'Eficiencia Riego']
    values = [val_ndvi, val_moist, val_dis, val_irrig]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.3)',
        line=dict(color='#10B981', width=2)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2A2F3D", tickfont=dict(color="#9CA3AF")),
            angularaxis=dict(gridcolor="#2A2F3D", tickfont=dict(color="white", size=12)),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=40, b=40, l=40, r=40)
    )
    return fig

# 6. Lógica de Pestañas
tab1, tab2 = st.tabs(["🗃️ Base de Datos (Kaggle)", "🎛️ Simulador Manual"])

# --- TAB 1: EVALUACIÓN KAGGLE ---
with tab1:
    col_sel, col_req = st.columns([2, 1])
    with col_sel:
        farm_id = st.selectbox("Buscar Cliente (Farm ID):", df_50['farm_id'].tolist())
        farm_data = df_50[df_50['farm_id'] == farm_id].iloc[0]
    with col_req:
        loan_requested = st.number_input("Monto de Crédito ($ USD):", 100, 5000, 800, step=100)

    # Botón Flotante Principal
    if st.button("🚀 Ejecutar Scoring Multi-Dimensional", type="primary", use_container_width=True):
        
        # Preparación de datos y predicción ML
        input_data = pd.DataFrame([farm_data])
        input_ml = pd.get_dummies(input_data, columns=['irrigation_type', 'crop_disease_status'], drop_first=False)
        input_ml = input_ml.reindex(columns=columnas_ml, fill_value=0)
        
        yield_pred = modelo_rf.predict(input_ml)[0]
        score, decision, color_hex, dscr, revenue = calcular_credit_score(
            yield_pred, loan_requested, farm_data['NDVI_index'], farm_data['crop_disease_status']
        )
        
        # === INICIO DEL BENTO GRID (UI) ===
        st.write("---")
        
        # Fila 1: KPIs Principales
        c1, c2, c3 = st.columns([1.2, 1, 1])
        
        with c1:
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color:#9CA3AF; margin-top:0;'>AI Credit Score</h4>", unsafe_allow_html=True)
            st.plotly_chart(plot_gauge_score(score), use_container_width=True)
            
            # Badge Dinámico
            badge_class = "badge-green" if score >= 720 else ("badge-yellow" if score >= 600 else "badge-red")
            st.markdown(f"<div style='text-align:center;'><span class='{badge_class}'>{decision}</span></div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color:#9CA3AF; margin-top:0;'>Proyección Financiera</h4>", unsafe_allow_html=True)
            st.metric("Ingreso Bruto Estimado", f"${revenue:,.2f} USD", f"{yield_pred:,.1f} kg")
            st.metric("Monto Solicitado", f"${loan_requested:,.2f} USD")
            
            dscr_color = "normal" if dscr > 1.2 else "inverse"
            st.metric("Ratio de Cobertura (DSCR)", f"{dscr:.2f}x", "Saludable" if dscr > 1.2 else "Riesgoso", delta_color=dscr_color)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color:#9CA3AF; margin-top:0;'>Radar Agrónomo</h4>", unsafe_allow_html=True)
            st.plotly_chart(plot_radar_agronomo(
                farm_data['NDVI_index'], farm_data['soil_moisture_%'], 
                farm_data['crop_disease_status'], farm_data['irrigation_type']
            ), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Fila 2: Copiloto de IA (Generación Estructurada)
        if api_key:
            st.markdown("### 🤖 Agente IA: Copiloto de Riesgo")
            with st.spinner("Analizando matrices de riesgo..."):
                prompt = f"""
                Eres el Copiloto de Riesgo Crediticio de un Banco Agrícola de primer nivel.
                Analiza este caso y devuelve EXCLUSIVAMENTE 3 secciones usando este formato markdown exacto (sin introducciones):

                ### 🟢 Fortalezas del Perfil
                (Usa viñetas para listar los puntos fuertes técnicos y financieros)

                ### 🔴 Alertas Tempranas y Sensibilidad
                (Usa viñetas para listar riesgos climáticos, biológicos o financieros)

                ### 📜 Cláusulas Sugeridas para el Contrato
                (Redacta 2 condiciones estrictas para desembolsar el crédito basadas en los datos)

                Datos de entrada:
                Cultivo: {farm_data['crop_type']} | Riego: {farm_data['irrigation_type']} | Enfermedad: {farm_data['crop_disease_status']}
                NDVI: {farm_data['NDVI_index']} | Lluvia: {farm_data['rainfall_mm']}mm
                Préstamo: ${loan_requested} | Ingresos Estimados: ${revenue:.2f} | DSCR: {dscr:.2f}x | Score: {score}/850
                """
                try:
                    resultado_ia = consultar_agente_gemini(prompt)
                    
                    # Mostrar resultado en una tarjeta Bento grande
                    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
                    
                    # Dividimos la respuesta de la IA en columnas si es posible, o usamos markdown nativo
                    c_ai1, c_ai2 = st.columns(2)
                    
                    # Parseo rápido por headers
                    secciones = resultado_ia.split('###')
                    fortalezas = "🟢 Fortalezas"
                    alertas = "🔴 Alertas"
                    clausulas = "📜 Cláusulas"
                    
                    for sec in secciones:
                        if "Fortalezas" in sec: fortalezas = "###" + sec
                        elif "Alertas" in sec: alertas = "###" + sec
                        elif "Cláusulas" in sec: clausulas = "###" + sec

                    with c_ai1:
                        st.markdown(fortalezas)
                        st.markdown(alertas)
                    with c_ai2:
                        st.info("💡 **Decisión del Motor Automático**")
                        st.markdown(clausulas)
                        
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as err:
                    st.error(f"Error IA: {err}")
        else:
            st.warning("⚠️ Requiere Gemini API Key en código (Secrets) o panel lateral para el Copiloto IA.")

# --- TAB 2: SIMULADOR MANUAL (Resumido para la UI) ---
with tab2:
    st.markdown("<h4 style='color:#9CA3AF;'>Consola de Parametrización Manual</h4>", unsafe_allow_html=True)
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        s_moisture = st.slider("Humedad Suelo (%)", 0.0, 100.0, 45.0)
        temp = st.slider("Temp (°C)", 10.0, 45.0, 26.0)
    with c_m2:
        ndvi = st.slider("NDVI (Salud Vigorosa)", 0.2, 0.95, 0.70)
        rain = st.slider("Lluvia (mm)", 50.0, 1500.0, 600.0)
    with c_m3:
        irrigation = st.selectbox("Riego", ["Drip", "Sprinkler", "Manual", "None"])
        disease = st.selectbox("Fitosanitario", ["None", "Mild", "Moderate", "Severe"])
        loan_manual = st.number_input("Crédito ($ USD)", 100, 10000, 1500, step=100)

    if st.button("Simular Escenario de Riesgo", type="primary", use_container_width=True):
        st.info("La lógica visual del Dashboard Bento se aplica de la misma forma que en la Pestaña 1. (Puedes replicar el código de visualización aquí si lo deseas expandir).")
        # Aquí puedes copiar/pegar fácilmente el bloque "INICIO DEL BENTO GRID" si quieres que el simulador manual también dibuje todo.
