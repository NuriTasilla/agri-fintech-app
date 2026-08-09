import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import google.generativeai as genai
import plotly.graph_objects as go

# 1. Configuración de Página
st.set_page_config(
    page_title="Agri-Fintech Credit Cockpit", 
    page_icon="🌾", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 2. CSS Minimalista y Limpio (Sin containers fantasma)
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Estilizado para contenedores nativos de Streamlit */
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        border-radius: 12px;
    }
    
    .main-title {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }
    
    .sub-title {
        color: #64748B;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }

    .badge-approved { background-color: #DCFCE7; color: #166534; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.9rem; border: 1px solid #BBF7D0; display: inline-block; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.9rem; border: 1px solid #FDE68A; display: inline-block; }
    .badge-rejected { background-color: #FEE2E2; color: #991B1B; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.9rem; border: 1px solid #FECACA; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# Encabezado
st.markdown('<div class="main-title">🌾CoFundo: Credit Cockpit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Plataforma de evaluación crediticia alternativa con analítica satelital, IOT e IA.</div>', unsafe_allow_html=True)

# 3. Cargar Recursos
@st.cache_resource
def cargar_recursos():
    with open('modelo_agrotech.pkl', 'rb') as f:
        datos_modelo = pickle.load(f)
    df_50 = pd.read_csv('50_ejemplares_kaggle.csv')
    return datos_modelo['modelo'], datos_modelo['columnas'], df_50

try:
    modelo_rf, columnas_ml, df_50 = cargar_recursos()
except Exception as e:
    st.error("⚠️ Error al cargar archivos del repositorio ('modelo_agrotech.pkl' o '50_ejemplares_kaggle.csv').")
    st.stop()

# 4. Configurar API Gemini
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

# FUNCIÓN CORREGIDA CON AUTO-DETECCIÓN DE MODELO ACTIVO
def consultar_agente_gemini_json(prompt_texto):
    modelos_disponibles = []
    
    # 1. Explorar dinámicamente qué modelos soporta tu API Key
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                modelos_disponibles.append(m.name)
    except Exception:
        pass
        
    # Nombres de respaldo estándar por si falla la lista
    modelos_respaldo = [
        'gemini-1.5-flash', 'gemini-1.5-pro',
        'models/gemini-1.5-flash', 'models/gemini-1.5-pro',
        'gemini-1.0-pro', 'models/gemini-1.0-pro'
    ]
    
    # Unir sin duplicados
    modelos_a_probar = modelos_disponibles + [m for m in modelos_respaldo if m not in modelos_disponibles]
    
    ultimo_error = None

    # Intento A: Con modo JSON nativo
    for mod in modelos_a_probar:
        try:
            model = genai.GenerativeModel(
                mod, 
                generation_config={"response_mime_type": "application/json"}
            )
            response = model.generate_content(prompt_texto)
            return json.loads(response.text)
        except Exception as e:
            ultimo_error = e
            continue

    # Intento B: Si el endpoint no soporta mime_type, forzar extracción del bloque de texto JSON
    for mod in modelos_a_probar:
        try:
            model = genai.GenerativeModel(mod)
            response = model.generate_content(prompt_texto + "\n\nResponde ÚNICAMENTE con un objeto JSON válido sin texto adicional.")
            texto_raw = response.text.strip()
            if "```json" in texto_raw:
                texto_raw = texto_raw.split("```json")[1].split("```")[0].strip()
            elif "```" in texto_raw:
                texto_raw = texto_raw.split("```")[1].split("```")[0].strip()
            return json.loads(texto_raw)
        except Exception as e:
            ultimo_error = e
            continue
            
    raise Exception(f"No se pudo conectar a Gemini. Detalle: {ultimo_error}")

# 5. Lógica Financiera y Gráficos
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
        number={'font': {'size': 46, 'color': '#0F172A', 'family': 'Inter'}},
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
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(t=10, b=10, l=15, r=15))
    return fig

# RADAR EXTENDIDO A 8 EJES
def plot_radar_agronomo_8_ejes(row):
    val_ndvi = min(row['NDVI_index'] * 100, 100)
    val_moist = min(row['soil_moisture_%'], 100)
    
    ph = row['soil_pH']
    val_ph = 100 - (abs(6.8 - ph) * 20)
    val_ph = max(min(val_ph, 100), 10)
    
    val_rain = min((row['rainfall_mm'] / 800) * 100, 100)
    
    temp = row['temperature_C']
    val_temp = 100 - (abs(25 - temp) * 4)
    val_temp = max(min(val_temp, 100), 20)
    
    val_sun = min((row['sunlight_hours'] / 12) * 100, 100)
    
    dict_disease = {"None": 100, "Mild": 75, "Moderate": 40, "Severe": 10}
    val_dis = dict_disease.get(row['crop_disease_status'], 50)
    
    dict_irrig = {"Drip": 100, "Sprinkler": 80, "Manual": 40, "None": 10}
    val_irrig = dict_irrig.get(row['irrigation_type'], 30)
    
    categories = [
        'Vigor (NDVI)', 'Humedad Suelo', 'Balance pH', 'Precipitación',
        'Confort Térmico', 'Horas Sol', 'Sanidad Cultivo', 'Eficiencia Riego'
    ]
    values = [val_ndvi, val_moist, val_ph, val_rain, val_temp, val_sun, val_dis, val_irrig]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.18)',
        line=dict(color='#2563EB', width=2)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#E2E8F0", tickfont=dict(color="#64748B", size=8)),
            angularaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#1E293B", size=10, family="Inter")),
            bgcolor="white"
        ),
        paper_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=25, b=25, l=25, r=25)
    )
    return fig

# 6. Pestañas
tab1, tab2 = st.tabs(["📊 Catálogo de Clientes (Kaggle)", "📝 Evaluación Manual"])

with tab1:
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        farm_id = st.selectbox("Seleccionar Parcela / Agricultor:", df_50['farm_id'].tolist())
        farm_data = df_50[df_50['farm_id'] == farm_id].iloc[0]
    with col_sel2:
        loan_requested = st.number_input("Monto Crédito Solicitado ($ USD):", 100, 10000, 800, step=50)

    # FICHA TÉCNICA
    with st.container(border=True):
        st.subheader("📍 Ficha Técnica & Sensores Agrónomos")
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Cultivo", str(farm_data['crop_type']))
        f2.metric("Región", str(farm_data['region']))
        f3.metric("Índice NDVI", f"{farm_data['NDVI_index']:.2f}")
        f4.metric("Sistema Riego", str(farm_data['irrigation_type']))
        f5.metric("Estado Salud", str(farm_data['crop_disease_status']))

        st.divider()
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.write(f"💧 **Humedad:** {farm_data['soil_moisture_%']}%")
        s2.write(f"🧪 **pH Suelo:** {farm_data['soil_pH']}")
        s3.write(f"🌧️ **Lluvia:** {farm_data['rainfall_mm']} mm")
        s4.write(f"🌡️ **Temp:** {farm_data['temperature_C']} °C")
        s5.write(f"☀️ **Horas Sol:** {farm_data['sunlight_hours']} hrs")

    # BOTÓN PRINCIPAL
    if st.button("⚡ Ejecutar Evaluador FinTech & Agente IA (JSON)", type="primary", use_container_width=True):
        
        # Machine Learning
        input_data = pd.DataFrame([farm_data])
        input_ml = pd.get_dummies(input_data, columns=['irrigation_type', 'crop_disease_status'], drop_first=False)
        input_ml = input_ml.reindex(columns=columnas_ml, fill_value=0)
        
        yield_pred = modelo_rf.predict(input_ml)[0]
        score, decision, badge_style, dscr, revenue = calcular_credit_score(
            yield_pred, loan_requested, farm_data['NDVI_index'], farm_data['crop_disease_status']
        )
        
        # PANEL BENTO PRINCIPAL
        col_res1, col_res2, col_res3 = st.columns([1.1, 1, 1.2])
        
        with col_res1:
            with st.container(border=True):
                st.markdown("##### AI Credit Score")
                st.plotly_chart(plot_gauge_score(score), use_container_width=True)
                st.markdown(f'<div style="text-align:center;"><span class="{badge_style}">{decision}</span></div>', unsafe_allow_html=True)

        with col_res2:
            with st.container(border=True):
                st.markdown("##### Capacidad Financiera")
                st.metric("Rendimiento Estimado", f"{yield_pred:,.1f} kg/ha")
                st.metric("Ingreso Neto Estimado", f"${revenue:,.2f} USD")
                st.metric("Cobertura Deuda (DSCR)", f"{dscr:.2f}x", "Apto" if dscr >= 1.2 else "Riesgo Mora")

        with col_res3:
            with st.container(border=True):
                st.markdown("##### Radar Agrónomo 360° (8 Ejes)")
                st.plotly_chart(plot_radar_agronomo_8_ejes(farm_data), use_container_width=True)

        # AGENTE IA CON JSON STRUCTURED OUTPUT
        if api_key:
            st.markdown("### 🤖 Dictamen Cualitativo e Stress Testing (Powered by Gemini JSON)")
            with st.spinner("Procesando JSON de sensibilidad de riesgo..."):
                prompt_json = f"""
                Eres el Director General de Riesgos en un Banco Agrícola.
                Analiza el cliente {farm_data['farm_id']} y responde ESTRICTAMENTE con la siguiente estructura JSON válida:

                {{
                    "indices_riesgo": {{
                        "riesgo_climatico_pct": 35,
                        "riesgo_mercado_pct": 20,
                        "resiliencia_operativa_pct": 80
                    }},
                    "fortalezas": [
                        "Punto fuerte 1 sobre el cultivo o la salud satelital",
                        "Punto fuerte 2 sobre finanzas o capacidad de pago"
                    ],
                    "alertas": [
                        "Alerta 1 sobre plagas, riego o clima",
                        "Alerta 2 sobre sensibilidad de ingresos"
                    ],
                    "stress_test": [
                        {{"escenario": "Caída del 20% en Precio del Cultivo", "nuevo_dscr": "{dscr*0.8:.2f}x", "viabilidad": "Aceptable"}},
                        {{"escenario": "Sequía Severa (-30% Rendimiento)", "nuevo_dscr": "{dscr*0.7:.2f}x", "viabilidad": "Riesgoso"}},
                        {{"escenario": "Aumento del 15% en Costo de Insumos", "nuevo_dscr": "{dscr*0.9:.2f}x", "viabilidad": "Aceptable"}}
                    ],
                    "clausulas_sugeridas": [
                        "Cláusula 1 obligatoria para desembolsar",
                        "Cláusula 2 obligatoria de monitoreo"
                    ]
                }}

                Datos del Cliente:
                Cultivo: {farm_data['crop_type']} | Región: {farm_data['region']} | NDVI: {farm_data['NDVI_index']}
                Lluvia: {farm_data['rainfall_mm']}mm | Riego: {farm_data['irrigation_type']} | Enfermedad: {farm_data['crop_disease_status']}
                Préstamo: ${loan_requested} USD | Rendimiento Estimado: {yield_pred:.1f} kg/ha | DSCR: {dscr:.2f}x | Score: {score}/850
                """
                
                try:
                    data_json = consultar_agente_gemini_json(prompt_json)
                    
                    # RENDERIZADO DESDE EL JSON
                    c_ai1, c_ai2 = st.columns(2)
                    
                    with c_ai1:
                        with st.container(border=True):
                            st.markdown("##### 🎯 Sub-Índices de Riesgo (Calculados por IA)")
                            idx = data_json.get("indices_riesgo", {})
                            st.write(f"**Riesgo Climático:** {idx.get('riesgo_climatico_pct', 0)}%")
                            st.progress(idx.get('riesgo_climatico_pct', 0) / 100)
                            
                            st.write(f"**Riesgo de Mercado:** {idx.get('riesgo_mercado_pct', 0)}%")
                            st.progress(idx.get('riesgo_mercado_pct', 0) / 100)
                            
                            st.write(f"**Resiliencia Operativa:** {idx.get('resiliencia_operativa_pct', 0)}%")
                            st.progress(idx.get('resiliencia_operativa_pct', 0) / 100)

                    with c_ai2:
                        with st.container(border=True):
                            st.markdown("##### 🧪 Matriz de Sensibilidad / Stress Test")
                            df_stress = pd.DataFrame(data_json.get("stress_test", []))
                            st.dataframe(df_stress, use_container_width=True, hide_index=True)

                    c_det1, c_det2 = st.columns(2)
                    with c_det1:
                        with st.container(border=True):
                            st.markdown("##### 🟢 Fortalezas Identificadas")
                            for f in data_json.get("fortalezas", []):
                                st.write(f"• {f}")
                                
                            st.markdown("##### ⚠️ Alertas Críticas")
                            for a in data_json.get("alertas", []):
                                st.write(f"• {a}")

                    with c_det2:
                        with st.container(border=True):
                            st.markdown("##### 📜 Checklist de Cláusulas para Contrato")
                            st.caption("Marque las cláusulas verificadas por el analista antes de autorizar:")
                            for clausula in data_json.get("clausulas_sugeridas", []):
                                st.checkbox(clausula, value=False)
                            
                            st.success("✅ Estructura JSON validada e integrada correctamente.")

                except Exception as err:
                    st.error(f"Error procesando la respuesta JSON de Gemini: {err}")
        else:
            st.warning("🔑 Configure la Gemini API Key en el panel lateral para activar el análisis JSON.")

# --- TAB 2: SIMULADOR MANUAL ---
with tab2:
    with st.container(border=True):
        st.subheader("Parametrización Manual de Sensores")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            s_moisture = st.slider("Humedad Suelo (%)", 0.0, 100.0, 45.0)
            s_ph = st.slider("pH del Suelo", 4.0, 9.0, 6.5)
            temp = st.slider("Temperatura (°C)", 10.0, 45.0, 26.0)
        with mc2:
            rain = st.slider("Precipitación (mm)", 50.0, 1500.0, 600.0)
            ndvi = st.slider("NDVI (Vigor)", 0.2, 0.95, 0.65)
            sun = st.slider("Horas Sol", 2.0, 14.0, 8.0)
        with mc3:
            irrigation = st.selectbox("Sistema Riego", ["Drip", "Sprinkler", "Manual", "None"])
            disease = st.selectbox("Estado Fitosanitario", ["None", "Mild", "Moderate", "Severe"])
            loan_manual = st.number_input("Crédito Solicitado ($ USD):", 100, 10000, 1200, step=100)

    if st.button("🧪 Evaluar Simulación", type="primary", use_container_width=True):
        data_m = {
            'soil_moisture_%': s_moisture, 'soil_pH': s_ph, 'temperature_C': temp,
            'rainfall_mm': rain, 'sunlight_hours': sun, 'NDVI_index': ndvi,
            'irrigation_type': irrigation, 'crop_disease_status': disease
        }
        yield_sim = 4200.0 * ndvi
        score, decision, badge_style, dscr, revenue = calcular_credit_score(yield_sim, loan_manual, ndvi, disease)
        
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            with st.container(border=True):
                st.plotly_chart(plot_gauge_score(score), use_container_width=True)
                st.markdown(f'<div style="text-align:center;"><span class="{badge_style}">{decision}</span></div>', unsafe_allow_html=True)
        with c_m2:
            with st.container(border=True):
                st.metric("Rendimiento Estimado", f"{yield_sim:,.1f} kg/ha")
                st.metric("Ingreso Estimado", f"${revenue:,.2f} USD")
                st.metric("DSCR", f"{dscr:.2f}x")
        with c_m3:
            with st.container(border=True):
                st.plotly_chart(plot_radar_agronomo_8_ejes(data_m), use_container_width=True)
