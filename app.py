import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import google.generativeai as genai
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Configuración de la Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="CoFundo | Agri-Fintech Credit & Governance Cockpit", 
    page_icon="🌾", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. CSS Estilizado Profesional y Limpio
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        border-radius: 12px;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
    }
    
    .main-title {
        color: #FFFFFF;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    
    .sub-title {
        color: #94A3B8;
        font-size: 1.0rem;
        font-weight: 400;
    }

    .badge-approved { background-color: #DCFCE7; color: #166534; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 0.9rem; border: 1px solid #BBF7D0; display: inline-block; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 0.9rem; border: 1px solid #FDE68A; display: inline-block; }
    .badge-rejected { background-color: #FEE2E2; color: #991B1B; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 0.9rem; border: 1px solid #FECACA; display: inline-block; }

    .audit-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado Principal
st.markdown("""
<div class="main-header">
    <div class="main-title">🌾 CoFundo: Credit Cockpit & Audit Engine</div>
    <div class="sub-title">Plataforma de Evaluación Crediticia Algorítmica, Telemetría Satelital IoT y Auditoría de Gobernanza Financiera</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Cargar Recursos (Modelo ML y Dataset)
# ---------------------------------------------------------
@st.cache_resource
def cargar_recursos():
    try:
        with open('modelo_agrotech.pkl', 'rb') as f:
            datos_modelo = pickle.load(f)
        modelo_rf = datos_modelo['modelo']
        columnas_ml = datos_modelo['columnas']
    except Exception:
        modelo_rf = None
        columnas_ml = None
        
    try:
        df_50 = pd.read_csv('50_ejemplares_kaggle.csv')
    except Exception:
        # Dataset sintético de respaldo en caso de no hallar el archivo
        data_sintetica = {
            'farm_id': [f'FARM-{100+i}' for i in range(10)],
            'crop_type': ['Corn', 'Wheat', 'Soybean', 'Coffee', 'Cotton', 'Corn', 'Rice', 'Sugarcane', 'Avocado', 'Potato'],
            'region': ['Cundinamarca', 'Sinaloa', 'Cordoba', 'Eje Cafetero', 'Ica', 'Jalisco', 'Tolima', 'Valle', 'Michoacán', 'Boyacá'],
            'NDVI_index': [0.72, 0.58, 0.81, 0.65, 0.49, 0.77, 0.63, 0.85, 0.70, 0.55],
            'soil_moisture_%': [45.0, 32.0, 55.0, 60.0, 25.0, 48.0, 65.0, 58.0, 42.0, 38.0],
            'soil_pH': [6.5, 7.2, 6.2, 5.8, 7.8, 6.8, 6.0, 6.4, 6.1, 5.5],
            'rainfall_mm': [650, 400, 850, 1100, 200, 720, 950, 1050, 800, 500],
            'temperature_C': [24.0, 28.0, 26.0, 21.0, 31.0, 25.0, 27.0, 29.0, 22.0, 17.0],
            'sunlight_hours': [7.5, 9.0, 8.0, 6.5, 10.0, 8.5, 7.0, 8.0, 7.8, 6.0],
            'irrigation_type': ['Drip', 'Sprinkler', 'Drip', 'Manual', 'None', 'Drip', 'Manual', 'Drip', 'Drip', 'Sprinkler'],
            'crop_disease_status': ['None', 'Mild', 'None', 'Moderate', 'Severe', 'None', 'Mild', 'None', 'None', 'Moderate']
        }
        df_50 = pd.DataFrame(data_sintetica)
        
    return modelo_rf, columnas_ml, df_50

modelo_rf, columnas_ml, df_50 = cargar_recursos()

# ---------------------------------------------------------
# 4. Configurar API Gemini
# ---------------------------------------------------------
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"].strip()
else:
    st.sidebar.markdown("### 🔑 Configuración de API")
    user_key = st.sidebar.text_input("Gemini API Key:", type="password", help="Ingrese su API key de Google Gemini")
    if user_key:
        api_key = user_key.strip()

if api_key:
    genai.configure(api_key=api_key)

# Función con Auto-Detección de Modelo y Extracción JSON Robusta
def consultar_agente_gemini_json(prompt_texto):
    modelos_disponibles = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                modelos_disponibles.append(m.name)
    except Exception:
        pass
        
    modelos_respaldo = [
        'gemini-1.5-flash', 'gemini-1.5-pro',
        'models/gemini-1.5-flash', 'models/gemini-1.5-pro',
        'gemini-1.0-pro', 'models/gemini-1.0-pro'
    ]
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

    # Intento B: Extracción de texto estructurado en JSON
    for mod in modelos_a_probar:
        try:
            model = genai.GenerativeModel(mod)
            response = model.generate_content(prompt_texto + "\n\nResponde ÚNICAMENTE con un objeto JSON válido sin bloques de código o texto explicativo fuera del JSON.")
            texto_raw = response.text.strip()
            if "```json" in texto_raw:
                texto_raw = texto_raw.split("```json")[1].split("```")[0].strip()
            elif "```" in texto_raw:
                texto_raw = texto_raw.split("```")[1].split("```")[0].strip()
            return json.loads(texto_raw)
        except Exception as e:
            ultimo_error = e
            continue
            
    raise Exception(f"No se pudo conectar a Gemini API. Detalle: {ultimo_error}")

# ---------------------------------------------------------
# 5. Motor de Cálculos Financieros y Agronómicos
# ---------------------------------------------------------
def calcular_metricas_financieras(crop_type, loan_amount, tcea_pct, yield_pred, price_per_kg, ndvi, disease_status):
    # Estimación de Costos Operativos y Financieros
    costos_operativos_total = loan_amount * 0.075 # 7.5% comisiones, seguro, fondo reserva
    plazo_meses = 6
    interes_nominal = loan_amount * (tcea_pct / 100) * (plazo_meses / 12)
    deuda_total = loan_amount + interes_nominal + costos_operativos_total
    
    # Ingresos y OPEX de Producción
    ingreso_bruto = yield_pred * price_per_kg
    costo_produccion_estimado = ingreso_bruto * 0.45 # ~45% OPEX promedio
    
    # Retorno Neto para el agricultor
    retorno_neto_usd = ingreso_bruto - costo_produccion_estimado - deuda_total
    retorno_neto_pct = (retorno_neto_usd / ingreso_bruto * 100) if ingreso_bruto > 0 else 0.0
    
    # DSCR Cobertura
    dscr = ingreso_bruto / max(deuda_total, 1.0)
    
    # Cálculo de Score Algorítmico (300 - 850)
    score_base = 580 + (dscr * 80) + (ndvi * 120)
    if disease_status in ['Severe', 'Moderate']:
        score_base -= 90
    if retorno_neto_usd < 0:
        score_base -= 70
        
    score = int(np.clip(score_base, 300, 850))
    
    if score >= 720 and dscr >= 1.25 and retorno_neto_usd > 0:
        decision = "APROBADO DEFINITIVO"
        badge_style = "badge-approved"
    elif score >= 610 and dscr >= 1.05:
        decision = "APROBACIÓN CONDICIONAL"
        badge_style = "badge-warning"
    else:
        decision = "RECHAZADO / INVIABLE"
        badge_style = "badge-rejected"
        
    return {
        'score': score,
        'decision': decision,
        'badge_style': badge_style,
        'dscr': dscr,
        'deuda_total': deuda_total,
        'costos_operativos_total': costos_operativos_total,
        'ingreso_bruto': ingreso_bruto,
        'costo_produccion_estimado': costo_produccion_estimado,
        'retorno_neto_usd': retorno_neto_usd,
        'retorno_neto_pct': retorno_neto_pct,
        # Stress Test Scenarios
        'dscr_p20': (ingreso_bruto * 0.80) / max(deuda_total, 1.0),
        'dscr_seq': (ingreso_bruto * 0.70) / max(deuda_total, 1.0),
        'dscr_ins': ingreso_bruto / max(deuda_total + (costo_produccion_estimado * 0.15), 1.0)
    }

# Visualización: Gauge de Credit Score
def plot_gauge_score(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 42, 'color': '#0F172A', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#2563EB", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [300, 609], 'color': "rgba(239, 68, 68, 0.15)"},
                {'range': [610, 719], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [720, 850], 'color': "rgba(16, 185, 129, 0.15)"}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(t=10, b=10, l=15, r=15))
    return fig

# Visualización: Radar Agrónomo 360° (8 Ejes)
def plot_radar_agronomo_8_ejes(row):
    val_ndvi = min(row['NDVI_index'] * 100, 100)
    val_moist = min(row['soil_moisture_%'], 100)
    
    ph = row['soil_pH']
    val_ph = max(min(100 - (abs(6.5 - ph) * 22), 100), 10)
    val_rain = min((row['rainfall_mm'] / 900) * 100, 100)
    
    temp = row['temperature_C']
    val_temp = max(min(100 - (abs(24 - temp) * 4.5), 100), 15)
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
            angularaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#1E293B", size=9, family="Inter")),
            bgcolor="white"
        ),
        paper_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(t=20, b=20, l=20, r=20)
    )
    return fig

# ---------------------------------------------------------
# 6. Interfaz Streamlit (Pestañas Principal y Simulador)
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Catálogo de Clientes (Auditoría Live)", "📝 Simulador de Parcela Manual"])

with tab1:
    col_sel1, col_sel2, col_sel3 = st.columns([1.5, 1, 1])
    with col_sel1:
        farm_id = st.selectbox("Seleccionar Parcela / Cliente:", df_50['farm_id'].tolist())
        farm_data = df_50[df_50['farm_id'] == farm_id].iloc[0]
    with col_sel2:
        loan_requested = st.number_input("Capital Solicitado ($ USD):", 200, 25000, 1500, step=100)
    with col_sel3:
        tcea_pct = st.slider("TCEA Aplicable (%):", 12.0, 45.0, 26.5, step=0.5)

    # Ficha Técnica y Telemetría
    with st.container(border=True):
        st.subheader("📍 Ficha Técnica de Parcela y Telemetría IoT Satelital")
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Cultivo / Especie", str(farm_data['crop_type']))
        f2.metric("Región Agrícola", str(farm_data['region']))
        f3.metric("Índice Vigor (NDVI)", f"{farm_data['NDVI_index']:.2f}")
        f4.metric("Sistema Riego", str(farm_data['irrigation_type']))
        f5.metric("Estado Salud", str(farm_data['crop_disease_status']))

        st.divider()
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.write(f"💧 **Humedad Suelo:** {farm_data['soil_moisture_%']}%")
        s2.write(f"🧪 **pH del Suelo:** {farm_data['soil_pH']}")
        s3.write(f"🌧️ **Precipitación:** {farm_data['rainfall_mm']} mm")
        s4.write(f"🌡️ **Temperatura:** {farm_data['temperature_C']} °C")
        s5.write(f"☀️ **Horas Sol:** {farm_data['sunlight_hours']} hrs")

    # Mapeo de Precios y Rendimiento ML
    precio_base_map = {'Corn': 0.35, 'Wheat': 0.40, 'Soybean': 0.55, 'Coffee': 3.20, 'Cotton': 1.80, 'Rice': 0.45, 'Sugarcane': 0.12, 'Avocado': 2.10, 'Potato': 0.30}
    precio_kg = precio_base_map.get(farm_data['crop_type'], 0.50)
    
    # Predicción de Rendimiento ML o Heurística
    if modelo_rf is not None and columnas_ml is not None:
        input_data = pd.DataFrame([farm_data])
        input_ml = pd.get_dummies(input_data, columns=['irrigation_type', 'crop_disease_status'], drop_first=False)
        input_ml = input_ml.reindex(columns=columnas_ml, fill_value=0)
        yield_pred = float(modelo_rf.predict(input_ml)[0])
    else:
        yield_pred = 3800.0 * farm_data['NDVI_index']

    # Cálculo Financiero Avanzado
    fin = calcular_metricas_financieras(
        farm_data['crop_type'], loan_requested, tcea_pct, yield_pred, precio_kg, 
        farm_data['NDVI_index'], farm_data['crop_disease_status']
    )

    st.markdown("---")

    # BOTÓN PRINCIPAL DE EJECUCIÓN Y AUDITORÍA
    if st.button("⚡ Ejecutar Evaluador FinTech & Agente Senior de Auditoría", type="primary", use_container_width=True):
        
        # DASHBOARD RESUMEN EJECUTIVO
        st.markdown("### 📈 Dashboard Evaluativo del Motor Algorítmico")
        col_res1, col_res2, col_res3 = st.columns([1.1, 1.1, 1.2])
        
        with col_res1:
            with st.container(border=True):
                st.markdown("##### AI Credit Score Base")
                st.plotly_chart(plot_gauge_score(fin['score']), use_container_width=True)
                st.markdown(f'<div style="text-align:center;"><span class="{fin["badge_style"]}">{fin["decision"]}</span></div>', unsafe_allow_html=True)

        with col_res2:
            with st.container(border=True):
                st.markdown("##### Servicio Deuda & Métricas Éticas")
                st.metric("Rendimiento Estimado", f"{yield_pred:,.1f} kg/ha")
                st.metric("Servicio Deuda Total", f"${fin['deuda_total']:,.2f} USD")
                st.metric("Retorno Neto Agricultor", f"${fin['retorno_neto_usd']:,.2f} USD", f"{fin['retorno_neto_pct']:.1f}% del ingreso")
                st.metric("DSCR Cobertura", f"{fin['dscr']:.2f}x", "Apto (>=1.25x)" if fin['dscr']>=1.25 else "Riesgo de Default")

        with col_res3:
            with st.container(border=True):
                st.markdown("##### Radar Agrónomo 360° (8 Ejes)")
                st.plotly_chart(plot_radar_agronomo_8_ejes(farm_data), use_container_width=True)

        # AGENTE DE AUDITORÍA CON GEMINI
        if api_key:
            st.markdown("---")
            st.markdown("### 📋 Informe Formal de Auditoría y Gobernanza Financiera")
            
            with st.spinner("Ejecutando auditoría senior, análisis de riesgo hiperpersonalizado y stress test..."):
                
                # Datos parametrizados para el Prompt Maestro
                prompt_maestro = f"""
                ROL: Auditor Senior de Riesgo Crediticio, Analítica Algorítmica y Gobernanza Financiera en Entidades FinTech Agrícolas.
                
                Audita la evaluación crediticia para el caso {farm_id}.
                
                DATOS DEL CASO:
                - Identificador: {farm_id} | Región: {farm_data['region']}
                - Cultivo: {farm_data['crop_type']} (Grupo: Granos / Cultivo Comercial)
                - Ventana Monitoreo IoT: Últimos 45 días (Fase Vegetativa Intermedia)
                - Perfil Riesgo Grupo: 3.2 / 5.0 (Moderado - Alto)
                - Crédito Solicitado (Capital): ${loan_requested:,.2f} USD
                - TCEA: {tcea_pct}% | Comisiones/Reserva: ${fin['costos_operativos_total']:,.2f} USD
                - Deuda Total Exigible: ${fin['deuda_total']:,.2f} USD
                - Rendimiento Estimado: {yield_pred:,.1f} kg/ha | Precio Base: ${precio_kg:.2f} USD/kg
                - Ingreso Bruto Estimado: ${fin['ingreso_bruto']:,.2f} USD
                - OPEX Estimado Producción: ${fin['costo_produccion_estimado']:,.2f} USD
                - Retorno Neto Agricultor: ${fin['retorno_neto_usd']:,.2f} USD ({fin['retorno_neto_pct']:.1f}% de Retorno)
                - DSCR Cobertura: {fin['dscr']:.2f}x
                - Score Algorítmico: {fin['score']} / 850 ({fin['decision']})
                - Indicadores Operativos: Clima (38%), Mercado (42%), Resiliencia (75%)
                - Pruebas de Estrés (Stress Test):
                  * Caída 20% Precio: DSCR {fin['dscr_p20']:.2f}x
                  * Sequía Severa (-30% Rendimiento): DSCR {fin['dscr_seq']:.2f}x
                  * Alza 15% Costo Insumos: DSCR {fin['dscr_ins']:.2f}x
                - Telemetría IoT: NDVI {farm_data['NDVI_index']}, Humedad {farm_data['soil_moisture_%']}%, Riego {farm_data['irrigation_type']}, Fitosanitario {farm_data['crop_disease_status']}

                Responde ESTRICTAMENTE con un objeto JSON válido con la siguiente estructura exacta:
                {{
                  "indices_riesgo": {{
                    "riesgo_climatico_pct": 38,
                    "riesgo_mercado_pct": 42,
                    "resiliencia_operativa_pct": 75
                  }},
                  "stress_test": [
                    {{"escenario": "Caída 20% Precio Cosecha", "nuevo_dscr": "{fin['dscr_p20']:.2f}x", "viabilidad": "{'Inviable' if fin['dscr_p20']<1.0 else 'Aceptable'}"}},
                    {{"escenario": "Sequía Severa (-30% Rendimiento)", "nuevo_dscr": "{fin['dscr_seq']:.2f}x", "viabilidad": "{'Crítico / Default' if fin['dscr_seq']<1.0 else 'Aceptable'}"}},
                    {{"escenario": "Alza 15% Insumos / Fertilizantes", "nuevo_dscr": "{fin['dscr_ins']:.2f}x", "viabilidad": "Aceptable"}}
                  ],
                  "fortalezas": ["Punto fuerte 1 sobre el cultivo", "Punto fuerte 2 sobre finanzas"],
                  "alertas": ["Alerta 1 sobre margen o clima", "Alerta 2 sobre sensibilidad de mercado"],
                  "clausulas_sugeridas": [
                    "Exigencia de Seguro Paramétrico Climático ajustado a la cuenca regional",
                    "Contrato Off-take con precio piso firmado previo al desembolso",
                    "Monitoreo satelital quincenal obligatorio con alertas NDVI"
                  ],
                  "informe_auditoria_markdown": "INFORME COMPLETO EN FORMATO MARKDOWN SIGUIENDO LA ESTRUCTURA OBLIGATORIA (Resumen ejecutivo, Datos evaluados, Hallazgos de auditoria 3.1 a 3.5 con profundización en riesgos hiperpersonalizados y geopolíticos para este cultivo y región, Cuadro resumen de hallazgos, Aspectos positivos y Conclusión con exigencias contractuales)"
                }}
                """
                
                try:
                    resultado_json = consultar_agente_gemini_json(prompt_maestro)
                    
                    # RENDERIZADO VISUAL DEL RESULTADO DE AUDITORÍA
                    idx = resultado_json.get("indices_riesgo", {})
                    c_ai1, c_ai2 = st.columns(2)
                    
                    with c_ai1:
                        with st.container(border=True):
                            st.markdown("##### 🎯 Sub-Índices de Riesgo Auditados")
                            st.write(f"**Riesgo Climático Local:** {idx.get('riesgo_climatico_pct', 38)}%")
                            st.progress(idx.get('riesgo_climatico_pct', 38) / 100)
                            
                            st.write(f"**Riesgo de Mercado / Volatilidad:** {idx.get('riesgo_mercado_pct', 42)}%")
                            st.progress(idx.get('riesgo_mercado_pct', 42) / 100)
                            
                            st.write(f"**Resiliencia Operativa Parcela:** {idx.get('resiliencia_operativa_pct', 75)}%")
                            st.progress(idx.get('resiliencia_operativa_pct', 75) / 100)

                    with c_ai2:
                        with st.container(border=True):
                            st.markdown("##### 🧪 Matriz de Sensibilidad / Stress Test")
                            df_stress = pd.DataFrame(resultado_json.get("stress_test", []))
                            st.dataframe(df_stress, use_container_width=True, hide_index=True)

                    c_det1, c_det2 = st.columns(2)
                    with c_det1:
                        with st.container(border=True):
                            st.markdown("##### 🟢 Fortalezas Clave")
                            for f in resultado_json.get("fortalezas", []):
                                st.write(f"• {f}")
                            
                            st.markdown("##### ⚠️ Alertas Críticas de Auditoría")
                            for a in resultado_json.get("alertas", []):
                                st.write(f"• {a}")

                    with c_det2:
                        with st.container(border=True):
                            st.markdown("##### 📜 Checklist Contractual & Governance Covenants")
                            st.caption("Marque las cláusulas verified por el oficial de cumplimiento antes del desembolso:")
                            for clausula in resultado_json.get("clausulas_sugeridas", []):
                                st.checkbox(clausula, value=False)

                    # RENDERIZADO DEL INFORME COMPLETO EN MARKDOWN
                    st.markdown("---")
                    reporte_markdown = resultado_json.get("informe_auditoria_markdown", "# Error al generar el informe en Markdown")
                    
                    with st.container(border=True):
                        st.markdown(reporte_markdown)
                        
                        # Botón para descargar el Informe de Auditoría
                        st.download_button(
                            label="📥 Descargar Informe de Auditoría Completo (Markdown)",
                            data=reporte_markdown,
                            file_name=f"Informe_Auditoria_CoFundo_{farm_id}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )

                except Exception as err:
                    st.error(f"Error al procesar el dictamen de auditoría con la API de Gemini: {err}")
        else:
            st.warning("🔑 Configure su API Key de Gemini en el panel lateral para desplegar la Auditoría Automática.")

# ---------------------------------------------------------
# TAB 2: SIMULADOR PARCELA MANUAL
# ---------------------------------------------------------
with tab2:
    st.subheader("🧪 Simulador Paramétrico de Parcela Manual")
    with st.container(border=True):
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            s_crop = st.selectbox("Especie de Cultivo:", ["Corn", "Wheat", "Soybean", "Coffee", "Avocado", "Potato"])
            s_moisture = st.slider("Humedad del Suelo (%)", 0.0, 100.0, 48.0)
            s_ph = st.slider("pH del Suelo", 4.0, 9.0, 6.4)
        with mc2:
            s_ndvi = st.slider("Índice NDVI (Vigor)", 0.20, 0.95, 0.68)
            s_rain = st.slider("Precipitación (mm)", 100, 1800, 750)
            s_temp = st.slider("Temperatura (°C)", 10.0, 42.0, 24.0)
        with mc3:
            s_irrigation = st.selectbox("Sistema de Riego", ["Drip", "Sprinkler", "Manual", "None"])
            s_disease = st.selectbox("Estado Fitosanitario", ["None", "Mild", "Moderate", "Severe"])
            s_loan = st.number_input("Crédito Solicitado ($ USD):", 100, 30000, 2000, step=100)

    if st.button("🧪 Evaluar Simulación Parcela Manual", type="primary", use_container_width=True):
        precio_m = precio_base_map.get(s_crop, 0.50)
        yield_sim = 4500.0 * s_ndvi
        fin_sim = calcular_metricas_financieras(s_crop, s_loan, 24.0, yield_sim, precio_m, s_ndvi, s_disease)
        
        sim_data = {
            'NDVI_index': s_ndvi, 'soil_moisture_%': s_moisture, 'soil_pH': s_ph,
            'rainfall_mm': s_rain, 'temperature_C': s_temp, 'sunlight_hours': 8.0,
            'crop_disease_status': s_disease, 'irrigation_type': s_irrigation
        }
        
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            with st.container(border=True):
                st.plotly_chart(plot_gauge_score(fin_sim['score']), use_container_width=True)
                st.markdown(f'<div style="text-align:center;"><span class="{fin_sim["badge_style"]}">{fin_sim["decision"]}</span></div>', unsafe_allow_html=True)
        with c_m2:
            with st.container(border=True):
                st.metric("Rendimiento Estimado", f"{yield_sim:,.1f} kg/ha")
                st.metric("Deuda Total", f"${fin_sim['deuda_total']:,.2f} USD")
                st.metric("Retorno Neto Agricultor", f"${fin_sim['retorno_neto_usd']:,.2f} USD")
                st.metric("DSCR Cobertura", f"{fin_sim['dscr']:.2f}x")
        with c_m3:
            with st.container(border=True):
                st.plotly_chart(plot_radar_agronomo_8_ejes(sim_data), use_container_width=True)
