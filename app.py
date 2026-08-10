import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import google.generativeai as genai
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Configuración de Página y Estilos Custom
# ---------------------------------------------------------
st.set_page_config(
    page_title="CoFundo | Credit Cockpit & Decision Engine", 
    page_icon="🌾", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; color: #0F172A; font-family: 'Inter', system-ui, sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 32px; border-radius: 16px; color: white; margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
    }
    .main-title { color: #FFFFFF; font-size: 2.1rem; font-weight: 800; margin-bottom: 4px; }
    .sub-title { color: #94A3B8; font-size: 1.0rem; font-weight: 400; }
    .human-decision-box {
        background-color: #EFF6FF; border: 2px solid #3B82F6; border-radius: 14px; padding: 24px; margin-top: 24px;
    }
    .badge-approved { background-color: #DCFCE7; color: #166534; padding: 6px 18px; border-radius: 20px; font-weight: 700; border: 1px solid #BBF7D0; display: inline-block; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; padding: 6px 18px; border-radius: 20px; font-weight: 700; border: 1px solid #FDE68A; display: inline-block; }
    .badge-rejected { background-color: #FEE2E2; color: #991B1B; padding: 6px 18px; border-radius: 20px; font-weight: 700; border: 1px solid #FECACA; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="main-title">🌾 CoFundo: Credit Cockpit & Governance Engine</div>
    <div class="sub-title">Evaluación Crediticia Integral, Telemetría Satelital IoT y Auditoría de Gobernanza Financiera (Human-in-the-Loop)</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Benchmarks y Parámetros Estándar por Cultivo
# ---------------------------------------------------------
DEFAULT_BENCHMARK = {
    'periodo_captura': 'Últimos 45 días (Monitoreo Estándar - Fase Vegetativa)',
    'tasa_min': 10.0, 'tasa_max': 18.0, 'tasa_def': 13.5,
    'costo_estructuracion_pct': 2.0, 'costo_seguro_pct': 2.5, 'costo_reserva_pct': 1.5,
    'precio_base_kg': 0.40, 'opex_por_ha': 450.0,
    'lat': 19.4326, 'lon': -99.1332,
    'recomendaciones': [
        "Mantener esquema de fertilización balanceado de acuerdo al análisis de suelo.",
        "Realizar monitoreo preventivo de plagas y enfermedades quincenalmente.",
        "Asegurar canales de comercialización previo a la etapa de cosecha."
    ]
}

CROP_BENCHMARKS = {
    'Corn': {
        'periodo_captura': '15 Ene 2026 – 01 Mar 2026 (45 días - Fase Vegetativa V6 a VT)',
        'tasa_min': 10.0, 'tasa_max': 18.0, 'tasa_def': 13.5,
        'costo_estructuracion_pct': 2.0, 'costo_seguro_pct': 2.5, 'costo_reserva_pct': 1.5,
        'precio_base_kg': 0.35, 'opex_por_ha': 480.0, 'lat': 25.6866, 'lon': -100.3161,
        'recomendaciones': [
            "Mantener índice NDVI > 0.65 durante la etapa crítica de diferenciación floral (VT).",
            "Aplicar fertilización nitrogenada fraccionada en estado V6-V8 para maximizar rendimiento.",
            "Monitorear la humedad del suelo para evitar niveles inferiores al 35% en floración."
        ]
    },
    'Wheat': {
        'periodo_captura': '01 Ene 2026 – 15 Feb 2026 (45 días - Encañado y Espigado)',
        'tasa_min': 11.0, 'tasa_max': 19.0, 'tasa_def': 14.0,
        'costo_estructuracion_pct': 2.0, 'costo_seguro_pct': 2.5, 'costo_reserva_pct': 1.5,
        'precio_base_kg': 0.40, 'opex_por_ha': 420.0, 'lat': 24.8059, 'lon': -107.3944,
        'recomendaciones': [
            "Vigilar la presencia de roya amarilla ante humedades relativas superiores al 70%.",
            "Asegurar un drenaje adecuado en suelo para evitar la asfixia radicular.",
            "Coordinar precio piso o contrato de cobertura en Bolsa antes de la cosecha."
        ]
    },
    'Soybean': {
        'periodo_captura': '10 Ene 2026 – 25 Feb 2026 (45 días - Floración y Llenado R1-R3)',
        'tasa_min': 9.5, 'tasa_max': 17.0, 'tasa_def': 12.5,
        'costo_estructuracion_pct': 1.8, 'costo_seguro_pct': 2.3, 'costo_reserva_pct': 1.4,
        'precio_base_kg': 0.55, 'opex_por_ha': 410.0, 'lat': -12.0463, 'lon': -77.0428,
        'recomendaciones': [
            "Inoculación con Bradyrhizobium para optimizar la fijación biológica de nitrógeno.",
            "Control preventivo de orugas desfoliadoras si el daño foliar supera el 15%.",
            "Mantener monitoreo térmico para prevenir aborto de flores por temperaturas > 35°C."
        ]
    },
    'Coffee': {
        'periodo_captura': '01 Dic 2025 – 15 Ene 2026 (45 días - Llenado del Grano)',
        'tasa_min': 12.0, 'tasa_max': 22.0, 'tasa_def': 16.0,
        'costo_estructuracion_pct': 2.5, 'costo_seguro_pct': 3.0, 'costo_reserva_pct': 2.0,
        'precio_base_kg': 3.20, 'opex_por_ha': 850.0, 'lat': 4.5709, 'lon': -74.2973,
        'recomendaciones': [
            "Muestreo de broca del café en frutos con nivel de infestación objetivo < 2%.",
            "Manejo de sombra regulada para preservar humedad en microclima.",
            "Planificar fertilización potásica en fase de maduración del grano."
        ]
    },
    'Potato': {
        'periodo_captura': '15 Ene 2026 – 01 Mar 2026 (45 días - Tuberización)',
        'tasa_min': 12.5, 'tasa_max': 22.0, 'tasa_def': 16.5,
        'costo_estructuracion_pct': 2.5, 'costo_seguro_pct': 3.0, 'costo_reserva_pct': 2.0,
        'precio_base_kg': 0.30, 'opex_por_ha': 650.0, 'lat': 5.5353, 'lon': -73.3678,
        'recomendaciones': [
            "Control estricto de Phytophthora infestans (gota) tras días lluviosos.",
            "Aporcado oportuno para proteger tubérculos de plagas de luz y calor.",
            "Análisis de fósforo disponible en suelo para asegurar tuberización homogénea."
        ]
    }
}

def get_crop_benchmark(crop_type):
    bench = DEFAULT_BENCHMARK.copy()
    if crop_type in CROP_BENCHMARKS:
        bench.update(CROP_BENCHMARKS[crop_type])
    return bench

# ---------------------------------------------------------
# 3. Carga de Datos y Configuración de API
# ---------------------------------------------------------
@st.cache_resource
def cargar_recursos():
    try:
        with open('modelo_agrotech.pkl', 'rb') as f:
            datos_modelo = pickle.load(f)
        modelo_rf, columnas_ml = datos_modelo['modelo'], datos_modelo['columnas']
    except Exception:
        modelo_rf, columnas_ml = None, None

    try:
        df_50 = pd.read_csv('50_ejemplares_kaggle.csv')
    except Exception:
        data_sintetica = {
            'farm_id': ['FARM0214', 'FARM0102', 'FARM0305', 'FARM0412', 'FARM0515'],
            'crop_type': ['Wheat', 'Corn', 'Soybean', 'Coffee', 'Potato'],
            'region': ['Sinaloa', 'Jalisco', 'Córdoba', 'Eje Cafetero', 'Boyacá'],
            'NDVI_index': [0.38, 0.75, 0.82, 0.61, 0.55],
            'soil_moisture_%': [28.0, 48.0, 52.0, 58.0, 38.0],
            'soil_pH': [6.5, 7.2, 6.2, 5.8, 5.5],
            'rainfall_mm': [400, 720, 850, 1100, 500],
            'temperature_C': [28.0, 25.0, 26.0, 21.0, 17.0],
            'sunlight_hours': [9.0, 8.5, 8.0, 6.5, 6.0],
            'irrigation_type': ['Sprinkler', 'Drip', 'Drip', 'Manual', 'Sprinkler'],
            'crop_disease_status': ['Severe', 'None', 'None', 'Moderate', 'Moderate']
        }
        df_50 = pd.DataFrame(data_sintetica)
    return modelo_rf, columnas_ml, df_50

modelo_rf, columnas_ml, df_50 = cargar_recursos()

# Configuración API Gemini
api_key = st.secrets.get("GEMINI_API_KEY", None)
if api_key:
    try: 
        genai.configure(api_key=api_key)
    except Exception: 
        pass

# ---------------------------------------------------------
# 4. Motor de Cálculo Financiero Unificado
# ---------------------------------------------------------
def calcular_financiamiento_detallado(crop_type, capital_solicitado, tasa_interes_annual, yield_pred, ndvi, disease_status):
    bench = get_crop_benchmark(crop_type)
    
    # Costos Operativos Fijos
    c_est = capital_solicitado * (bench['costo_estructuracion_pct'] / 100)
    c_seg = capital_solicitado * (bench['costo_seguro_pct'] / 100)
    c_res = capital_solicitado * (bench['costo_reserva_pct'] / 100)
    total_costos_operativos = c_est + c_seg + c_res
    
    # Interés Nominal (Plazo Estándar: 6 meses / 0.5 años)
    plazo_meses = 6
    interes_monto = capital_solicitado * (tasa_interes_annual / 100) * (plazo_meses / 12)
    
    # Suma Total a Devolver
    total_a_devolver = capital_solicitado + total_costos_operativos + interes_monto
    
    # Ingreso Bruto y OPEX
    precio_kg = bench['precio_base_kg']
    ingreso_bruto = yield_pred * precio_kg
    opex_produccion = bench['opex_por_ha']
    
    # Retorno Neto del Agricultor
    retorno_neto_usd = ingreso_bruto - opex_produccion - total_a_devolver
    retorno_neto_pct = (retorno_neto_usd / ingreso_bruto * 100) if ingreso_bruto > 0 else 0.0
    
    # Cobertura del Servicio de Deuda (DSCR) y Credit Score
    dscr = ingreso_bruto / max(total_a_devolver, 1.0)
    score_base = 590 + (dscr * 75) + (ndvi * 110)
    if disease_status in ['Severe', 'Moderate']: score_base -= 70
    if retorno_neto_usd < 0: score_base -= 80
    score = int(np.clip(score_base, 300, 850))
    
    if score >= 710 and dscr >= 1.25 and retorno_neto_usd > 0:
        sugerencia = "APROBACIÓN SUGERIDA"
        badge_style = "badge-approved"
    elif score >= 600 and dscr >= 1.02:
        sugerencia = "REVISIÓN REQUERIDA (CONDICIONAL)"
        badge_style = "badge-warning"
    else:
        sugerencia = "ALTO RIESGO (REESTRUCTURAR)"
        badge_style = "badge-rejected"
        
    return {
        'capital': capital_solicitado,
        'c_est': c_est, 'c_seg': c_seg, 'c_res': c_res,
        'total_costos_operativos': total_costos_operativos,
        'interes_monto': interes_monto,
        'total_a_devolver': total_a_devolver,
        'ingreso_bruto': ingreso_bruto,
        'opex_produccion': opex_produccion,
        'retorno_neto_usd': retorno_neto_usd,
        'retorno_neto_pct': retorno_neto_pct,
        'dscr': dscr,
        'score': score,
        'sugerencia': sugerencia,
        'badge_style': badge_style,
        'dscr_p20': (ingreso_bruto * 0.80) / max(total_a_devolver, 1.0),
        'dscr_seq': (ingreso_bruto * 0.70) / max(total_a_devolver, 1.0),
        'dscr_ins': ingreso_bruto / max(total_a_devolver + (opex_produccion * 0.15), 1.0)
    }

# ---------------------------------------------------------
# 5. Visualizaciones (Plotly Gauge, Pie & Radar)
# ---------------------------------------------------------
def fig_tacometro(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={'font': {'size': 38, 'color': '#0F172A'}},
        gauge={
            'axis': {'range': [300, 850]},
            'bar': {'color': "#1E293B", 'thickness': 0.25},
            'steps': [
                {'range': [300, 599], 'color': "rgba(239, 68, 68, 0.2)"},
                {'range': [600, 709], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [710, 850], 'color': "rgba(16, 185, 129, 0.2)"}
            ]
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(t=20, b=10, l=20, r=20))
    return fig

def fig_anillo_financiero(fin):
    labels = ['Capital Solicitado', 'Costos Operativos', 'Intereses Generados', 'Margen Neto Agricultor']
    values = [fin['capital'], fin['total_costos_operativos'], fin['interes_monto'], max(0, fin['retorno_neto_usd'])]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.5,
        marker_colors=['#3B82F6', '#F59E0B', '#EF4444', '#10B981']
    )])
    fig.update_layout(
        title_text="Distribución del Flujo de Caja (USD)", 
        title_font=dict(size=13),
        height=200, margin=dict(t=35, b=10, l=10, r=10), showlegend=False
    )
    return fig

def fig_radar_riesgo(ndvi, dscr_val, disease_status):
    r_agro = 30 if ndvi < 0.5 else 85
    r_liquidez = min(100, int(dscr_val * 60))
    r_fitosanitario = 20 if disease_status == 'Severe' else (60 if disease_status == 'Moderate' else 90)
    r_mercado = 70 
    r_cambiario = 75 
    
    categories = ['Salud Agro (NDVI)', 'Liquidez (DSCR)', 'Fitosaguridad', 'Estabilidad Mercado', 'Cobertura FX']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[r_agro, r_liquidez, r_fitosanitario, r_mercado, r_cambiario],
        theta=categories, fill='toself', name='Perfil de Riesgo',
        fillcolor='rgba(59, 130, 246, 0.25)', line_color='#2563EB'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=200, margin=dict(t=20, b=20, l=30, r=30)
    )
    return fig

# ---------------------------------------------------------
# 6. Generación de Informes & Fallback para Gemini
# ---------------------------------------------------------
def generar_fallback_local_report(fin_data, bench, farm_id, farm_data):
    """Genera un informe ejecutivo robusto analizando el impacto de riesgos en la decisión final."""
    
    # Análisis de sensibilidad de la decisión según DSCR en estrés
    impacto_decision = ""
    if fin_data['dscr_seq'] < 1.0 or fin_data['dscr_p20'] < 1.0:
        impacto_decision = "⚠️ **RIESGO ELEVADO DE INCUMPLIMIENTO:** Ante escenarios de sequía o caída de precio, la cobertura de deuda cae por debajo de 1.0x (insolvencia). Se **recomienda condicionalidad obligatoria**: aplicar la Estrategia A (reducir capital) o exigir seguro paramétrico al 100% antes de desembolsar."
    else:
        impacto_decision = "✅ **RESILIENCIA FINANCIERA ADECUADA:** El crédito soporta fluctuaciones moderadas. Se recomienda **Aprobación Estándar** manteniendo el monitoreo satelital continuo."

    markdown_content = f"""# 📋 Informe de Auditoría de Riesgo, Escenarios y Gobernanza
**CoFundo Credit Cockpit** | Parcela: **{farm_id}** | Región: **{farm_data.get('region', 'N/A')}** | Cultivo: **{farm_data['crop_type']}**

---

### 1. Resumen Ejecutivo
* **Sugerencia del Motor IA:** **{fin_data['sugerencia']}** (Credit Score: **{fin_data['score']}/850**)
* **Estructura Financiera:** Capital de **${fin_data['capital']:,.2f} USD** + Intereses (**${fin_data['interes_monto']:,.2f} USD**) + Costos Operativos (**${fin_data['total_costos_operativos']:,.2f} USD**).
* **Monto Total Exigible a Devolver:** **${fin_data['total_a_devolver']:,.2f} USD**.
* **Retorno Neto Proyectado del Agricultor:** **${fin_data['retorno_neto_usd']:,.2f} USD** (Margen del {fin_data['retorno_neto_pct']:.1f}%).

---

### 2. Cuadro Integrado de Devolución
| Concepto | Monto (USD) | % del Capital | Impacto Financiero |
| :--- | :--- | :--- | :--- |
| **Capital Solicitado** | ${fin_data['capital']:,.2f} | 100.0% | Principal prestado |
| **Intereses Generados** | ${fin_data['interes_monto']:,.2f} | {(fin_data['interes_monto']/fin_data['capital']*100):.1f}% | Costo de financiamiento (6 meses) |
| **Costos Operativos (Estructuración/Seguro/Reserva)** | ${fin_data['total_costos_operativos']:,.2f} | {(fin_data['total_costos_operativos']/fin_data['capital']*100):.1f}% | Gastos fijos institucionales |
| **TOTAL A DEVOLVER** | **${fin_data['total_a_devolver']:,.2f}** | **{(fin_data['total_a_devolver']/fin_data['capital']*100):.1f}%** | **Deuda Total Contratada** |

---

### 3. Evaluación de Escenarios de Riesgo y Pruebas de Estrés (Stress Testing)
Si las condiciones de mercado o clima empeoran, la capacidad de pago cambia significativamente:

1. **Escenario Caída de Precio (-20% en Mercado):**
   * El DSCR baja de **{fin_data['dscr']:.2f}x** a **{fin_data['dscr_p20']:.2f}x**.
   * *Impacto:* Reducción directa en el flujo de caja operativo del agricultor.
2. **Escenario Sequía Severa (-30% en Rendimiento de Cosecha):**
   * El DSCR cae a **{fin_data['dscr_seq']:.2f}x**.
   * *Impacto:* Riesgo alto de pérdida parcial de la cosecha si el NDVI cae de **{farm_data['NDVI_index']}**.
3. **Escenario Incremento de Insumos / Fertilizantes (+15% OPEX):**
   * El DSCR se ajusta a **{fin_data['dscr_ins']:.2f}x**.

---

### 4. Recomendación Formal para la Decisión Final (Human-in-the-Loop)
{impacto_decision}
"""
    return {"informe_auditoria_markdown": markdown_content}

def consultar_agente_gemini(prompt_texto, fin_data, bench, farm_id, farm_data):
    if not api_key:
        return generar_fallback_local_report(fin_data, bench, farm_id, farm_data)
        
    modelos = ['gemini-1.5-flash', 'gemini-1.5-pro']
    for mod in modelos:
        try:
            model = genai.GenerativeModel(mod, generation_config={"response_mime_type": "application/json"})
            res = model.generate_content(prompt_texto)
            return json.loads(res.text)
        except Exception:
            continue
            
    return generar_fallback_local_report(fin_data, bench, farm_id, farm_data)

# ---------------------------------------------------------
# 7. Sidebar: Chatbot Copiloto para el Analista
# ---------------------------------------------------------
with st.sidebar:
    st.title("🤖 Chatbot Copiloto")
    st.caption("Asistente en tiempo real para estructuración y análisis de crédito agrícola.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy tu Copiloto CoFundo. ¿En qué te ayudo a evaluar o reestructurar este expediente crediticio?"}
        ]
        
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if user_input := st.chat_input("Escribe tu consulta..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        respuesta = f"Entendido. Respecto a tu consulta sobre '{user_input}': Recuerda que si el DSCR es inferior a 1.25x o la condición fitosanitaria es desfavorable, es altamente recomendable solicitar la Estrategia A (reducción de capital) o exigir la cobertura de seguro paramétrico al 100%."
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        st.chat_message("assistant").write(respuesta)

# ---------------------------------------------------------
# 8. Panel Principal: Selección e Insumos
# ---------------------------------------------------------
col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    farm_id = st.selectbox("Seleccionar Parcela / Cliente:", df_50['farm_id'].tolist())
    farm_data = df_50[df_50['farm_id'] == farm_id].iloc[0]
    bench = get_crop_benchmark(farm_data['crop_type'])
with col2:
    capital_req = st.number_input("Capital Solicitado ($ USD):", 300, 30000, 1100, step=100)
with col3:
    st.caption(f"Rango Tasa Sugerida ({farm_data['crop_type']}): **{bench['tasa_min']}% – {bench['tasa_max']}%**")
    tasa_interes = st.slider("Tasa de Interés Nominal Anual (%):", float(bench['tasa_min']), float(bench['tasa_max']), float(bench['tasa_def']), step=0.5)

# Diagnóstico de Parcela (Datos Ampliados y Telemetría IoT)
with st.container(border=True):
    st.subheader(f"📍 Parcela {farm_id} ({farm_data.get('region', 'N/A')}) — Telemetría Agronómica Integrada")
    
    # Fila 1: Datos de Cultivo, Monitoreo y Vigor
    c1, c2, c3, c4 = st.columns([1, 1.8, 1, 1])
    c1.metric("Cultivo", str(farm_data['crop_type']))
    c2.metric("🗓️ Periodo de Monitoreo", bench['periodo_captura'])
    c3.metric("Vigor (NDVI)", f"{farm_data['NDVI_index']:.2f}")
    c4.metric("Estado Fitosanitario", str(farm_data['crop_disease_status']))

    # Fila 2: Condiciones Meteorológicas y de Suelo
    c5, c6, c7, c8, c9 = st.columns(5)
    c5.metric("🌡️ Temperatura", f"{farm_data.get('temperature_C', 'N/A')} °C")
    c6.metric("💧 Humedad Suelo", f"{farm_data.get('soil_moisture_%', 'N/A')}%")
    c7.metric("🌧️ Lluvia Acumulada", f"{farm_data.get('rainfall_mm', 'N/A')} mm")
    c8.metric("🧪 pH del Suelo", f"{farm_data.get('soil_pH', 'N/A')}")
    c9.metric("🚿 Sistema Riego", str(farm_data.get('irrigation_type', 'N/A')))

    st.markdown("**🌱 Recomendaciones Agronómicas Específicas:**")
    for reco in bench['recomendaciones']:
        st.write(f"• {reco}")

# Estimación de Rendimiento
if modelo_rf is not None and columnas_ml is not None:
    input_data = pd.DataFrame([farm_data])
    input_ml = pd.get_dummies(input_data, columns=['irrigation_type', 'crop_disease_status'], drop_first=False).reindex(columns=columnas_ml, fill_value=0)
    yield_pred = float(modelo_rf.predict(input_ml)[0])
else:
    yield_pred = 4200.0 * farm_data['NDVI_index']

fin = calcular_financiamiento_detallado(
    farm_data['crop_type'], capital_req, tasa_interes, yield_pred, farm_data['NDVI_index'], farm_data['crop_disease_status']
)

st.markdown("---")

# ---------------------------------------------------------
# Estructura Financiera Completa (Capital, Intereses y Costos)
# ---------------------------------------------------------
st.markdown("### 💳 Estructura de Crédito, Intereses y Costos Operativos")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Capital Solicitado", f"${fin['capital']:,.2f} USD")
m2.metric("Intereses Generados", f"${fin['interes_monto']:,.2f} USD", help=f"Tasa anual del {tasa_interes}% proyectada a 6 meses")
m3.metric("Costos Operativos Fijos", f"${fin['total_costos_operativos']:,.2f} USD", help="Suma de Estructuración, Seguro y Reserva")
m4.metric("TOTAL A DEVOLVER", f"${fin['total_a_devolver']:,.2f} USD", delta="Obligación Exigible", delta_color="inverse")
m5.metric("Retorno Neto Agricultor", f"${fin['retorno_neto_usd']:,.2f} USD", f"{fin['retorno_neto_pct']:.1f}% Margen")

# Desglose transparente en tabla rápida
st.markdown("""
| Concepto | Porcentaje / Tasa | Monto USD |
| :--- | :--- | :--- |
| **Tasa de Interés Nominal (6 meses)** | **{tasa_pct:.1f}% Anual** | **${interes:,.2f} USD** |
| Comisión de Estructuración | {c_est_pct:.1f}% | ${c_est:,.2f} USD |
| Seguro Agrícola Paramétrico | {c_seg_pct:.1f}% | ${c_seg:,.2f} USD |
| Fondo Reserva de Contingencia | {c_res_pct:.1f}% | ${c_res:,.2f} USD |
""".format(
    tasa_pct=tasa_interes,
    interes=fin['interes_monto'],
    c_est_pct=bench['costo_estructuracion_pct'], c_est=fin['c_est'],
    c_seg_pct=bench['costo_seguro_pct'], c_seg=fin['c_seg'],
    c_res_pct=bench['costo_reserva_pct'], c_res=fin['c_res']
))

with st.expander("🔍 Ver desglose de Costos Fijos institucionales"):
    c_f1, c_f2, c_f3 = st.columns(3)
    c_f1.write(f"• **Comisión Estructuración ({bench['costo_estructuracion_pct']}%):** ${fin['c_est']:,.2f} USD")
    c_f2.write(f"• **Seguro Agrícola Paramétrico ({bench['costo_seguro_pct']}%):** ${fin['c_seg']:,.2f} USD")
    c_f3.write(f"• **Fondo Reserva Contingencia ({bench['costo_reserva_pct']}%):** ${fin['c_res']:,.2f} USD")

st.markdown("---")

# ---------------------------------------------------------
# 9. Suite Visual (Dashboard Integrado)
# ---------------------------------------------------------
st.markdown("### 📈 Diagnóstico Visual y Cobertura de Riesgos")

g1, g2, g3 = st.columns(3)
with g1:
    st.plotly_chart(fig_tacometro(fin['score']), use_container_width=True)
    st.markdown(f"<div style='text-align:center;'>Sugerencia Motor IA:<br><span class='{fin['badge_style']}'>{fin['sugerencia']}</span></div>", unsafe_allow_html=True)
with g2:
    st.plotly_chart(fig_anillo_financiero(fin), use_container_width=True)
with g3:
    st.plotly_chart(fig_radar_riesgo(farm_data['NDVI_index'], fin['dscr'], farm_data['crop_disease_status']), use_container_width=True)

# ---------------------------------------------------------
# 10. Motor de Estrategias "What-If" (Reestructuración)
# ---------------------------------------------------------
if fin['sugerencia'] != "APROBACIÓN SUGERIDA":
    st.markdown("---")
    st.warning("⚠️ **Alerta de Viabilidad Crediticia:** El crédito presenta fragilidad financiera o fitosanitaria. El motor propone las siguientes alternativas de reestructuración:")
    
    st1, st2, st3 = st.columns(3)
    with st1:
        st.info("**Estrategia A: Reducción de Capital**")
        cap_a = capital_req * 0.8
        st.write(f"• Ajustar monto a: **${cap_a:,.0f} USD**")
        st.write(f"• Eleva el DSCR a: **{(fin['ingreso_bruto']/max(1, cap_a*1.13)):.2f}x**")
    with st2:
        st.info("**Estrategia B: Seguro Paramétrico Reforzado**")
        st.write("• Exigir cobertura contra sequía/roya al 100%.")
        st.write("• Tasa ajustada preferencial: **-1.0%**")
    with st3:
        st.info("**Estrategia C: Contrato de Cosecha Futura**")
        st.write("• Vincular desembolso a contrato de venta firmado con comprador verificado.")
        st.write("• Reduce el riesgo de volatilidad de mercado.")

# ---------------------------------------------------------
# 11. Auditoría Integrada y Dictamen Humano (Gobernanza)
# ---------------------------------------------------------
st.markdown("---")
if st.button("⚡ Generar Informe Integrado de Auditoría e Institucionalidad", type="primary", use_container_width=True):
    prompt_informe = f"""Genera un informe institucional en formato JSON con la clave 'informe_auditoria_markdown' para la parcela {farm_id} con cultivo {farm_data['crop_type']}.
    Datos: Capital={fin['capital']}, Devolución Total={fin['total_a_devolver']}, Score={fin['score']}, DSCR={fin['dscr']:.2f}."""
    
    with st.spinner("Procesando auditoría y análisis de estrés..."):
        res_json = consultar_agente_gemini(prompt_informe, fin, bench, farm_id, farm_data)
        st.markdown(res_json.get("informe_auditoria_markdown", "No se pudo generar el informe."))

# Panel Ético de Gobernanza (Human-in-the-Loop)
st.markdown('<div class="human-decision-box">', unsafe_allow_html=True)
st.subheader("👨‍💼 Panel de Decisiones de Gobernanza (Human-in-the-Loop)")
st.write("La recomendación algorítmica es un insumo técnico. La decisión contractual final es responsabilidad exclusiva del analista de crédito responsable.")

col_dec1, col_dec2 = st.columns([1, 2])

with col_dec1:
    decision_humana = st.radio(
        "Dictamen Final del Analista:",
        ["Aprobar Crédito Original", "Aprobar con Reestructuración (Estrategia A/B/C)", "Rechazar Solicitud"],
        index=1 if fin['sugerencia'] != "APROBACIÓN SUGERIDA" else 0
    )

with col_dec2:
    justificacion = st.text_area(
        "Justificación del Dictamen / Observaciones Contractuales:", 
        placeholder="Escribe aquí los motivos de tu decisión (ej. 'Se aprueba condicionado a aplicar tratamiento fitosanitario y aceptar la Estrategia A de reducción de capital')..."
    )
    
    if st.button("💾 Guardar Dictamen y Registrar en Gobernanza", type="primary"):
        if justificacion.strip() == "":
            st.error("⚠️ Por favor ingresa una justificación antes de registrar el dictamen.")
        else:
            st.success(f"✅ Dictamen registrado exitosamente: **{decision_humana}**. Expediente de auditoría actualizado.")
st.markdown('</div>', unsafe_allow_html=True)
