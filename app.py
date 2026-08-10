import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import google.generativeai as genai
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Configuración de la Página y Estilos
# ---------------------------------------------------------
st.set_page_config(
    page_title="CoFundo | Agri-Fintech Risk & Governance Cockpit", 
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
    .badge-approved { background-color: #DCFCE7; color: #166534; padding: 6px 18px; border-radius: 20px; font-weight: 700; border: 1px solid #BBF7D0; display: inline-block; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; padding: 6px 18px; border-radius: 20px; font-weight: 700; border: 1px solid #FDE68A; display: inline-block; }
    .badge-rejected { background-color: #FEE2E2; color: #991B1B; padding: 6px 18px; border-radius: 20px; font-weight: 700; border: 1px solid #FECACA; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="main-title">🌾 CoFundo: Credit Cockpit & Governance Engine</div>
    <div class="sub-title">Evaluación Crediticia Integral, Telemetría Satelital IoT y Auditoría de Gobernanza Financiera</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. CASO ESTÁNDAR DEFAULT Y DICCIONARIO BENCHMARKS
# ---------------------------------------------------------
DEFAULT_BENCHMARK = {
    'periodo_captura': 'Últimos 45 días (Monitoreo Estándar - Fase Vegetativa)',
    'tasa_min': 10.0, 'tasa_max': 18.0, 'tasa_def': 13.5,
    'costo_estructuracion_pct': 2.0,
    'costo_seguro_pct': 2.5,
    'costo_reserva_pct': 1.5,
    'precio_base_kg': 0.40,
    'opex_por_ha': 450.0,  # OPEX realista
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
        'precio_base_kg': 0.35, 'opex_por_ha': 480.0,
        'recomendaciones': [
            "Mantener índice NDVI > 0.65 durante la etapa crítica de diferenciación floral (VT).",
            "Aplicar fertilización nitrogenada fraccionada en estado V6-V8 para maximizar rendimiento.",
            "Monitorear la humedad del suelo para evitar humedad por debajo del 35% en floración."
        ]
    },
    'Wheat': {
        'periodo_captura': '01 Ene 2026 – 15 Feb 2026 (45 días - Encañado y Espigado)',
        'tasa_min': 11.0, 'tasa_max': 19.0, 'tasa_def': 14.0,
        'costo_estructuracion_pct': 2.0, 'costo_seguro_pct': 2.5, 'costo_reserva_pct': 1.5,
        'precio_base_kg': 0.40, 'opex_por_ha': 420.0,
        'recomendaciones': [
            "Vigilar la presencia de roya amarilla ante humedades relativas superiores al 70%.",
            "Asegurar un drenaje adecuado en suelo para evitar la asfixia radicular.",
            "Coordinar precio piso o contrato de cobertura (futures) en Bolsa antes de cosecha."
        ]
    },
    'Soybean': {
        'periodo_captura': '10 Ene 2026 – 25 Feb 2026 (45 días - Floración y Llenado R1-R3)',
        'tasa_min': 9.5, 'tasa_max': 17.0, 'tasa_def': 12.5,
        'costo_estructuracion_pct': 1.8, 'costo_seguro_pct': 2.3, 'costo_reserva_pct': 1.4,
        'precio_base_kg': 0.55, 'opex_por_ha': 410.0,
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
        'precio_base_kg': 3.20, 'opex_por_ha': 850.0,
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
        'precio_base_kg': 0.30, 'opex_por_ha': 650.0,
        'recomendaciones': [
            "Control estricto de Phytophthora infestans (gota) tras días lluviosos.",
            "Aporcado oportuno para proteger tubérculos de plagas de luz y calor.",
            "Análisis de fósforo disponible en suelo para asegurar tuberización homogénea."
        ]
    }
}

def get_crop_benchmark(crop_type):
    """Garantiza un resultado válido devolviendo un default estándar si falta alguna clave."""
    bench = DEFAULT_BENCHMARK.copy()
    if crop_type in CROP_BENCHMARKS:
        bench.update(CROP_BENCHMARKS[crop_type])
    return bench

# ---------------------------------------------------------
# 3. Carga de Datos y Configuración Gemini API Robusta
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

# Configuración API Key
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"].strip()
else:
    st.sidebar.markdown("### 🔑 API Key")
    user_key = st.sidebar.text_input("Gemini Key:", type="password")
    if user_key: api_key = user_key.strip()

if api_key: 
    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass

def generar_fallback_local_report(fin_data, bench, farm_id, farm_data):
    """Genera una respuesta estructurada en JSON local si la API de Gemini falla o excede cuota."""
    markdown_content = f"""# Informe Integrado de Auditoría de Riesgo y Gobernanza
**CoFundo Credit Cockpit** | Caso: {farm_id} | Cultivo: {farm_data['crop_type']}

---

### 1. Resumen Ejecutivo
Dictamen institucional evaluado mediante el motor algorítmico de CoFundo. La solicitud presenta una decisión de **{fin_data['decision']}** con un score de **{fin_data['score']}/850**. La estructura crediticia requiere un desembolso total a devolver de **${fin_data['total_a_devolver']:,.2f} USD**, dejando un retorno neto para el agricultor de **${fin_data['retorno_neto_usd']:,.2f} USD** ({fin_data['retorno_neto_pct']:.1f}%).

---

### 2. Cuadro de Conformación Financiera de Devolución
| Concepto Financiero | Valor (USD) | % sobre Capital | Observación de Auditoría |
| :--- | :--- | :--- | :--- |
| **Capital Solicitado** | ${fin_data['capital']:,.2f} | 100.0% | Base desembolsada |
| **Costos Operativos (Estructuración/Seguro/Reserva)** | ${fin_data['total_costos_operativos']:,.2f} | {(fin_data['total_costos_operativos']/fin_data['capital']*100):.1f}% | Gastos fijos institucionales |
| **Interés Nominal Generado** | ${fin_data['interes_monto']:,.2f} | {(fin_data['interes_monto']/fin_data['capital']*100):.1f}% | Intereses nominales acumulados |
| **TOTAL A DEVOLVER** | **${fin_data['total_a_devolver']:,.2f}** | **{(fin_data['total_a_devolver']/fin_data['capital']*100):.1f}%** | **Deuda total exigible** |
| **Retorno Neto Agricultor** | **${fin_data['retorno_neto_usd']:,.2f}** | **{fin_data['retorno_neto_pct']:.1f}% Margen** | **Utilidad neta post-OPEX** |

---

### 3. Riesgos de Cola e Hiperpersonalización Agronómica / Geopolítica
- **Inflación e Insumos N-P-K / Urea:** Presión potencial en costos de fertilizantes nitrogenados que podrían incrementar el OPEX previsto.
- **Fenómenos Climáticos de Cola (El Niño / La Niña):** Monitoreo adicional en la cuenca hidrológica durante la ventana telemétrica de *{bench['periodo_captura']}*.
- **Cuellos de Botella Logísticos y Riesgo Cambiario (FX):** Exposición moderada a fletes internacionales y volatilidad cambiaria regional.

---

### 4. Pruebas de Estrés (Stress Testing)
- **Caída 20% en Precio de Cosecha:** DSCR resultante **{fin_data['dscr_p20']:.2f}x**
- **Sequía Severa (-30% Rendimiento):** DSCR resultante **{fin_data['dscr_seq']:.2f}x**
- **Alza 15% Insumos / Fertilizantes:** DSCR resultante **{fin_data['dscr_ins']:.2f}x**

---

### 5. Dictamen Final y Cláusulas de Gobernanza
El crédito ha sido categorizado como **{fin_data['decision']}**. Se exige cumplimiento de las garantías de seguro paramétrico y monitoreo constante en el panel telemétrico.
"""
    return {"informe_auditoria_markdown": markdown_content}

def consultar_agente_gemini_json(prompt_texto, fin_data, bench, farm_id, farm_data):
    """Consulta Gemini en forma segura y si falla entrega una respuesta fallback sin colapsar Streamlit."""
    if not api_key:
        return generar_fallback_local_report(fin_data, bench, farm_id, farm_data)
        
    modelos = ['gemini-1.5-flash', 'gemini-1.5-pro', 'models/gemini-1.5-flash', 'models/gemini-1.5-pro']
    for mod in modelos:
        try:
            model = genai.GenerativeModel(mod, generation_config={"response_mime_type": "application/json"})
            res = model.generate_content(prompt_texto)
            return json.loads(res.text)
        except Exception:
            continue
            
    # Fallback si fallan las llamadas a la API
    return generar_fallback_local_report(fin_data, bench, farm_id, farm_data)

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
    
    # Interés Nominal Generado (Plazo Estándar: 6 meses)
    plazo_meses = 6
    interes_monto = capital_solicitado * (tasa_interes_annual / 100) * (plazo_meses / 12)
    
    # SUMA TOTAL A DEVOLVER
    total_a_devolver = capital_solicitado + total_costos_operativos + interes_monto
    
    # Ingreso Bruto y OPEX
    precio_kg = bench['precio_base_kg']
    ingreso_bruto = yield_pred * precio_kg
    opex_produccion = bench['opex_por_ha']
    
    # Retorno Neto del Agricultor
    retorno_neto_usd = ingreso_bruto - opex_produccion - total_a_devolver
    retorno_neto_pct = (retorno_neto_usd / ingreso_bruto * 100) if ingreso_bruto > 0 else 0.0
    
    # DSCR y Score
    dscr = ingreso_bruto / max(total_a_devolver, 1.0)
    score_base = 590 + (dscr * 75) + (ndvi * 110)
    if disease_status in ['Severe', 'Moderate']: score_base -= 70
    if retorno_neto_usd < 0: score_base -= 80
    score = int(np.clip(score_base, 300, 850))
    
    if score >= 710 and dscr >= 1.25 and retorno_neto_usd > 0:
        decision = "APROBADO DEFINITIVO"
        badge_style = "badge-approved"
    elif score >= 600 and dscr >= 1.02:
        decision = "APROBACIÓN CONDICIONAL"
        badge_style = "badge-warning"
    else:
        decision = "RECHAZADO / INVIABLE"
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
        'decision': decision,
        'badge_style': badge_style,
        'dscr_p20': (ingreso_bruto * 0.80) / max(total_a_devolver, 1.0),
        'dscr_seq': (ingreso_bruto * 0.70) / max(total_a_devolver, 1.0),
        'dscr_ins': ingreso_bruto / max(total_a_devolver + (opex_produccion * 0.15), 1.0)
    }

def plot_gauge_score(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={'font': {'size': 40, 'color': '#0F172A'}},
        gauge={
            'axis': {'range': [300, 850]},
            'bar': {'color': "#2563EB", 'thickness': 0.25},
            'steps': [
                {'range': [300, 599], 'color': "rgba(239, 68, 68, 0.15)"},
                {'range': [600, 709], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [710, 850], 'color': "rgba(16, 185, 129, 0.15)"}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=190, margin=dict(t=10, b=10, l=15, r=15))
    return fig

# ---------------------------------------------------------
# 5. UI Principal Streamlit
# ---------------------------------------------------------
col_sel1, col_sel2, col_sel3 = st.columns([1.5, 1, 1])
with col_sel1:
    farm_id = st.selectbox("Seleccionar Parcela / Cliente:", df_50['farm_id'].tolist())
    farm_data = df_50[df_50['farm_id'] == farm_id].iloc[0]
    crop_info = get_crop_benchmark(farm_data['crop_type'])
with col_sel2:
    capital_req = st.number_input("Capital Solicitado ($ USD):", 300, 30000, 1500, step=100)
with col_sel3:
    st.caption(f"Rango Tasa Sugerida ({farm_data['crop_type']}): **{crop_info['tasa_min']}% – {crop_info['tasa_max']}%**")
    tasa_interes = st.slider("Tasa de Interés Nominal Annual (%):", float(crop_info['tasa_min']), float(crop_info['tasa_max']), float(crop_info['tasa_def']), step=0.5)

# Diagnóstico de Parcela
with st.container(border=True):
    st.subheader(f"📍 Parcela {farm_id} — Telemetría y Diagnóstico Agronómico")
    t1, t2, t3, t4 = st.columns([1.2, 1.5, 1, 1])
    t1.metric("Cultivo", str(farm_data['crop_type']))
    t2.metric("Periodo Monitoreo IoT", crop_info['periodo_captura'])
    t3.metric("Vigor (NDVI)", f"{farm_data['NDVI_index']:.2f}")
    t4.metric("Estado Fitosanitario", str(farm_data['crop_disease_status']))

    st.markdown("**🌱 Recomendaciones Agronómicas Específicas:**")
    for reco in crop_info['recomendaciones']:
        st.write(f"• {reco}")

# Predicción Rendimiento
if modelo_rf is not None and columnas_ml is not None:
    input_data = pd.DataFrame([farm_data])
    input_ml = pd.get_dummies(input_data, columns=['irrigation_type', 'crop_disease_status'], drop_first=False).reindex(columns=columnas_ml, fill_value=0)
    yield_pred = float(modelo_rf.predict(input_ml)[0])
else:
    yield_pred = 4200.0 * farm_data['NDVI_index']

fin = calcular_financiamiento_detallado(farm_data['crop_type'], capital_req, tasa_interes, yield_pred, farm_data['NDVI_index'], farm_data['crop_disease_status'])

st.markdown("---")

# Desglose Financiero Explicito
st.subheader("💳 Estructura y Desglose Financiero de Devolución Total")

d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Capital Solicitado", f"${fin['capital']:,.2f} USD")
d2.metric("Costos Operativos Fijos", f"${fin['total_costos_operativos']:,.2f} USD", help="Estructuración + Seguro Paramétrico + Reserva")
d3.metric("Intereses Generados", f"${fin['interes_monto']:,.2f} USD", help=f"Calculado a {tasa_interes}% anual a 6 meses")
d4.metric("TOTAL A DEVOLVER", f"${fin['total_a_devolver']:,.2f} USD", delta="Obligación Final Exigible", delta_color="inverse")
d5.metric("Retorno Neto Agricultor", f"${fin['retorno_neto_usd']:,.2f} USD", f"{fin['retorno_neto_pct']:.1f}% Margen")

with st.expander("🔍 Ver desglose de Costos Fijos no alterables"):
    c_f1, c_f2, c_f3 = st.columns(3)
    c_f1.write(f"• **Comisión Estructuración ({crop_info['costo_estructuracion_pct']}%):** ${fin['c_est']:,.2f} USD")
    c_f2.write(f"• **Seguro Agrícola Paramétrico ({crop_info['costo_seguro_pct']}%):** ${fin['c_seg']:,.2f} USD")
    c_f3.write(f"• **Fondo de Reserva de Contingencia ({crop_info['costo_reserva_pct']}%):** ${fin['c_res']:,.2f} USD")

st.markdown("---")

# Botón Ejecutar Auditoría
if st.button("⚡ Generar Informe Integrado de Auditoría e Institucionalidad", type="primary", use_container_width=True):
    col_dash1, col_dash2 = st.columns([1, 2])
    with col_dash1:
        with st.container(border=True):
            st.markdown("##### Credit Score Algorítmico")
            st.plotly_chart(plot_gauge_score(fin['score']), use_container_width=True)
            st.markdown(f'<div style="text-align:center;"><span class="{fin["badge_style"]}">{fin["decision"]}</span></div>', unsafe_allow_html=True)
    with col_dash2:
        with st.container(border=True):
            st.markdown("##### Cobertura Financiera y OPEX")
            c_op1, c_op2 = st.columns(2)
            c_op1.metric("Rendimiento Proyectado", f"{yield_pred:,.1f} kg/ha")
            c_op1.metric("Ingreso Bruto Estimado", f"${fin['ingreso_bruto']:,.2f} USD")
            c_op2.metric("DSCR (Cobertura Deuda)", f"{fin['dscr']:.2f}x")
            c_op2.metric("OPEX Estimado Producción", f"${fin['opex_produccion']:,.2f} USD")

    st.markdown("---")
    with st.spinner("Generando dictamen institucional de auditoría de gobernanza..."):
        prompt_agente = f"""
        ROL: Actúa como el Auditor Senior de Riesgo Crediticio y Gobernanza en CoFundo.
        Tono: Institucional, técnico, transparente y analítico. No contradigas la decisión del algoritmo, sino respalda la evaluación profundizando en riesgos macro y geopolíticos.

        DATOS EVALUADOS DEL CASO:
        - Cliente / Parcela: {farm_id} | Región: {farm_data['region']} | Cultivo: {farm_data['crop_type']}
        - Periodo Captura Telemetría: {crop_info['periodo_captura']}
        - Capital Solicitado: ${fin['capital']:,.2f} USD
        - Costos Operativos Fijos: ${fin['total_costos_operativos']:,.2f} USD
        - Intereses ({tasa_interes}% anual a 6m): ${fin['interes_monto']:,.2f} USD
        - TOTAL A DEVOLVER: ${fin['total_a_devolver']:,.2f} USD
        - Ingreso Bruto: ${fin['ingreso_bruto']:,.2f} USD | OPEX Producción: ${fin['opex_produccion']:,.2f} USD
        - Retorno Neto Agricultor: ${fin['retorno_neto_usd']:,.2f} USD ({fin['retorno_neto_pct']:.1f}%)
        - DSCR: {fin['dscr']:.2f}x | Score: {fin['score']}/850 | Decisión: {fin['decision']}
        - Stress Test: Precio -20% (DSCR {fin['dscr_p20']:.2f}x), Sequía -30% (DSCR {fin['dscr_seq']:.2f}x), Insumos +15% (DSCR {fin['dscr_ins']:.2f}x)

        Responde ÚNICAMENTE con un JSON con la siguiente clave:
        {{
          "informe_auditoria_markdown": "CADENA MARKDOWN DEL INFORME COMPLETO SIGUIENDO ESTA ESTRUCTURA:"
        }}

        ESTRUCTURA OBLIGATORIA DEL MARKDOWN EN 'informe_auditoria_markdown':
        # Informe Integrado de Auditoría de Riesgo y Gobernanza
        **CoFundo Credit Cockpit** | Caso: {farm_id} | Cultivo: {farm_data['crop_type']}

        ---

        ### 1. Resumen Ejecutivo
        Síntesis institucional unificada respaldando la decisión {fin['decision']} (Score: {fin['score']}/850). Evaluación de la devolución total exigible de US$ {fin['total_a_devolver']:,.2f} y la preservación del retorno neto del agricultor de US$ {fin['retorno_neto_usd']:,.2f}.

        ---

        ### 2. Cuadro de Conformación Financiera de Devolución
        | Concepto Financiero | Valor (USD) | % sobre Capital | Observación de Auditoría |
        | :--- | :--- | :--- | :--- |
        | **Capital Solicitado** | ${fin['capital']:,.2f} | 100.0% | Base desembolsada |
        | **Costos Operativos (Estructuración/Seguro/Reserva)** | ${fin['total_costos_operativos']:,.2f} | {(fin['total_costos_operativos']/fin['capital']*100):.1f}% | Gastos fijos institucionales |
        | **Interés Nominal Generado** | ${fin['interes_monto']:,.2f} | {(fin['interes_monto']/fin['capital']*100):.1f}% | Calculado a tasa del {tasa_interes}% anual |
        | **TOTAL A DEVOLVER** | **${fin['total_a_devolver']:,.2f}** | **{(fin['total_a_devolver']/fin['capital']*100):.1f}%** | **Obligación total exigible** |
        | **Retorno Neto Agricultor** | **${fin['retorno_neto_usd']:,.2f}** | **{fin['retorno_neto_pct']:.1f}% Margen** | **Utilidad neta post-OPEX** |

        ---

        ### 3. Riesgos de Cola e Hiperpersonalización Agronómica / Geopolítica
        - **Inflación e Interrupción de Insumos N-P-K / Urea:** Evaluación del impacto del costo de fertilizantes y agroquímicos sobre el OPEX regional.
        - **Microclimas y Fenómenos Extremos (El Niño / La Niña):** Riesgos climáticos no detectados en la ventana telemétrica de {crop_info['periodo_captura']}.
        - **Logística y Cuellos de Botella de Exportación:** Volatilidad en fletes, almacenamiento y contratos de venta.
        - **Descalce Cambiario (FX Risk):** Exposición a divisas locales frente a créditos denominados en USD.

        ---

        ### 4. Pruebas de Estrés (Stress Testing)
        - **Caída 20% en Precio de Cosecha:** DSCR resultante {fin['dscr_p20']:.2f}x
        - **Sequía Severa (-30% Rendimiento):** DSCR resultante {fin['dscr_seq']:.2f}x
        - **Alza 15% Insumos / Fertilizantes:** DSCR resultante {fin['dscr_ins']:.2f}x

        ---

        ### 5. Dictamen Final y Cláusulas de Gobernanza Contractual
        Veredicto unificado e instrucciones de desembolso o mitigantes requeridos.
        """
        
        res_agent = consultar_agente_gemini_json(prompt_agente, fin, crop_info, farm_id, farm_data)
        st.markdown(res_agent.get("informe_auditoria_markdown", "# Error en la generación del informe"))
