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
# ---------------------------------------------------------
# Generación de Informes Senior e Integración con Gemini IA
# ---------------------------------------------------------

def construir_prompt_senior_ia(fin_data, bench, farm_id, farm_data, yield_pred):
    """Construye un prompt de alta precisión para que el LLM actúe como un Director de Riesgo Agrícola Senior."""
    
    precio_base = bench['precio_base_kg']
    ingreso_minimo_requerido = fin_data['total_a_devolver'] + fin_data['opex_produccion']
    precio_breakeven = ingreso_minimo_requerido / max(1.0, yield_pred)
    rendimiento_breakeven = ingreso_minimo_requerido / max(0.01, precio_base)

    prompt = f"""
    Actúa como un Vicepresidente Senior de Riesgo Crediticio Agropecuario y Estructuración Financiera en un fondo AgTech internacional.
    Debes generar un dictamen de auditoría crediticia en formato JSON con la única clave "informe_auditoria_markdown".

    DATOS DE LA PARCELA Y CLIENTE:
    - ID Parcela: {farm_id}
    - Región / Localización: {farm_data.get('region', 'N/A')}
    - Cultivo: {farm_data['crop_type']}
    - Telemetría IoT / Satelital: NDVI = {farm_data['NDVI_index']}, Humedad Suelo = {farm_data.get('soil_moisture_%', 'N/A')}%, Lluvia Acumulada = {farm_data.get('rainfall_mm', 'N/A')} mm, pH = {farm_data.get('soil_pH', 'N/A')}, Temp = {farm_data.get('temperature_C', 'N/A')}°C
    - Estado Fitosanitario: {farm_data['crop_disease_status']}
    - Riego: {farm_data.get('irrigation_type', 'N/A')}

    MÉTRICAS FINANCIERAS BASE:
    - Capital Solicitado: ${fin_data['capital']:,.2f} USD
    - Intereses (6 meses): ${fin_data['interes_monto']:,.2f} USD
    - Costos Operativos Crédito: ${fin_data['total_costos_operativos']:,.2f} USD
    - Monto Total a Devolver: ${fin_data['total_a_devolver']:,.2f} USD
    - Rendimiento Estimado ML: {yield_pred:,.1f} kg/ha
    - Precio Base Mercado: ${precio_base:,.2f} USD/kg
    - Ingreso Bruto Proyectado: ${fin_data['ingreso_bruto']:,.2f} USD
    - OPEX Producción: ${fin_data['opex_produccion']:,.2f} USD
    - Ganancia Neta Agricultor: ${fin_data['retorno_neto_usd']:,.2f} USD (Margen: {fin_data['retorno_neto_pct']:.1f}%)
    - DSCR Base: {fin_data['dscr']:.2f}x
    - Credit Score IA: {fin_data['score']}/850 ({fin_data['sugerencia']})
    - Precio Mínimo Equilibrio (DSCR=1.0x): ${precio_breakeven:,.2f} USD/kg
    - Rendimiento Mínimo Equilibrio (DSCR=1.0x): {rendimiento_breakeven:,.1f} kg/ha

    INSTRUCCIONES DE ESTRUCTURA DEL INFORME (Markdown Senior):
    Genera un informe institucional exhaustivo con las siguientes 5 secciones:

    1. 🎯 Executive Risk Summary & Credit Scoring
       - Dictamen formal, DSCR base, Credit Score y evaluación del margen de seguridad financiero.
    2. 🌿 Radiografía Agronómica y Análisis de Vulnerabilidad Regional
       - Análisis del impacto del clima, región ({farm_data.get('region', 'N/A')}), tipo de cultivo ({farm_data['crop_type']}) y métricas IoT (NDVI, humedad, plagas).
    3. ⚖️ Análisis de Punto de Equilibrio (Breakeven & Stress Margins)
       - Explicación de los límites de tolerancia antes del impago (Precio y Rendimiento Mínimo).
    4. 🌩️ Matriz de Estrés Dinámica por Región y Cultivo (AI-Generated Stress Testing)
       - Genera 3 ESCENARIOS DE ESTRÉS ESPECÍFICOS para {farm_data['crop_type']} en la región {farm_data.get('region', 'N/A')} (ej. sequía regional, plaga focalizada, caída de precio internacional o shock de insumos).
       - Para CADA escenario debes incluir:
         * Nombre del Evento y Probabilidad
         * Caída estimada en DSCR y Margen del Agricultor
         * **ACCIÓN DE MITIGACIÓN SUGERIDA** (ej. contrato futures, seguro paramétrico, fondo de reserva, desembolsos parciales).
    5. 🛡️ Dictamen de Gobernanza y Convenios Binding (Covenants & Disbursement Rules)
       - 3 a 4 cláusulas obligatorias que el analista debe exigir antes de desembolsar la firma del contrato.

    Utiliza formato Markdown profesional, tablas, negritas y bloques de citas. Sé riguroso, analítico y técnico.
    """
    return prompt


def generar_fallback_local_report(fin_data, bench, farm_id, farm_data, yield_pred):
    """Fallback robusto institucional en caso de no conectar con la API de IA."""
    precio_base = bench['precio_base_kg']
    ingreso_minimo = fin_data['total_a_devolver'] + fin_data['opex_produccion']
    precio_breakeven = ingreso_minimo / max(1.0, yield_pred)
    rendimiento_breakeven = ingreso_minimo / max(0.01, precio_base)

    markdown_content = f"""# 📋 Auditoría de Riesgo Crediticio e Informe de Gobernanza
**CoFundo Credit Cockpit** | Parcela: **{farm_id}** | Región: **{farm_data.get('region', 'N/A')}** | Cultivo: **{farm_data['crop_type']}**

---

### 1. 🎯 Executive Risk Summary & Credit Scoring
* **Dictamen Algorítmico:** **{fin_data['sugerencia']}** (Credit Score: **{fin_data['score']}/850**)
* **Ratio de Cobertura de Deuda (DSCR Base):** **{fin_data['dscr']:.2f}x** *(Capacidad de pago de la cosecha)*
* **Estructura de la Obligación Financiera:**
  * Capital Prestado: **${fin_data['capital']:,.2f} USD**
  * Costo Financiero (Intereses 6m): **${fin_data['interes_monto']:,.2f} USD**
  * Gastos Operativos (Estructuración/Seguro/Reserva): **${fin_data['total_costos_operativos']:,.2f} USD**
  * **Exigible Total a Devolver:** **${fin_data['total_a_devolver']:,.2f} USD**
* **Retorno Neto Proyectado del Productor:** **${fin_data['retorno_neto_usd']:,.2f} USD** (Margen Neto: **{fin_data['retorno_neto_pct']:.1f}%**)

---

### 2. 🌿 Radiografía Agronómica y Vulnerabilidad Regional ({farm_data.get('region', 'N/A')})
| Métrica Evaluada | Valor Medido | Parámetro Esperado | Evaluación de Riesgo |
| :--- | :--- | :--- | :--- |
| **Índice Vigor (NDVI)** | **{farm_data['NDVI_index']:.2f}** | > 0.65 | Desarrollo vegetativo adecuado |
| **Rendimiento Estimado** | **{yield_pred:,.1f} kg/ha** | Promedio Regional | Productividad dentro de rango base |
| **Humedad y Clima** | **{farm_data.get('soil_moisture_%', 'N/A')}% Suelo** | 40% - 60% | Balance hídrico monitoreado |
| **Fitosanidad** | **{farm_data['crop_disease_status']}** | Sin plaga severa | Riesgo controlado por telemetría |

---

### 3. ⚖️ Análisis de Punto de Equilibrio y Margen de Seguridad
* **Precio Mínimo de Equilibrio:** El precio del **{farm_data['crop_type']}** puede caer de **${precio_base:,.2f} USD/kg** hasta **${precio_breakeven:,.2f} USD/kg** (caída máx: {((precio_base-precio_breakeven)/precio_base*100):.1f}%) sin entrar en insolvencia (DSCR < 1.0x).
* **Rendimiento Mínimo de Equilibrio:** La parcela debe cosechar al menos **{rendimiento_breakeven:,.1f} kg/ha** para cubrir el costo operativo y el crédito.

---

### 4. 🌩️ Matriz de Estrés Regional y Estrategias de Mitigación
| Escenario de Estrés | Impacto Financiero | DSCR Ajustado | Acción Mitigadora Sugerida |
| :--- | :--- | :--- | :--- |
| **Caída de Precio (-20%)** | Caída en ingreso bruto | **{fin_data['dscr_p20']:.2f}x** | Cobertura de precio mediante contrato de opción en Bolsa o venta futura. |
| **Evento Climático / Sequía (-30% Rendimiento)** | Pérdida parcial de cosecha | **{fin_data['dscr_seq']:.2f}x** | Exigir póliza de seguro paramétrico indexada al NDVI antes del desembolso. |
| **Shock Inflacionario Insumos (+15% OPEX)** | Reducción de margen neto | **{fin_data['dscr_ins']:.2f}x** | Desembolso directo a proveedores autorizados de insumos. |

---

### 5. 🛡️ Dictamen de Gobernanza y Convenios (Covenants)
> **DISPOSICIONES OBLIGATORIAS PARA EL ANALISTA DE CRÉDITO:**
> 1. Registro formal de garantía prendaria sobre la cosecha de {farm_data['crop_type']}.
> 2. Desembolso fraccionado en 2 ministraciones sujeto a la validación satelital del NDVI.
> 3. Verificación de contrato de seguro activo contra eventos hidrometeorológicos.
"""
    return {"informe_auditoria_markdown": markdown_content}


def consultar_agente_gemini(prompt_texto, fin_data, bench, farm_id, farm_data, yield_pred):
    if not api_key:
        return generar_fallback_local_report(fin_data, bench, farm_id, farm_data, yield_pred)
        
    modelos = ['gemini-1.5-pro', 'gemini-1.5-flash']
    for mod in modelos:
        try:
            model = genai.GenerativeModel(mod, generation_config={"response_mime_type": "application/json"})
            res = model.generate_content(prompt_texto)
            return json.loads(res.text)
        except Exception:
            continue
            
    return generar_fallback_local_report(fin_data, bench, farm_id, farm_data, yield_pred)
    
# ---------------------------------------------------------
# 7. Motor Inteligente del Chatbot Copiloto
# ---------------------------------------------------------
def responder_chat_copiloto(historial_mensajes, farm_id, farm_data, fin_data, bench, yield_pred, tasa_interes):
    """Genera respuestas del chatbot garantizando autenticación y autodetectando modelos disponibles."""
    
    # 1. Obtener y verificar la API Key
    api_key = st.secrets.get("GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY", None)
    if not api_key:
        return "⚠️ **API Key no configurada:** Agrega `GEMINI_API_KEY` en tus `secrets.toml` para activar el copiloto."

    # 2. Autenticar la librería
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return f"❌ Error al configurar la API Key: {str(e)}"

    # 3. Contexto técnico y financiero de la parcela
    contexto_sistema = f"""
    [INSTRUCCIONES DE COFUNDO COPILOT]
    Eres CoFundo Copilot, un analista experto en riesgo crediticio agropecuario y estructuración financiera.
    Estás asistiendo en tiempo real al oficial de crédito que evalúa la siguiente parcela activa:

    FICHA TÉCNICA Y FINANCIERA EN TIEMPO REAL:
    - ID Parcela: {farm_id} | Región: {farm_data.get('region', 'N/A')} | Cultivo: {farm_data['crop_type']}
    - Telemetría IoT: NDVI = {farm_data['NDVI_index']:.2f}, Humedad = {farm_data.get('soil_moisture_%', 'N/A')}%, Temp = {farm_data.get('temperature_C', 'N/A')}°C
    - Estado Fitosanitario: {farm_data['crop_disease_status']}
    - Crédito Solicitado: ${fin_data['capital']:,.2f} USD | Total a Devolver: ${fin_data['total_a_devolver']:,.2f} USD (Tasa: {tasa_interes}%)
    - Métricas Riesgo: DSCR Base = {fin_data['dscr']:.2f}x | Credit Score = {fin_data['score']}/850 ({fin_data['sugerencia']})
    - Ganancia Neta Productor: ${fin_data['retorno_neto_usd']:,.2f} USD (Margen: {fin_data['retorno_neto_pct']:.1f}%)
    - Productividad Estimada ML: {yield_pred:,.1f} kg/ha | Precio Base: ${bench['precio_base_kg']:,.2f} USD/kg

    INSTRUCCIONES:
    - Responde de forma concisa, profesional y técnica en español (máximo 2 párrafos cortos o viñetas).
    - Argumenta utilizando siempre los datos específicos citados arriba.
    - Si te preguntan sobre la caída de precio (-20%), explica que el DSCR cae a {fin_data['dscr_p20']:.2f}x y sugiere estrategias de mitigación.
    """

    pregunta_usuario = historial_mensajes[-1]["content"]
    prompt_completo = f"{contexto_sistema}\n\nPregunta del analista: {pregunta_usuario}"

    # 4. Descubrimiento dinámico del modelo disponible en tu API Key
    candidatos_modelo = []
    try:
        modelos_remotos = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        # Priorizar modelos rápidos de Gemini
        for objetivo in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro']:
            if objetivo in modelos_remotos:
                candidatos_modelo.append(objetivo)
        
        # Añadir resto de modelos encontrados
        candidatos_modelo.extend(modelos_remotos)
    except Exception:
        pass

    # Modelos por defecto en caso de fallback
    if not candidatos_modelo:
        candidatos_modelo = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']

    # 5. Intento de generación con el primer modelo funcional
    for nombre_mod in candidatos_modelo:
        try:
            model = genai.GenerativeModel(nombre_mod)
            respuesta = model.generate_content(prompt_completo)
            if respuesta and respuesta.text:
                return respuesta.text
        except Exception:
            continue

    return "❌ No se pudo conectar con la API de Gemini. Verifica que tu API Key esté activa en Google AI Studio."

def renderizar_sidebar_copiloto(farm_id, farm_data, fin, bench, yield_pred, tasa_interes):
    """Renderiza el Sidebar del Chatbot con contexto dinámico."""
    # 🎨 Inyección de CSS para ensanchar el Sidebar del chatbot
    st.markdown(
        """
        <style>
            /* Cambia el ancho mínimo y máximo del sidebar */
            [data-testid="stSidebar"] {
                min-width: 480px !important;
                max-width: 520px !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.sidebar:
        st.title("🤖 Copiloto de Riesgo IA")
        st.caption(f"Asistente en vivo para **{farm_id}** ({farm_data['crop_type']})")
        
        # Botones de consulta rápida
        st.markdown("**💡 Consultas Rápidas:**")
        cq1, cq2 = st.columns(2)
        preg_rapida = None
        with cq1:
            if st.button("❓ ¿Es viable?", use_container_width=True):
                preg_rapida = "¿Cuáles son los mayores riesgos de aprobar este crédito y cómo los mitigamos?"
        with cq2:
            if st.button("🔄 Reestructurar", use_container_width=True):
                preg_rapida = "¿Cómo reestructurar el crédito si el rendimiento cae un 20%?"

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": f"¡Hola! Soy tu Copiloto CoFundo. Tengo en pantalla la parcela **{farm_id}** ({farm_data['crop_type']}). ¿En qué te ayudo a evaluar esta operación?"}
            ]
            
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        user_input = st.chat_input("Pregunta al copiloto...")
        
        if preg_rapida:
            user_input = preg_rapida

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            with st.spinner("Analizando expediente con Gemini..."):
                respuesta_ia = responder_chat_copiloto(
                    st.session_state.messages, farm_id, farm_data, fin, bench, yield_pred, tasa_interes
                )
            
            st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})
            st.rerun()

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


# ✅ MODIFICACIÓN: Desplegable opcional para las recomendaciones agronómicas
    with st.expander("🌱 Ver Recomendaciones Agronómicas Específicas"):
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

# Llama a la barra lateral pasándole los datos ya calculados de la parcela
renderizar_sidebar_copiloto(farm_id, farm_data, fin, bench, yield_pred, tasa_interes)
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

# ✅ MODIFICACIÓN: Explicación y cálculo transparente de Retorno Bruto y Neto
with st.container(border=True):
    st.markdown("#### 🌾 ¿Cómo se calcula la ganancia del agricultor?")
    
    col_ret1, col_ret2 = st.columns(2)
    
    with col_ret1:
        st.markdown(f"""
        **1. Ingreso Bruto Proyectado (Ventas de Cosecha)**
        * **Rendimiento Estimado:** `{yield_pred:,.1f} kg/ha`
        * **Precio Base Estimado:** `${bench['precio_base_kg']:,.2f} USD/kg`
        * **Fórmula:** `Rendimiento × Precio Base`
        
        👉 **Ingreso Bruto Total:** **`${fin['ingreso_bruto']:,.2f} USD`**
        """)
        
    with col_ret2:
        st.markdown(f"""
        **2. Retorno Neto Estimado (Ganancia Limpia)**
        * **(+) Ingreso Bruto:** `${fin['ingreso_bruto']:,.2f} USD`
        * **(-) Costos de Producción (OPEX):** `${fin['opex_produccion']:,.2f} USD`
        * **(-) Crédito Total a Devolver:** `${fin['total_a_devolver']:,.2f} USD`
        
        👉 **Ganancia Neta Final:** **`${fin['retorno_neto_usd']:,.2f} USD`** *(Margen: {fin['retorno_neto_pct']:.1f}%)*
        """)
        
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

if st.button("⚡ Generar Informe Integrado de Auditoría e Institucionalidad (IA Senior)", type="primary", use_container_width=True):
    # 1. Construir el prompt detallado de nivel bancario/senior
    prompt_informe = construir_prompt_senior_ia(fin, bench, farm_id, farm_data, yield_pred)
    
    # 2. Iniciar el spinner de procesando
    with st.spinner("Procesando auditoría de riesgo, simulación de estrés regional y convenios..."):
        # 3. Llamar a la función pasando los 6 argumentos correctos
        res_json = consultar_agente_gemini(prompt_informe, fin, bench, farm_id, farm_data, yield_pred)
        
        # 4. Renderizar el resultado en pantalla en formato Markdown
        st.markdown(res_json.get("informe_auditoria_markdown", "No se pudo generar el informe de auditoría."))

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
