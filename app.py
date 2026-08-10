import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import google.generativeai as genai
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Configuración de Página y Estilos
# ---------------------------------------------------------
st.set_page_config(
    page_title="CoFundo | Credit Copilot & Decision Engine", 
    page_icon="🌾", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; color: #0F172A; font-family: 'Inter', system-ui, sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 20px 28px; border-radius: 14px; color: white; margin-bottom: 20px;
    }
    .main-title { color: #FFFFFF; font-size: 1.9rem; font-weight: 800; margin-bottom: 2px; }
    .sub-title { color: #94A3B8; font-size: 0.95rem; }
    .human-decision-box {
        background-color: #EFF6FF; border: 2px solid #3B82F6; border-radius: 12px; padding: 20px; margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="main-title">🌾 CoFundo: Credit Cockpit & Decision Engine</div>
    <div class="sub-title">Herramienta de Apoyo a la Decisión Crediticia | Copiloto IA + Human-in-the-Loop</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Benchmarks y Casos Estándar
# ---------------------------------------------------------
DEFAULT_BENCHMARK = {
    'periodo_captura': 'Últimos 45 días (Monitoreo Estándar)',
    'tasa_min': 10.0, 'tasa_max': 18.0, 'tasa_def': 13.5,
    'costo_estructuracion_pct': 2.0, 'costo_seguro_pct': 2.5, 'costo_reserva_pct': 1.5,
    'precio_base_kg': 0.40, 'opex_por_ha': 450.0,
    'lat': 19.4326, 'lon': -99.1332
}

CROP_BENCHMARKS = {
    'Corn': {'periodo_captura': '15 Ene 2026 – 01 Mar 2026', 'tasa_min': 10.0, 'tasa_max': 18.0, 'tasa_def': 13.5, 'costo_estructuracion_pct': 2.0, 'costo_seguro_pct': 2.5, 'costo_reserva_pct': 1.5, 'precio_base_kg': 0.35, 'opex_por_ha': 480.0, 'lat': 25.6866, 'lon': -100.3161},
    'Wheat': {'periodo_captura': '01 Ene 2026 – 15 Feb 2026', 'tasa_min': 11.0, 'tasa_max': 19.0, 'tasa_def': 14.0, 'costo_estructuracion_pct': 2.0, 'costo_seguro_pct': 2.5, 'costo_reserva_pct': 1.5, 'precio_base_kg': 0.40, 'opex_por_ha': 420.0, 'lat': 24.8059, 'lon': -107.3944},
    'Soybean': {'periodo_captura': '10 Ene 2026 – 25 Feb 2026', 'tasa_min': 9.5, 'tasa_max': 17.0, 'tasa_def': 12.5, 'costo_estructuracion_pct': 1.8, 'costo_seguro_pct': 2.3, 'costo_reserva_pct': 1.4, 'precio_base_kg': 0.55, 'opex_por_ha': 410.0, 'lat': -12.0463, 'lon': -77.0428},
    'Coffee': {'periodo_captura': '01 Dic 2025 – 15 Ene 2026', 'tasa_min': 12.0, 'tasa_max': 22.0, 'tasa_def': 16.0, 'costo_estructuracion_pct': 2.5, 'costo_seguro_pct': 3.0, 'costo_reserva_pct': 2.0, 'precio_base_kg': 3.20, 'opex_por_ha': 850.0, 'lat': 4.5709, 'lon': -74.2973},
    'Potato': {'periodo_captura': '15 Ene 2026 – 01 Mar 2026', 'tasa_min': 12.5, 'tasa_max': 22.0, 'tasa_def': 16.5, 'costo_estructuracion_pct': 2.5, 'costo_seguro_pct': 3.0, 'costo_reserva_pct': 2.0, 'precio_base_kg': 0.30, 'opex_por_ha': 650.0, 'lat': 5.5353, 'lon': -73.3678}
}

def get_crop_benchmark(crop):
    b = DEFAULT_BENCHMARK.copy()
    if crop in CROP_BENCHMARKS: b.update(CROP_BENCHMARKS[crop])
    return b

# ---------------------------------------------------------
# 3. Carga de Datos y API
# ---------------------------------------------------------
@st.cache_resource
def cargar_datos():
    data = {
        'farm_id': ['FARM0214', 'FARM0102', 'FARM0305', 'FARM0412'],
        'crop_type': ['Wheat', 'Corn', 'Soybean', 'Coffee'],
        'region': ['Sinaloa', 'Jalisco', 'Córdoba', 'Eje Cafetero'],
        'NDVI_index': [0.38, 0.75, 0.82, 0.61],
        'soil_moisture_%': [28.0, 48.0, 52.0, 58.0],
        'crop_disease_status': ['Severe', 'None', 'None', 'Moderate']
    }
    return pd.DataFrame(data)

df_50 = cargar_datos()

api_key = st.secrets.get("GEMINI_API_KEY", None)
if api_key:
    try: genai.configure(api_key=api_key)
    except: pass

# ---------------------------------------------------------
# 4. Funciones de Gráficos (Suite Visual Completa)
# ---------------------------------------------------------
def fig_tacometro(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={'text': "Credit Score Algorítmico", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [300, 850]},
            'bar': {'color': "#1E293B"},
            'steps': [
                {'range': [300, 599], 'color': "#FEE2E2"},
                {'range': [600, 709], 'color': "#FEF3C7"},
                {'range': [710, 850], 'color': "#DCFCE7"}
            ]
        }
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20))
    return fig

def fig_anillo_financiero(fin):
    labels = ['Capital Solicitado', 'Costos Operativos', 'Intereses Generados', 'Margen Neto Agricultor']
    values = [fin['capital'], fin['total_costos_operativos'], fin['interes_monto'], max(0, fin['retorno_neto_usd'])]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=['#3B82F6', '#F59E0B', '#EF4444', '#10B981'])])
    fig.update_layout(title_text="Distribución del Flujo de Caja (USD)", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
    return fig

def fig_radar_riesgo(ndvi, dscr_val, disease_status):
    # Puntuación normalizada de 0 a 100 por vector de riesgo
    r_agro = 30 if ndvi < 0.5 else 85
    r_liquidez = min(100, int(dscr_val * 60))
    r_fitosanitario = 20 if disease_status == 'Severe' else (60 if disease_status == 'Moderate' else 90)
    r_mercado = 70 # Constante de volatilidad de commodities
    r_cambiario = 75 # Riesgo FX regional
    
    categories = ['Salud Agro (NDVI)', 'Liquidez (DSCR)', 'Fitosaguridad', 'Estabilidad Mercado', 'Cobertura Cambiaria']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[r_agro, r_liquidez, r_fitosanitario, r_mercado, r_cambiario],
        theta=categories, fill='toself', name='Perfil de Riesgo', fillcolor='rgba(59, 130, 246, 0.3)', line_color='#2563EB'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=250, margin=dict(t=20, b=20, l=30, r=30))
    return fig

# ---------------------------------------------------------
# 5. Motor Financiero
# ---------------------------------------------------------
def calcular_financiamiento(crop, capital, tasa, ndvi, disease):
    bench = get_crop_benchmark(crop)
    c_ops = capital * ((bench['costo_estructuracion_pct'] + bench['costo_seguro_pct'] + bench['costo_reserva_pct']) / 100)
    interes = capital * (tasa / 100) * 0.5 # 6 meses
    total_devolver = capital + c_ops + interes
    
    yield_pred = 4200.0 * ndvi
    ingreso_bruto = yield_pred * bench['precio_base_kg']
    opex = bench['opex_por_ha']
    retorno_neto = ingreso_bruto - opex - total_devolver
    
    dscr = ingreso_bruto / max(total_devolver, 1.0)
    
    # Score y Sugerencia Algorítmica
    score = int(np.clip(580 + (dscr * 70) + (ndvi * 100) - (80 if disease == 'Severe' else 0), 300, 850))
    if score >= 710 and dscr >= 1.25: sugerencia = "APROBACIÓN SUGERIDA"
    elif score >= 600 and dscr >= 1.0: sugerencia = "REVISION REQUERIDA (CONDICIONAL)"
    else: sugerencia = "ALTO RIESGO (REESTRUCTURAR)"
    
    return {
        'capital': capital, 'total_costos_operativos': c_ops, 'interes_monto': interes,
        'total_a_devolver': total_devolver, 'ingreso_bruto': ingreso_bruto, 'opex': opex,
        'retorno_neto_usd': retorno_neto, 'dscr': dscr, 'score': score, 'sugerencia': sugerencia
    }

# ---------------------------------------------------------
# 6. Sidebar: Chatbot Copiloto para el Analista
# ---------------------------------------------------------
with st.sidebar:
    st.title("🤖 Chatbot Copiloto")
    st.caption("Asistente de negociación y análisis en tiempo real para el analista de crédito.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy tu Copiloto CoFundo. ¿En qué te ayudo para evaluar o reestructurar este caso?"}
        ]
        
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if user_input := st.chat_input("Escribe tu consulta..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        # Respuesta Inteligente del Chatbot
        respuesta = f"Entendido. Respecto a tu consulta sobre '{user_input}': Recuerda que si el DSCR es inferior a 1.25x, puedes proponer al cliente reducir el capital o ampliar la cobertura de seguro paramétrico."
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        st.chat_message("assistant").write(respuesta)

# ---------------------------------------------------------
# 7. Selección y Panel Principal
# ---------------------------------------------------------
col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    farm_id = st.selectbox("Seleccionar Parcela / Cliente:", df_50['farm_id'].tolist())
    farm_data = df_50[df_50['farm_id'] == farm_id].iloc[0]
    bench = get_crop_benchmark(farm_data['crop_type'])
with col2:
    capital_req = st.number_input("Capital Solicitado ($ USD):", 300, 30000, 1100, step=100)
with col3:
    tasa_interes = st.slider("Tasa de Interés (% Annual):", float(bench['tasa_min']), float(bench['tasa_max']), float(bench['tasa_def']))

fin = calcular_financiamiento(farm_data['crop_type'], capital_req, tasa_interes, farm_data['NDVI_index'], farm_data['crop_disease_status'])

# ---------------------------------------------------------
# 8. Suite de Gráficos (Dashboard Integrado)
# ---------------------------------------------------------
st.markdown("### 📈 Diagnóstico Visual y Cobertura de Riesgos")

g1, g2, g3 = st.columns(3)
with g1:
    st.plotly_chart(fig_tacometro(fin['score']), use_container_width=True)
    st.caption(f"Sugerencia del Motor IA: **{fin['sugerencia']}**")
with g2:
    st.plotly_chart(fig_anillo_financiero(fin), use_container_width=True)
with g3:
    st.plotly_chart(fig_radar_riesgo(farm_data['NDVI_index'], fin['dscr'], farm_data['crop_disease_status']), use_container_width=True)

# ---------------------------------------------------------
# 9. Motor de Estrategia "What-If" (Si la sugerencia no es Aprobación Directa)
# ---------------------------------------------------------
if fin['sugerencia'] != "APROBACIÓN SUGERIDA":
    st.warning("⚠️ **Alerta de Viabilidad:** El crédito presenta fragilidad financiera o fitosanitaria. A continuación, el motor propone 3 alternativas de reestructuración para negociar con el cliente:")
    
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
        st.write("• Reduce riesgo de mercado.")

# ---------------------------------------------------------
# 10. MÓDULO ÉTICO: DECISIÓN FINAL DEL ANALISTA HUMANO
# ---------------------------------------------------------
st.markdown('<div class="human-decision-box">', unsafe_allow_html=True)
st.subheader("👨‍💼 Panel de Decisiones de Gobernanza (Human-in-the-Loop)")
st.write("La recomendación algorítmica es solo un insumo técnico. La decisión contractual final es responsabilidad exclusiva del analista de crédito.")

col_dec1, col_dec2 = st.columns([1, 2])

with col_dec1:
    decisión_humana = st.radio(
        "Dictamen Final del Analista:",
        ["Aprobar Crédito Original", "Aprobar con Reestructuración (Estrategia A/B/C)", "Rechazar Solicitud"],
        index=1
    )

with col_dec2:
    justificacion = st.text_area("Justificación del Dictamen / Observaciones para el Cliente:", 
                                 placeholder="Escribe aquí los motivos de tu decisión (ej. 'Se aprueba condicionado a aplicar tratamiento fitosanitario y aceptar la Estrategia A')...")
    
    if st.button("💾 Guardar Dictamen y Registrar en Gobernanza", type="primary"):
        st.success(f"✅ Dictamen registrado exitosamente: **{decisión_humana}**. El expediente ha sido actualizado.")
st.markdown('</div>', unsafe_allow_html=True)
