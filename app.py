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
    page_title="CoFundo | Credit Cockpit & Copilot", 
    page_icon="🌾", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; color: #0F172A; font-family: 'Inter', system-ui, sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 22px 28px; border-radius: 14px; color: white; margin-bottom: 20px;
    }
    .main-title { color: #FFFFFF; font-size: 1.9rem; font-weight: 800; margin-bottom: 2px; }
    .sub-title { color: #94A3B8; font-size: 0.95rem; }
    .alert-card {
        padding: 14px; border-radius: 10px; margin-bottom: 12px; font-size: 0.9rem; font-weight: 500;
    }
    .alert-danger { background-color: #FEE2E2; border-left: 5px solid #EF4444; color: #991B1B; }
    .alert-warning { background-color: #FEF3C7; border-left: 5px solid #F59E0B; color: #92400E; }
    .alert-success { background-color: #DCFCE7; border-left: 5px solid #10B981; color: #166534; }
    .exec-summary {
        background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px; margin-bottom: 20px;
    }
    .human-decision-box {
        background-color: #EFF6FF; border: 2px solid #3B82F6; border-radius: 12px; padding: 20px; margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="main-title">🌾 CoFundo: Credit Cockpit & Decision Engine</div>
    <div class="sub-title">Plataforma Agéntica de Análisis de Crédito Agrícola | Human-in-the-Loop</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Benchmarks y Base de Datos Integrada
# ---------------------------------------------------------
DEFAULT_BENCHMARK = {
    'periodo_captura': 'Últimos 45 días (Monitoreo Satelital)',
    'tasa_min': 10.0, 'tasa_max': 18.0, 'tasa_def': 13.5,
    'costo_estructuracion_pct': 2.0, 'costo_seguro_pct': 2.5, 'costo_reserva_pct': 1.5,
    'precio_base_kg': 0.40, 'opex_por_ha': 450.0, 'hectareas_def': 3.0
}

CROP_BENCHMARKS = {
    'Corn': {'periodo_captura': '15 Ene 2026 – 01 Mar 2026', 'tasa_min': 10.0, 'tasa_max': 18.0, 'tasa_def': 13.5, 'costo_estructuracion_pct': 2.0, 'costo_seguro_pct': 2.5, 'costo_reserva_pct': 1.5, 'precio_base_kg': 0.35, 'opex_por_ha': 480.0, 'hectareas_def': 5.0},
    'Wheat': {'periodo_captura': '01 Ene 2026 – 15 Feb 2026', 'tasa_min': 11.0, 'tasa_max': 19.0, 'tasa_def': 14.0, 'costo_estructuracion_pct': 2.0, 'costo_seguro_pct': 2.5, 'costo_reserva_pct': 1.5, 'precio_base_kg': 0.40, 'opex_por_ha': 420.0, 'hectareas_def': 4.0},
    'Soybean': {'periodo_captura': '10 Ene 2026 – 25 Feb 2026', 'tasa_min': 9.5, 'tasa_max': 17.0, 'tasa_def': 12.5, 'costo_estructuracion_pct': 1.8, 'costo_seguro_pct': 2.3, 'costo_reserva_pct': 1.4, 'precio_base_kg': 0.55, 'opex_por_ha': 410.0, 'hectareas_def': 6.0},
    'Coffee': {'periodo_captura': '01 Dic 2025 – 15 Ene 2026', 'tasa_min': 12.0, 'tasa_max': 22.0, 'tasa_def': 16.0, 'costo_estructuracion_pct': 2.5, 'costo_seguro_pct': 3.0, 'costo_reserva_pct': 2.0, 'precio_base_kg': 3.20, 'opex_por_ha': 850.0, 'hectareas_def': 2.5},
    'Potato': {'periodo_captura': '15 Ene 2026 – 01 Mar 2026', 'tasa_min': 12.5, 'tasa_max': 22.0, 'tasa_def': 16.5, 'costo_estructuracion_pct': 2.5, 'costo_seguro_pct': 3.0, 'costo_reserva_pct': 2.0, 'precio_base_kg': 0.30, 'opex_por_ha': 650.0, 'hectareas_def': 2.0}
}

def get_crop_benchmark(crop):
    b = DEFAULT_BENCHMARK.copy()
    if crop in CROP_BENCHMARKS: b.update(CROP_BENCHMARKS[crop])
    return b

@st.cache_resource
def cargar_datos():
    data = {
        'farm_id': ['FARM0214', 'FARM0102', 'FARM0305', 'FARM0412'],
        'farmer_name': ['José Donoso', 'Carlos Mendoza', 'Elena Rostova', 'Santiago Gómez'],
        'crop_type': ['Wheat', 'Corn', 'Soybean', 'Coffee'],
        'region': ['Sinaloa, MX', 'Jalisco, MX', 'Córdoba, AR', 'Eje Cafetero, CO'],
        'NDVI_index': [0.38, 0.75, 0.82, 0.61],
        'soil_moisture_%': [28.0, 48.0, 52.0, 58.0],
        'crop_disease_status': ['Severe', 'None', 'None', 'Moderate']
    }
    return pd.DataFrame(data)

df_farms = cargar_datos()

# Configuración API Gemini
api_key = st.secrets.get("GEMINI_API_KEY", None)
if api_key:
    try: genai.configure(api_key=api_key)
    except: pass

# ---------------------------------------------------------
# 3. Motor Financiero y Agronómico
# ---------------------------------------------------------
def calcular_financiamiento(crop, capital, tasa, ndvi, disease, hectareas):
    bench = get_crop_benchmark(crop)
    
    # Costos de deuda
    c_ops = capital * ((bench['costo_estructuracion_pct'] + bench['costo_seguro_pct'] + bench['costo_reserva_pct']) / 100)
    interes = capital * (tasa / 100) * 0.5 # Plazo estándar 6 meses
    total_devolver = capital + c_ops + interes
    
    # Rendimientos y Escala
    yield_pred_ha = 4200.0 * ndvi
    produccion_total_kg = yield_pred_ha * hectareas
    produccion_total_ton = produccion_total_kg / 1000.0
    
    ingreso_bruto_total = produccion_total_kg * bench['precio_base_kg']
    opex_total = bench['opex_por_ha'] * hectareas
    
    retorno_neto_usd = ingreso_bruto_total - opex_total - total_devolver
    flujo_caja_operativo = max(0, ingreso_bruto_total - opex_total)
    dscr = flujo_caja_operativo / max(total_devolver, 1.0)
    
    # Score y Diagnóstico Algorítmico
    score = int(np.clip(580 + (dscr * 70) + (ndvi * 100) - (80 if disease == 'Severe' else 0), 300, 850))
    if score >= 710 and dscr >= 1.25 and retorno_neto_usd > 0: 
        sugerencia = "APROBACIÓN SUGERIDA"
    elif score >= 600 and dscr >= 0.95: 
        sugerencia = "REVISIÓN REQUERIDA (CONDICIONAL)"
    else: 
        sugerencia = "ALTO RIESGO (REESTRUCTURAR)"
    
    return {
        'capital': capital, 'total_costos_operativos': c_ops, 'interes_monto': interes,
        'total_a_devolver': total_devolver, 'yield_ha': yield_pred_ha,
        'produccion_total_kg': produccion_total_kg, 'produccion_total_ton': produccion_total_ton,
        'ingreso_bruto': ingreso_bruto_total, 'opex_total': opex_total,
        'retorno_neto_usd': retorno_neto_usd, 'dscr': dscr, 'score': score, 'sugerencia': sugerencia
    }

# ---------------------------------------------------------
# 4. Funciones de Gráficos Visuales
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
    fig.update_layout(height=210, margin=dict(t=30, b=10, l=20, r=20))
    return fig

def fig_anillo_financiero(fin):
    labels = ['Capital Solicitado', 'Costos Operativos', 'Intereses Generados', 'OPEX Producción', 'Margen Neto Agricultor']
    values = [fin['capital'], fin['total_costos_operativos'], fin['interes_monto'], fin['opex_total'], max(0, fin['retorno_neto_usd'])]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=['#3B82F6', '#F59E0B', '#EF4444', '#64748B', '#10B981'])])
    fig.update_layout(title_text="Estructura de Flujo de Caja (USD)", height=250, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
    return fig

def fig_radar_riesgo(ndvi, dscr_val, disease_status):
    r_agro = 30 if ndvi < 0.5 else 85
    r_liquidez = min(100, int(dscr_val * 60))
    r_fitosanitario = 20 if disease_status == 'Severe' else (60 if disease_status == 'Moderate' else 90)
    r_mercado = 70 
    r_cambiario = 75 
    
    categories = ['Salud Agro (NDVI)', 'Liquidez (DSCR)', 'Fitosaguridad', 'Estabilidad Mercado', 'Cobertura Cambiaria']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[r_agro, r_liquidez, r_fitosanitario, r_mercado, r_cambiario],
        theta=categories, fill='toself', name='Perfil de Riesgo', fillcolor='rgba(59, 130, 246, 0.3)', line_color='#2563EB'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=250, margin=dict(t=20, b=20, l=30, r=30))
    return fig

# ---------------------------------------------------------
# 5. Selección Principal de Caso
# ---------------------------------------------------------
col_sel1, col_sel2, col_sel3, col_sel4 = st.columns([1.5, 1, 1, 1])

with col_sel1:
    farm_id = st.selectbox("Seleccionar Parcela / Expediente:", df_farms['farm_id'].tolist())
    farm_data = df_farms[df_farms['farm_id'] == farm_id].iloc[0]
    bench = get_crop_benchmark(farm_data['crop_type'])

with col_sel2:
    hectareas = st.number_input("Superficie (ha):", 0.5, 50.0, float(bench['hectareas_def']), step=0.5)

with col_sel3:
    capital_req = st.number_input("Capital Solicitado ($ USD):", 300, 30000, 1100, step=100)

with col_sel4:
    tasa_interes = st.slider("Tasa Interés (% Anual):", float(bench['tasa_min']), float(bench['tasa_max']), float(bench['tasa_def']))

fin = calcular_financiamiento(farm_data['crop_type'], capital_req, tasa_interes, farm_data['NDVI_index'], farm_data['crop_disease_status'], hectareas)

# ---------------------------------------------------------
# 6. Sidebar: Chatbot Copiloto (Con Limpieza por Cultivo)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🤖 Copiloto de Negociación")
    st.caption(f"Evaluando: **{farm_id}** ({farm_data['crop_type']})")
    st.caption(f"Cliente: **{farm_data['farmer_name']}** | {farm_data['region']}")
    
    # REGLA DE LIMPIEZA AUTOMÁTICA DEL CHAT AL CAMBIAR DE PARCELA/CULTIVO
    if "current_farm" not in st.session_state:
        st.session_state.current_farm = farm_id

    if st.session_state.current_farm != farm_id:
        st.session_state.current_farm = farm_id
        st.session_state.messages = [
            {"role": "assistant", "content": f"¡Hola! He actualizado mi contexto para la parcela **{farm_id}** ({farm_data['crop_type']}).\n\n• **NDVI:** {farm_data['NDVI_index']}\n• **Enfermedad:** {farm_data['crop_disease_status']}\n• **Score:** {fin['score']}\n• **DSCR:** {fin['dscr']:.2f}x\n\n¿En qué deseas profundizar para este expediente?"}
        ]
    elif "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"¡Hola! Soy tu Copiloto CoFundo. Estoy analizando **{farm_id}** ({farm_data['crop_type']}). ¿Cómo puedo ayudarte en la negociación?"}
        ]

    # Mostrar Historial de Mensajes
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Entrada de Usuario
    if user_input := st.chat_input("Escribe tu consulta al copiloto..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Generación de Respuesta con Gemini (si está la API Key) o Respuestas Lógicas
        if api_key:
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
                Eres CoFundo, un copiloto experto en crédito agropecuario para bancos y fintechs.
                Estás asistiendo a un analista de crédito humano. El analista toma la decisión final.
                
                DATOS DE LA PARCELA ACTUAL:
                - Parcela ID: {farm_id}
                - Cliente: {farm_data['farmer_name']}
                - Cultivo: {farm_data['crop_type']}
                - Región: {farm_data['region']}
                - NDVI (Salud Biológica): {farm_data['NDVI_index']}
                - Humedad del Suelo: {farm_data['soil_moisture_%']}%
                - Estado Fitosanitario: {farm_data['crop_disease_status']}
                - Capital Solicitado: ${capital_req} USD para {hectareas} ha
                - Tasa de Interés: {tasa_interes}%
                - Score Algorítmico: {fin['score']}
                - Cobertura DSCR: {fin['dscr']:.2f}x
                - Sugerencia del Sistema: {fin['sugerencia']}
                
                PREGUNTA DEL ANALISTA: {user_input}
                
                Instrucciones: Responde de forma concisa, comercial y profesional. Si el crédito es riesgoso, sugiere cómo negociar (garantías, seguro paramétrico, reducción de capital) manteniendo siempre el enfoque ético de apoyo al analista.
                """
                response = model.generate_content(prompt)
                respuesta_text = response.text
            except Exception as e:
                respuesta_text = f"Respuesta basada en reglas: Para {farm_data['crop_type']} con DSCR {fin['dscr']:.2f}x, te sugiero revisar las alternativas de la sección 'What-If' si requieres ajustar la estructura de pago."
        else:
            respuesta_text = f"Copiloto: Analizando la consulta sobre '{user_input}' para {farm_data['crop_type']}. Si buscas mejorar la cobertura financiera, recuerda que elevar la superficie o reducir el capital amortiza mejor el crédito."

        st.session_state.messages.append({"role": "assistant", "content": respuesta_text})
        st.chat_message("assistant").write(respuesta_text)

# ---------------------------------------------------------
# 7. Resumen Ejecutivo y Métricas Clave
# ---------------------------------------------------------
st.markdown('<div class="exec-summary">', unsafe_allow_html=True)
st.markdown(f"### 📋 Resumen Ejecutivo | Cliente: **{farm_data['farmer_name']}** ({farm_id})")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Superficie Financiar", f"{hectareas:.1f} ha")
m2.metric("Producción Estimada", f"{fin['produccion_total_ton']:.2f} Ton", help=f"{fin['yield_ha']:,.0f} kg/ha")
m3.metric("Ingreso Bruto Parcela", f"${fin['ingreso_bruto']:,.2f} USD")
m4.metric("Cobertura Deuda (DSCR)", f"{fin['dscr']:.2f}x", delta="Seguro" if fin['dscr'] >= 1.25 else "Bajo Margen", delta_color="normal" if fin['dscr']>=1.25 else "inverse")
m5.metric("Retorno Neto Agricultor", f"${fin['retorno_neto_usd']:,.2f} USD")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 8. Sistema de Alertas Automáticas Dinámicas
# ---------------------------------------------------------
st.markdown("### 🚨 Panel de Alertas Fitosanitarias y Financieras")

col_a1, col_a2 = st.columns(2)

with col_a1:
    if farm_data['NDVI_index'] < 0.50:
        st.markdown(f"""
        <div class="alert-card alert-danger">
            🚨 <b>Alerta Crítica Agronómica:</b> NDVI de <b>{farm_data['NDVI_index']}</b> (Estrés hídrico o daño severo foliar). Rendimiento proyectado recortado.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-card alert-success">
            🟢 <b>Salud Vegetal Adecuada:</b> Índice NDVI de <b>{farm_data['NDVI_index']}</b> dentro de parámetros óptimos de desarrollo vegetal.
        </div>
        """, unsafe_allow_html=True)

    if farm_data['crop_disease_status'] == 'Severe':
        st.markdown("""
        <div class="alert-card alert-danger">
            🚨 <b>Riesgo Fitosanitario Grave:</b> Se reporta afectación SEVERA por plaga/enfermedad en la zona. Requiere póliza de seguro paramétrico obligatoria.
        </div>
        """, unsafe_allow_html=True)
    elif farm_data['crop_disease_status'] == 'Moderate':
        st.markdown("""
        <div class="alert-card alert-warning">
            ⚠️ <b>Riesgo Fitosanitario Moderado:</b> Afectación leve observada. Se recomienda monitoreo foliar a los 30 días del desembolso.
        </div>
        """, unsafe_allow_html=True)

with col_a2:
    if fin['dscr'] < 1.0:
        st.markdown(f"""
        <div class="alert-card alert-danger">
            🚨 <b>Alerta de Insolvencia:</b> DSCR de <b>{fin['dscr']:.2f}x</b>. La parcela no genera suficiente flujo libre para repagar capital e intereses.
        </div>
        """, unsafe_allow_html=True)
    elif fin['dscr'] < 1.25:
        st.markdown(f"""
        <div class="alert-card alert-warning">
            ⚠️ <b>Margen Estrecho de Liquidez:</b> DSCR de <b>{fin['dscr']:.2f}x</b> (Inferior al estándar mínimo deseable de 1.25x).
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-card alert-success">
            🟢 <b>Sólida Capacidad de Pago:</b> DSCR de <b>{fin['dscr']:.2f}x</b> garantiza capacidad de amortización holgada.
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 9. Suite Visual (Tacómetro, Anillo y Radar)
# ---------------------------------------------------------
st.markdown("### 📈 Diagnóstico Visual del Expediente")

g1, g2, g3 = st.columns(3)
with g1:
    st.plotly_chart(fig_tacometro(fin['score']), use_container_width=True)
    st.caption(f"Sugerencia del Motor IA: **{fin['sugerencia']}**")
with g2:
    st.plotly_chart(fig_anillo_financiero(fin), use_container_width=True)
with g3:
    st.plotly_chart(fig_radar_riesgo(farm_data['NDVI_index'], fin['dscr'], farm_data['crop_disease_status']), use_container_width=True)

# ---------------------------------------------------------
# 10. Motor de Estrategia "What-If" (Estrategias de Reestructuración)
# ---------------------------------------------------------
if fin['sugerencia'] != "APROBACIÓN SUGERIDA":
    st.markdown("---")
    st.subheader("🛠️ Motor de Estrategias de Reestructuración (Estrategias 'What-If')")
    st.info("El crédito presenta fragilidad técnica o financiera. A continuación, el motor agéntico sugiere 3 opciones para renegociar con el cliente:")
    
    st1, st2, st3 = st.columns(3)
    with st1:
        st.warning("**Estrategia A: Ajuste de Capital**")
        cap_a = capital_req * 0.8
        st.write(f"• Reducir monto a: **${cap_a:,.0f} USD**")
        flujo_op = max(0, fin['ingreso_bruto'] - fin['opex_total'])
        st.write(f"• Eleva el DSCR a: **{(flujo_op / max(1, cap_a * 1.13)):.2f}x**")
    with st2:
        st.warning("**Estrategia B: Seguro Paramétrico Activado**")
        st.write("• Cobertura satelital contra sequía o plagas.")
        st.write("• Tasa ajustada preferencial: **-1.0%**")
    with st3:
        st.warning("**Estrategia C: Compromiso de Cosecha Futura**")
        st.write("• Vincular pago a contrato de compra directa (*Offtake Agreement*) con comprador certificado.")
        st.write("• Mitiga riesgo de precio.")

# ---------------------------------------------------------
# 11. MÓDULO ÉTICO DE GOBERNANZA: DECISIÓN DEL ANALISTA HUMANO
# ---------------------------------------------------------
st.markdown('<div class="human-decision-box">', unsafe_allow_html=True)
st.subheader("👨‍💼 Panel de Decisiones y Gobernanza (Human-in-the-Loop)")
st.write("El dictamen algorítmico es solo una recomendación técnica. **La decisión final sobre el otorgamiento del crédito corresponde únicamente al analista autorizado.**")

col_dec1, col_dec2 = st.columns([1, 2])

with col_dec1:
    decision_humana = st.radio(
        "Dictamen Final del Analista:",
        ["Aprobar Crédito Original", "Aprobar con Reestructuración (Estrategia A/B/C)", "Rechazar Solicitud"],
        index=1
    )

with col_dec2:
    justificacion = st.text_area("Justificación del Dictamen / Observaciones para la Ficha Técnica:", 
                                 placeholder="Escribe la motivación de tu dictamen (ej. 'Se aprueba condicionado a adquirir seguro paramétrico contra roya y ajustar el desembolso según la Estrategia A')...")
    
    if st.button("💾 Registrar Dictamen Final en Gobernanza", type="primary"):
        st.success(f"✅ Dictamen registrado oficialmente como: **{decision_humana}**. Expediente {farm_id} actualizado en base de datos auditada.")
st.markdown('</div>', unsafe_allow_html=True)
