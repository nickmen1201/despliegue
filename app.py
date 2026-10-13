"""
CardioRisk · Despliegue del modelo de clasificación de ataque al corazón.

Ejecutar con:  streamlit run app.py
Requiere en la misma carpeta: modelo-class.pkl (y opcionalmente modelo-info.json)
"""
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / "modelo-class.pkl"
INFO_PATH = BASE / "modelo-info.json"

VARIABLES_NUMERICAS = ["age", "avg_glucose_level"]
COLUMNAS_ENTRADA = ["age", "hypertension", "heart_disease", "ever_married",
                    "avg_glucose_level", "smoking_status"]
SMOKING_OPCIONES = {
    "Nunca ha fumado": "never smoked",
    "Exfumador": "formerly smoked",
    "Fumador actual": "smokes",
    "Desconocido": "Unknown",
}

st.set_page_config(page_title="CardioRisk · Predicción de ataque al corazón",
                   page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }

.hero {
  background: linear-gradient(120deg, #0b1f3a 0%, #16325c 55%, #b3213b 130%);
  border-radius: 20px; padding: 30px 36px; color: #fff; margin-bottom: 26px;
  box-shadow: 0 12px 30px rgba(11,31,58,.25); position: relative; overflow: hidden;
}
.hero:after { content: "🫀"; position: absolute; right: 34px; top: 12px; font-size: 96px; opacity: .18; }
.hero h1 { font-size: 2.05rem; font-weight: 800; margin: 0 0 6px 0; color: #fff; letter-spacing: -.5px; }
.hero p  { margin: 0; color: #d6e0f0; font-size: 1.02rem; max-width: 760px; }
.badge { display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
  padding: 3px 12px; border-radius: 999px; font-size: .75rem; font-weight: 600; letter-spacing: .6px;
  text-transform: uppercase; margin-bottom: 12px; }

.card { background: var(--card-bg, #ffffff); border: 1px solid rgba(128,140,160,.22); border-radius: 16px;
  padding: 22px 24px; box-shadow: 0 2px 10px rgba(16,24,40,.05); height: 100%; }
.card h3 { margin-top: 0; font-size: 1.05rem; font-weight: 700; }
.muted { color: #6b7a90; font-size: .9rem; }

.result { border-radius: 16px; padding: 22px 24px; color: #fff; margin-bottom: 14px; }
.result .label { font-size: .78rem; text-transform: uppercase; letter-spacing: 1px; opacity: .85; font-weight: 600; }
.result .value { font-size: 1.9rem; font-weight: 800; margin: 2px 0 4px 0; }
.result.alto     { background: linear-gradient(135deg, #b3213b, #e0485f); }
.result.moderado { background: linear-gradient(135deg, #c2620a, #f0a020); }
.result.bajo     { background: linear-gradient(135deg, #0f7a55, #23b07d); }

.factor { display: flex; align-items: center; gap: 10px; padding: 9px 12px; border-radius: 10px;
  margin-bottom: 7px; font-size: .93rem; background: rgba(128,140,160,.08); }
.dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.dot.r { background: #e0485f; } .dot.g { background: #23b07d; } .dot.a { background: #f0a020; }

.kpi { background: rgba(128,140,160,.08); border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; }
.kpi .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .8px; color: #6b7a90; font-weight: 600; }
.kpi .v { font-size: 1.35rem; font-weight: 800; }

.stButton > button, .stDownloadButton > button {
  background: linear-gradient(135deg, #b3213b, #e0485f); color: #fff; border: 0; border-radius: 12px;
  padding: .7rem 1.4rem; font-weight: 700; font-size: 1rem; width: 100%;
  box-shadow: 0 6px 16px rgba(179,33,59,.3); transition: transform .08s ease, box-shadow .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { color: #fff; transform: translateY(-1px);
  box-shadow: 0 10px 22px rgba(179,33,59,.38); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600; }
.disclaimer { font-size: .8rem; color: #6b7a90; border-top: 1px solid rgba(128,140,160,.25);
  padding-top: 14px; margin-top: 34px; }
@media (prefers-color-scheme: dark) { .card { --card-bg: #111827; } }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
@st.cache_resource
def cargar_modelo():
    with open(MODEL_PATH, "rb") as f:
        model, labelencoder, variables, min_max_scaler = pickle.load(f)
    info = json.loads(INFO_PATH.read_text(encoding="utf-8")) if INFO_PATH.exists() else {}
    return model, labelencoder, list(variables), min_max_scaler, info


def preparar(df: pd.DataFrame, variables, scaler) -> pd.DataFrame:
    """Aplica la misma preparación que en el notebook: limpieza, dummies, columnas y normalización."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include="object"):
        df[c] = df[c].astype(str).str.strip().str.strip("'\"")
    df[VARIABLES_NUMERICAS] = df[VARIABLES_NUMERICAS].apply(pd.to_numeric, errors="raise")
    X = pd.get_dummies(df[COLUMNAS_ENTRADA], dtype=int)
    X = X.reindex(columns=variables, fill_value=0)  # mismas columnas y orden que en entrenamiento
    X[VARIABLES_NUMERICAS] = scaler.transform(X[VARIABLES_NUMERICAS])
    return X


def predecir(df, model, labelencoder, variables, scaler, umbral):
    X = preparar(df, variables, scaler)
    idx_yes = list(labelencoder.classes_).index("Yes")
    prob = model.predict_proba(X)[:, idx_yes]
    pred = np.where(prob >= umbral, "Yes", "No")
    return prob, pred


def nivel_riesgo(p, umbral):
    if p >= max(umbral + 0.25, 0.75):
        return "alto", "Riesgo alto"
    if p >= umbral:
        return "moderado", "Riesgo moderado"
    return "bajo", "Riesgo bajo"


def gauge(p, umbral):
    color = {"alto": "#e0485f", "moderado": "#f0a020", "bajo": "#23b07d"}[nivel_riesgo(p, umbral)[0]]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 46}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9aa6b8"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [{"range": [0, umbral * 100], "color": "rgba(35,176,125,.15)"},
                      {"range": [umbral * 100, 100], "color": "rgba(224,72,95,.12)"}],
            "threshold": {"line": {"color": "#16325c", "width": 3}, "thickness": 0.8, "value": umbral * 100},
        }))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"})
    return fig


def factores(fila):
    out = []
    edad = fila["age"]
    out.append(("r" if edad >= 60 else "a" if edad >= 45 else "g", f"Edad: {edad} años"))
    out.append(("r" if fila["hypertension"] == "Yes" else "g",
                "Hipertensión" if fila["hypertension"] == "Yes" else "Sin hipertensión"))
    out.append(("r" if fila["heart_disease"] == "Yes" else "g",
                "Enfermedad cardíaca previa" if fila["heart_disease"] == "Yes" else "Sin enfermedad cardíaca previa"))
    g = fila["avg_glucose_level"]
    out.append(("r" if g >= 200 else "a" if g >= 126 else "g", f"Glucosa media: {g:.1f} mg/dL"))
    fum = {v: k for k, v in SMOKING_OPCIONES.items()}.get(fila["smoking_status"], fila["smoking_status"])
    out.append(("r" if fila["smoking_status"] == "smokes" else "a" if fila["smoking_status"] == "formerly smoked" else "g",
                f"Tabaquismo: {fum}"))
    return out


# --------------------------------------------------------------------------- #
# Interfaz
# --------------------------------------------------------------------------- #
st.markdown("""
<div class="hero">
  <span class="badge">Modelo predictivo · Machine Learning</span>
  <h1>CardioRisk</h1>
  <p>Estimación del riesgo de ataque al corazón a partir de variables clínicas básicas.
     Evalúe un paciente individual o cargue un archivo con varios pacientes y obtenga la predicción al instante.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("No se encontró **modelo-class.pkl**. Ejecute primero el notebook de entrenamiento para generarlo.")
    st.stop()

model, labelencoder, variables, scaler, info = cargar_modelo()
umbral_def = float(info.get("umbral", 0.5))

with st.sidebar:
    st.markdown("### ⚙️ Modelo en producción")
    st.markdown(f"""
    <div class="kpi"><div class="k">Algoritmo</div><div class="v" style="font-size:1.05rem">{info.get('modelo', type(model).__name__)}</div></div>
    """, unsafe_allow_html=True)
    met = info.get("metricas_test", {})
    if met:
        c1, c2 = st.columns(2)
        for i, (k, lab) in enumerate([("auc", "AUC ROC"), ("recall_yes", "Sensibilidad"),
                                       ("accuracy", "Exactitud"), ("balanced_accuracy", "Exact. balanceada")]):
            if k in met:
                (c1 if i % 2 == 0 else c2).markdown(
                    f'<div class="kpi"><div class="k">{lab}</div><div class="v">{met[k]*100:.0f}%</div></div>',
                    unsafe_allow_html=True)
        st.caption("Métricas sobre el 30% de prueba (datos no vistos en el entrenamiento).")
    st.markdown("---")
    st.markdown("### 🎚️ Umbral de decisión")
    umbral = st.slider("Probabilidad a partir de la cual se clasifica como *riesgo*",
                       0.05, 0.95, umbral_def, 0.05)
    st.caption("Un umbral menor detecta más casos (más sensibilidad) a costa de más falsas alarmas.")

tab1, tab2, tab3 = st.tabs(["🩺  Paciente individual", "📂  Carga masiva (CSV / Excel)", "ℹ️  Acerca del modelo"])

# ------------------------------- Individual -------------------------------- #
with tab1:
    col_form, col_res = st.columns([1.05, 1], gap="large")
    with col_form:
        st.markdown("#### Datos del paciente")
        with st.form("form_paciente", border=True):
            c1, c2 = st.columns(2)
            age = c1.number_input("Edad (años)", min_value=0, max_value=120, value=55, step=1)
            glucosa = c2.number_input("Glucosa media en sangre (mg/dL)", min_value=40.0, max_value=400.0,
                                      value=105.0, step=0.5, format="%.1f")
            c3, c4 = st.columns(2)
            hip = c3.radio("¿Hipertensión?", ["No", "Sí"], horizontal=True)
            car = c4.radio("¿Enfermedad cardíaca previa?", ["No", "Sí"], horizontal=True)
            c5, c6 = st.columns(2)
            cas = c5.radio("¿Alguna vez casado/a?", ["No", "Sí"], horizontal=True, index=1)
            fum = c6.selectbox("Hábito de fumar", list(SMOKING_OPCIONES.keys()))
            enviar = st.form_submit_button("Calcular riesgo", width="stretch")

    with col_res:
        st.markdown("#### Resultado")
        if enviar:
            yn = lambda v: "Yes" if v == "Sí" else "No"
            paciente = pd.DataFrame([{
                "age": int(age), "hypertension": yn(hip), "heart_disease": yn(car),
                "ever_married": yn(cas), "avg_glucose_level": float(glucosa),
                "smoking_status": SMOKING_OPCIONES[fum],
            }])
            prob, pred = predecir(paciente, model, labelencoder, variables, scaler, umbral)
            p = float(prob[0])
            clase, texto = nivel_riesgo(p, umbral)
            mensaje = ("El modelo clasifica al paciente en el grupo de <b>riesgo de ataque al corazón</b>. "
                       "Se recomienda valoración médica." if pred[0] == "Yes" else
                       "El modelo <b>no</b> clasifica al paciente en el grupo de riesgo.")
            st.markdown(f"""<div class="result {clase}"><div class="label">Predicción</div>
                <div class="value">{texto}</div><div>{mensaje}</div></div>""", unsafe_allow_html=True)
            st.plotly_chart(gauge(p, umbral), width="stretch", config={"displayModeBar": False})
            st.markdown("**Factores del paciente**")
            st.markdown("".join(f'<div class="factor"><span class="dot {c}"></span>{t}</div>'
                                for c, t in factores(paciente.iloc[0])), unsafe_allow_html=True)
        else:
            st.markdown("""<div class="card"><h3>Esperando datos</h3>
              <p class="muted">Complete el formulario y pulse <b>Calcular riesgo</b>.
              El resultado aparecerá aquí con la probabilidad estimada y los factores relevantes.</p></div>""",
                        unsafe_allow_html=True)

# --------------------------------- Masivo ---------------------------------- #
with tab2:
    st.markdown("#### Predicción por lotes")
    st.markdown('<p class="muted">Suba un archivo <b>.csv</b> o <b>.xlsx</b> con una fila por paciente y las columnas: '
                '<code>age, hypertension, heart_disease, ever_married, avg_glucose_level, smoking_status</code>. '
                'Valores categóricos: <code>Yes/No</code> y <code>never smoked / formerly smoked / smokes / Unknown</code>.</p>',
                unsafe_allow_html=True)

    plantilla = pd.DataFrame([
        {"age": 81, "hypertension": "No", "heart_disease": "No", "ever_married": "Yes",
         "avg_glucose_level": 81.95, "smoking_status": "never smoked"},
        {"age": 60, "hypertension": "Yes", "heart_disease": "No", "ever_married": "Yes",
         "avg_glucose_level": 213.03, "smoking_status": "smokes"},
    ])
    cA, cB = st.columns([3, 1])
    archivo = cA.file_uploader("Archivo de pacientes", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    cB.download_button("⬇ Plantilla CSV", plantilla.to_csv(index=False).encode("utf-8"),
                       "plantilla_pacientes.csv", "text/csv", width="stretch")

    if archivo is not None:
        try:
            if archivo.name.lower().endswith(".csv"):
                datos = pd.read_csv(archivo, sep=None, engine="python")
            else:
                xls = pd.ExcelFile(archivo)
                hoja = st.selectbox("Hoja del Excel", xls.sheet_names,
                                    index=len(xls.sheet_names) - 1 if len(xls.sheet_names) > 1 else 0)
                datos = pd.read_excel(xls, hoja)
            datos.columns = [str(c).strip() for c in datos.columns]
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            st.stop()

        faltan = [c for c in COLUMNAS_ENTRADA if c not in datos.columns]
        if faltan:
            st.error(f"Faltan columnas obligatorias: **{', '.join(faltan)}**")
        else:
            st.markdown(f"Vista previa · **{len(datos)}** pacientes")
            st.dataframe(datos.head(10), width="stretch", hide_index=True)
            if st.button("Generar predicciones"):
                try:
                    prob, pred = predecir(datos, model, labelencoder, variables, scaler, umbral)
                except Exception as e:
                    st.error(f"Error al procesar los datos: {e}")
                    st.stop()
                res = datos.copy()
                res["probabilidad_riesgo"] = prob.round(4)
                res["prediccion"] = pred
                res["nivel_riesgo"] = [nivel_riesgo(p, umbral)[1] for p in prob]

                n_pos = int((pred == "Yes").sum())
                k1, k2, k3, k4 = st.columns(4)
                for col, k, v in [(k1, "Pacientes evaluados", f"{len(res)}"),
                                  (k2, "En riesgo", f"{n_pos}"),
                                  (k3, "% en riesgo", f"{n_pos / len(res) * 100:.1f}%"),
                                  (k4, "Prob. media", f"{prob.mean() * 100:.1f}%")]:
                    col.markdown(f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div></div>',
                                 unsafe_allow_html=True)

                def color_fila(v):
                    return {"Riesgo alto": "background-color: rgba(224,72,95,.18)",
                            "Riesgo moderado": "background-color: rgba(240,160,32,.18)",
                            "Riesgo bajo": "background-color: rgba(35,176,125,.14)"}.get(v, "")

                st.dataframe(
                    res.style.map(color_fila, subset=["nivel_riesgo"])
                       .format({"probabilidad_riesgo": "{:.1%}", "avg_glucose_level": "{:.2f}"}),
                    width="stretch", hide_index=True)

                g1, g2 = st.columns([1.4, 1])
                fig_h = go.Figure(go.Histogram(x=prob * 100, nbinsx=20, marker_color="#16325c"))
                fig_h.add_vline(x=umbral * 100, line_dash="dash", line_color="#e0485f",
                                annotation_text="umbral", annotation_position="top")
                fig_h.update_layout(title="Distribución de la probabilidad de riesgo", height=320,
                                    xaxis_title="Probabilidad (%)", yaxis_title="Pacientes",
                                    margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"})
                g1.plotly_chart(fig_h, width="stretch", config={"displayModeBar": False})
                cnt = res["nivel_riesgo"].value_counts().reindex(
                    ["Riesgo bajo", "Riesgo moderado", "Riesgo alto"], fill_value=0)
                fig_p = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=.6,
                                         marker_colors=["#23b07d", "#f0a020", "#e0485f"], sort=False))
                fig_p.update_layout(title="Pacientes por nivel", height=320, margin=dict(l=10, r=10, t=50, b=10),
                                    paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"},
                                    legend=dict(orientation="h", y=-0.1))
                g2.plotly_chart(fig_p, width="stretch", config={"displayModeBar": False})

                buf = io.BytesIO()
                res.to_excel(buf, index=False, sheet_name="Predicciones")
                d1, d2, _ = st.columns([1, 1, 2])
                d1.download_button("⬇ Descargar Excel", buf.getvalue(), "predicciones_cardiorisk.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   width="stretch")
                d2.download_button("⬇ Descargar CSV", res.to_csv(index=False).encode("utf-8"),
                                   "predicciones_cardiorisk.csv", "text/csv", width="stretch")

# --------------------------------- Acerca ---------------------------------- #
with tab3:
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown(f"""<div class="card"><h3>¿Cómo funciona?</h3>
        <p class="muted">El modelo (<b>{info.get('modelo', type(model).__name__)}</b>) se entrenó con
        {info.get('n_registros', '5.110')} registros clínicos, de los cuales solo ~5% corresponden a pacientes
        que sufrieron un ataque. Para que el modelo no ignore la clase minoritaria se entrenó con
        <b>pesos de clase balanceados</b>, priorizando la detección de casos de riesgo (sensibilidad).</p>
        <p class="muted">Pasos aplicados a los datos de entrada: limpieza de texto → variables dummy →
        normalización Min-Max de edad y glucosa → predicción de probabilidad → comparación con el umbral.</p>
        </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown("""<div class="card"><h3>Variables de entrada</h3>
        <div class="factor"><b>age</b>&nbsp;· edad en años</div>
        <div class="factor"><b>hypertension</b>&nbsp;· Yes / No</div>
        <div class="factor"><b>heart_disease</b>&nbsp;· Yes / No</div>
        <div class="factor"><b>ever_married</b>&nbsp;· Yes / No</div>
        <div class="factor"><b>avg_glucose_level</b>&nbsp;· glucosa media (mg/dL)</div>
        <div class="factor"><b>smoking_status</b>&nbsp;· never smoked / formerly smoked / smokes / Unknown</div>
        </div>""", unsafe_allow_html=True)
    st.info("La probabilidad mostrada es una **puntuación de riesgo relativa**: al entrenar con clases balanceadas "
            "tiende a sobreestimar la probabilidad real, por lo que debe interpretarse comparativamente.")

st.markdown("""<div class="disclaimer">⚕️ Herramienta con fines académicos y de apoyo a la decisión.
No sustituye el diagnóstico de un profesional de la salud.</div>""", unsafe_allow_html=True)
