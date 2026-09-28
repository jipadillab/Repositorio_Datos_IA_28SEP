"""
Panel interactivo de análisis del rendimiento académico (maestro.csv)

Ejecutar con:
    pip install -r requirements.txt
    streamlit run main_app.py

Columnas esperadas en el CSV:
    id_estudiante, facultad, horario_estudio, participacion_clase, asistencia_pct,
    horas_biblioteca_virtual, clics_material_extra, uso_tutorias,
    nota_corte1, nota_corte2, nota_final, comentario_retro
"""

import io
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# =========================================================================== #
# Configuración general
# =========================================================================== #
st.set_page_config(page_title="Análisis académico", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

DEFAULT_CSV = Path(__file__).parent / "maestro.csv"

COLUMNAS_REQUERIDAS = [
    "id_estudiante", "facultad", "horario_estudio", "participacion_clase",
    "asistencia_pct", "horas_biblioteca_virtual", "clics_material_extra",
    "uso_tutorias", "nota_corte1", "nota_corte2", "nota_final", "comentario_retro",
]
NUMERICAS = ["asistencia_pct", "horas_biblioteca_virtual", "clics_material_extra",
             "nota_corte1", "nota_corte2", "nota_final"]
CATEGORICAS = ["facultad", "horario_estudio", "participacion_clase", "uso_tutorias"]
CATEGORICAS_EXT = CATEGORICAS + ["estado", "tema_comentario"]
NUMERICAS_EXT = NUMERICAS + ["variacion_nota"]
ORDEN_PARTICIPACION = ["Nula", "Baja", "Media", "Alta"]
ORDEN_HORARIO = ["Mañana", "Tarde", "Noche"]

ETIQUETAS = {
    "id_estudiante": "ID estudiante",
    "facultad": "Facultad",
    "horario_estudio": "Horario de estudio",
    "participacion_clase": "Participación en clase",
    "uso_tutorias": "Uso de tutorías",
    "asistencia_pct": "Asistencia (%)",
    "horas_biblioteca_virtual": "Horas en biblioteca virtual",
    "clics_material_extra": "Clics en material extra",
    "nota_corte1": "Nota corte 1",
    "nota_corte2": "Nota corte 2",
    "nota_final": "Nota final",
    "variacion_nota": "Variación (final − corte 1)",
    "comentario_retro": "Comentario",
    "estado": "Estado (aprobado/reprobado)",
    "tema_comentario": "Tema del comentario",
}

PALETAS = {
    "Vivo": px.colors.qualitative.Vivid,
    "Clásica": px.colors.qualitative.Plotly,
    "Suave": px.colors.qualitative.Pastel,
    "Sobria": px.colors.qualitative.Safe,
    "Oscura": px.colors.qualitative.Dark24,
    "Daltónico": px.colors.qualitative.Bold,
}
ESCALAS = ["Blues", "Viridis", "Plasma", "Teal", "Oranges", "Purples"]
PLANTILLAS = {"Claro": "plotly_white", "Oscuro": "plotly_dark", "Limpio": "simple_white",
              "Clásico": "plotly"}

TEMAS_COMENTARIOS = {
    "Barrera: trabajo en grupo": ["en grupo", "grupo de trabajo", "grupal",
                                  "compañeros", "resto del grupo"],
    "Barrera: conectividad / tecnología": ["internet", "conexión", "conectarme",
                                           "computador", "plataforma", "señal", "app "],
    "Barrera: tiempo / trabajo / familia": ["trabajo", "laborales", "tiempo", "familiar",
                                            "lejos", "desplazamiento", "horarios"],
}
TEMA_POSITIVO = "Valoración positiva / neutral"

STOPWORDS = set("""a al algo ante como con contra cual cuando de del desde donde durante e el ella
ellos en entre era eran es esa ese eso esta estas este esto fue fueron ha han hasta la las le les lo
los me mi mis mucho muy no nos o para pero por porque que se si sin sobre su sus te tuve un una uno
unas unos y ya yo tuvo varias varios fue ser estuvo siempre bien más mas todo toda también""".split())

PAGINAS = {
    "🏠 Resumen": "resumen",
    "🗂️ Explorador de datos": "explorador",
    "🧑‍🎓 Ficha de estudiante": "ficha",
    "👥 Comparar grupos": "grupos",
    "🔗 Relaciones": "relaciones",
    "📈 Evolución de notas": "evolucion",
    "🌲 Factores y riesgo": "factores",
    "💬 Comentarios": "comentarios",
    "🛠️ Constructor de gráficos": "constructor",
    "❓ Ayuda": "ayuda",
}


# =========================================================================== #
# Carga y preparación de datos
# =========================================================================== #
def _leer_csv(contenido: bytes) -> pd.DataFrame:
    ultimo = None
    for enc in ("utf-8-sig", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(io.BytesIO(contenido), encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception as e:  # noqa: BLE001
                ultimo = e
    raise ValueError(f"No se pudo leer el archivo: {ultimo}")


def clasificar_comentario(texto: str) -> str:
    t = f" {str(texto).lower()} "
    for tema, palabras in TEMAS_COMENTARIOS.items():
        if any(p in t for p in palabras):
            return tema
    return TEMA_POSITIVO


@st.cache_data(show_spinner=False)
def cargar_datos(contenido: bytes) -> pd.DataFrame:
    df = _leer_csv(contenido)
    df.columns = [c.strip() for c in df.columns]
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el archivo: {', '.join(faltantes)}")
    for c in NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAS:
        df[c] = df[c].astype(str).str.strip()
    df["participacion_clase"] = pd.Categorical(df["participacion_clase"],
                                               ORDEN_PARTICIPACION, ordered=True)
    df["horario_estudio"] = pd.Categorical(df["horario_estudio"], ORDEN_HORARIO,
                                           ordered=True)
    df["variacion_nota"] = df["nota_final"] - df["nota_corte1"]
    df["tema_comentario"] = df["comentario_retro"].apply(clasificar_comentario)
    return df


# =========================================================================== #
# Utilidades
# =========================================================================== #
def et(col) -> str:
    return ETIQUETAS.get(col, col) if col is not None else "—"


def orden_de(cat):
    if cat == "participacion_clase":
        return ORDEN_PARTICIPACION
    if cat == "horario_estudio":
        return ORDEN_HORARIO
    return None


def paleta():
    return PALETAS[st.session_state.get("ap_paleta", "Vivo")]


def escala():
    return st.session_state.get("ap_escala", "Blues")


def grafico(fig, key=None):
    """Aplica la apariencia elegida en el panel lateral y muestra el gráfico."""
    fig.update_layout(
        template=PLANTILLAS[st.session_state.get("ap_plantilla", "Claro")],
        height=st.session_state.get("ap_altura", 420),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, width="stretch", key=key)


def seg(label, opciones, key, default=None, fmt=None):
    """Control segmentado (botones) que nunca devuelve None."""
    kwargs = dict(options=opciones, default=default or opciones[0], key=key)
    if fmt:
        kwargs["format_func"] = fmt
    val = st.segmented_control(label, **kwargs)
    return val if val is not None else (default or opciones[0])


def prueba_grupos(df, cat, num):
    grupos = [g[num].dropna().values for _, g in df.groupby(cat, observed=True)]
    grupos = [g for g in grupos if len(g) >= 2]
    if len(grupos) < 2:
        return None
    if len(grupos) == 2:
        s, p = stats.ttest_ind(grupos[0], grupos[1], equal_var=False)
        return "t de Welch", s, p
    s, p = stats.f_oneway(*grupos)
    return "ANOVA", s, p


def interpreta_p(p):
    if p < 0.001:
        return "diferencia estadísticamente muy significativa (p < 0.001)"
    if p < 0.05:
        return f"diferencia estadísticamente significativa (p = {p:.3f})"
    return f"sin evidencia de diferencia significativa (p = {p:.3f})"


def fuerza_corr(r):
    a = abs(r)
    return "nula" if a < 0.1 else "débil" if a < 0.3 else "moderada" if a < 0.5 else "fuerte"


def nota_estadistica(texto):
    if st.session_state.get("avanzado", True):
        st.info(texto)


def csv_bytes(d):
    return d.to_csv(index=False).encode("utf-8-sig")


# =========================================================================== #
# Panel izquierdo
# =========================================================================== #
st.sidebar.title("🎓 Análisis académico")

# ---- 1) Navegación -------------------------------------------------------- #
pagina_sel = st.sidebar.radio("📍 Ir a", list(PAGINAS.keys()), key="nav")

# ---- 2) Datos ------------------------------------------------------------- #
with st.sidebar.expander("📁 Datos", expanded=False):
    archivo = st.file_uploader("Cargar otro CSV", type=["csv"], key="uploader")
    if st.button("🔄 Recargar datos", width="stretch"):
        st.cache_data.clear()
        st.rerun()

try:
    if archivo is not None:
        contenido, origen = archivo.getvalue(), archivo.name
    elif DEFAULT_CSV.exists():
        contenido, origen = DEFAULT_CSV.read_bytes(), DEFAULT_CSV.name
    else:
        st.warning("No se encontró `maestro.csv` junto a la aplicación. "
                   "Ábrelo desde 📁 Datos en el panel izquierdo.")
        st.stop()
    data = cargar_datos(contenido)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.sidebar.caption(f"Archivo: **{origen}** · {len(data)} filas")

# ---- 3) Herramientas globales -------------------------------------------- #
with st.sidebar.expander("⚙️ Opciones de análisis", expanded=False):
    umbral = st.slider("Nota mínima para aprobar", 0.0, 5.0, 3.0, 0.1, key="umbral")
    st.toggle("Mostrar pruebas estadísticas", value=True, key="avanzado",
              help="Muestra interpretaciones (ANOVA, t, correlaciones) bajo cada gráfico.")
    st.toggle("Mostrar valores sobre las barras", value=True, key="ap_valores")

with st.sidebar.expander("🎨 Apariencia de gráficos", expanded=False):
    st.selectbox("Tema", list(PLANTILLAS), key="ap_plantilla")
    st.selectbox("Paleta de colores", list(PALETAS), key="ap_paleta")
    st.selectbox("Escala continua", ESCALAS, key="ap_escala")
    st.slider("Altura de gráficos (px)", 300, 800, 420, 20, key="ap_altura")

# ---- 4) Filtros ------------------------------------------------------------ #
FAC = sorted(data["facultad"].unique())
TUT = sorted(data["uso_tutorias"].unique())
A_MIN, A_MAX = float(data["asistencia_pct"].min()), float(data["asistencia_pct"].max())
N_MIN, N_MAX = float(data["nota_final"].min()), float(data["nota_final"].max())


def reset_filtros():
    st.session_state["f_fac"] = list(FAC)
    st.session_state["f_hor"] = list(ORDEN_HORARIO)
    st.session_state["f_par"] = list(ORDEN_PARTICIPACION)
    st.session_state["f_tut"] = list(TUT)
    st.session_state["f_asi"] = (A_MIN, A_MAX)
    st.session_state["f_nota"] = (N_MIN, N_MAX)


def preset(nombre):
    reset_filtros()
    if nombre == "riesgo":
        st.session_state["f_nota"] = (N_MIN, max(N_MIN, round(st.session_state["umbral"] - 0.1, 1)))
    elif nombre == "destacados":
        st.session_state["f_nota"] = (max(N_MIN, 4.0), N_MAX)
    elif nombre == "tutorias":
        st.session_state["f_tut"] = ["Sí"] if "Sí" in TUT else list(TUT)
    elif nombre == "asistencia":
        st.session_state["f_asi"] = (A_MIN, min(A_MAX, 70.0))


firma = (origen, len(data))
if st.session_state.get("_firma") != firma or "f_fac" not in st.session_state:
    reset_filtros()
    st.session_state["_firma"] = firma

with st.sidebar.expander("🔍 Filtros", expanded=True):
    st.caption("Atajos rápidos")
    b1, b2 = st.columns(2)
    b1.button("⚠️ En riesgo", on_click=preset, args=("riesgo",), width="stretch")
    b2.button("🏆 Destacados", on_click=preset, args=("destacados",), width="stretch")
    b3, b4 = st.columns(2)
    b3.button("🧑‍🏫 Con tutorías", on_click=preset, args=("tutorias",), width="stretch")
    b4.button("📉 Baja asistencia", on_click=preset, args=("asistencia",), width="stretch")
    st.button("🧹 Limpiar filtros", on_click=reset_filtros, type="primary", width="stretch")
    st.divider()
    st.multiselect("Facultad", FAC, key="f_fac")
    st.multiselect("Horario de estudio", ORDEN_HORARIO, key="f_hor")
    st.multiselect("Participación en clase", ORDEN_PARTICIPACION, key="f_par")
    st.multiselect("Uso de tutorías", TUT, key="f_tut")
    st.slider("Asistencia (%)", A_MIN, A_MAX, key="f_asi")
    st.slider("Nota final", N_MIN, N_MAX, step=0.1, key="f_nota")

df = data[
    data["facultad"].isin(st.session_state["f_fac"])
    & data["horario_estudio"].isin(st.session_state["f_hor"])
    & data["participacion_clase"].isin(st.session_state["f_par"])
    & data["uso_tutorias"].isin(st.session_state["f_tut"])
    & data["asistencia_pct"].between(*st.session_state["f_asi"])
    & data["nota_final"].between(*st.session_state["f_nota"])
].copy()
df["aprobado"] = df["nota_final"] >= umbral
df["estado"] = np.where(df["aprobado"], "Aprobado", "Reprobado")

# ---- 5) Estado y descargas ------------------------------------------------- #
st.sidebar.divider()
st.sidebar.metric("Estudiantes en la selección", f"{len(df)} / {len(data)}",
                  delta=f"{len(df) / len(data) * 100:.0f}% del total", delta_color="off")
if len(df):
    st.sidebar.progress(float(df["aprobado"].mean()),
                        text=f"Aprobación: {df['aprobado'].mean() * 100:.1f}%")
    st.sidebar.download_button("⬇️ Descargar selección (CSV)", csv_bytes(df),
                               "seleccion.csv", "text/csv", width="stretch")

# =========================================================================== #
# Páginas
# =========================================================================== #


def pg_resumen():
    c = st.columns(5)
    c[0].metric("Estudiantes", len(df))
    c[1].metric("Nota final promedio", f"{df['nota_final'].mean():.2f}")
    c[2].metric("Aprobación", f"{df['aprobado'].mean() * 100:.1f}%")
    c[3].metric("Asistencia media", f"{df['asistencia_pct'].mean():.1f}%")
    c[4].metric("Usan tutorías", f"{(df['uso_tutorias'] == 'Sí').mean() * 100:.1f}%")

    st.divider()
    izq, der = st.columns(2)
    with izq:
        st.subheader("Distribución de una variable")
        var = st.selectbox("Variable", NUMERICAS_EXT, index=5, format_func=et, key="res_var")
        tipo = seg("Tipo de gráfico", ["Histograma", "Caja", "Violín", "Acumulada"], "res_tipo")
        if tipo == "Histograma":
            fig = px.histogram(df, x=var, nbins=20, marginal="box",
                               color_discrete_sequence=paleta(), labels={var: et(var)})
        elif tipo == "Caja":
            fig = px.box(df, y=var, points="all", color_discrete_sequence=paleta(),
                         labels={var: et(var)})
        elif tipo == "Violín":
            fig = px.violin(df, y=var, box=True, points="all",
                            color_discrete_sequence=paleta(), labels={var: et(var)})
        else:
            fig = px.ecdf(df, x=var, color_discrete_sequence=paleta(), labels={var: et(var)})
        if var.startswith("nota") and tipo in ("Histograma", "Acumulada"):
            fig.add_vline(x=umbral, line_dash="dash", line_color="red",
                          annotation_text=f"Umbral {umbral:.1f}")
        grafico(fig)
    with der:
        st.subheader("Composición")
        cat = st.selectbox("Categoría", CATEGORICAS_EXT, format_func=et, key="res_cat")
        forma = seg("Forma", ["Dona", "Barras", "Treemap"], "res_forma")
        conteo = df[cat].value_counts().reset_index()
        conteo.columns = [cat, "estudiantes"]
        if forma == "Dona":
            fig = px.pie(conteo, names=cat, values="estudiantes", hole=0.45,
                         color_discrete_sequence=paleta())
        elif forma == "Barras":
            fig = px.bar(conteo, x=cat, y="estudiantes",
                         text="estudiantes" if st.session_state["ap_valores"] else None,
                         color=cat, color_discrete_sequence=paleta(),
                         labels={cat: et(cat)})
            fig.update_layout(showlegend=False)
        else:
            fig = px.treemap(conteo, path=[cat], values="estudiantes",
                             color_discrete_sequence=paleta())
        grafico(fig)

    with st.expander("📋 Estadística descriptiva"):
        desc = df[NUMERICAS_EXT].describe().T.rename(index=ETIQUETAS)
        desc = desc.rename(columns={"count": "n", "mean": "media", "std": "desv.",
                                    "min": "mín", "max": "máx"})
        st.dataframe(desc.round(2), width="stretch")
    with st.expander("🩺 Calidad de los datos"):
        q = st.columns(3)
        q[0].metric("Valores nulos", int(df[COLUMNAS_REQUERIDAS].isna().sum().sum()))
        q[1].metric("IDs duplicados", int(df["id_estudiante"].duplicated().sum()))
        nn = df[["nota_corte1", "nota_corte2", "nota_final"]]
        q[2].metric("Notas fuera de 0–5", int(((nn < 0) | (nn > 5)).sum().sum()))


def pg_explorador():
    st.subheader("Explorador de datos")
    f1, f2 = st.columns([2, 3])
    buscar = f1.text_input("🔎 Buscar (ID o texto del comentario)", key="ex_buscar")
    default_cols = ["id_estudiante", "facultad", "horario_estudio", "participacion_clase",
                    "asistencia_pct", "uso_tutorias", "nota_corte1", "nota_corte2",
                    "nota_final", "estado"]
    todas = [c for c in df.columns if c != "aprobado"]
    cols = f2.multiselect("Columnas a mostrar", todas, default=default_cols,
                          format_func=et, key="ex_cols")
    g1, g2, g3 = st.columns([2, 1, 2])
    orden = g1.selectbox("Ordenar por", cols or todas, format_func=et, key="ex_orden")
    desc = g2.toggle("Descendente", value=True, key="ex_desc")
    top = g3.slider("Filas a mostrar", 5, max(len(df), 6), min(len(df), 50), key="ex_top")

    vista = df.copy()
    if buscar:
        m = (vista["id_estudiante"].str.contains(buscar, case=False, na=False)
             | vista["comentario_retro"].str.contains(buscar, case=False, na=False))
        vista = vista[m]
    if orden in vista.columns:
        vista = vista.sort_values(orden, ascending=not desc)
    vista = vista.head(top)
    mostrar = vista[cols] if cols else vista

    st.caption(f"{len(vista)} filas mostradas de {len(df)} en la selección")
    st.dataframe(
        mostrar, width="stretch", hide_index=True,
        column_config={
            "asistencia_pct": st.column_config.ProgressColumn(
                "Asistencia (%)", min_value=0, max_value=100, format="%.1f"),
            "nota_final": st.column_config.NumberColumn("Nota final", format="%.1f"),
            "nota_corte1": st.column_config.NumberColumn("Corte 1", format="%.1f"),
            "nota_corte2": st.column_config.NumberColumn("Corte 2", format="%.1f"),
            "id_estudiante": st.column_config.TextColumn("ID"),
        },
    )
    st.download_button("⬇️ Descargar esta vista (CSV)", csv_bytes(mostrar),
                       "vista.csv", "text/csv")


def pg_ficha():
    st.subheader("Ficha de estudiante")
    ids = sorted(df["id_estudiante"].unique())
    c1, c2 = st.columns([2, 3])
    sel = c1.selectbox("Estudiante", ids, key="fi_id")
    comparar = c2.selectbox("Comparar con el promedio de…",
                            ["Toda la selección", "Su facultad", "Su horario", "Su nivel de participación"],
                            key="fi_cmp")
    e = df[df["id_estudiante"] == sel].iloc[0]
    if comparar == "Su facultad":
        ref, nombre = data[data["facultad"] == e["facultad"]], f"Facultad {e['facultad']}"
    elif comparar == "Su horario":
        ref, nombre = data[data["horario_estudio"] == e["horario_estudio"]], f"Horario {e['horario_estudio']}"
    elif comparar == "Su nivel de participación":
        ref, nombre = (data[data["participacion_clase"] == e["participacion_clase"]],
                       f"Participación {e['participacion_clase']}")
    else:
        ref, nombre = df, "Selección"

    k = st.columns(5)
    k[0].metric("Facultad", e["facultad"])
    k[1].metric("Horario", str(e["horario_estudio"]))
    k[2].metric("Participación", str(e["participacion_clase"]))
    k[3].metric("Tutorías", e["uso_tutorias"])
    k[4].metric("Nota final", f"{e['nota_final']:.1f}",
                delta=f"{e['nota_final'] - ref['nota_final'].mean():+.2f} vs {nombre}")

    izq, der = st.columns(2)
    with izq:
        ejes = ["Asistencia", "Biblioteca virtual", "Material extra",
                "Corte 1", "Corte 2", "Nota final"]

        def norm(r):
            return [r["asistencia_pct"] / 100,
                    r["horas_biblioteca_virtual"] / max(data["horas_biblioteca_virtual"].max(), 1),
                    r["clics_material_extra"] / max(data["clics_material_extra"].max(), 1),
                    r["nota_corte1"] / 5, r["nota_corte2"] / 5, r["nota_final"] / 5]

        media = ref[NUMERICAS].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=norm(e) + [norm(e)[0]], theta=ejes + [ejes[0]],
                                      fill="toself", name=sel,
                                      line_color=paleta()[0]))
        fig.add_trace(go.Scatterpolar(r=norm(media) + [norm(media)[0]], theta=ejes + [ejes[0]],
                                      fill="toself", name=nombre, opacity=0.5,
                                      line_color=paleta()[1]))
        fig.update_layout(title="Perfil normalizado (0–1)",
                          polar=dict(radialaxis=dict(range=[0, 1])))
        grafico(fig)
    with der:
        tabla = pd.DataFrame({
            "Variable": [et(c) for c in NUMERICAS],
            "Estudiante": [e[c] for c in NUMERICAS],
            nombre: [ref[c].mean() for c in NUMERICAS],
        })
        tabla["Diferencia"] = tabla["Estudiante"] - tabla[nombre]
        st.dataframe(tabla.round(2), width="stretch", hide_index=True)
        st.markdown("**Comentario del estudiante**")
        st.info(e["comentario_retro"])
        st.caption(f"Tema detectado: {e['tema_comentario']}")


def pg_grupos():
    st.subheader("Comparar grupos")
    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Agrupar por", CATEGORICAS_EXT, format_func=et, key="gr_cat")
    num = c2.selectbox("Variable a comparar",
                       [c for c in NUMERICAS_EXT], index=5, format_func=et, key="gr_num")
    sub = c3.selectbox("Subdividir por (opcional)",
                       [None] + [c for c in CATEGORICAS_EXT if c != cat],
                       format_func=lambda c: "Sin subdivisión" if c is None else et(c),
                       key="gr_sub")
    tipo = seg("Tipo de gráfico", ["Caja", "Violín", "Barras (media)", "Puntos"], "gr_tipo")

    orden = {cat: orden_de(cat)} if orden_de(cat) else {}
    if sub and orden_de(sub):
        orden[sub] = orden_de(sub)
    color = sub or cat
    lab = {cat: et(cat), num: et(num), color: et(color)}
    if tipo == "Caja":
        fig = px.box(df, x=cat, y=num, color=color, points="all", category_orders=orden,
                     color_discrete_sequence=paleta(), labels=lab)
    elif tipo == "Violín":
        fig = px.violin(df, x=cat, y=num, color=color, box=True, category_orders=orden,
                        color_discrete_sequence=paleta(), labels=lab)
    elif tipo == "Puntos":
        fig = px.strip(df, x=cat, y=num, color=color, category_orders=orden,
                       color_discrete_sequence=paleta(), labels=lab)
    else:
        keys = [cat] + ([sub] if sub else [])
        g = df.groupby(keys, observed=True)[num].mean().reset_index()
        fig = px.bar(g, x=cat, y=num, color=color, barmode="group", category_orders=orden,
                     text_auto=".2f" if st.session_state["ap_valores"] else False,
                     color_discrete_sequence=paleta(), labels=lab)
    fig.update_layout(title=f"{et(num)} según {et(cat).lower()}")
    grafico(fig)

    resumen = (df.groupby(cat, observed=True)
                 .agg(n=("id_estudiante", "count"), media=(num, "mean"),
                      mediana=(num, "median"), desv=(num, "std"),
                      aprobacion=("aprobado", "mean")).reset_index())
    resumen["aprobacion"] = (resumen["aprobacion"] * 100).round(1)
    t1, t2 = st.columns([3, 2])
    with t1:
        st.dataframe(resumen.round(2).rename(columns={cat: et(cat), "desv": "desv.",
                                                      "aprobacion": "% aprobación"}),
                     width="stretch", hide_index=True)
        st.download_button("⬇️ Descargar tabla", csv_bytes(resumen), "comparacion.csv",
                           "text/csv")
    with t2:
        fig = px.bar(resumen, x=cat, y="aprobacion",
                     text="aprobacion" if st.session_state["ap_valores"] else None,
                     category_orders=orden, color=cat, color_discrete_sequence=paleta(),
                     labels={cat: et(cat), "aprobacion": "% aprobación"},
                     title="% de aprobación por grupo")
        fig.update_yaxes(range=[0, 105])
        fig.update_layout(showlegend=False)
        grafico(fig)

    res = prueba_grupos(df, cat, num)
    if res:
        nota_estadistica(f"**{res[0]}** (estadístico = {res[1]:.2f}): {interpreta_p(res[2])}.")

    if sub:
        with st.expander("🔥 Tabla cruzada (media)"):
            piv = df.pivot_table(index=cat, columns=sub, values=num, aggfunc="mean",
                                 observed=True)
            fig = px.imshow(piv.round(2), text_auto=".2f", aspect="auto",
                            color_continuous_scale=escala())
            grafico(fig)
    if cat == "uso_tutorias" and num.startswith("nota"):
        st.warning("⚠️ Quienes van a tutoría suelen ser quienes ya tienen dificultades: una "
                   "nota menor en ese grupo **no implica** que la tutoría perjudique. "
                   "Revisa *Evolución de notas* para ver si mejoran entre cortes.")


def pg_relaciones():
    st.subheader("Relaciones entre variables")
    vista = seg("Vista", ["Matriz de correlación", "Diagrama de dispersión"], "re_vista")

    if vista == "Matriz de correlación":
        c1, c2 = st.columns([1, 3])
        metodo = c1.radio("Método", ["pearson", "spearman"], key="re_metodo")
        vars_ = c2.multiselect("Variables", NUMERICAS_EXT, default=NUMERICAS,
                               format_func=et, key="re_vars")
        if len(vars_) < 2:
            st.info("Elige al menos dos variables.")
            return
        corr = df[vars_].corr(method=metodo)
        corr.index = [et(c) for c in corr.index]
        corr.columns = [et(c) for c in corr.columns]
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title=f"Correlación ({metodo.capitalize()})")
        grafico(fig)
        pares = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
                     .reset_index())
        pares.columns = ["Variable A", "Variable B", "r"]
        pares["|r|"] = pares["r"].abs()
        st.markdown("**Pares más correlacionados**")
        st.dataframe(pares.sort_values("|r|", ascending=False).head(5).drop(columns="|r|")
                     .round(3), width="stretch", hide_index=True)
    else:
        c = st.columns(4)
        x = c[0].selectbox("Eje X", NUMERICAS_EXT, index=0, format_func=et, key="re_x")
        y = c[1].selectbox("Eje Y", NUMERICAS_EXT, index=5, format_func=et, key="re_y")
        color = c[2].selectbox("Color", [None] + CATEGORICAS_EXT,
                               format_func=lambda v: "Sin color" if v is None else et(v),
                               key="re_col")
        tam = c[3].selectbox("Tamaño", [None] + NUMERICAS,
                             format_func=lambda v: "Fijo" if v is None else et(v), key="re_tam")
        o1, o2 = st.columns(2)
        tendencia = o1.toggle("Línea de tendencia", value=True, key="re_tend")
        marg = o2.toggle("Distribuciones en los márgenes", value=False, key="re_marg")

        fig = px.scatter(df, x=x, y=y, color=color, size=tam,
                         marginal_x="box" if marg else None,
                         marginal_y="box" if marg else None,
                         hover_data=["id_estudiante", "facultad"],
                         color_discrete_sequence=paleta(),
                         labels={x: et(x), y: et(y)})
        sub = df[[x, y]].dropna()
        if len(sub) > 2 and x != y:
            m, b = np.polyfit(sub[x], sub[y], 1)
            if tendencia:
                xs = np.linspace(sub[x].min(), sub[x].max(), 50)
                fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines",
                                         name="Tendencia", line=dict(color="black", dash="dash")),
                              row=1 if marg else None, col=1 if marg else None)
            grafico(fig)
            r, p = stats.pearsonr(sub[x], sub[y])
            rho, _ = stats.spearmanr(sub[x], sub[y])
            nota_estadistica(f"Pearson **r = {r:.2f}** ({fuerza_corr(r)}), Spearman **ρ = {rho:.2f}**; "
                             f"{interpreta_p(p)}. Pendiente: {m:.3f} de «{et(y)}» por unidad de «{et(x)}».")
        else:
            grafico(fig)


def pg_evolucion():
    st.subheader("Evolución de notas")
    k = st.columns(4)
    k[0].metric("Corte 1", f"{df['nota_corte1'].mean():.2f}")
    k[1].metric("Corte 2", f"{df['nota_corte2'].mean():.2f}",
                delta=f"{df['nota_corte2'].mean() - df['nota_corte1'].mean():+.2f}")
    k[2].metric("Final", f"{df['nota_final'].mean():.2f}",
                delta=f"{df['nota_final'].mean() - df['nota_corte2'].mean():+.2f}")
    k[3].metric("Mejoraron", f"{(df['variacion_nota'] > 0).mean() * 100:.1f}%")

    c1, c2 = st.columns([2, 3])
    dim = c1.selectbox("Ver por", CATEGORICAS_EXT, format_func=et, key="ev_dim")
    vista = c2.radio("Vista", ["Promedio por grupo", "Trayectorias individuales",
                               "Distribución del cambio"], horizontal=True, key="ev_vista")
    momentos = {"nota_corte1": "Corte 1", "nota_corte2": "Corte 2", "nota_final": "Final"}

    if vista == "Promedio por grupo":
        evo = (df.groupby(dim, observed=True)[list(momentos)].mean().reset_index()
                 .melt(id_vars=dim, var_name="momento", value_name="nota"))
        evo["momento"] = evo["momento"].map(momentos)
        fig = px.line(evo, x="momento", y="nota", color=dim, markers=True,
                      category_orders={"momento": list(momentos.values())},
                      color_discrete_sequence=paleta(),
                      labels={"nota": "Nota promedio", "momento": "", dim: et(dim)})
        grafico(fig)
    elif vista == "Trayectorias individuales":
        n = st.slider("Estudiantes a dibujar", 5, min(100, len(df)) if len(df) > 5 else 6,
                      min(30, len(df)), key="ev_n")
        muestra = df.sample(min(n, len(df)), random_state=1)
        largo = muestra.melt(id_vars=["id_estudiante", dim], value_vars=list(momentos),
                             var_name="momento", value_name="nota")
        largo["momento"] = largo["momento"].map(momentos)
        fig = px.line(largo, x="momento", y="nota", color=dim, line_group="id_estudiante",
                      hover_name="id_estudiante", markers=True,
                      category_orders={"momento": list(momentos.values())},
                      color_discrete_sequence=paleta())
        grafico(fig)
    else:
        fig = px.histogram(df, x="variacion_nota", nbins=25, marginal="box", color=dim,
                           barmode="overlay", opacity=0.7, color_discrete_sequence=paleta(),
                           labels={"variacion_nota": et("variacion_nota"), dim: et(dim)})
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        grafico(fig)

    var_g = (df.groupby(dim, observed=True)["variacion_nota"].agg(["count", "mean", "std"])
               .reset_index().rename(columns={dim: et(dim), "count": "n",
                                              "mean": "variación media", "std": "desv."}))
    st.dataframe(var_g.round(3), width="stretch", hide_index=True)
    r = prueba_grupos(df, dim, "variacion_nota")
    if r:
        nota_estadistica(f"{r[0]} sobre la variación de nota: {interpreta_p(r[2])}.")
    if len(df) > 2:
        _, p = stats.ttest_rel(df["nota_final"], df["nota_corte1"], nan_policy="omit")
        nota_estadistica(f"Prueba t pareada final vs corte 1: {interpreta_p(p)} "
                         f"(cambio medio {df['variacion_nota'].mean():+.3f}).")


def pg_factores():
    st.subheader("Factores asociados y estudiantes en riesgo")
    modo = seg("Herramienta", ["🌲 Importancia de variables", "⚠️ Estudiantes en riesgo"],
               "fa_modo")

    if modo.startswith("🌲"):
        st.caption("Análisis exploratorio: la importancia indica asociación, no causalidad.")
        c1, c2, c3 = st.columns(3)
        objetivo = c1.selectbox("Variable a explicar", ["nota_final", "nota_corte2", "nota_corte1"],
                                format_func=et, key="fa_obj")
        usar_cortes = c2.checkbox("Incluir notas previas como predictores", value=False,
                                  key="fa_cortes",
                                  help="Si se incluyen dominan el modelo.")
        n_arb = c3.slider("Árboles del modelo", 50, 500, 200, 50, key="fa_arboles")

        feats = ["asistencia_pct", "horas_biblioteca_virtual", "clics_material_extra"]
        if usar_cortes:
            feats += [c for c in ["nota_corte1", "nota_corte2"] if c != objetivo]
        X = pd.get_dummies(df[feats + CATEGORICAS].astype({c: str for c in CATEGORICAS}),
                           columns=CATEGORICAS, dtype=float)
        y = df[objetivo]
        if len(df) < 30:
            st.warning("Se necesitan al menos 30 estudiantes en la selección.")
            return
        rf = RandomForestRegressor(n_estimators=n_arb, random_state=42, min_samples_leaf=3,
                                   n_jobs=-1)
        k = 5 if len(df) >= 50 else 3
        r2 = cross_val_score(rf, X, y, cv=k, scoring="r2")
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().reset_index()
        imp.columns = ["variable", "importancia"]
        imp["variable"] = imp["variable"].map(
            lambda v: et(v) if v in ETIQUETAS else v.replace("_", ": ", 1))
        m1, m2 = st.columns([2, 1])
        with m1:
            fig = px.bar(imp, x="importancia", y="variable", orientation="h",
                         color_discrete_sequence=paleta(), title="Importancia de variables")
            grafico(fig)
        with m2:
            st.metric("R² (validación cruzada)", f"{r2.mean():.2f}",
                      help=f"Promedio de {k} particiones ±{r2.std():.2f}")
            st.caption("Poder explicativo " + ("bajo." if r2.mean() < 0.2 else
                                               "moderado." if r2.mean() < 0.5 else "alto."))
            st.download_button("⬇️ Descargar importancias", csv_bytes(imp), "importancias.csv",
                               "text/csv")
    else:
        c1, c2 = st.columns(2)
        criterio = c1.selectbox("Criterio de riesgo",
                                ["Nota final bajo el umbral", "Nota corte 2 bajo el umbral",
                                 "Nota corte 1 bajo el umbral", "Baja asistencia",
                                 "Empeoró respecto al corte 1"], key="fa_crit")
        lim_asis = c2.slider("Asistencia mínima (%)", 40, 100, 70, key="fa_asis",
                             disabled=criterio != "Baja asistencia")
        if criterio.startswith("Nota final"):
            mask = df["nota_final"] < umbral
        elif criterio.startswith("Nota corte 2"):
            mask = df["nota_corte2"] < umbral
        elif criterio.startswith("Nota corte 1"):
            mask = df["nota_corte1"] < umbral
        elif criterio == "Baja asistencia":
            mask = df["asistencia_pct"] < lim_asis
        else:
            mask = df["variacion_nota"] < 0
        riesgo = df[mask]
        st.metric("Estudiantes en riesgo", len(riesgo),
                  delta=f"{len(riesgo) / len(df) * 100:.1f}% de la selección", delta_color="off")
        if riesgo.empty:
            st.success("Ningún estudiante cumple el criterio en la selección actual.")
            return
        cols = ["id_estudiante", "facultad", "horario_estudio", "participacion_clase",
                "asistencia_pct", "uso_tutorias", "nota_corte1", "nota_corte2", "nota_final",
                "tema_comentario"]
        st.dataframe(riesgo[cols].sort_values("nota_final"), width="stretch", hide_index=True,
                     column_config={"asistencia_pct": st.column_config.ProgressColumn(
                         "Asistencia (%)", min_value=0, max_value=100, format="%.1f")})
        st.download_button("⬇️ Descargar lista de riesgo", csv_bytes(riesgo[cols]),
                           "estudiantes_en_riesgo.csv", "text/csv")
        with st.expander("📊 Perfil: en riesgo vs. resto"):
            comp = pd.DataFrame({
                "En riesgo": riesgo[NUMERICAS].mean(),
                "Resto": df[~mask][NUMERICAS].mean() if (~mask).any() else np.nan,
            })
            comp.index = [et(i) for i in comp.index]
            st.dataframe(comp.round(2), width="stretch")
            cat = st.selectbox("Distribución por", CATEGORICAS, format_func=et, key="fa_dist")
            g = (df.assign(riesgo=np.where(mask, "En riesgo", "Resto"))
                   .groupby([cat, "riesgo"], observed=True).size().reset_index(name="n"))
            fig = px.bar(g, x=cat, y="n", color="riesgo", barmode="group",
                         color_discrete_sequence=paleta(), labels={cat: et(cat)})
            grafico(fig)


def pg_comentarios():
    st.subheader("Comentarios de retroalimentación")
    st.caption("Clasificación por reglas de palabras clave: es una aproximación, revisa los textos.")
    vista = seg("Vista", ["Temas", "Palabras frecuentes", "Explorar textos"], "co_vista")

    if vista == "Temas":
        temas = (df.groupby("tema_comentario")
                   .agg(estudiantes=("id_estudiante", "count"), nota_media=("nota_final", "mean"),
                        aprobacion=("aprobado", "mean"), asistencia=("asistencia_pct", "mean"))
                   .reset_index().sort_values("estudiantes", ascending=False))
        temas["aprobacion"] = (temas["aprobacion"] * 100).round(1)
        k1, k2 = st.columns(2)
        with k1:
            fig = px.bar(temas, x="tema_comentario", y="estudiantes", color="tema_comentario",
                         text="estudiantes" if st.session_state["ap_valores"] else None,
                         color_discrete_sequence=paleta(),
                         labels={"tema_comentario": "", "estudiantes": "Estudiantes"},
                         title="Comentarios por tema")
            fig.update_layout(showlegend=False)
            grafico(fig)
        with k2:
            fig = px.box(df, x="tema_comentario", y="nota_final", color="tema_comentario",
                         points="all", color_discrete_sequence=paleta(),
                         labels={"tema_comentario": "", "nota_final": et("nota_final")},
                         title="Nota final según tema")
            fig.update_layout(showlegend=False)
            grafico(fig)
        st.dataframe(temas.round(2).rename(columns={
            "tema_comentario": "Tema", "nota_media": "nota final media",
            "aprobacion": "% aprobación", "asistencia": "asistencia media"}),
            width="stretch", hide_index=True)
        r = prueba_grupos(df, "tema_comentario", "nota_final")
        if r:
            nota_estadistica(f"{r[0]} de nota final entre temas: {interpreta_p(r[2])}.")
        cruce = st.selectbox("Cruzar temas con", CATEGORICAS, format_func=et, key="co_cruce")
        tabla = pd.crosstab(df[cruce], df["tema_comentario"], normalize="index") * 100
        fig = px.imshow(tabla.round(1), text_auto=".1f", aspect="auto",
                        color_continuous_scale=escala(),
                        labels={"color": "% dentro del grupo", "x": "", "y": et(cruce)},
                        title=f"% de cada tema dentro de {et(cruce).lower()}")
        grafico(fig)
    elif vista == "Palabras frecuentes":
        c1, c2 = st.columns(2)
        n = c1.slider("Palabras a mostrar", 5, 40, 15, key="co_n")
        solo = c2.selectbox("Solo comentarios de tema", ["Todos"] + sorted(df["tema_comentario"].unique()),
                            key="co_solo")
        base = df if solo == "Todos" else df[df["tema_comentario"] == solo]
        palabras = Counter(
            w for t in base["comentario_retro"].astype(str)
            for w in re.findall(r"[a-záéíóúñü]{4,}", t.lower()) if w not in STOPWORDS)
        top = pd.DataFrame(palabras.most_common(n), columns=["palabra", "frecuencia"])
        if top.empty:
            st.info("Sin comentarios para mostrar.")
        else:
            fig = px.bar(top.sort_values("frecuencia"), x="frecuencia", y="palabra",
                         orientation="h", color_discrete_sequence=paleta(),
                         title="Palabras más frecuentes")
            grafico(fig)
    else:
        temas_sel = st.multiselect("Filtrar por tema", sorted(df["tema_comentario"].unique()),
                                   default=sorted(df["tema_comentario"].unique()), key="co_temas")
        buscar = st.text_input("Buscar texto", key="co_buscar")
        v = df[df["tema_comentario"].isin(temas_sel)]
        if buscar:
            v = v[v["comentario_retro"].str.contains(buscar, case=False, na=False)]
        st.caption(f"{len(v)} comentarios")
        st.dataframe(v[["id_estudiante", "facultad", "nota_final", "tema_comentario",
                        "comentario_retro"]], width="stretch", hide_index=True)
        st.download_button("⬇️ Descargar comentarios", csv_bytes(v), "comentarios.csv", "text/csv")


def pg_constructor():
    st.subheader("Constructor de gráficos")
    st.caption("Elige tipo, ejes y color: el gráfico se actualiza al instante con la selección actual.")
    todas = NUMERICAS_EXT + CATEGORICAS_EXT
    fmt_none = lambda v: "— ninguno —" if v is None else et(v)  # noqa: E731

    tipo = seg("Tipo de gráfico", ["Dispersión", "Barras", "Líneas", "Histograma", "Caja",
                                   "Violín", "Mapa de calor 2D", "Pastel"], "co2_tipo")
    c = st.columns(4)
    x = c[0].selectbox("Eje X", todas, index=todas.index("facultad"), format_func=et, key="cb_x")
    y = c[1].selectbox("Eje Y", NUMERICAS_EXT, index=5, format_func=et, key="cb_y")
    color = c[2].selectbox("Color", [None] + CATEGORICAS_EXT, format_func=fmt_none, key="cb_c")
    agg = c[3].selectbox("Agregación", ["Media", "Mediana", "Suma", "Máximo", "Mínimo", "Conteo"],
                         key="cb_agg", disabled=tipo not in ("Barras", "Líneas", "Pastel"))
    fn = {"Media": "mean", "Mediana": "median", "Suma": "sum", "Máximo": "max",
          "Mínimo": "min", "Conteo": "count"}[agg]
    lab = {v: et(v) for v in todas}
    seq = paleta()

    try:
        if tipo == "Dispersión":
            fig = px.scatter(df, x=x, y=y, color=color, color_discrete_sequence=seq,
                             hover_data=["id_estudiante"], labels=lab)
        elif tipo in ("Barras", "Líneas"):
            keys = [x] + ([color] if color and color != x else [])
            if fn == "count":
                g = df.groupby(keys, observed=True).size().reset_index(name="valor")
            else:
                g = df.groupby(keys, observed=True)[y].agg(fn).reset_index(name="valor")
            g = g.sort_values(x)
            if tipo == "Barras":
                fig = px.bar(g, x=x, y="valor", color=color, barmode="group",
                             text_auto=".2f" if st.session_state["ap_valores"] else False,
                             color_discrete_sequence=seq, labels={**lab, "valor": f"{agg} de {et(y)}"})
            else:
                fig = px.line(g, x=x, y="valor", color=color, markers=True,
                              color_discrete_sequence=seq, labels={**lab, "valor": f"{agg} de {et(y)}"})
        elif tipo == "Histograma":
            bins = st.slider("Intervalos", 5, 60, 20, key="cb_bins")
            fig = px.histogram(df, x=x, color=color, nbins=bins, barmode="overlay", opacity=0.75,
                               color_discrete_sequence=seq, labels=lab)
        elif tipo == "Caja":
            fig = px.box(df, x=x if x in CATEGORICAS_EXT else None, y=y, color=color,
                         points="all", color_discrete_sequence=seq, labels=lab)
        elif tipo == "Violín":
            fig = px.violin(df, x=x if x in CATEGORICAS_EXT else None, y=y, color=color, box=True,
                            color_discrete_sequence=seq, labels=lab)
        elif tipo == "Mapa de calor 2D":
            fig = px.density_heatmap(df, x=x, y=y, color_continuous_scale=escala(), labels=lab,
                                     text_auto=True)
        else:
            if fn == "count":
                g = df.groupby(x, observed=True).size().reset_index(name="valor")
            else:
                g = df.groupby(x, observed=True)[y].agg(fn).reset_index(name="valor")
            fig = px.pie(g, names=x, values="valor", hole=0.4, color_discrete_sequence=seq)
        grafico(fig)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Esta combinación de variables no se puede graficar: {e}")


def pg_ayuda():
    st.subheader("Guía rápida")
    with st.expander("🧭 ¿Cómo navego?", expanded=True):
        st.markdown(
            "- **Panel izquierdo → «Ir a»**: cambia de sección.\n"
            "- **🔍 Filtros**: acota la muestra; todo el análisis se actualiza. Los **atajos** "
            "(En riesgo, Destacados, Con tutorías, Baja asistencia) aplican filtros comunes con un clic.\n"
            "- **📁 Datos**: carga otro CSV con las mismas columnas o recarga el archivo.\n"
            "- **⚙️ Opciones de análisis**: umbral de aprobación y pruebas estadísticas.\n"
            "- **🎨 Apariencia**: tema, paleta, escala de color y altura de los gráficos.")
    with st.expander("🧰 ¿Qué hace cada sección?"):
        st.markdown(
            "- **Resumen**: indicadores y distribuciones.\n"
            "- **Explorador de datos**: tabla con búsqueda, columnas y orden a elección.\n"
            "- **Ficha de estudiante**: perfil individual comparado con un grupo.\n"
            "- **Comparar grupos**: cajas, violines o barras con pruebas estadísticas.\n"
            "- **Relaciones**: correlaciones y dispersión.\n"
            "- **Evolución de notas**: corte 1 → corte 2 → final.\n"
            "- **Factores y riesgo**: importancia de variables y lista de estudiantes en riesgo.\n"
            "- **Comentarios**: temas, palabras frecuentes y búsqueda de textos.\n"
            "- **Constructor de gráficos**: arma tu propio gráfico.")
    with st.expander("⚠️ Precauciones al interpretar"):
        st.markdown(
            "- Correlación no es causalidad.\n"
            "- Los temas de los comentarios se asignan por palabras clave.\n"
            "- Con pocos estudiantes filtrados, las pruebas estadísticas pierden fiabilidad.")
    with st.expander("📑 Columnas esperadas en el CSV"):
        st.code(", ".join(COLUMNAS_REQUERIDAS))


# =========================================================================== #
# Encabezado y despacho
# =========================================================================== #
st.title(pagina_sel)

if df.empty and PAGINAS[pagina_sel] != "ayuda":
    st.warning("Los filtros no dejan ningún estudiante.")
    st.button("🧹 Limpiar filtros", on_click=reset_filtros, key="limpiar_main", type="primary")
    st.stop()

{
    "resumen": pg_resumen,
    "explorador": pg_explorador,
    "ficha": pg_ficha,
    "grupos": pg_grupos,
    "relaciones": pg_relaciones,
    "evolucion": pg_evolucion,
    "factores": pg_factores,
    "comentarios": pg_comentarios,
    "constructor": pg_constructor,
    "ayuda": pg_ayuda,
}[PAGINAS[pagina_sel]]()
