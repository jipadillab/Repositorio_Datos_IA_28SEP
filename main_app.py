"""
Dashboard de análisis del rendimiento académico (maestro.csv)

Ejecutar con:
    pip install -r requirements.txt
    streamlit run main_app.py

Columnas esperadas en el CSV:
    id_estudiante, facultad, horario_estudio, participacion_clase, asistencia_pct,
    horas_biblioteca_virtual, clics_material_extra, uso_tutorias,
    nota_corte1, nota_corte2, nota_final, comentario_retro
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Análisis académico",
    page_icon="🎓",
    layout="wide",
)

DEFAULT_CSV = Path(__file__).parent / "maestro.csv"

COLUMNAS_REQUERIDAS = [
    "id_estudiante", "facultad", "horario_estudio", "participacion_clase",
    "asistencia_pct", "horas_biblioteca_virtual", "clics_material_extra",
    "uso_tutorias", "nota_corte1", "nota_corte2", "nota_final", "comentario_retro",
]
NUMERICAS = [
    "asistencia_pct", "horas_biblioteca_virtual", "clics_material_extra",
    "nota_corte1", "nota_corte2", "nota_final",
]
CATEGORICAS = ["facultad", "horario_estudio", "participacion_clase", "uso_tutorias"]
ORDEN_PARTICIPACION = ["Nula", "Baja", "Media", "Alta"]
ORDEN_HORARIO = ["Mañana", "Tarde", "Noche"]

ETIQUETAS = {
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
}

# Categorías temáticas para los comentarios (se evalúan en este orden)
TEMAS_COMENTARIOS = {
    "Barrera: trabajo en grupo": [
        "en grupo", "grupo de trabajo", "grupal", "compañeros", "resto del grupo",
    ],
    "Barrera: conectividad / tecnología": [
        "internet", "conexión", "conectarme", "computador", "plataforma",
        "señal", "app ",
    ],
    "Barrera: tiempo / trabajo / familia": [
        "trabajo", "laborales", "tiempo", "familiar", "lejos", "desplazamiento",
        "horarios",
    ],
}
TEMA_POSITIVO = "Valoración positiva / neutral"


# --------------------------------------------------------------------------- #
# Carga y preparación de datos
# --------------------------------------------------------------------------- #
def _leer_csv(fuente) -> pd.DataFrame:
    """Lee el CSV probando codificaciones y separadores habituales."""
    ultimo_error = None
    for enc in ("utf-8-sig", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                if hasattr(fuente, "seek"):
                    fuente.seek(0)
                df = pd.read_csv(fuente, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception as e:  # noqa: BLE001
                ultimo_error = e
    raise ValueError(f"No se pudo leer el archivo: {ultimo_error}")


def clasificar_comentario(texto: str) -> str:
    t = f" {str(texto).lower()} "
    for tema, palabras in TEMAS_COMENTARIOS.items():
        if any(p in t for p in palabras):
            return tema
    return TEMA_POSITIVO


@st.cache_data(show_spinner=False)
def cargar_datos(fuente) -> pd.DataFrame:
    df = _leer_csv(fuente)
    df.columns = [c.strip() for c in df.columns]

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el archivo: {', '.join(faltantes)}")

    for c in NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAS:
        df[c] = df[c].astype(str).str.strip()

    df["participacion_clase"] = pd.Categorical(
        df["participacion_clase"], ORDEN_PARTICIPACION, ordered=True
    )
    df["horario_estudio"] = pd.Categorical(
        df["horario_estudio"], ORDEN_HORARIO, ordered=True
    )
    df["variacion_nota"] = df["nota_final"] - df["nota_corte1"]
    df["tema_comentario"] = df["comentario_retro"].apply(clasificar_comentario)
    return df


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def et(col: str) -> str:
    return ETIQUETAS.get(col, col)


def prueba_grupos(df: pd.DataFrame, cat: str, num: str):
    """t de Welch (2 grupos) o ANOVA (3+ grupos). Devuelve (nombre, estadístico, p)."""
    grupos = [g[num].dropna().values for _, g in df.groupby(cat, observed=True)]
    grupos = [g for g in grupos if len(g) >= 2]
    if len(grupos) < 2:
        return None
    if len(grupos) == 2:
        s, p = stats.ttest_ind(grupos[0], grupos[1], equal_var=False)
        return "t de Welch", s, p
    s, p = stats.f_oneway(*grupos)
    return "ANOVA", s, p


def interpreta_p(p: float) -> str:
    if p < 0.001:
        return "diferencia estadísticamente muy significativa (p < 0.001)"
    if p < 0.05:
        return f"diferencia estadísticamente significativa (p = {p:.3f})"
    return f"sin evidencia de diferencia significativa (p = {p:.3f})"


def fuerza_corr(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "nula"
    if a < 0.3:
        return "débil"
    if a < 0.5:
        return "moderada"
    return "fuerte"


# --------------------------------------------------------------------------- #
# Sidebar: datos y filtros
# --------------------------------------------------------------------------- #
st.sidebar.title("🎓 Análisis académico")

archivo = st.sidebar.file_uploader("Cargar CSV (opcional)", type=["csv"])
try:
    if archivo is not None:
        data = cargar_datos(archivo)
        st.sidebar.success(f"Usando: {archivo.name}")
    elif DEFAULT_CSV.exists():
        data = cargar_datos(str(DEFAULT_CSV))
        st.sidebar.info(f"Usando: {DEFAULT_CSV.name}")
    else:
        st.warning(
            "No se encontró `maestro.csv` junto a la aplicación. "
            "Sube el archivo desde la barra lateral."
        )
        st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")

umbral = st.sidebar.slider("Nota mínima para aprobar", 0.0, 5.0, 3.0, 0.1)

f_fac = st.sidebar.multiselect("Facultad", sorted(data["facultad"].unique()),
                               default=sorted(data["facultad"].unique()))
f_hor = st.sidebar.multiselect("Horario de estudio", ORDEN_HORARIO, default=ORDEN_HORARIO)
f_par = st.sidebar.multiselect("Participación en clase", ORDEN_PARTICIPACION,
                               default=ORDEN_PARTICIPACION)
f_tut = st.sidebar.multiselect("Uso de tutorías", sorted(data["uso_tutorias"].unique()),
                               default=sorted(data["uso_tutorias"].unique()))
amin, amax = float(data["asistencia_pct"].min()), float(data["asistencia_pct"].max())
f_asi = st.sidebar.slider("Asistencia (%)", amin, amax, (amin, amax))

df = data[
    data["facultad"].isin(f_fac)
    & data["horario_estudio"].isin(f_hor)
    & data["participacion_clase"].isin(f_par)
    & data["uso_tutorias"].isin(f_tut)
    & data["asistencia_pct"].between(*f_asi)
].copy()
df["aprobado"] = df["nota_final"] >= umbral

st.sidebar.caption(f"{len(df)} de {len(data)} estudiantes seleccionados")

st.title("Análisis del rendimiento académico")

if df.empty:
    st.warning("Los filtros no dejan ningún estudiante. Amplía la selección.")
    st.stop()

tabs = st.tabs([
    "📊 Resumen",
    "👥 Comparación por grupos",
    "🔗 Relaciones",
    "📈 Evolución de notas",
    "🌲 Factores y riesgo",
    "💬 Comentarios",
    "🗂️ Datos",
])

# --------------------------------------------------------------------------- #
# 1. Resumen
# --------------------------------------------------------------------------- #
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Estudiantes", len(df))
    c2.metric("Nota final promedio", f"{df['nota_final'].mean():.2f}")
    c3.metric("Aprobación", f"{df['aprobado'].mean() * 100:.1f}%")
    c4.metric("Asistencia media", f"{df['asistencia_pct'].mean():.1f}%")
    c5.metric("Usan tutorías", f"{(df['uso_tutorias'] == 'Sí').mean() * 100:.1f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x="nota_final", nbins=20, marginal="box",
                           labels={"nota_final": et("nota_final")},
                           title="Distribución de la nota final")
        fig.add_vline(x=umbral, line_dash="dash", line_color="red",
                      annotation_text=f"Umbral {umbral:.1f}")
        st.plotly_chart(fig, width="stretch")
    with col_b:
        cat_resumen = st.selectbox("Distribución de…", CATEGORICAS,
                                   format_func=et, key="resumen_cat")
        conteo = df[cat_resumen].value_counts().reset_index()
        conteo.columns = [cat_resumen, "estudiantes"]
        fig = px.pie(conteo, names=cat_resumen, values="estudiantes", hole=0.45,
                     title=f"Estudiantes por {et(cat_resumen).lower()}")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Estadística descriptiva")
    desc = df[NUMERICAS].describe().T.rename(index=ETIQUETAS)
    desc = desc.rename(columns={"count": "n", "mean": "media", "std": "desv.",
                                "min": "mín", "max": "máx"})
    st.dataframe(desc.round(2), width="stretch")

    with st.expander("Calidad de los datos"):
        q1, q2, q3 = st.columns(3)
        q1.metric("Valores nulos", int(df[COLUMNAS_REQUERIDAS].isna().sum().sum()))
        q2.metric("IDs duplicados", int(df["id_estudiante"].duplicated().sum()))
        fuera = ((df[["nota_corte1", "nota_corte2", "nota_final"]] < 0)
                 | (df[["nota_corte1", "nota_corte2", "nota_final"]] > 5)).sum().sum()
        q3.metric("Notas fuera de 0–5", int(fuera))

# --------------------------------------------------------------------------- #
# 2. Comparación por grupos
# --------------------------------------------------------------------------- #
with tabs[1]:
    g1, g2 = st.columns(2)
    cat = g1.selectbox("Agrupar por", CATEGORICAS, format_func=et, key="grp_cat")
    num = g2.selectbox("Variable a comparar",
                       ["nota_final", "nota_corte1", "nota_corte2", "asistencia_pct",
                        "horas_biblioteca_virtual", "clics_material_extra"],
                       format_func=et, key="grp_num")

    orden = None
    if cat == "participacion_clase":
        orden = ORDEN_PARTICIPACION
    elif cat == "horario_estudio":
        orden = ORDEN_HORARIO

    fig = px.box(df, x=cat, y=num, color=cat, points="all",
                 category_orders={cat: orden} if orden else None,
                 labels={cat: et(cat), num: et(num)},
                 title=f"{et(num)} según {et(cat).lower()}")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    resumen = (df.groupby(cat, observed=True)
                 .agg(n=("id_estudiante", "count"),
                      media=(num, "mean"),
                      mediana=(num, "median"),
                      desv=(num, "std"),
                      aprobacion=("aprobado", "mean"))
                 .reset_index())
    resumen["aprobacion"] = (resumen["aprobacion"] * 100).round(1)

    t1, t2 = st.columns([3, 2])
    with t1:
        st.dataframe(resumen.round(2).rename(columns={
            cat: et(cat), "desv": "desv.", "aprobacion": "% aprobación"}),
            width="stretch", hide_index=True)
    with t2:
        fig = px.bar(resumen, x=cat, y="aprobacion", text="aprobacion",
                     category_orders={cat: orden} if orden else None,
                     labels={cat: et(cat), "aprobacion": "% aprobación"},
                     title="% de aprobación por grupo")
        fig.update_yaxes(range=[0, 105])
        st.plotly_chart(fig, width="stretch")

    res = prueba_grupos(df, cat, num)
    if res:
        nombre, s, p = res
        st.info(f"**{nombre}** (estadístico = {s:.2f}): {interpreta_p(p)}.")
    else:
        st.caption("No hay suficientes datos por grupo para una prueba estadística.")

    if cat == "uso_tutorias" and num in ("nota_final", "nota_corte1", "nota_corte2"):
        st.warning(
            "⚠️ Cuidado al interpretar las tutorías: es común que quienes van a tutoría "
            "sean quienes ya tienen dificultades, por lo que una nota más baja en ese "
            "grupo **no implica** que la tutoría perjudique. Revisa la pestaña "
            "*Evolución de notas* para ver si mejoran entre cortes."
        )

# --------------------------------------------------------------------------- #
# 3. Relaciones
# --------------------------------------------------------------------------- #
with tabs[2]:
    corr_metodo = st.radio("Método de correlación", ["pearson", "spearman"],
                           horizontal=True)
    corr = df[NUMERICAS].corr(method=corr_metodo)
    corr.index = [et(c) for c in corr.index]
    corr.columns = [et(c) for c in corr.columns]
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", aspect="auto",
                    title=f"Matriz de correlación ({corr_metodo.capitalize()})")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Diagrama de dispersión")
    s1, s2, s3 = st.columns(3)
    x = s1.selectbox("Eje X", NUMERICAS, index=0, format_func=et, key="sx")
    y = s2.selectbox("Eje Y", NUMERICAS, index=5, format_func=et, key="sy")
    color = s3.selectbox("Color por", [None] + CATEGORICAS,
                         format_func=lambda c: "Sin color" if c is None else et(c),
                         key="sc")

    fig = px.scatter(df, x=x, y=y, color=color,
                     hover_data=["id_estudiante", "facultad"],
                     labels={x: et(x), y: et(y)})
    sub = df[[x, y]].dropna()
    if len(sub) > 2 and x != y:
        m, b = np.polyfit(sub[x], sub[y], 1)
        xs = np.linspace(sub[x].min(), sub[x].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines",
                                 name="Tendencia lineal",
                                 line=dict(color="black", dash="dash")))
        r, p = stats.pearsonr(sub[x], sub[y])
        rho, _ = stats.spearmanr(sub[x], sub[y])
        st.plotly_chart(fig, width="stretch")
        st.info(
            f"Correlación de Pearson **r = {r:.2f}** ({fuerza_corr(r)}), "
            f"Spearman **ρ = {rho:.2f}**; {interpreta_p(p)}. "
            f"Pendiente: {m:.3f} unidades de «{et(y)}» por cada unidad de «{et(x)}»."
        )
    else:
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# 4. Evolución de notas
# --------------------------------------------------------------------------- #
with tabs[3]:
    e1, e2, e3 = st.columns(3)
    e1.metric("Corte 1 (media)", f"{df['nota_corte1'].mean():.2f}")
    e2.metric("Corte 2 (media)", f"{df['nota_corte2'].mean():.2f}")
    e3.metric("Nota final (media)", f"{df['nota_final'].mean():.2f}")

    dim = st.selectbox("Ver evolución por", CATEGORICAS, format_func=et, key="evo_cat")
    evo = (df.groupby(dim, observed=True)[["nota_corte1", "nota_corte2", "nota_final"]]
             .mean().reset_index()
             .melt(id_vars=dim, var_name="momento", value_name="nota"))
    evo["momento"] = evo["momento"].map(
        {"nota_corte1": "Corte 1", "nota_corte2": "Corte 2", "nota_final": "Final"})
    fig = px.line(evo, x="momento", y="nota", color=dim, markers=True,
                  category_orders={"momento": ["Corte 1", "Corte 2", "Final"]},
                  labels={"nota": "Nota promedio", "momento": "", dim: et(dim)},
                  title=f"Nota promedio por corte según {et(dim).lower()}")
    st.plotly_chart(fig, width="stretch")

    v1, v2 = st.columns(2)
    with v1:
        fig = px.histogram(df, x="variacion_nota", nbins=25, marginal="box",
                           labels={"variacion_nota": et("variacion_nota")},
                           title="Cambio entre corte 1 y nota final")
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, width="stretch")
    with v2:
        mejora = (df["variacion_nota"] > 0).mean() * 100
        igual = (df["variacion_nota"] == 0).mean() * 100
        empeora = (df["variacion_nota"] < 0).mean() * 100
        st.metric("Mejoraron", f"{mejora:.1f}%")
        st.metric("Se mantuvieron", f"{igual:.1f}%")
        st.metric("Empeoraron", f"{empeora:.1f}%")

    st.subheader("Variación por grupo")
    var_g = (df.groupby(dim, observed=True)["variacion_nota"]
               .agg(["count", "mean", "std"]).reset_index()
               .rename(columns={dim: et(dim), "count": "n",
                                "mean": "variación media", "std": "desv."}))
    st.dataframe(var_g.round(3), width="stretch", hide_index=True)

    t_res = prueba_grupos(df, dim, "variacion_nota")
    if t_res:
        st.info(f"{t_res[0]} sobre la variación de nota: {interpreta_p(t_res[2])}.")

    if len(df) > 2:
        t, p = stats.ttest_rel(df["nota_final"], df["nota_corte1"], nan_policy="omit")
        st.caption(
            f"Prueba t pareada final vs corte 1: {interpreta_p(p)} "
            f"(cambio medio {df['variacion_nota'].mean():+.3f})."
        )

# --------------------------------------------------------------------------- #
# 5. Factores y riesgo
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.markdown(
        "Modelo **Random Forest** para estimar qué variables se asocian más con la "
        "nota final. Es un análisis exploratorio: la importancia indica asociación, "
        "no causalidad."
    )
    usar_cortes = st.checkbox(
        "Incluir nota corte 1 y corte 2 como predictores", value=False,
        help="Si se incluyen, dominarán el modelo. Desmarcado se evalúan solo "
             "hábitos y contexto del estudiante.",
    )

    feats_num = ["asistencia_pct", "horas_biblioteca_virtual", "clics_material_extra"]
    if usar_cortes:
        feats_num += ["nota_corte1", "nota_corte2"]
    X = pd.get_dummies(df[feats_num + CATEGORICAS].astype({c: str for c in CATEGORICAS}),
                       columns=CATEGORICAS, dtype=float)
    yv = df["nota_final"]

    if len(df) < 30:
        st.warning("Se necesitan al menos 30 estudiantes filtrados para ajustar el modelo.")
    else:
        rf = RandomForestRegressor(n_estimators=300, random_state=42,
                                   min_samples_leaf=3, n_jobs=-1)
        k = 5 if len(df) >= 50 else 3
        r2 = cross_val_score(rf, X, yv, cv=k, scoring="r2")
        rf.fit(X, yv)
        imp = (pd.Series(rf.feature_importances_, index=X.columns)
                 .sort_values().reset_index())
        imp.columns = ["variable", "importancia"]
        imp["variable"] = imp["variable"].map(lambda v: et(v) if v in ETIQUETAS else
                                              v.replace("_", ": ", 1))

        m1, m2 = st.columns([2, 1])
        with m1:
            fig = px.bar(imp, x="importancia", y="variable", orientation="h",
                         title="Importancia de variables")
            st.plotly_chart(fig, width="stretch")
        with m2:
            st.metric("R² (validación cruzada)", f"{r2.mean():.2f}",
                      help=f"Promedio de {k} particiones; ±{r2.std():.2f}")
            if r2.mean() < 0.2:
                st.caption("Poder explicativo bajo: estas variables por sí solas "
                           "explican poco de la nota final.")
            elif r2.mean() < 0.5:
                st.caption("Poder explicativo moderado.")
            else:
                st.caption("Buen poder explicativo.")

    st.subheader("Estudiantes en riesgo")
    criterio = st.radio(
        "Criterio",
        ["Nota final por debajo del umbral", "Nota corte 2 por debajo del umbral",
         "Baja asistencia (< 70 %)"],
        horizontal=True,
    )
    if criterio.startswith("Nota final"):
        riesgo = df[df["nota_final"] < umbral]
    elif criterio.startswith("Nota corte 2"):
        riesgo = df[df["nota_corte2"] < umbral]
    else:
        riesgo = df[df["asistencia_pct"] < 70]
    st.write(f"**{len(riesgo)}** estudiantes ({len(riesgo) / len(df) * 100:.1f}% de la selección).")
    cols_r = ["id_estudiante", "facultad", "horario_estudio", "participacion_clase",
              "asistencia_pct", "uso_tutorias", "nota_corte1", "nota_corte2", "nota_final"]
    st.dataframe(riesgo[cols_r].sort_values("nota_final"), width="stretch",
                 hide_index=True)
    st.download_button("Descargar lista de riesgo (CSV)",
                       riesgo[cols_r].to_csv(index=False).encode("utf-8-sig"),
                       "estudiantes_en_riesgo.csv", "text/csv")

# --------------------------------------------------------------------------- #
# 6. Comentarios
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.markdown(
        "Los comentarios se clasifican con **reglas de palabras clave** en barreras "
        "(trabajo en grupo, conectividad/tecnología, tiempo/trabajo/familia) o "
        "valoración positiva/neutral. Es una aproximación: revisa la tabla inferior."
    )
    temas = (df.groupby("tema_comentario")
               .agg(estudiantes=("id_estudiante", "count"),
                    nota_media=("nota_final", "mean"),
                    aprobacion=("aprobado", "mean"),
                    asistencia=("asistencia_pct", "mean"))
               .reset_index().sort_values("estudiantes", ascending=False))
    temas["aprobacion"] = (temas["aprobacion"] * 100).round(1)

    k1, k2 = st.columns(2)
    with k1:
        fig = px.bar(temas, x="tema_comentario", y="estudiantes", text="estudiantes",
                     color="tema_comentario",
                     labels={"tema_comentario": "", "estudiantes": "Estudiantes"},
                     title="Comentarios por tema")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with k2:
        fig = px.box(df, x="tema_comentario", y="nota_final", color="tema_comentario",
                     points="all",
                     labels={"tema_comentario": "", "nota_final": et("nota_final")},
                     title="Nota final según tema del comentario")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.dataframe(temas.round(2).rename(columns={
        "tema_comentario": "Tema", "nota_media": "nota final media",
        "aprobacion": "% aprobación", "asistencia": "asistencia media"}),
        width="stretch", hide_index=True)

    res = prueba_grupos(df, "tema_comentario", "nota_final")
    if res:
        st.info(f"{res[0]} de nota final entre temas: {interpreta_p(res[2])}.")

    cruce = st.selectbox("Cruzar temas con", CATEGORICAS, format_func=et, key="cruce")
    tabla = pd.crosstab(df[cruce], df["tema_comentario"], normalize="index") * 100
    fig = px.imshow(tabla.round(1), text_auto=".1f", aspect="auto",
                    color_continuous_scale="Blues",
                    labels={"color": "% dentro del grupo", "x": "", "y": et(cruce)},
                    title=f"% de cada tema dentro de {et(cruce).lower()}")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Explorar comentarios")
    tema_sel = st.multiselect("Filtrar por tema", temas["tema_comentario"].tolist(),
                              default=temas["tema_comentario"].tolist())
    buscar = st.text_input("Buscar texto")
    vista = df[df["tema_comentario"].isin(tema_sel)]
    if buscar:
        vista = vista[vista["comentario_retro"].str.contains(buscar, case=False, na=False)]
    st.dataframe(vista[["id_estudiante", "facultad", "nota_final", "tema_comentario",
                        "comentario_retro"]],
                 width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# 7. Datos
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.write(f"{len(df)} filas · {df.shape[1]} columnas (incluye columnas derivadas).")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Descargar datos filtrados (CSV)",
                       df.to_csv(index=False).encode("utf-8-sig"),
                       "maestro_filtrado.csv", "text/csv")
