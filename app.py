import streamlit as st

from src.analisis import AnalizadorVentas, DatasetInvalidoError
from src.modelo import PredictorVentas, ModeloNoEntrenadoError

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

st.set_page_config(page_title="Analizador de Ventas", layout="wide")

if "analizador" not in st.session_state:
    st.session_state.analizador = None
if "predictor" not in st.session_state:
    st.session_state.predictor = None
if "metricas" not in st.session_state:
    st.session_state.metricas = None
if "df_crudo" not in st.session_state:
    st.session_state.df_crudo = None
if "nombre_archivo_actual" not in st.session_state:
    st.session_state.nombre_archivo_actual = None

# nombre interno -> como se lo muestro al usuario
CAMPOS_SISTEMA = {
    "fecha": "Fecha de la venta",
    "producto": "Producto",
    "categoria": "Categoría",
    "precio": "Precio unitario",
    "cantidad": "Cantidad vendida",
    "cliente": "Cliente (opcional)",
}

st.title("Analizador y Predictor de Ventas")
st.write("TP Python para IA - UAI 2026. Combina análisis con Pandas y una red neuronal para predecir ventas.")

st.header("1. Cargar dataset")
st.write("Subí tu archivo de ventas (csv o excel). En el paso siguiente indicás qué columna es cada dato.")

col1, col2 = st.columns([2, 1])
with col1:
    archivo = st.file_uploader("Archivo .csv, .xlsx o .xls", type=["csv", "xlsx", "xls"])
with col2:
    usar_ejemplo = st.button("Usar dataset de ejemplo", use_container_width=True)

if usar_ejemplo:
    st.session_state.df_crudo = AnalizadorVentas.leer_archivo("data/ejemplo_ventas.csv", nombre_archivo="ejemplo_ventas.csv")
    st.session_state.nombre_archivo_actual = "ejemplo_ventas.csv"
    st.session_state.analizador = None
    st.session_state.predictor = None
    st.session_state.metricas = None

elif archivo is not None and archivo.name != st.session_state.nombre_archivo_actual:
    try:
        if archivo.name.lower().endswith((".xlsx", ".xls")):
            hojas = AnalizadorVentas.nombres_de_hojas(archivo)
            hoja_elegida = hojas[0]
            if len(hojas) > 1:
                hoja_elegida = st.selectbox("El Excel tiene varias hojas, elegí una:", hojas)
            archivo.seek(0)
            df_crudo = AnalizadorVentas.leer_archivo(archivo, archivo.name, hoja=hoja_elegida)
        else:
            df_crudo = AnalizadorVentas.leer_archivo(archivo, archivo.name)

        st.session_state.df_crudo = df_crudo
        st.session_state.nombre_archivo_actual = archivo.name
        st.session_state.analizador = None
        st.session_state.predictor = None
        st.session_state.metricas = None
    except DatasetInvalidoError as err:
        st.error(f"No se pudo leer el archivo: {err}")
        st.session_state.df_crudo = None

df_crudo = st.session_state.df_crudo

if df_crudo is not None:
    st.subheader("Elegí qué columna es cada cosa")
    st.dataframe(df_crudo.head(5), use_container_width=True)

    columnas_archivo = list(df_crudo.columns)
    opciones = ["-- Elegir --"] + columnas_archivo

    def sugerir(campo):
        for col in columnas_archivo:
            if campo.lower() in str(col).lower():
                return col
        return "-- Elegir --"

    mapeo = {}
    colA, colB = st.columns(2)
    campos = list(CAMPOS_SISTEMA.items())
    mitad = len(campos) // 2 + len(campos) % 2

    for columna_ui, subset in [(colA, campos[:mitad]), (colB, campos[mitad:])]:
        with columna_ui:
            for campo, etiqueta in subset:
                sugerencia = sugerir(campo)
                idx = opciones.index(sugerencia) if sugerencia in opciones else 0
                seleccion = st.selectbox(etiqueta, opciones, index=idx, key=f"map_{campo}")
                mapeo[campo] = None if seleccion == "-- Elegir --" else seleccion

    if st.button("Confirmar y analizar", type="primary"):
        try:
            st.session_state.analizador = AnalizadorVentas.desde_mapeo(df_crudo, mapeo)
            st.session_state.predictor = None
            st.session_state.metricas = None
            st.success("Listo, dataset cargado.")
        except DatasetInvalidoError as err:
            st.error(f"Error en el mapeo: {err}")
        except Exception as err:
            st.error(f"Error al procesar el archivo: {err}")

analizador = st.session_state.analizador

if analizador is None:
    st.info("Subí un archivo o usá el de ejemplo para seguir.")
    st.stop()

st.header("2. Análisis exploratorio")

calidad = analizador.resumen_calidad()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas originales", calidad["filas_originales"])
c2.metric("Filas tras limpieza", calidad["filas_limpias"])
c3.metric("Duplicados eliminados", calidad["duplicados_eliminados"])
c4.metric("Outliers de precio", calidad["outliers_detectados"])

colf1, colf2 = st.columns(2)
with colf1:
    categoria_filtro = st.selectbox("Filtrar por categoría", ["Todas"] + analizador.categorias_disponibles())
with colf2:
    fecha_min, fecha_max = analizador.df["fecha"].min(), analizador.df["fecha"].max()
    rango_fechas = st.date_input("Rango de fechas", value=(fecha_min.date(), fecha_max.date()))

fecha_desde, fecha_hasta = (rango_fechas if len(rango_fechas) == 2 else (fecha_min, fecha_max))
df_filtrado = analizador.filtrar(categoria_filtro, fecha_desde, fecha_hasta)

tab1, tab2, tab3 = st.tabs(["Ventas por mes", "Ventas por producto", "Ventas por categoría"])

with tab1:
    ventas_mes = df_filtrado.groupby("mes")["monto_total"].sum().reset_index().sort_values("mes")
    st.bar_chart(ventas_mes.set_index("mes"))

with tab2:
    ventas_prod = (
        df_filtrado.groupby("producto")
        .agg(unidades=("cantidad", "sum"), monto_total=("monto_total", "sum"))
        .reset_index()
        .sort_values("unidades", ascending=False)
    )
    st.dataframe(ventas_prod, use_container_width=True)
    st.bar_chart(ventas_prod.set_index("producto")["unidades"])

with tab3:
    ventas_cat = df_filtrado.groupby("categoria")["monto_total"].sum().reset_index()
    st.bar_chart(ventas_cat.set_index("categoria"))

colm1, colm2 = st.columns(2)
colm1.metric("Ticket promedio", f"${analizador.ticket_promedio():,.0f}")
colm2.metric("Producto más vendido", analizador.producto_mas_vendido())

with st.expander("Ver dataset limpio completo"):
    st.dataframe(df_filtrado, use_container_width=True)

st.header("3. Entrenar la red neuronal")
st.write("Se entrena un MLPRegressor de scikit-learn con las ventas por producto/mes, para predecir unidades vendidas.")

if st.button("Entrenar modelo", type="primary"):
    try:
        predictor = PredictorVentas()
        with st.spinner("Entrenando..."):
            metricas = predictor.entrenar(analizador.df)
        st.session_state.predictor = predictor
        st.session_state.metricas = metricas
        st.success("Modelo entrenado.")
    except ValueError as err:
        st.warning(str(err))
    except RuntimeError as err:
        st.error(str(err))

if st.session_state.metricas:
    m = st.session_state.metricas
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("MAE", m["mae"])
    mc2.metric("R2", m["r2"])
    mc3.metric("Muestras train", m["n_muestras_train"])
    mc4.metric("Muestras test", m["n_muestras_test"])

st.header("4. Predecir ventas de un producto")

with st.form("form_prediccion"):
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        categoria_pred = st.selectbox("Categoría", analizador.categorias_disponibles())
    with pc2:
        precio_pred = st.number_input("Precio ($)", min_value=0.0, value=50000.0, step=1000.0)
    with pc3:
        mes_pred = st.selectbox("Mes", list(MESES_ES.keys()), format_func=lambda m: MESES_ES[m])

    enviado = st.form_submit_button("Predecir")

if enviado:
    predictor = st.session_state.predictor
    if predictor is None:
        st.warning("Primero entrená el modelo en el paso 3.")
    else:
        try:
            unidades = predictor.predecir(precio_pred, categoria_pred, mes_pred)
            rotacion = predictor.clasificar_rotacion(precio_pred, categoria_pred, mes_pred)
            rc1, rc2 = st.columns(2)
            rc1.metric("Unidades predichas", unidades)
            rc2.metric("Rotación estimada", rotacion)
        except ModeloNoEntrenadoError as err:
            st.warning(str(err))

st.divider()
st.caption("TP Python para IA - UAI 2026")
