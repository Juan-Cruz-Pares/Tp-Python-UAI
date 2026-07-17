# Analizador y Predictor de Ventas

TP de Python - UAI 2026.

## Cómo funciona

El usuario sube su archivo de ventas (csv o excel, no importa cómo se llamen las columnas) y en la app elige qué columna corresponde a fecha, producto, categoría, precio y cantidad. Con eso se arma el análisis con Pandas: ventas por mes, por producto, por categoría, ticket promedio, etc. Después se entrena la red neuronal con esos mismos datos, y el usuario puede cargar un producto nuevo (precio, categoría, mes) para que el modelo le prediga cuántas unidades se venderían y si es de alta, media o baja rotación.

Incluye un dataset de ejemplo en `data/ejemplo_ventas.csv`, y otro en `data/ejemplo_ventas_columnas_distintas.xlsx` con las mismas ventas pero con nombres de columna distintos, para probar el paso del mapeo.

## Estructura

```
app.py                 -> la app de Streamlit
src/analisis.py         -> clase AnalizadorVentas (limpieza y stats con Pandas)
src/modelo.py            -> clase PredictorVentas (red neuronal)
data/                    -> datasets de ejemplo
requirements.txt
```

## Instalación

```bash
git clone <URL_DEL_REPO>
cd proyecto-ventas
python -m venv venv
venv\Scripts\activate      (en Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## Correrlo

```bash
streamlit run app.py
```

Abre en http://localhost:8501