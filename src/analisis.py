import re
import pandas as pd


class DatasetInvalidoError(Exception):
    pass


class AnalizadorVentas:

    COLUMNAS_REQUERIDAS = ["fecha", "producto", "categoria", "precio", "cantidad"]
    COLUMNA_OPCIONAL = "cliente"

    def __init__(self, df):
        self._validar_columnas(df)
        self.df_original = df.copy()
        self.df = self._limpiar(df.copy())

    @staticmethod
    def leer_archivo(ruta_o_buffer, nombre_archivo=None, hoja=None):
        nombre = (nombre_archivo or str(ruta_o_buffer)).lower()
        try:
            if nombre.endswith((".xlsx", ".xls")):
                df = pd.read_excel(ruta_o_buffer, sheet_name=hoja if hoja is not None else 0)
            else:
                df = pd.read_csv(ruta_o_buffer)
        except Exception as err:
            raise DatasetInvalidoError(f"No se pudo leer el archivo ({nombre}): {err}")

        if isinstance(df, dict):
            raise DatasetInvalidoError("Elegí una sola hoja del Excel.")
        return df

    @staticmethod
    def nombres_de_hojas(ruta_o_buffer):
        try:
            excel = pd.ExcelFile(ruta_o_buffer)
            return excel.sheet_names
        except Exception as err:
            raise DatasetInvalidoError(f"No se pudo abrir el Excel: {err}")

    @classmethod
    def desde_csv(cls, ruta_o_buffer):
        df = cls.leer_archivo(ruta_o_buffer)
        return cls(df)

    @classmethod
    def desde_mapeo(cls, df_crudo, mapeo):
        # mapeo = {"fecha": "Fecha de Venta", "producto": "Articulo", ...}
        faltantes = [c for c in cls.COLUMNAS_REQUERIDAS if not mapeo.get(c)]
        if faltantes:
            raise DatasetInvalidoError("Faltan mapear columnas: " + ", ".join(faltantes))

        columnas_necesarias = set(cls.COLUMNAS_REQUERIDAS) | {cls.COLUMNA_OPCIONAL}
        columnas_a_tomar = {d: o for d, o in mapeo.items() if d in columnas_necesarias and o}

        df_renombrado = df_crudo[list(columnas_a_tomar.values())].copy()
        df_renombrado.columns = list(columnas_a_tomar.keys())
        return cls(df_renombrado)

    def _validar_columnas(self, df):
        presentes = {c.lower().strip() for c in df.columns}
        faltantes = set(self.COLUMNAS_REQUERIDAS) - presentes
        if faltantes:
            raise DatasetInvalidoError("Faltan columnas: " + ", ".join(sorted(faltantes)))

    def _limpiar(self, df):
        df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]

        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

        limpiar_texto = lambda s: re.sub(r"\s+", " ", str(s).strip()).title()
        df["producto"] = df["producto"].apply(limpiar_texto)
        df["categoria"] = df["categoria"].apply(limpiar_texto)

        df["precio"] = pd.to_numeric(df["precio"], errors="coerce")
        df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")

        df = df.drop_duplicates()
        df = df.dropna(subset=["fecha", "precio", "cantidad"])

        # marco como outlier lo que se va de 3 desvios del precio promedio del producto
        def marcar_outlier(grupo):
            media, std = grupo["precio"].mean(), grupo["precio"].std()
            if std == 0 or pd.isna(std):
                return pd.Series([False] * len(grupo), index=grupo.index)
            return (grupo["precio"] - media).abs() > 3 * std

        df["es_outlier_precio"] = df.groupby("producto", group_keys=False).apply(marcar_outlier)
        df["monto_total"] = df["precio"] * df["cantidad"]
        df["mes"] = df["fecha"].dt.to_period("M").astype(str)

        return df.reset_index(drop=True)

    def resumen_calidad(self):
        return {
            "filas_originales": len(self.df_original),
            "filas_limpias": len(self.df),
            "filas_descartadas": len(self.df_original) - len(self.df),
            "duplicados_eliminados": int(self.df_original.duplicated().sum()),
            "outliers_detectados": int(self.df["es_outlier_precio"].sum()),
        }

    def ventas_por_mes(self):
        return self.df.groupby("mes")["monto_total"].sum().reset_index().sort_values("mes")

    def ventas_por_producto(self):
        return (
            self.df.groupby("producto")
            .agg(unidades=("cantidad", "sum"), monto_total=("monto_total", "sum"))
            .reset_index()
            .sort_values("unidades", ascending=False)
        )

    def ventas_por_categoria(self):
        return self.df.groupby("categoria")["monto_total"].sum().reset_index().sort_values("monto_total", ascending=False)

    def ticket_promedio(self):
        return round(self.df["monto_total"].mean(), 2)

    def producto_mas_vendido(self):
        return self.ventas_por_producto().iloc[0]["producto"]

    def categorias_disponibles(self):
        return sorted({c for c in self.df["categoria"].unique()})

    def productos_disponibles(self):
        return sorted({p for p in self.df["producto"].unique()})

    def filtrar(self, categoria=None, fecha_desde=None, fecha_hasta=None):
        resultado = self.df
        if categoria and categoria != "Todas":
            resultado = resultado[resultado["categoria"] == categoria]
        if fecha_desde is not None:
            resultado = resultado[resultado["fecha"] >= pd.Timestamp(fecha_desde)]
        if fecha_hasta is not None:
            resultado = resultado[resultado["fecha"] <= pd.Timestamp(fecha_hasta)]
        return resultado
