import pickle
from pathlib import Path

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


class ModeloNoEntrenadoError(Exception):
    pass


class PredictorVentas:

    COLUMNAS_NUMERICAS = ["precio", "mes_numero"]
    COLUMNAS_CATEGORICAS = ["categoria"]

    def __init__(self):
        self.pipeline = None
        self.metricas = {}
        self._entrenado = False

    def _preparar_features(self, df):
        df = df.copy()
        df["mes_numero"] = df["fecha"].dt.month
        # agrupo por producto/mes sacar las "unidades vendidas ese mes"
        agrupado = (
            df.groupby(["producto", "categoria", "mes_numero"])
            .agg(precio=("precio", "mean"), cantidad=("cantidad", "sum"))
            .reset_index()
        )
        return agrupado

    def entrenar(self, df, test_size=0.2, random_state=42):
        datos = self._preparar_features(df)

        if len(datos) < 10:
            raise ValueError(f"Hay muy pocos datos para entrenar (min 10, hay {len(datos)}).")

        X = datos[self.COLUMNAS_NUMERICAS + self.COLUMNAS_CATEGORICAS]
        y = datos["cantidad"]

        preprocesador = ColumnTransformer([
            ("num", StandardScaler(), self.COLUMNAS_NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.COLUMNAS_CATEGORICAS),
        ])

        self.pipeline = Pipeline([
            ("preproc", preprocesador),
            ("modelo", MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                                     max_iter=2000, random_state=random_state)),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        try:
            self.pipeline.fit(X_train, y_train)
        except Exception as err:
            raise RuntimeError(f"Falló el entrenamiento: {err}")

        y_pred = self.pipeline.predict(X_test)
        self.metricas = {
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
            "r2": round(float(r2_score(y_test, y_pred)), 3),
            "n_muestras_train": len(X_train),
            "n_muestras_test": len(X_test),
        }
        self._entrenado = True
        return self.metricas

    def predecir(self, precio, categoria, mes_numero):
        if not self._entrenado or self.pipeline is None:
            raise ModeloNoEntrenadoError("El modelo todavía no fue entrenado.")
        entrada = pd.DataFrame([{"precio": precio, "categoria": categoria, "mes_numero": mes_numero}])
        prediccion = self.pipeline.predict(entrada)[0]
        return max(0.0, round(float(prediccion), 1))

    def clasificar_rotacion(self, precio, categoria, mes_numero):
        unidades = self.predecir(precio, categoria, mes_numero)
        if unidades < 3:
            return "Baja rotación"
        elif unidades < 8:
            return "Media rotación"
        return "Alta rotación"

    def guardar(self, ruta="modelo_entrenado.pkl"):
        if not self._entrenado:
            raise ModeloNoEntrenadoError("No hay modelo entrenado para guardar.")
        with open(ruta, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def cargar(ruta="modelo_entrenado.pkl"):
        path = Path(ruta)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {ruta}")
        with open(path, "rb") as f:
            return pickle.load(f)
