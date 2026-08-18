"""modell.py — ajuste y entrenamiento de modelos.

El arco predictivo sigue los siete pasos de la estrategia de validación cruzada anidada
que documenta el manuscrito: cada paso es una función nombrada, de modo que el código pueda
leerse junto a su diagrama de flujo.

    1 preparar_datos            5 reentrenar
    2 particion_externa         6 aplicar_preprocesamiento
    3 ajustar_preprocesamiento  7 predecir_fuera_de_pliegue
    4 seleccionar_hiperparametros        · validar_cruzado orquesta 1→7

Las funciones reciben datos y parámetros: nada específico del estudio vive aquí. Lo
particular —predictores, desenlace, grillas— entra por `config` y por el catálogo, lo que
permite reproducir el flujo bajo otras configuraciones.

La evaluación del desempeño, la interpretabilidad y la transportabilidad viven en `eval.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


# ─────────────────────────────────────────────────────────────────────────────
# Codificación de variables ordinales
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Codifica las variables ordinales respetando su orden clínico.

    El orden proviene de las categorías declaradas en el dato, no del alfabeto: codificar
    `[No, 50, 80, 100, > 100 mcg]` por orden alfabético descoloca la escala y altera el
    signo de las asociaciones. Se leen todas las categorías definidas, no solo las
    observadas en el pliegue, de modo que un pliegue sin algún nivel no altere la
    codificación. Imputa la moda antes de codificar.
    """

    def __init__(self, valor_desconocido: int = -1):
        self.valor_desconocido = valor_desconocido

    @staticmethod
    def _marco(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def _imputado(self, X):
        X = self._marco(X).astype(object)
        X = X.where(pd.notna(X), np.nan)
        for c in self.columnas_:
            X[c] = X[c].fillna(self.relleno_[c])
        return X

    def fit(self, X, y=None):
        from sklearn.preprocessing import OrdinalEncoder

        X = self._marco(X)
        self.columnas_ = list(X.columns)
        self.categorias_, self.relleno_ = [], {}
        for c in X.columns:
            col = X[c]
            cats = (list(col.cat.categories) if isinstance(col.dtype, pd.CategoricalDtype)
                    else sorted(pd.Series(col).dropna().astype(object).unique(), key=str))
            self.categorias_.append([str(x) for x in cats])
            moda = pd.Series(col).astype(object).mode()
            self.relleno_[c] = str(moda.iloc[0]) if len(moda) else (str(cats[0]) if cats else np.nan)
        self.encoder_ = OrdinalEncoder(categories=self.categorias_,
                                       handle_unknown="use_encoded_value",
                                       unknown_value=self.valor_desconocido)
        self.encoder_.fit(self._imputado(X))
        return self

    def transform(self, X):
        return self.encoder_.transform(self._imputado(X))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.columnas_)


def _a_objeto(X):
    """Convierte a objeto y normaliza los faltantes a un valor que sklearn reconozca.

    Los tipos anulables de pandas usan un marcador propio de ausencia que no admite
    evaluación booleana, de modo que el imputador falla si no se normaliza.
    """
    out = pd.DataFrame(X).astype(object)
    return out.where(pd.notna(out), np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Catálogo de algoritmos
# ─────────────────────────────────────────────────────────────────────────────

def catalogo_algoritmos(semilla: int | None = None) -> dict:
    """Instancia los algoritmos candidatos con su grilla de hiperparámetros.

    La regresión logística clásica no tiene grilla: omite el paso de selección.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    s = config.SEED if semilla is None else semilla
    estimadores = {
        "LR": LogisticRegression(penalty=None, max_iter=2000, solver="lbfgs", random_state=s),
        "LASSO": LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=s),
        "ElasticNet": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                         max_iter=2000, random_state=s),
        "RF": RandomForestClassifier(random_state=s, n_jobs=-1),
        "GBM": GradientBoostingClassifier(random_state=s),
        "SVM": SVC(probability=True, random_state=s),
    }
    try:
        from xgboost import XGBClassifier
        estimadores["XGB"] = XGBClassifier(eval_metric="logloss", random_state=s, n_jobs=-1)
    except ImportError:
        print("⚠ XGBoost no disponible: se omite del catálogo")
    return {k: {"estimador": v, "grilla": config.GRILLAS.get(k)}
            for k, v in estimadores.items() if k in config.GRILLAS}


# ─────────────────────────────────────────────────────────────────────────────
# Paso 1 · Preparación de los datos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Datos:
    """Matriz de predictores y desenlace de una especificación."""
    X: pd.DataFrame
    y: np.ndarray
    tipos: dict
    especificacion: str
    escenario: str
    cohorte: str = "principal"

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def eventos(self) -> int:
        return int(self.y.sum())

    def __repr__(self) -> str:
        return (f"Datos({self.cohorte}/{self.escenario}/{self.especificacion}: "
                f"n={self.n}, eventos={self.eventos}, predictores={self.X.shape[1]})")


def predictores(cat: pd.DataFrame, especificacion: str, escenario: str,
                cohorte: str = "principal") -> list[str]:
    """Predictores de una especificación, según las inclusiones del catálogo."""
    col = {"principal": "include_model", "espinal": "include_espinal"}[cohorte]
    sel = cat[cat[col].fillna(False)]
    if especificacion == "preop":
        sel = sel[sel["include_preop"].fillna(False)]
    vs = [v for v in sel["var_rename"] if v != config.CENTER_VAR]
    return vs + [config.CENTER_VAR] if escenario == "con_centro" else vs


def preparar_datos(cohorte_df: pd.DataFrame, cat: pd.DataFrame, especificacion: str,
                   escenario: str = "sin_centro", cohorte: str = "principal",
                   desenlace: str | None = None) -> Datos:
    """**Paso 1.** Reúne la matriz de predictores y el desenlace de una especificación.

    Agrupa además los predictores por tipo, que es lo que determina la transformación que
    recibirá cada uno en el preprocesamiento.
    """
    desenlace = desenlace or config.OUTCOME
    vs = [v for v in predictores(cat, especificacion, escenario, cohorte)
          if v in cohorte_df.columns]
    meta = cat.set_index("var_rename")
    tipos = {t: [] for t in ("num", "bin", "ord", "nom")}
    for v in vs:
        t = str(meta.loc[v, "conceptual_type"]).lower()
        tipos[t if t in tipos else "nom"].append(v)
    obs = cohorte_df[desenlace].notna()
    return Datos(X=cohorte_df.loc[obs, vs].copy(),
                 y=cohorte_df.loc[obs, desenlace].astype(int).to_numpy(),
                 tipos={k: v for k, v in tipos.items() if v},
                 especificacion=especificacion, escenario=escenario, cohorte=cohorte)


# ─────────────────────────────────────────────────────────────────────────────
# Paso 2 · Partición externa
# ─────────────────────────────────────────────────────────────────────────────

def particion_externa(y: np.ndarray, k: int | None = None, semilla: int | None = None):
    """**Paso 2.** Define los pliegues externos, estratificados por el desenlace.

    Devuelve la lista de particiones para que la asignación sea explícita y reutilizable
    entre algoritmos: todos se evalúan sobre exactamente los mismos pliegues.
    """
    from sklearn.model_selection import StratifiedKFold

    k = k or config.CV_EXTERNO
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.SEED if semilla is None else semilla)
    return list(cv.split(np.zeros(len(y)), y))


# ─────────────────────────────────────────────────────────────────────────────
# Paso 3 · Preprocesamiento
# ─────────────────────────────────────────────────────────────────────────────

def construir_preprocesamiento(tipos: dict):
    """Define la transformación que recibe cada tipo de variable.

    Numéricas: imputación por mediana y estandarización. Binarias y ordinales: imputación
    por moda y codificación, respetando el orden clínico en las ordinales. Nominales:
    indicadores binarios, agrupando los niveles infrecuentes.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

    ts = []
    if tipos.get("num"):
        ts.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler())]), tipos["num"]))
    if tipos.get("bin"):
        ts.append(("bin", Pipeline([
            ("objeto", FunctionTransformer(_a_objeto, validate=False, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), tipos["bin"]))
    if tipos.get("ord"):
        ts.append(("ord", ClinicalOrdinalEncoder(), tipos["ord"]))
    if tipos.get("nom"):
        ts.append(("nom", Pipeline([
            ("objeto", FunctionTransformer(_a_objeto, validate=False, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False,
                                      handle_unknown="infrequent_if_exist", min_frequency=5)),
        ]), tipos["nom"]))
    return ColumnTransformer(ts, remainder="drop")


def ajustar_preprocesamiento(prep, X_entrenamiento: pd.DataFrame):
    """**Paso 3.** Ajusta el preprocesamiento **dentro** del pliegue de entrenamiento.

    Estimar medianas, modas o escalas sobre el conjunto completo filtraría información del
    pliegue de validación al entrenamiento y produciría un desempeño optimista.
    """
    from sklearn.base import clone

    return clone(prep).fit(X_entrenamiento)


# ─────────────────────────────────────────────────────────────────────────────
# Pasos 4 y 5 · Selección de hiperparámetros y reentrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def seleccionar_hiperparametros(estimador, grilla, X, y, k: int | None = None,
                                semilla: int | None = None, metrica: str | None = None):
    """**Paso 4.** Selecciona los hiperparámetros por validación cruzada interna.

    La búsqueda ocurre solo sobre los datos de entrenamiento del pliegue externo. Si el
    algoritmo no tiene grilla, este paso se omite.
    """
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    if not grilla:
        return clone(estimador), {}
    cv = StratifiedKFold(n_splits=k or config.CV_INTERNO, shuffle=True,
                         random_state=config.SEED if semilla is None else semilla)
    busqueda = GridSearchCV(clone(estimador), grilla, cv=cv,
                            scoring=metrica or config.METRICA, n_jobs=-1, refit=True)
    busqueda.fit(X, y)
    return busqueda.best_estimator_, busqueda.best_params_


def reentrenar(estimador, X, y):
    """**Paso 5.** Reentrena sobre el pliegue de entrenamiento completo."""
    from sklearn.base import clone

    return clone(estimador).fit(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# Pasos 6 y 7 · Aplicación al pliegue retenido
# ─────────────────────────────────────────────────────────────────────────────

def aplicar_preprocesamiento(prep_ajustado, X_validacion: pd.DataFrame):
    """**Paso 6.** Aplica al pliegue retenido el preprocesamiento del entrenamiento."""
    return prep_ajustado.transform(X_validacion)


def predecir_fuera_de_pliegue(modelo, X_validacion) -> np.ndarray:
    """**Paso 7.** Probabilidad predicha para el pliegue de validación."""
    return modelo.predict_proba(X_validacion)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Orquestación
# ─────────────────────────────────────────────────────────────────────────────

def validar_cruzado(datos: Datos, algoritmos: dict | None = None, *,
                    particiones=None, semilla: int | None = None,
                    verbose: bool = True) -> pd.DataFrame:
    """Recorre los siete pasos y consolida las predicciones fuera de pliegue.

    Todos los algoritmos se evalúan sobre los mismos pliegues, de modo que sus diferencias
    de desempeño no provengan de particiones distintas. Devuelve una fila por paciente y
    algoritmo, con la probabilidad predicha cuando esa paciente estuvo en el pliegue
    retenido, junto a los hiperparámetros elegidos en cada pliegue.
    """
    algoritmos = algoritmos or catalogo_algoritmos(semilla)
    # `particiones` admite el número de pliegues o los pliegues ya construidos: en el resto
    # del módulo el nombre designa una cantidad, y recibir aquí solo una lista invitaba al error.
    if particiones is None or isinstance(particiones, int):
        particiones = particion_externa(datos.y, particiones, semilla=semilla)
    particiones = list(particiones)
    prep = construir_preprocesamiento(datos.tipos)

    filas, elegidos = [], []
    for nombre, alg in algoritmos.items():
        if verbose:
            print(f"  {nombre:11s}", end=" ", flush=True)
        oof = np.full(datos.n, np.nan)
        for i, (tr, va) in enumerate(particiones):
            X_tr, X_va = datos.X.iloc[tr], datos.X.iloc[va]
            prep_i = ajustar_preprocesamiento(prep, X_tr)               # paso 3
            Z_tr = prep_i.transform(X_tr)
            est, params = seleccionar_hiperparametros(                   # paso 4
                alg["estimador"], alg["grilla"], Z_tr, datos.y[tr], semilla=semilla)
            modelo = reentrenar(est, Z_tr, datos.y[tr])                  # paso 5
            Z_va = aplicar_preprocesamiento(prep_i, X_va)                # paso 6
            oof[va] = predecir_fuera_de_pliegue(modelo, Z_va)            # paso 7
            # Formato largo: cada algoritmo tiene hiperparámetros distintos, y algunos
            # mezclan texto con números en un mismo parámetro. Una columna por parámetro
            # daría un marco disperso y de tipos mixtos, que ningún formato columnar admite.
            elegidos += [{"algoritmo": nombre, "pliegue": i, "hiperparametro": k,
                          "valor": str(v)} for k, v in (params or {}).items()] or \
                        [{"algoritmo": nombre, "pliegue": i, "hiperparametro": "—",
                          "valor": "sin hiperparámetros"}]
            if verbose:
                print("·", end="", flush=True)
        filas.append(pd.DataFrame({"algoritmo": nombre, "indice": np.arange(datos.n),
                                   "y": datos.y, "p": oof}))
        if verbose:
            print(" ✔")

    d = pd.concat(filas, ignore_index=True)
    d["cohorte"], d["escenario"] = datos.cohorte, datos.escenario
    d["especificacion"] = datos.especificacion
    # Lista de diccionarios, no un marco: un marco dentro de `attrs` rompe tanto la
    # escritura columnar como la concatenación, porque pandas compara los atributos con
    # una igualdad que sobre un marco no devuelve un booleano.
    d.attrs["hiperparametros"] = elegidos
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Diagnósticos del diseño
# ─────────────────────────────────────────────────────────────────────────────

def epv(datos: Datos, n_parametros: int | None = None) -> dict:
    """Eventos por variable: relación entre eventos observados y parámetros estimados.

    Una razón baja favorece el sobreajuste y vuelve inestables las estimaciones; suele
    tomarse diez como mínimo orientador.
    """
    k = n_parametros if n_parametros is not None else datos.X.shape[1]
    return {"n": datos.n, "eventos": datos.eventos, "n_parametros": k,
            "EPV": round(datos.eventos / k, 2) if k else np.nan,
            "cumple": bool(k and datos.eventos / k >= config.EPV_MIN)}


def vif_explicativo(datos: pd.DataFrame, esp: "Configuracion", cat: pd.DataFrame,
                    desenlace: str | None = None, **kw) -> pd.DataFrame:
    """Colinealidad del modelo explicativo, por variable.

    El factor de inflación de la varianza clásico se define por término, de modo que en
    una variable de tres o más niveles depende de cuál se haya tomado como referencia:
    una categoría de referencia poco frecuente infla el factor de todas las demás sin que
    exista colinealidad sustantiva, porque los indicadores de una misma variable son
    mutuamente excluyentes por construcción. Se reporta entonces el factor generalizado
    (Fox y Monette, 1992), que es invariante a la parametrización y entrega un valor por
    variable. Su forma escalada, elevada al cuadrado, es comparable con el factor clásico
    y coincide con él en las variables numéricas y binarias, que aportan un solo término.
    """
    desenlace = desenlace or config.OUTCOME
    obs = pd.to_numeric(datos[desenlace], errors="coerce").notna()
    M, mapa = diseno(datos.loc[obs], esp, cat, **kw)
    M = M.loc[M.notna().all(axis=1)].astype(float).drop(columns="Intercept")
    R = np.corrcoef(M.values, rowvar=False)
    meta = cat.set_index("var_rename")
    det = np.linalg.det

    filas = []
    for v, g in mapa.groupby("var_rename", sort=False):
        i = [M.columns.get_loc(t) for t in g["termino"] if t in M.columns]
        if not i:
            continue
        j = [k for k in range(R.shape[0]) if k not in i]
        gvif = det(R[np.ix_(i, i)]) * det(R[np.ix_(j, j)]) / det(R) if j else 1.0
        gl = len(i)
        filas.append({"var_rename": v, "label": str(meta.loc[v, "label"]) if v in meta.index else v,
                      "gl": gl, "GVIF": round(float(gvif), 3),
                      "GVIF_escalado": round(float(gvif) ** (1 / (2 * gl)), 3),
                      "VIF_equivalente": round(float(gvif) ** (1 / gl), 2)})
    return pd.DataFrame(filas).sort_values("VIF_equivalente", ascending=False).reset_index(drop=True)


def vif(datos: Datos) -> pd.DataFrame:
    """Factor de inflación de la varianza sobre la matriz preprocesada.

    Se calcula con intercepto: sin él, la colinealidad con la constante infla los valores
    de variables perfectamente razonables. Los indicadores de una nominal expandida por
    completo son linealmente dependientes entre sí, de modo que su factor resulta
    infinito por construcción y no expresa colinealidad real.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    prep = construir_preprocesamiento(datos.tipos).fit(datos.X)
    Z = pd.DataFrame(prep.transform(datos.X),
                     columns=[c.split("__", 1)[-1] for c in prep.get_feature_names_out()])
    Z = Z.loc[:, Z.std() > 0]
    M = np.column_stack([np.ones(len(Z)), Z.to_numpy(float)])
    filas = [{"variable": c, "VIF": round(variance_inflation_factor(M, i + 1), 2)}
             for i, c in enumerate(Z.columns)]
    return pd.DataFrame(filas).sort_values("VIF", ascending=False).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Arco explicativo (docs/cleanse.md §4c)
#
# La unidad es la ESPECIFICACIÓN: un conjunto nombrado de covariables. Sobre ella
# operan seis funciones —ajustar, escalonar, contrastar, consistir, diagnosticar y
# estabilizar— que son también los pasos que el notebook orquesta.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Configuracion:
    """Conjunto nombrado de covariables de un modelo explicativo.

    Se llama configuración y no especificación porque este trabajo reserva ese término
    para los dos conjuntos de información contrastados en el arco predictivo, el
    preoperatorio y el perioperatorio. La configuración es dato, no código: declarar una
    nueva no exige tocar el módulo. Sobre esa propiedad descansan la escalera de modelos
    anidados y la reutilización del arco en la subcohorte espinal.
    """
    nombre: str
    covariables: list[str]
    descripcion: str = ""
    foco: tuple[str, ...] = ()
    padre: str | None = None

    @property
    def formula(self) -> str:
        """Lado derecho de la fórmula del modelo."""
        return " + ".join(self.covariables)

    def mas(self, nombre: str, extra, descripcion: str = "", *,
            padre: str | None = None) -> "Configuracion":
        """Nueva configuración con covariables añadidas: un peldaño de la escalera.

        Las covariables añadidas quedan registradas como foco del peldaño, porque son las
        únicas cuya estimación admite lectura ajustada en ese modelo.
        """
        extra = [extra] if isinstance(extra, str) else list(extra)
        nuevas = [v for v in extra if v not in self.covariables]
        return Configuracion(nombre, self.covariables + nuevas, descripcion,
                             foco=tuple(nuevas), padre=padre or self.nombre)

    def menos(self, nombre: str, quitar, descripcion: str = "", *,
              padre: str | None = None) -> "Configuracion":
        """Nueva configuración sin ciertas covariables.

        Conserva el foco, de modo que pueda observarse cómo cambia su estimación al
        retirar una covariable del ajuste. Qué signifique ese cambio depende del diseño
        del estudio y no se decide aquí.
        """
        quitar = {quitar} if isinstance(quitar, str) else set(quitar)
        return Configuracion(nombre, [v for v in self.covariables if v not in quitar],
                             descripcion, foco=self.foco, padre=padre or self.nombre)

    def __repr__(self) -> str:
        f = f", foco={list(self.foco)}" if self.foco else ""
        return f"Configuracion({self.nombre}: {len(self.covariables)} covariables{f})"


def configuracion_del_catalogo(cat: pd.DataFrame, nombre: str = "base",
                               columna: str = "include_explicativo") -> "Configuracion":
    """Construye la configuración declarada en el catálogo."""
    vs = cat.loc[cat[columna].fillna(False), "var_rename"].tolist()
    return Configuracion(nombre, vs, "Conjunto declarado en el catálogo")


# ── Diseño del modelo ────────────────────────────────────────────────────────

def niveles_declarados(cat: pd.DataFrame, var: str, serie: pd.Series | None = None) -> list[str]:
    """Niveles de una variable categórica, en su orden clínico.

    El orden proviene del catálogo, no del dato: si dependiera del dato, la referencia
    del modelo cambiaría al reordenar las filas o al cambiar los faltantes.
    """
    import ast

    meta = cat.set_index("var_rename")
    tipo = str(meta.loc[var, "conceptual_type"]).lower() if var in meta.index else "nom"
    if tipo == "bin":
        return ["No", "S\u00ed"]
    decl = meta.loc[var, "categories"] if var in meta.index else None
    if isinstance(decl, str) and decl.strip():
        try:
            vs = ast.literal_eval(decl)
            if isinstance(vs, (list, tuple)):
                return [str(x).strip() for x in vs]
        except (ValueError, SyntaxError):
            return [x.strip() for x in decl.split("|")]
    if serie is not None:
        if isinstance(serie.dtype, pd.CategoricalDtype):
            return [str(x) for x in serie.cat.categories]
        return sorted(str(x) for x in serie.dropna().unique())
    return []


def referencia(cat: pd.DataFrame, var: str, serie: pd.Series | None = None,
               overrides: dict | None = None) -> str:
    """Nivel de referencia de una variable categórica.

    Es la primera categoría declarada, salvo anulación explícita. Fijarla por catálogo
    y no por el dato evita que la base de comparación dependa de qué fila viene primero
    o de cuántos faltantes tenga la variable.
    """
    if overrides and var in overrides:
        return str(overrides[var])
    vs = niveles_declarados(cat, var, serie)
    return vs[0] if vs else ""


def _a_categoria(s: pd.Series) -> pd.Series:
    """Normaliza una columna categórica a texto, con las binarias como No/Si."""
    if pd.api.types.is_bool_dtype(s) or isinstance(s.dtype, pd.BooleanDtype):
        b = s.astype("boolean")
        out = pd.Series(np.nan, index=s.index, dtype=object)
        out[b.eq(True)], out[b.eq(False)] = "S\u00ed", "No"
        return out
    return s.astype(object).where(pd.notna(s), np.nan)


def diseno(datos: pd.DataFrame, esp: "Configuracion", cat: pd.DataFrame, *,
           imputacion: str = "simple", referencias: dict | None = None,
           intercepto: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matriz de diseno del modelo explicativo y el mapa de sus terminos.

    Cada variable categorica se expande en indicadores contra su nivel de referencia.
    Con `imputacion="simple"` los faltantes se completan con la mediana o la moda, de
    modo que el modelo conserve la cohorte completa, con `imputacion="completos"` se
    ajusta solo sobre casos completos. La eleccion se declara: la primera preserva el
    tamano muestral a costa de tratar como observado lo que no lo fue, la segunda es
    fiel al dato observado a costa de perder casos.
    """
    meta = cat.set_index("var_rename")
    X, mapa = [], []
    for v in esp.covariables:
        if v not in datos.columns:
            continue
        tipo = str(meta.loc[v, "conceptual_type"]).lower() if v in meta.index else "nom"
        s = datos[v]
        if tipo == "num":
            x = pd.to_numeric(s, errors="coerce")
            if imputacion == "simple":
                x = x.fillna(x.median())
            X.append(x.rename(v).astype(float))
            mapa.append({"termino": v, "var_rename": v, "nivel": np.nan,
                         "referencia": np.nan, "tipo": "numerica"})
            continue
        c = _a_categoria(s)
        if imputacion == "simple":
            moda = c.dropna().mode()
            if len(moda):
                c = c.fillna(moda.iloc[0])
        obs = set(c.dropna())
        niveles = [n for n in niveles_declarados(cat, v, s) if n in obs]
        ref = referencia(cat, v, s, referencias)
        ref = ref if ref in niveles else (niveles[0] if niveles else "")
        for niv in [n for n in niveles if n != ref]:
            X.append(pd.Series(np.where(c.eq(niv), 1.0, np.where(c.isna(), np.nan, 0.0)),
                               index=datos.index, name=f"{v}__{niv}"))
            mapa.append({"termino": f"{v}__{niv}", "var_rename": v, "nivel": niv,
                         "referencia": ref, "tipo": "categorica"})
    M = pd.concat(X, axis=1) if X else pd.DataFrame(index=datos.index)
    if intercepto:
        M.insert(0, "Intercept", 1.0)
    return M, pd.DataFrame(mapa)


def _etiqueta(fila: pd.Series, meta: pd.DataFrame) -> str:
    """Etiqueta legible de un termino, con su nivel de referencia."""
    v = fila["var_rename"]
    if v == "Intercept":
        return "Intercepto"
    lbl = str(meta.loc[v, "label"]) if v in meta.index else v
    if isinstance(fila.get("nivel"), str):
        return f"{lbl}: {fila['nivel']} (ref: {fila['referencia']})"
    return lbl


FAMILIAS = ("logit", "ordinal", "lineal")


def ajustar(datos: pd.DataFrame, esp: Configuracion, cat: pd.DataFrame,
            desenlace: str | None = None, *, familia: str = "logit",
            imputacion: str = "simple", referencias: dict | None = None) -> pd.DataFrame:
    """**Ajustar.** Estima el modelo multivariable de una configuración.

    La familia se elige según la escala del desenlace, no según el gusto: `logit` para un
    desenlace binario, `ordinal` para uno de categorías ordenadas —modelo de razones
    proporcionales, sin intercepto porque los umbrales lo absorben— y `lineal` para uno
    continuo, ajustado de forma robusta para que unos pocos valores extremos no gobiernen
    la pendiente. Devuelve una fila por término, con la medida de efecto que corresponde a
    cada familia y su intervalo de confianza al 95 %.
    """
    import statsmodels.api as sm

    if familia not in FAMILIAS:
        raise ValueError(f"familia debe ser una de {FAMILIAS}, se recibió {familia!r}")
    desenlace = desenlace or config.OUTCOME
    y = pd.to_numeric(datos[desenlace], errors="coerce")
    obs = y.notna()
    M, mapa = diseno(datos.loc[obs], esp, cat, imputacion=imputacion,
                     referencias=referencias, intercepto=familia != "ordinal")
    completos = M.notna().all(axis=1)
    M = M.loc[completos].astype(float)
    yv = y[obs][completos]

    if familia == "logit":
        modelo = sm.GLM(yv.astype(float), M, family=sm.families.Binomial()).fit()
    elif familia == "ordinal":
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        modelo = OrderedModel(yv.astype(int), M, distr="logit").fit(
            method="bfgs", disp=False, maxiter=500)
    else:
        modelo = sm.RLM(yv.astype(float), M, M=sm.robust.norms.HuberT()).fit(maxiter=200)

    ic = modelo.conf_int()
    out = pd.DataFrame({"termino": modelo.params.index, "beta": modelo.params.values,
                        "ic_beta_low": ic.iloc[:, 0].values,
                        "ic_beta_high": ic.iloc[:, 1].values,
                        "p": modelo.pvalues.values})
    # Los umbrales del modelo ordinal no son términos del diseño: se separan.
    out["umbral"] = out["termino"].str.contains("/", regex=False)
    if familia == "lineal":
        out["efecto"], out["ic_low"], out["ic_high"] = (out["beta"], out["ic_beta_low"],
                                                        out["ic_beta_high"])
    else:
        out["OR"] = np.exp(out["beta"])
        out["ic_low"], out["ic_high"] = np.exp(out["ic_beta_low"]), np.exp(out["ic_beta_high"])
        out["efecto"] = out["OR"]

    meta = cat.set_index("var_rename")
    mapa = pd.concat([pd.DataFrame([{"termino": "Intercept", "var_rename": "Intercept"}]), mapa],
                     ignore_index=True)
    out = out.merge(mapa, on="termino", how="left")
    out["var_rename"] = out["var_rename"].fillna(
        out["termino"].where(out["umbral"], out["termino"].str.split("__").str[0]))
    out["label"] = [_etiqueta(f, meta) if not f["umbral"] else f["termino"]
                    for _, f in out.iterrows()]
    out["en_foco"] = out["var_rename"].isin(esp.foco) if esp.foco else False

    a = {"configuracion": esp.nombre, "desenlace": desenlace, "familia": familia,
         "n": len(yv), "n_parametros": M.shape[1], "imputacion": imputacion, "modelo": modelo}
    if familia == "logit":
        nula = sm.GLM(yv.astype(float), np.ones((len(yv), 1)), family=sm.families.Binomial()).fit()
        ll, ll0, n = modelo.llf, nula.llf, len(yv)
        a.update(eventos=int(yv.sum()), llf=ll, aic=modelo.aic, bic=modelo.bic_llf,
                 r2_mcfadden=1 - ll / ll0,
                 r2_nagelkerke=(1 - np.exp((ll0 - ll) * 2 / n)) / (1 - np.exp(ll0 * 2 / n)))
    elif familia == "ordinal":
        a.update(eventos=int((yv > yv.min()).sum()), llf=modelo.llf,
                 aic=modelo.aic, bic=modelo.bic)
    else:
        a.update(eventos=len(yv), escala=float(modelo.scale))
    out.attrs.update(a)
    return out


def efecto_por_variable(t: pd.DataFrame) -> pd.DataFrame:
    """Resume un ajuste a un efecto por variable, el del término más significativo.

    Una variable de varios niveles aporta varios términos y no un único número. Para
    comparar entre desenlaces se conserva el término de menor valor p y, ante empate, el de
    mayor magnitud. El resumen sirve para contrastar dirección y consistencia, no para
    reportar la estimación: el término elegido depende del nivel de referencia.
    """
    d = t[~t["umbral"] & t["termino"].ne("Intercept")].copy()
    if d.empty:
        return d
    d["_p"] = d["p"].fillna(np.inf)
    d["_m"] = d["beta"].abs()
    d = d.sort_values(["var_rename", "_p", "_m"], ascending=[True, True, False])
    d = d.groupby("var_rename", as_index=False).head(1)
    d["direccion"] = np.sign(d["beta"]).astype(int)
    return d.drop(columns=["_p", "_m"]).reset_index(drop=True)


def escalera(datos: pd.DataFrame, configuraciones: list[Configuracion], cat: pd.DataFrame,
             desenlace: str | None = None, *, padres: dict | None = None, **kw) -> pd.DataFrame:
    """**Escalonar.** Compara configuraciones anidadas y el aporte de cada bloque.

    La escalera no es necesariamente una cadena. Cada peldaño declara su padre, de modo que
    varios bloques distintos puedan contrastarse contra un mismo modelo base y que un peldaño
    pueda además **restar** una covariable de su padre para distinguir la asociación total de
    la que no transita por ella. El contraste es la razón de verosimilitud entre el modelo con
    más parámetros y el que tiene menos, cualquiera sea el orden en que se declararon.

    Se acompaña de los criterios de información y de los eventos por parámetro, porque un
    bloque puede mejorar la verosimilitud a costa de una complejidad que la muestra no sostiene.
    """
    from scipy import stats

    ajustes = {c.nombre: ajustar(datos, c, cat, desenlace, **kw) for c in configuraciones}
    padres = padres or {c.nombre: c.padre for c in configuraciones}

    filas = []
    for c in configuraciones:
        a = ajustes[c.nombre].attrs
        fila = {"peldano": c.nombre, "descripcion": c.descripcion,
                "foco": ", ".join(c.foco), "n": a["n"], "eventos": a["eventos"],
                "n_parametros": a["n_parametros"], "AIC": round(a["aic"], 2),
                "BIC": round(a["bic"], 2), "R2_McFadden": round(a["r2_mcfadden"], 4),
                "R2_Nagelkerke": round(a["r2_nagelkerke"], 4),
                "EPV": round(a["eventos"] / (a["n_parametros"] - 1), 2)}
        p = padres.get(c.nombre)
        if p and p in ajustes:
            b = ajustes[p].attrs
            completo, reducido = ((a, b) if a["n_parametros"] >= b["n_parametros"] else (b, a))
            gl = completo["n_parametros"] - reducido["n_parametros"]
            est = 2 * (completo["llf"] - reducido["llf"])
            fila.update({"contraste_vs": p, "lrt": round(est, 3), "gl": gl,
                         "p_lrt": float(stats.chi2.sf(est, gl)) if gl > 0 else np.nan})
        filas.append(fila)
    return pd.DataFrame(filas)


def coeficientes(datos: "Datos", cat: pd.DataFrame) -> pd.DataFrame:
    """Odds ratio de una regresión logística ajustada sobre la matriz **ya preprocesada**.

    No es lo mismo que `ajustar`. Aquí el diseño es el que consume el arco predictivo, de
    modo que las ordinales entran como una sola columna con su orden clínico y las numéricas
    estandarizadas. En consecuencia el OR de una ordinal es el paso de un nivel y el de una
    numérica una desviación estándar, mientras que `ajustar` expande cada nivel contra su
    referencia. Reportar una parametrización y citar la otra produce cifras que no cuadran.

    El preprocesamiento se estima sobre el conjunto completo, que es lo correcto para una
    lectura descriptiva de la dirección de las asociaciones y no para estimar desempeño.
    """
    import statsmodels.api as sm

    prep = construir_preprocesamiento(datos.tipos).fit(datos.X)
    M = pd.DataFrame(prep.transform(datos.X), columns=prep.get_feature_names_out())
    M = sm.add_constant(M.astype(float))
    r = sm.GLM(np.asarray(datos.y, dtype=float), M, family=sm.families.Binomial()).fit()
    ic = r.conf_int()

    vs = list(datos.X.columns)
    meta = cat.set_index("var_rename")
    out = pd.DataFrame({"termino": r.params.index, "beta": r.params.values,
                        "p": r.pvalues.values,
                        "OR": np.exp(r.params.values),
                        "ic_low": np.exp(ic.iloc[:, 0].values),
                        "ic_high": np.exp(ic.iloc[:, 1].values)})
    out = out[out["termino"].ne("const")].reset_index(drop=True)
    out["var_rename"] = [_raiz(t, vs) for t in out["termino"]]
    out["tipo"] = out["var_rename"].map(
        lambda v: str(meta.loc[v, "conceptual_type"]).lower() if v in meta.index else "")
    out["label"] = out["var_rename"].map(
        lambda v: str(meta.loc[v, "label"]) if v in meta.index else v)
    # Una nominal aporta un término por nivel: se distingue por el sufijo del diseño.
    resto = [t.split("__", 1)[-1] for t in out["termino"]]
    out["nivel"] = [r_.replace(f"{v}_", "", 1) if r_ != v else None
                    for r_, v in zip(resto, out["var_rename"])]
    return out


def cambio_en_estimacion(datos: pd.DataFrame, configuraciones: list[Configuracion],
                         cat: pd.DataFrame, desenlace: str | None = None, *,
                         contra: str = "base", **kw) -> pd.DataFrame:
    """**Contrastar.** Mide cuánto se desplaza cada odds ratio entre peldaños.

    Un desplazamiento apreciable al incorporar un bloque indica que las estimaciones del
    modelo previo no eran independientes de ese bloque. Su lectura complementa a la razón de
    verosimilitud: un bloque puede no mejorar el ajuste y aun así desplazar sustantivamente
    las estimaciones. La función describe el desplazamiento, no lo interpreta.

    Con `contra="base"` todos los peldaños se comparan con el primero, que es lo que
    corresponde cuando la escalera abre varios bloques alternativos sobre un mismo modelo.
    Con `contra="consecutivo"` cada uno se compara con el anterior, que es la lectura propia
    de una cadena acumulativa.
    """
    ajustes = [ajustar(datos, c, cat, desenlace, **kw).set_index("termino")
               for c in configuraciones]
    filas = []
    for k in range(1, len(configuraciones)):
        ref = ajustes[0] if contra == "base" else ajustes[k - 1]
        nom = configuraciones[0].nombre if contra == "base" else configuraciones[k - 1].nombre
        cur = ajustes[k]
        for t in ref.index:
            if t == "Intercept" or t not in cur.index:
                continue
            a, b = ref.loc[t, "OR"], cur.loc[t, "OR"]
            filas.append({"peldano": configuraciones[k].nombre, "referencia": nom,
                          "termino": t, "label": ref.loc[t, "label"],
                          "OR_referencia": round(a, 3), "OR_peldano": round(b, 3),
                          "cambio_pct": round(100 * (b - a) / a, 2) if a else np.nan})
    d = pd.DataFrame(filas)
    if not d.empty:
        d["desplazamiento_relevante"] = d["cambio_pct"].abs().ge(config.CAMBIO_ESTIMACION_UMBRAL)
    return d


def dejar_centro_fuera(datos: pd.DataFrame, esp: Configuracion, cat: pd.DataFrame,
                       termino: str, desenlace: str | None = None, *,
                       centro: str | None = None, **kw) -> pd.DataFrame:
    """Reajusta la configuración excluyendo cada centro, uno a la vez.

    Sirve para saber si una asociación depende de un conglomerado en particular. Cuando la
    exposición se distribuye de forma muy desigual entre conglomerados, su coeficiente puede
    estar recogiendo la práctica del conglomerado, y eso se hace visible si la estimación se
    desploma o cambia de signo al retirarlo.
    """
    centro = centro or config.CENTER_VAR
    filas = []
    for excluido in [None] + sorted(datos[centro].dropna().unique().tolist()):
        d = datos if excluido is None else datos[datos[centro].ne(excluido)].copy()
        if excluido is not None and isinstance(d[centro].dtype, pd.CategoricalDtype):
            d[centro] = d[centro].cat.remove_unused_categories()
        t = ajustar(d, esp, cat, desenlace, **kw)
        for _, r in t[t["var_rename"].eq(termino)].iterrows():
            filas.append({"centro_excluido": "Ninguno" if excluido is None else excluido,
                          "n": t.attrs["n"], "eventos": t.attrs["eventos"],
                          "termino": r["termino"], "label": r["label"],
                          "OR": round(r["OR"], 3), "ic_low": round(r["ic_low"], 3),
                          "ic_high": round(r["ic_high"], 3), "p": r["p"]})
    d = pd.DataFrame(filas)
    if not d.empty:
        # La referencia se toma dentro de cada término, no de la primera fila del marco: una
        # variable de varios niveles aporta un término por nivel y compararlos todos contra
        # uno solo mezclaría contrastes distintos.
        ref = d.groupby("termino")["OR"].transform("first")
        d["cambio_pct"] = (100 * (d["OR"] - ref) / ref).round(2)
        d["signo_invertido"] = (d["OR"] - 1).mul(ref - 1).lt(0)
    return d


def consistencia(datos: pd.DataFrame, esp: Configuracion, cat: pd.DataFrame,
                 desenlaces, *, umbral: float = 0.05, **kw) -> pd.DataFrame:
    """**Consistir.** Reajusta la misma configuración sobre varias definiciones del desenlace.

    Una asociación que conserva dirección y significación bajo distintas
    operacionalizaciones resulta más difícil de atribuir al azar o a la definición elegida.
    Cada desenlace se ajusta con la familia que corresponde a su escala, de modo que la
    comparación no imponga una escala común artificial.

    `desenlaces` admite una lista, en cuyo caso se asume familia binomial, o un diccionario
    que asigna a cada desenlace su familia. Se resume un efecto por variable, el del término
    más significativo, porque una variable de varios niveles no tiene un único coeficiente.
    """
    if not isinstance(desenlaces, dict):
        desenlaces = {d: "logit" for d in desenlaces}

    filas = []
    for des, fam in desenlaces.items():
        if des not in datos.columns:
            continue
        t = ajustar(datos, esp, cat, des, familia=fam, **kw)
        for _, r in efecto_por_variable(t).iterrows():
            filas.append({"desenlace": des, "familia": fam, "var_rename": r["var_rename"],
                          "label": r["label"], "termino": r["termino"],
                          "efecto": round(float(r["efecto"]), 4), "p": float(r["p"]),
                          "senal": bool(r["p"] < umbral), "direccion": int(r["direccion"])})
    d = pd.DataFrame(filas)
    if d.empty:
        return d

    g = d.groupby("var_rename")
    # La consistencia se evalúa sobre TODAS las definiciones, no solo sobre aquellas donde
    # hay señal: una variable que invierte su signo donde no alcanza significación tampoco
    # sostiene una lectura estable.
    resumen = pd.DataFrame({
        "n_desenlaces_con_senal": g["senal"].sum().astype(int),
        "direccion_consistente": g["direccion"].apply(
            lambda x: x[x.ne(0)].nunique() <= 1)})
    return d.merge(resumen, left_on="var_rename", right_index=True, how="left")


def clasificar_consistencia(d: pd.DataFrame, *, temprano=(), tardio=(), gravedad=(),
                            minimo: int = 3) -> pd.DataFrame:
    """Clasifica cada variable según el patrón de señal a lo largo de las definiciones.

    Distingue a las variables que se asocian de forma estable en varias definiciones de
    aquellas cuya señal aparece solo en un momento del seguimiento o solo en las
    definiciones de gravedad. Qué desenlace es temprano, cuál tardío y cuáles miden
    gravedad lo declara quien llama, no el módulo.
    """
    filas = []
    for v, g in d.groupby("var_rename", sort=False):
        p = g.set_index("desenlace")["senal"]
        pre = any(bool(p.get(x, False)) for x in temprano)
        pos = any(bool(p.get(x, False)) for x in tardio)
        gra = any(bool(p.get(x, False)) for x in gravedad)
        n = int(g["n_desenlaces_con_senal"].iloc[0])
        cons = bool(g["direccion_consistente"].iloc[0])
        if n >= minimo and cons:      c = "consistente_global"
        elif pre and not pos:         c = "asociado_dolor_temprano"
        elif pos and not pre:         c = "asociado_dolor_tardio"
        elif gra and n >= 1:          c = "asociado_gravedad"
        elif n >= 2 and not cons:     c = "asociacion_inestable"
        else:                         c = "exploratorio"
        filas.append({"var_rename": v, "clasificacion": c})
    return d.merge(pd.DataFrame(filas), on="var_rename", how="left")


def estabilidad_seleccion(datos: pd.DataFrame, variables: list[str], cat: pd.DataFrame,
                          desenlace: str | None = None, *, particiones: int | None = None,
                          repeticiones: int | None = None, semilla: int | None = None
                          ) -> pd.DataFrame:
    """**Estabilizar.** Frecuencia con que cada variable sobrevive al remuestreo.

    Se ajusta repetidamente un modelo penalizado sobre particiones estratificadas de la
    cohorte y se registra en qué proporción cada variable conserva un coeficiente no nulo.
    La penalización se elige por validación interna dentro de cada remuestra, de modo que
    la frecuencia mida la persistencia de la variable y no el efecto de un valor fijado de
    antemano. Se acompaña del signo dominante y de su consistencia, porque una variable
    seleccionada con frecuencia pero con signo alternante no sostiene una interpretación.

    El criterio informa la decisión, no la reemplaza, una variable inestable puede seguir
    en el modelo por plausibilidad clínica, declarada como tal en el catálogo.
    """
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
    from sklearn.pipeline import Pipeline

    k = particiones or config.ESTABILIDAD_PARTICIONES
    r = repeticiones or config.ESTABILIDAD_REPETICIONES
    s = config.SEED if semilla is None else semilla
    desenlace = desenlace or config.OUTCOME

    vs = [v for v in variables if v in datos.columns]
    y = pd.to_numeric(datos[desenlace], errors="coerce")
    obs = y.notna()
    X, y = datos.loc[obs, vs].copy(), y[obs].astype(int)

    meta = cat.set_index("var_rename")
    tipos = {t: [] for t in ("num", "bin", "ord", "nom")}
    for v in vs:
        t = str(meta.loc[v, "conceptual_type"]).lower() if v in meta.index else "nom"
        tipos[t if t in tipos else "nom"].append(v)
    prep = construir_preprocesamiento({t: c for t, c in tipos.items() if c})
    base = LogisticRegression(penalty="l1", solver="saga", max_iter=10_000, random_state=s)

    registros = []
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=r, random_state=s)
    for i, (ent, _) in enumerate(cv.split(X, y), start=1):
        busqueda = GridSearchCV(
            Pipeline([("pre", clone(prep)), ("clf", clone(base))]),
            {"clf__C": list(config.ESTABILIDAD_C)}, scoring=config.METRICA,
            cv=StratifiedKFold(k, shuffle=True, random_state=s + i), refit=True, n_jobs=-1)
        busqueda.fit(X.iloc[ent], y.iloc[ent])
        mejor = busqueda.best_estimator_
        nombres = mejor.named_steps["pre"].get_feature_names_out()
        coef = mejor.named_steps["clf"].coef_.ravel()
        for v in vs:
            m = [j for j, n in enumerate(nombres) if _raiz(n, vs) == v]
            act = [j for j in m if coef[j] != 0]
            if act:
                top = max(act, key=lambda j: abs(coef[j]))
                registros.append({"var_rename": v, "seleccionada": 1,
                                  "signo": float(np.sign(coef[top]))})
            else:
                registros.append({"var_rename": v, "seleccionada": 0, "signo": np.nan})

    reg = pd.DataFrame(registros)
    filas = []
    for v in vs:
        sub = reg[reg["var_rename"].eq(v)]
        freq = float(sub["seleccionada"].mean())
        sig = sub.loc[sub["seleccionada"].eq(1), "signo"]
        pos, neg = int(sig.eq(1).sum()), int(sig.eq(-1).sum())
        dom = "positivo" if pos > neg else ("negativo" if neg > pos else "indefinido")
        cons = max(pos, neg) / len(sig) if len(sig) else np.nan
        filas.append({"var_rename": v,
                      "label": str(meta.loc[v, "label"]) if v in meta.index else v,
                      "frecuencia": round(freq, 2), "signo": dom,
                      "consistencia_signo": round(cons, 2) if len(sig) else np.nan,
                      "estabilidad": ("fuerte" if freq >= config.ESTABILIDAD_UMBRAL_FUERTE
                                      else "moderada" if freq >= config.ESTABILIDAD_UMBRAL
                                      else "inestable")})
    d = pd.DataFrame(filas)
    d["estable"] = d["frecuencia"].ge(config.ESTABILIDAD_UMBRAL)
    return d.sort_values("frecuencia", ascending=False).reset_index(drop=True)


def _raiz(termino: str, variables: list[str]) -> str:
    """Variable de origen de un termino del preprocesamiento."""
    n = termino.split("__", 1)[-1]
    if n in variables:
        return n
    candidatos = [v for v in variables if n.startswith(v)]
    return max(candidatos, key=len) if candidatos else n
