"""evaluar.py — desempeño, incertidumbre y comparación de los modelos predictivos.

Opera sobre las predicciones fuera de pliegue que produce `modell.validar_cruzado`, nunca
sobre modelos reentrenados: la separación permite recalcular métricas e intervalos sin
repetir la validación cruzada.

**Una sola extracción alimenta toda la incertidumbre.** Los puntos estimados son
deterministas y se calculan una vez sobre el vector completo. Los intervalos y los valores p
provienen de un único remuestreo estratificado por paciente, compartido entre las medidas
marginales y las diferencias pareadas. Estimarlos por separado, con semillas distintas,
rompe la correspondencia entre un intervalo y la diferencia que lo acompaña, y hace que dos
cifras del mismo cuadro no provengan de las mismas remuestras.

El remuestreo reproduce la variabilidad de la muestra de evaluación, no la del
entrenamiento: no reajusta los modelos en cada réplica, de modo que los intervalos del
desempeño absoluto pueden resultar algo estrechos. La comparación pareada es menos sensible
a esa limitación, porque al estimarse sobre las mismas remuestras cancela el ruido común.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_EPS = 1e-9
CLAVES = ("cohorte", "escenario", "especificacion", "algoritmo")
MEDIDAS = ("auc", "brier", "pendiente", "intercepto", "citl")


# ─────────────────────────────────────────────────────────────────────────────
# Medidas deterministas
# ─────────────────────────────────────────────────────────────────────────────

def _logit(x):
    x = np.clip(np.asarray(x, dtype=float), _EPS, 1 - _EPS)
    return np.log(x / (1 - x))


def recalibracion(y, p, *, iteraciones: int = 50, tol: float = 1e-10) -> tuple[float, float]:
    """Pendiente e intercepto de la recta de recalibración logística.

    Se reajusta el desenlace sobre el logit de la probabilidad predicha. Una pendiente menor
    que uno indica predicciones demasiado extremas, que es la forma habitual del
    sobreajuste, y el intercepto recoge el desplazamiento sistemático del nivel de riesgo.

    Se resuelve por mínimos cuadrados reponderados sobre dos parámetros. Es el mismo
    estimador de máxima verosimilitud que ajustaría una rutina general, pero sin su
    sobrecarga: el remuestreo lo invoca decenas de miles de veces y esa diferencia decide
    si el cálculo de los intervalos toma minutos u horas.
    """
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), _logit(p)])
    b = np.zeros(2)
    for _ in range(iteraciones):
        eta = X @ b
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        w = np.clip(mu * (1 - mu), _EPS, None)
        H = X.T @ (X * w[:, None])
        try:
            paso = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        b = b + paso
        if np.max(np.abs(paso)) < tol:
            break
    else:
        return np.nan, np.nan
    return float(b[1]), float(b[0])


def metricas(y, p) -> dict:
    """Discriminación y calibración de un vector de predicciones.

    El área bajo la curva mide el orden que el modelo impone entre pacientes; el resto mide
    si la probabilidad que emite es creíble en su magnitud, que es una propiedad distinta y
    que un modelo puede perder sin que su discriminación se resienta.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pendiente, intercepto = recalibracion(y, p)
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "pendiente": pendiente, "intercepto": intercepto,
            "citl": float(_logit(y.mean()) - _logit(p.mean()))}


def dispersion_entre_pliegues(oof: pd.DataFrame, *, particiones: int | None = None,
                              semilla: int | None = None) -> pd.DataFrame:
    """Media y desviación del área bajo la curva entre los pliegues externos.

    Es la variación del desempeño de un pliegue a otro, distinta del intervalo por
    remuestreo: informa cuánto depende la estimación de qué pacientes quedaron fuera.
    Los pliegues se reconstruyen de forma determinista con la misma semilla que los generó.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    k = particiones or config.CV_EXTERNO
    s = config.SEED if semilla is None else semilla
    filas = []
    for clave, g in oof.groupby(list(CLAVES), sort=False):
        g = g.sort_values("indice")
        y, p = g["y"].to_numpy(), g["p"].to_numpy()
        cortes = StratifiedKFold(k, shuffle=True, random_state=s).split(np.zeros(len(y)), y)
        aucs = [roc_auc_score(y[v], p[v]) for _, v in cortes if len(np.unique(y[v])) > 1]
        filas.append({**dict(zip(CLAVES, clave)),
                      "auc_pliegues": float(np.mean(aucs)),
                      "auc_pliegues_de": float(np.std(aucs, ddof=1))})
    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────
# Remuestreo único
# ─────────────────────────────────────────────────────────────────────────────

def _matriz(oof: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """Predicciones por combinación, alineadas al mismo vector de desenlace."""
    pred, y = {}, None
    for clave, g in oof.groupby(list(CLAVES), sort=False):
        g = g.sort_values("indice")
        if y is None:
            y = g["y"].to_numpy()
        elif len(g) != len(y) or not np.array_equal(g["y"].to_numpy(), y):
            raise ValueError(f"el desenlace de {clave} no coincide con el de las demás "
                             "combinaciones: las predicciones no son comparables")
        pred[clave] = g["p"].to_numpy()
    return pred, y


def _indices_remuestra(y: np.ndarray, rng) -> np.ndarray:
    """Remuestra con reemplazo dentro de cada estrato del desenlace.

    Estratificar preserva la frecuencia observada en cada réplica; sin ello, algunas
    remuestras quedarían con muy pocos eventos y el área bajo la curva se volvería inestable
    por una razón ajena al modelo.
    """
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, len(pos), replace=True),
                           rng.choice(neg, len(neg), replace=True)])


def _ic(muestras, lo=2.5, hi=97.5) -> tuple[float, float]:
    a = np.asarray(muestras, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return np.nan, np.nan
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def remuestrear(oof: pd.DataFrame, pares: list | None = None, *,
                remuestras: int | None = None, semilla: int | None = None,
                verbose: bool = True) -> dict:
    """Una extracción estratificada alimenta las medidas marginales y las pareadas.

    `pares` declara qué combinaciones se contrastan, como tuplas `(referencia, alternativa)`
    de claves. La diferencia se calcula dentro de cada réplica, de modo que el intervalo
    recoja la correlación entre ambas predicciones, que están hechas sobre las mismas
    pacientes.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    B = remuestras or config.BOOTSTRAP_B
    rng = np.random.RandomState(config.SEED if semilla is None else semilla)
    pred, y = _matriz(oof)
    pares = pares or []

    marg = {k: {m: [] for m in MEDIDAS} for k in pred}
    dif = {p: {"dauc": [], "dbrier": []} for p in pares}

    for b in range(B):
        i = _indices_remuestra(y, rng)
        yb = y[i]
        if len(np.unique(yb)) < 2:
            continue
        auc_b, brier_b = {}, {}
        for k, p in pred.items():
            pb = p[i]
            auc_b[k] = float(roc_auc_score(yb, pb))
            brier_b[k] = float(brier_score_loss(yb, pb))
            pend, inter = recalibracion(yb, pb)
            marg[k]["auc"].append(auc_b[k]);   marg[k]["brier"].append(brier_b[k])
            marg[k]["pendiente"].append(pend); marg[k]["intercepto"].append(inter)
            marg[k]["citl"].append(float(_logit(yb.mean()) - _logit(pb.mean())))
        for ref, alt in pares:
            if ref in auc_b and alt in auc_b:
                dif[(ref, alt)]["dauc"].append(auc_b[alt] - auc_b[ref])
                dif[(ref, alt)]["dbrier"].append(brier_b[alt] - brier_b[ref])
        if verbose and (b + 1) % max(1, B // 20) == 0:
            print("·", end="", flush=True)
    if verbose:
        print(f" ✔ {B} remuestras sobre {len(pred)} combinaciones")
    return {"marginal": marg, "diferencia": dif, "remuestras": B}


# ─────────────────────────────────────────────────────────────────────────────
# Ensamblado
# ─────────────────────────────────────────────────────────────────────────────

def desempeno(oof: pd.DataFrame, extraccion: dict | None = None, **kw) -> pd.DataFrame:
    """Desempeño de cada combinación: punto estimado, intervalo y dispersión entre pliegues.

    Los puntos no dependen del remuestreo y son exactamente reproducibles; el intervalo
    proviene de la extracción compartida.
    """
    pred, y = _matriz(oof)
    extraccion = extraccion or remuestrear(oof, **kw)
    pliegues = dispersion_entre_pliegues(oof).set_index(list(CLAVES))

    filas = []
    for k, p in pred.items():
        fila = {**dict(zip(CLAVES, k)), **metricas(y, p)}
        for m in MEDIDAS:
            fila[f"{m}_ic_low"], fila[f"{m}_ic_high"] = _ic(extraccion["marginal"][k][m])
        if k in pliegues.index:
            fila.update(pliegues.loc[k].to_dict())
        filas.append(fila)
    d = pd.DataFrame(filas)
    d.attrs["remuestras"] = extraccion["remuestras"]
    return d.sort_values(list(CLAVES)).reset_index(drop=True)


def comparacion_pareada(oof: pd.DataFrame, pares: list, extraccion: dict | None = None,
                        *, etiquetas: tuple = ("referencia", "alternativa"),
                        **kw) -> pd.DataFrame:
    """Diferencia de desempeño entre dos combinaciones, sobre las mismas pacientes.

    El valor p es unilateral y se lee como la proporción de remuestras en que la diferencia
    no favorece a la alternativa. Responde a una hipótesis direccional, que es la que el
    trabajo declara.
    """
    pred, y = _matriz(oof)
    extraccion = extraccion or remuestrear(oof, pares, **kw)

    filas = []
    for ref, alt in pares:
        if ref not in pred or alt not in pred:
            continue
        mr, ma = metricas(y, pred[ref]), metricas(y, pred[alt])
        b = extraccion["diferencia"].get((ref, alt), {"dauc": [], "dbrier": []})
        dauc, dbrier = np.asarray(b["dauc"]), np.asarray(b["dbrier"])
        lo, hi = _ic(dauc)
        blo, bhi = _ic(dbrier)
        filas.append({
            etiquetas[0]: " · ".join(map(str, ref)), etiquetas[1]: " · ".join(map(str, alt)),
            "auc_referencia": mr["auc"], "auc_alternativa": ma["auc"],
            "delta_auc": ma["auc"] - mr["auc"], "delta_auc_ic_low": lo, "delta_auc_ic_high": hi,
            "p_unilateral": float(np.mean(dauc <= 0)) if len(dauc) else np.nan,
            "delta_brier": ma["brier"] - mr["brier"],
            "delta_brier_ic_low": blo, "delta_brier_ic_high": bhi,
            "p_brier_unilateral": float(np.mean(dbrier >= 0)) if len(dbrier) else np.nan})
    return pd.DataFrame(filas)


def pares_por_especificacion(oof: pd.DataFrame, referencia: str = "preop",
                             alternativa: str = "periop") -> list:
    """Pares que contrastan dos especificaciones, dentro de cada cohorte y escenario."""
    claves = oof[list(CLAVES)].drop_duplicates()
    pares = []
    for (coh, esc, alg), _ in claves.groupby(["cohorte", "escenario", "algoritmo"], sort=False):
        r, a = (coh, esc, referencia, alg), (coh, esc, alternativa, alg)
        if {r, a} <= set(map(tuple, claves.to_numpy())):
            pares.append((r, a))
    return pares


def pares_por_escenario(oof: pd.DataFrame, referencia: str = "sin_centro",
                        alternativa: str = "con_centro") -> list:
    """Pares que contrastan la incorporación del centro, dentro de cada especificación."""
    claves = set(map(tuple, oof[list(CLAVES)].drop_duplicates().to_numpy()))
    pares = []
    for coh, esp, alg in {(c, e, a) for c, _, e, a in claves}:
        r, a = (coh, referencia, esp, alg), (coh, alternativa, esp, alg)
        if r in claves and a in claves:
            pares.append((r, a))
    return sorted(pares)


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia
# ─────────────────────────────────────────────────────────────────────────────

def persistir(ruta, calcular, *, recalcular: bool = False, verbose: bool = True):
    """Devuelve el artefacto guardado en `ruta`, o lo calcula y lo deja escrito.

    Las etapas de interpretabilidad y transportabilidad reajustan modelos y tardan minutos.
    Encapsular el cálculo tras un archivo permite que el notebook se ejecute entero muchas
    veces sin repetirlas, y que una interrupción no obligue a rehacer lo ya obtenido. El
    notebook queda con una sola sentencia por artefacto, que es lo que debe leerse.

    Los metadatos que acompañan al resultado —los hiperparámetros elegidos por pliegue, por
    ejemplo— se escriben como archivos aparte y se restituyen al recuperarlo. Sin eso se
    perderían dos veces: el formato columnar no admite objetos arbitrarios, y concatenar
    varios marcos descarta lo que cuelga de cada uno.
    """
    from pathlib import Path

    ruta = Path(ruta)
    _tabular = lambda v: isinstance(v, list) and v and isinstance(v[0], dict)

    if ruta.exists() and not recalcular:
        d = pd.read_parquet(ruta)
        for a in sorted(ruta.parent.glob(f"{ruta.stem}__*.parquet")):
            d.attrs[a.stem.split("__", 1)[1]] = pd.read_parquet(a).to_dict("records")
        if verbose:
            print(f"✔ recuperado {ruta.name}")
        return d

    d = calcular()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    extra = {k: v for k, v in d.attrs.items() if _tabular(v) or isinstance(v, pd.DataFrame)}
    for k, v in extra.items():
        marco = v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)
        marco.to_parquet(ruta.parent / f"{ruta.stem}__{k}.parquet", index=False)
    d.attrs = {k: v for k, v in d.attrs.items() if k not in extra}
    d.to_parquet(ruta, index=False)
    d.attrs.update({k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                    for k, v in extra.items()})
    if verbose:
        acomp = f" + {len(extra)} acompañante(s)" if extra else ""
        print(f"✔ generado {ruta.name}  ({len(d)} filas){acomp}")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Interpretabilidad
# ─────────────────────────────────────────────────────────────────────────────

def _modelo_final(datos, alg: dict, semilla: int | None = None):
    """Ajusta un algoritmo sobre la cohorte completa, con su preprocesamiento.

    Devuelve el modelo, la matriz transformada y el nombre de cada columna. Las medidas de
    importancia describen al modelo, de modo que se calculan sobre el ajuste completo y no
    sobre un pliegue: la pregunta es qué información usa, no cuánto generaliza.
    """
    from . import modell

    s = config.SEED if semilla is None else semilla
    prep = modell.ajustar_preprocesamiento(
        modell.construir_preprocesamiento(datos.tipos), datos.X)
    Z = modell.aplicar_preprocesamiento(prep, datos.X)
    nombres = [c.split("__", 1)[-1] for c in prep.get_feature_names_out()]
    est = modell.seleccionar_hiperparametros(
        alg["estimador"], alg["grilla"], Z, datos.y, semilla=s)[0]
    return modell.reentrenar(est, Z, datos.y), Z, nombres


def _variable_origen(columnas: list[str], variables: list[str]) -> dict:
    """Asocia cada columna del diseño con la variable de la que proviene."""
    salida = {}
    for c in columnas:
        candidatas = [v for v in variables if c == v or c.startswith(v)]
        salida[c] = max(candidatas, key=len) if candidatas else c
    return salida


def importancia_permutacion(datos, algoritmos: dict | None = None, *,
                            repeticiones: int | None = None, semilla: int | None = None,
                            verbose: bool = True) -> pd.DataFrame:
    """Pérdida de discriminación al permutar cada columna, en cada algoritmo.

    Permutar una columna rompe su relación con el desenlace conservando su distribución: la
    caída del área bajo la curva mide cuánto se apoyaba el modelo en esa información. Es una
    medida sobre el modelo, no sobre el mundo, y por eso no autoriza a leer las variables de
    mayor pérdida como factores de riesgo.
    """
    from sklearn.inspection import permutation_importance
    from . import modell

    algoritmos = algoritmos or modell.catalogo_algoritmos(semilla)
    reps = repeticiones or config.PERMUT_REPS
    s = config.SEED if semilla is None else semilla

    filas = []
    for nombre, alg in algoritmos.items():
        modelo, Z, columnas = _modelo_final(datos, alg, s)
        origen = _variable_origen(columnas, list(datos.X.columns))
        r = permutation_importance(modelo, Z, datos.y, n_repeats=reps,
                                   random_state=s, scoring=config.METRICA, n_jobs=-1)
        filas += [{"cohorte": datos.cohorte, "escenario": datos.escenario,
                   "especificacion": datos.especificacion, "algoritmo": nombre,
                   "columna": c, "var_rename": origen[c],
                   "importancia": float(r.importances_mean[i]),
                   "importancia_de": float(r.importances_std[i])}
                  for i, c in enumerate(columnas)]
        if verbose:
            print(f"  {nombre:11s} ✔")
    return pd.DataFrame(filas)


def consenso(permutacion: pd.DataFrame, *, top: int = 10) -> pd.DataFrame:
    """Ordenamiento de las variables por acuerdo entre algoritmos.

    Cada algoritmo ordena las variables a su manera, y quedarse con uno hace depender la
    lectura de esa elección. El consenso promedia el puesto que cada variable ocupa en los
    distintos modelos y registra en cuántos entra a las primeras posiciones, de modo que la
    conclusión no dependa de qué algoritmo se mire.
    """
    agregada = (permutacion.groupby(["cohorte", "escenario", "especificacion",
                                     "algoritmo", "var_rename"], as_index=False)["importancia"]
                .max())
    filas = []
    for clave, g in agregada.groupby(["cohorte", "escenario", "especificacion"], sort=False):
        ancha = g.pivot(index="var_rename", columns="algoritmo", values="importancia")
        puestos = ancha.rank(ascending=False, method="min")
        filas.append(pd.DataFrame({
            "cohorte": clave[0], "escenario": clave[1], "especificacion": clave[2],
            "var_rename": ancha.index,
            "rango_medio": puestos.mean(axis=1).to_numpy(),
            "rango_de": puestos.std(axis=1, ddof=1).to_numpy(),
            f"n_modelos_top{top}": (puestos <= top).sum(axis=1).to_numpy(),
            "importancia_media": ancha.mean(axis=1).to_numpy()}))
    d = pd.concat(filas, ignore_index=True)
    return d.sort_values(["cohorte", "escenario", "rango_medio"]).reset_index(drop=True)


def shap_valores(datos, algoritmo: str | None = None, *, semilla: int | None = None,
                 muestra: int | None = None) -> pd.DataFrame:
    """Contribución de cada columna a la predicción de cada paciente.

    A diferencia de la permutación, que entrega un número por variable, descompone cada
    predicción individual, lo que permite ver la forma de la relación y no solo su peso.
    Se calcula sobre el modelo de árboles, donde el cálculo es exacto y no una aproximación.
    """
    import shap
    from . import modell

    nombre = algoritmo or config.SHAP_ALGORITMO
    alg = modell.catalogo_algoritmos(semilla)[nombre]
    modelo, Z, columnas = _modelo_final(datos, alg, semilla)
    filas = np.arange(len(Z))
    if muestra and muestra < len(Z):
        rng = np.random.RandomState(config.SEED if semilla is None else semilla)
        filas = rng.choice(len(Z), muestra, replace=False)
        Z = Z[filas]
    valores = shap.TreeExplainer(modelo).shap_values(Z)
    if isinstance(valores, list):
        valores = valores[1]
    if valores.ndim == 3:
        valores = valores[:, :, 1]

    # `Z` está preprocesada: las numéricas van estandarizadas y las ordinales codificadas,
    # de modo que graficar su valor muestra una escala que no es la de la variable. Se
    # recupera el valor original cuando la columna proviene de una sola variable; en las
    # indicadoras de una nominal el valor preprocesado ya es el 0/1 que corresponde.
    from . import modell as _m
    vs = list(datos.X.columns)
    original = np.empty(Z.shape, dtype=object)
    for j, c in enumerate(columnas):
        raiz = _m._raiz(c, vs)
        directa = raiz in datos.X.columns and c.split("__")[-1] == raiz
        original[:, j] = (datos.X[raiz].to_numpy()[filas] if directa else Z[:, j])

    # Columna combinada: numérica en unas variables (edad, IMC), booleana en otras (tbq,
    # d_io), categórica en el resto (asa, ne...). Parquet exige un solo tipo por columna, así
    # que se uniforma a texto -conservando "Sí"/"No" en las booleanas, la convención del resto
    # del proyecto- y quien la consume vuelve a `pd.to_numeric` donde corresponde.
    def _texto_original(v):
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (bool, np.bool_)):
            return "Sí" if v else "No"
        return str(v)

    original_txt = np.vectorize(_texto_original, otypes=[object])(original)

    return pd.DataFrame({
        "cohorte": datos.cohorte, "escenario": datos.escenario,
        "especificacion": datos.especificacion, "algoritmo": nombre,
        "fila": np.repeat(np.arange(len(Z)), len(columnas)),
        "columna": np.tile(columnas, len(Z)),
        "valor": Z.ravel(), "valor_original": original_txt.ravel(),
        "shap": valores.ravel()})


CRITERIOS_PANELES = ("union", "consenso", "shap")


def _clave_unica(d: pd.DataFrame, que: str) -> tuple:
    """Combinación cohorte/escenario/especificación del marco, exigiendo que sea una sola."""
    cs = [c for c in ("cohorte", "escenario", "especificacion") if c in d]
    claves = sorted(map(tuple, d[cs].drop_duplicates().astype(str).to_numpy())) if cs else [()]
    if len(claves) != 1:
        raise ValueError(f"{que} trae {len(claves)} combinaciones de cohorte, escenario y "
                         f"especificación; se esperaba una sola: {claves}")
    return claves[0]


def paneles_shap(consenso: pd.DataFrame, shap: pd.DataFrame, *, criterio: str = "union",
                 top: int = 6, verbose: bool = True) -> pd.DataFrame:
    """Qué predictores reciben panel de dependencia, y por qué criterio entra cada uno.

    El bosque ordena sus columnas por contribución media absoluta y ese orden no es el del
    consenso entre los siete algoritmos. Seleccionar los paneles solo por el bosque produce
    una figura que ilustra a predictores que el resto del análisis no destaca, y omite a los
    que sí; seleccionarlos solo por el consenso descarta la forma de la relación justo donde
    el modelo graficado se apoya. `"union"` conserva ambas lecturas y deja registrado en la
    columna `criterio` -`"consenso"`, `"shap"` o `"ambos"`- en cuál de los dos primeros
    lugares figura cada panel, de modo que la figura no afirme un criterio distinto del que
    la produjo.

    El puesto es la posición en el ordenamiento del consenso, la misma que ocupa la fila en
    `rendern.consenso_permutacion`, para que ambas figuras se lean en paralelo. Un predictor
    sin puesto -presente en los valores SHAP pero ausente del consenso- va al final.

    Ambos marcos deben provenir de la misma cohorte y especificación: la espinal excluye la
    analgesia postoperatoria y recodifica el manejo intraoperatorio, de modo que sus columnas
    no son las de la principal y cruzarlas selecciona paneles que no existen.
    """
    if criterio not in CRITERIOS_PANELES:
        raise ValueError(f"criterio debe ser uno de {CRITERIOS_PANELES}, no {criterio!r}")
    if _clave_unica(consenso, "el consenso") != _clave_unica(shap, "los valores SHAP"):
        raise ValueError("el consenso y los valores SHAP no son de la misma cohorte")

    orden = consenso.sort_values("rango_medio", kind="stable")["var_rename"].tolist()
    puesto = {v: i + 1 for i, v in enumerate(orden)}
    magnitud = shap.groupby("columna")["shap"].apply(lambda s: s.abs().mean())

    por_consenso = orden[:top]
    por_bosque = magnitud.sort_values(ascending=False).head(top).index.tolist()
    elegidas = {"union": set(por_consenso) | set(por_bosque),
                "consenso": set(por_consenso),
                "shap": set(por_bosque)}[criterio]

    # Un predictor del consenso puede no tener valores SHAP en esta cohorte: se avisa en vez
    # de dejar el panel vacío, que es la forma en que la omisión pasaría inadvertida.
    ausentes = sorted(elegidas - set(magnitud.index))
    if ausentes and verbose:
        print(f"⚠ sin valores SHAP en esta cohorte, quedan fuera de la figura: "
              f"{', '.join(ausentes)}")

    d = pd.DataFrame({"var_rename": sorted(elegidas - set(ausentes))})
    d["puesto"] = d["var_rename"].map(puesto)
    d["rango_medio"] = d["var_rename"].map(dict(zip(consenso["var_rename"],
                                                    consenso["rango_medio"])))
    d["shap_abs_medio"] = d["var_rename"].map(magnitud)
    d["criterio"] = ["ambos" if v in por_consenso and v in por_bosque
                     else "consenso" if v in por_consenso else "shap"
                     for v in d["var_rename"]]
    return (d.sort_values(["puesto", "shap_abs_medio"], ascending=[True, False],
                          na_position="last").reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# Transportabilidad
# ─────────────────────────────────────────────────────────────────────────────

def transportabilidad(datos, grupo, algoritmos: dict | None = None, *,
                      semilla: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Desempeño al evaluar en un conglomerado que no participó del entrenamiento.

    Cada iteración retira un centro, entrena en los restantes y evalúa en el retirado. Es la
    aproximación más cercana a una validación externa que permite una muestra multicéntrica,
    y responde a una pregunta distinta de la validación cruzada habitual: no cuánto acierta
    en pacientes nuevos del mismo centro, sino en un centro nuevo.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from . import modell

    algoritmos = algoritmos or modell.catalogo_algoritmos(semilla)
    s = config.SEED if semilla is None else semilla
    g = pd.Series(np.asarray(grupo), index=datos.X.index)

    filas = []
    for nombre, alg in algoritmos.items():
        for retirado in sorted(g.dropna().unique()):
            fuera = (g == retirado).to_numpy()
            if len(np.unique(datos.y[fuera])) < 2:
                continue
            prep = modell.ajustar_preprocesamiento(
                modell.construir_preprocesamiento(datos.tipos), datos.X[~fuera])
            Z_tr = modell.aplicar_preprocesamiento(prep, datos.X[~fuera])
            est = modell.seleccionar_hiperparametros(
                alg["estimador"], alg["grilla"], Z_tr, datos.y[~fuera], semilla=s)[0]
            modelo = modell.reentrenar(est, Z_tr, datos.y[~fuera])
            p = modell.predecir_fuera_de_pliegue(
                modelo, modell.aplicar_preprocesamiento(prep, datos.X[fuera]))
            y = datos.y[fuera]
            filas.append({"cohorte": datos.cohorte, "escenario": datos.escenario,
                          "especificacion": datos.especificacion, "algoritmo": nombre,
                          "centro": retirado, "n": int(fuera.sum()),
                          "eventos": int(y.sum()), "prevalencia": float(y.mean()),
                          "auc": float(roc_auc_score(y, p)),
                          "brier": float(brier_score_loss(y, p))})
        if verbose:
            print(f"  {nombre:11s} ✔")
    return pd.DataFrame(filas)


def promedio_por_centro(iecv: pd.DataFrame, *, columna: str = "auc") -> pd.DataFrame:
    """Resumen descriptivo del desempeño entre centros: promedio y rango observado.

    Es el resumen declarado para la transportabilidad. Deliberadamente no pondera ni agrupa:
    con cuatro centros, el promedio simple y el rango transmiten la caída de discriminación y
    su dispersión sin comprometer al lector con un modelo de agrupamiento cuyo supuesto la
    muestra no permite examinar.
    """
    d = (iecv.groupby(["cohorte", "escenario", "especificacion", "algoritmo"], as_index=False)
         .agg(centros=("centro", "nunique"),
              **{f"{columna}_promedio": (columna, "mean"),
                 f"{columna}_min": (columna, "min"),
                 f"{columna}_max": (columna, "max")}))
    return d.sort_values(f"{columna}_promedio", ascending=False).reset_index(drop=True)


def resumen_aleatorio(iecv: pd.DataFrame, *, columna: str = "auc") -> pd.DataFrame:
    """Agrupa el desempeño entre centros y estima el rango esperable en uno nuevo.

    **Excede el resumen declarado en el protocolo**, que es el promedio simple de
    `promedio_por_centro`. Se conserva como capacidad disponible, no como parte del análisis
    reportado: agrupar cuatro conglomerados con un modelo de efectos aleatorios exige
    declarar un método adicional cuya conclusión no difiere de la del promedio.

    Con pocos centros el promedio simple oculta cuánto varían entre sí. El modelo de efectos
    aleatorios separa la variación por muestreo de la variación real entre centros, y el
    intervalo de predicción, más ancho que el del promedio, es el que responde a la pregunta
    relevante: qué desempeño cabría esperar en un centro no observado. Con cuatro centros
    ese intervalo es ancho por construcción, de modo que se lee su dirección, no su extremo.
    """
    from scipy import stats

    filas = []
    for clave, g in iecv.groupby(["cohorte", "escenario", "especificacion", "algoritmo"],
                                 sort=False):
        v = g[columna].to_numpy(dtype=float)
        k = len(v)
        # Varianza aproximada del AUC por centro (Hanley y McNeil), suficiente como peso.
        n1, n0 = g["eventos"].to_numpy(), (g["n"] - g["eventos"]).to_numpy()
        q1, q2 = v / (2 - v), 2 * v ** 2 / (1 + v)
        var = np.clip((v * (1 - v) + (n1 - 1) * (q1 - v ** 2)
                       + (n0 - 1) * (q2 - v ** 2)) / np.clip(n1 * n0, 1, None), 1e-6, None)
        w = 1 / var
        media_fija = float(np.sum(w * v) / np.sum(w))
        Q = float(np.sum(w * (v - media_fija) ** 2))
        c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
        tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
        wr = 1 / (var + tau2)
        media = float(np.sum(wr * v) / np.sum(wr))
        ee = float(np.sqrt(1 / np.sum(wr)))
        t = stats.t.ppf(0.975, max(k - 2, 1))
        filas.append({**dict(zip(("cohorte", "escenario", "especificacion", "algoritmo"), clave)),
                      "centros": k, f"{columna}_medio": media,
                      "ic_low": media - 1.96 * ee, "ic_high": media + 1.96 * ee,
                      "prediccion_low": media - t * np.sqrt(tau2 + ee ** 2),
                      "prediccion_high": media + t * np.sqrt(tau2 + ee ** 2),
                      "tau2": tau2,
                      "i2_pct": 100 * max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0})
    return pd.DataFrame(filas)
