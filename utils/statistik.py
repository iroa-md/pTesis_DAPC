"""statistik.py — pruebas bivariadas, tamaños de efecto y balance.

Alcance: el análisis descriptivo y bivariado se realiza sobre la **cohorte completa**,
con independencia del cribado de predictores (que rige solo la selección para los modelos).

Correcciones respecto de la implementación previa:
  · Mann-Whitney con `method="auto"`: el modo exacto asume ausencia de empates y con
    empates devuelve un valor p sobre la distribución nula equivocada, sin advertirlo.
  · Los faltantes nunca entran como categoría en las tablas de contingencia: hacerlo
    rompía la tabla 2×2, impedía que se ejecutara la prueba de Fisher y hacía que la
    prueba detectara el desbalance de faltantes en vez del de la variable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def or_2x2(a: float, b: float, c: float, d: float) -> tuple[float, float, float, str]:
    """OR del desenlace en expuestas frente a la referencia, con IC 95 %.

    Con la variable en filas y el desenlace en columnas: a = ref/sin · b = ref/con ·
    c = expuesta/sin · d = expuesta/con. De ahí OR = (d/c)/(b/a) = (d·a)/(c·b); invertir
    el cociente devuelve el recíproco y da por protector lo que es de riesgo. Ante una
    celda vacía se aplica la corrección de Haldane-Anscombe.
    """
    nota = ""
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        nota = "Haldane-Anscombe"
    orr = (d * a) / (c * b) if c * b else np.nan
    if not np.isfinite(orr) or orr <= 0:
        return np.nan, np.nan, np.nan, "no_estimable"
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, float(np.exp(np.log(orr) - 1.96 * se)), float(np.exp(np.log(orr) + 1.96 * se)), nota


def cramers_v(obs: np.ndarray, chi2: float) -> float:
    """V de Cramér para tablas de contingencia mayores que 2×2."""
    n = obs.sum()
    r, k = obs.shape
    return float(np.sqrt((chi2 / n) / min(r - 1, k - 1))) if n and min(r - 1, k - 1) else np.nan


def ic_wilson(k: int, n: int) -> tuple[float, float]:
    """Intervalo de confianza al 95 % de una proporción (método de Wilson)."""
    if n == 0:
        return (np.nan, np.nan)
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def resolver_tipo(s: pd.Series, tipo: str) -> str:
    """Tipo efectivo de una variable para el análisis bivariado.

    Una ordinal con solo dos niveles observados es, a efectos del contraste, binaria: la
    medida informativa es el odds ratio y no una correlación de rangos. El tipo declarado
    en el catálogo describe la variable; aquí interesa su estructura observada.
    """
    tipo = str(tipo).strip().lower()
    if tipo in ("ord", "nom") and s.dropna().nunique() <= 2:
        return "bin"
    return tipo


def _rangos(s: pd.Series, tipo: str) -> pd.Series:
    """Valores rankeables: códigos ordenados en ordinales, numérico en continuas."""
    if tipo == "ord" and isinstance(s.dtype, pd.CategoricalDtype):
        return s.cat.codes.where(s.notna()).astype("Float64")
    return pd.to_numeric(s.astype("string").str.replace(",", ".", regex=False), errors="coerce")


def _shapiro(arr, max_n: int | None = None) -> tuple[float, float, str]:
    """Shapiro-Wilk sobre un grupo, con las salvaguardas del análisis original.

    Con menos de tres observaciones la prueba no está definida y se devuelve NA, lo que
    aguas arriba cuenta como «no evaluable». Sobre `max_n` se evalúa en una submuestra,
    porque con muestras muy grandes la prueba rechaza por desviaciones irrelevantes.
    """
    from scipy import stats

    x = pd.to_numeric(pd.Series(arr), errors="coerce").dropna()
    if len(x) < 3:
        return np.nan, np.nan, "no_evaluable"
    tope = config.SHAPIRO_MAX_N if max_n is None else max_n
    nota = "muestra_completa"
    if len(x) > tope:
        x, nota = x.sample(tope, random_state=config.SEED), f"submuestra_{tope}"
    w, p = stats.shapiro(x)
    return float(w), float(p), nota


def cribar_normalidad(grupos, etiquetas=None, *, alpha: float | None = None,
                      max_n: int | None = None) -> tuple[bool, list[dict]]:
    """Tamizaje de normalidad por grupo. Devuelve `(compatible, detalle)`.

    No demuestra normalidad: hace explícita la regla con que se elige entre la prueba
    paramétrica y la de rangos. Se considera compatible **solo** si ningún grupo la rechaza
    y todos pudieron evaluarse, de modo que la duda favorece siempre a la prueba de rangos.
    """
    a = config.NORMALIDAD_ALPHA if alpha is None else alpha
    etiquetas = etiquetas if etiquetas is not None else range(len(grupos))
    detalle, compatible = [], True
    for etq, arr in zip(etiquetas, grupos):
        w, p, nota = _shapiro(arr, max_n)
        ok = bool(pd.notna(p) and float(p) >= a)
        detalle.append({"grupo": str(etq), "n": int(len(arr)), "prueba": "shapiro_wilk",
                        "w": None if pd.isna(w) else float(w),
                        "p": None if pd.isna(p) else float(p),
                        "alpha": float(a), "compatible": ok, "nota": nota})
        if not ok:
            compatible = False
    return compatible, detalle


def _cohens_d(g0: np.ndarray, g1: np.ndarray) -> float:
    """d de Cohen con desviación combinada. Positiva si el grupo 1 tiene valores mayores."""
    n0, n1 = len(g0), len(g1)
    s0, s1 = np.nanstd(g0, ddof=1), np.nanstd(g1, ddof=1)
    sp = np.sqrt(((n0 - 1) * s0 ** 2 + (n1 - 1) * s1 ** 2) / max(1, n0 + n1 - 2))
    if not np.isfinite(sp) or sp == 0:
        return np.nan
    return float((np.nanmean(g1) - np.nanmean(g0)) / sp)


def bivariado(x: pd.Series, y: pd.Series, tipo: str) -> dict:
    """Asociación cruda de una variable con el desenlace binario.

    Categóricas: χ² sin corrección, o Fisher en las 2×2 con alguna frecuencia esperada
    menor que 5; efecto como OR [IC 95 %] en las 2×2 y V de Cramér en las demás.

    Numéricas: la prueba se elige por un **cribado de normalidad de Shapiro-Wilk sobre cada
    grupo del desenlace**. Si ningún grupo la rechaza, t de Welch con d de Cohen; en caso
    contrario, Mann-Whitney con correlación rango-biserial. El detalle del cribado se
    devuelve en la clave `normalidad`, de modo que la elección quede auditable.

    Ordinales: Mann-Whitney directo. El cribado no se les aplica, porque la normalidad no
    es un supuesto interpretable sobre códigos de orden.

    En ambos casos Mann-Whitney usa `method="auto"`: el modo exacto asume ausencia de
    empates y con empates devuelve un p sobre la distribución nula equivocada.

    Se opera sobre casos completos: los faltantes no forman una categoría propia.
    """
    from scipy import stats

    tipo = resolver_tipo(x, tipo)
    vacio = {"prueba": "—", "p": np.nan, "efecto": "—", "or": np.nan,
             "ic_low": np.nan, "ic_high": np.nan, "n": 0, "nota": ""}
    cc = x.notna() & y.notna()
    n = int(cc.sum())
    if n == 0 or x[cc].nunique() < 2:
        return {**vacio, "n": n, "nota": "un_solo_nivel_observado"}

    if tipo in ("bin", "nom"):
        obs = pd.crosstab(x[cc], y[cc]).values
        chi2, p_chi = stats.chi2_contingency(obs, correction=False)[:2]
        esp = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / obs.sum()
        if obs.shape == (2, 2) and esp.min() < 5:
            p, prueba, nota = float(stats.fisher_exact(obs)[1]), "Fisher", "esperada<5"
        else:
            p, prueba = float(p_chi), "χ²"
            nota = "esperada<5; Fisher no aplica a RxC" if esp.min() < 5 else ""
        if obs.shape == (2, 2):
            orr, lo, hi, n_or = or_2x2(*map(float, (obs[0, 0], obs[0, 1], obs[1, 0], obs[1, 1])))
            ef = f"{orr:.2f} [{lo:.2f}–{hi:.2f}]" if np.isfinite(orr) else "—"
            return {"prueba": prueba, "p": p, "efecto": ef, "or": orr, "ic_low": lo,
                    "ic_high": hi, "n": n, "nota": "; ".join(t for t in (nota, n_or) if t)}
        return {"prueba": prueba, "p": p, "efecto": f"V = {cramers_v(obs, chi2):.2f}",
                "or": np.nan, "ic_low": np.nan, "ic_high": np.nan, "n": n, "nota": nota}

    xr = _rangos(x, tipo)[cc]
    yy = y[cc]
    ks = sorted(pd.unique(yy))
    g0, g1 = xr[yy == ks[0]].to_numpy(float), xr[yy == ks[1]].to_numpy(float)
    if len(g0) < 2 or len(g1) < 2:
        return {**vacio, "n": n, "nota": "grupo_insuficiente"}

    # El cribado de normalidad solo rige sobre las numéricas. En las ordinales se rankean
    # códigos de orden, sobre los que la normalidad no es un supuesto interpretable.
    normal, detalle = (False, [])
    if tipo == "num":
        normal, detalle = cribar_normalidad((g0, g1), (str(ks[0]), str(ks[1])))

    if normal:
        t, p = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
        d = _cohens_d(g0, g1)
        return {"prueba": "t-Welch", "p": float(p), "efecto": f"d = {d:+.2f}", "or": np.nan,
                "ic_low": np.nan, "ic_high": np.nan, "n": n, "nota": "",
                "normalidad": detalle}

    u, p = stats.mannwhitneyu(g0, g1, alternative="two-sided", method="auto")
    r = 1 - 2 * u / (len(g0) * len(g1))
    return {"prueba": "M-W", "p": float(p), "efecto": f"r = {r:+.2f}", "or": np.nan,
            "ic_low": np.nan, "ic_high": np.nan, "n": n, "nota": "",
            "normalidad": detalle}


# ─────────────────────────────────────────────────────────────────────────────
# Resúmenes descriptivos
# ─────────────────────────────────────────────────────────────────────────────

def mediana_ric(s: pd.Series, dec: int = 1) -> str:
    """Mediana [rango intercuartílico] de una variable numérica."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return "—"
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    return f"{x.median():.{dec}f} [{q1:.{dec}f}–{q3:.{dec}f}]"


def n_pct(k: int, n: int, dec: int = 1) -> str:
    """Recuento y porcentaje sobre el total evaluado."""
    return f"{k}/{n} ({100 * k / n:.{dec}f}%)" if n else "—"


def resumen(s: pd.Series, tipo: str) -> dict:
    """Resumen descriptivo según la naturaleza de la variable.

    Numéricas: mediana [RIC]. Binarias: n (%) del nivel positivo. Categóricas: recuento y
    porcentaje por nivel. Siempre sobre casos observados, informando cuántos son.
    """
    tipo = resolver_tipo(s, tipo)
    obs = s.notna()
    n = int(obs.sum())
    base = {"n_observado": n, "n_faltante": int((~obs).sum())}
    if n == 0:
        return {**base, "tipo": tipo, "resumen": "—", "niveles": {}}
    if tipo == "num":
        return {**base, "tipo": tipo, "resumen": mediana_ric(s), "niveles": {}}
    conteo = s[obs].astype(str).value_counts()
    if tipo == "bin":
        pos = [k for k in conteo.index if k not in ("False", "No", "0", "0.0", "no")]
        k = int(conteo.get(pos[0], 0)) if pos else 0
        return {**base, "tipo": tipo, "resumen": n_pct(k, n),
                "niveles": {str(i): int(v) for i, v in conteo.items()}}
    niveles = {str(i): f"{int(v)}/{n} ({100 * v / n:.1f}%)" for i, v in conteo.items()}
    return {**base, "tipo": tipo, "resumen": "; ".join(f"{k}: {v}" for k, v in niveles.items()),
            "niveles": niveles}


def frecuencia_por_centro(y: pd.Series, centro: pd.Series) -> pd.DataFrame:
    """Incidencia del desenlace por centro con IC 95 % de Wilson, más la fila global.

    Acompaña la prueba χ² global entre centros, que contrasta si la frecuencia difiere
    entre instituciones.
    """
    from scipy import stats

    filas = []
    for c in sorted(centro.dropna().astype(str).unique()):
        m = centro.astype(str).eq(c)
        yy = y[m].dropna()
        n, ev = len(yy), int(yy.sum())
        lo, hi = ic_wilson(ev, n)
        filas.append({"centro": c, "n": n, "eventos": ev, "no_eventos": n - ev,
                      "incidencia": ev / n if n else np.nan, "ic_low": lo, "ic_high": hi})
    yy = y.dropna()
    n, ev = len(yy), int(yy.sum())
    lo, hi = ic_wilson(ev, n)
    d = pd.DataFrame(filas)
    p = stats.chi2_contingency(d[["eventos", "no_eventos"]].to_numpy())[1] if len(d) > 1 else np.nan
    d.loc[len(d)] = {"centro": "GLOBAL", "n": n, "eventos": ev, "no_eventos": n - ev,
                     "incidencia": ev / n, "ic_low": lo, "ic_high": hi}
    d.attrs["p_global"] = float(p)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Heterogeneidad entre centros
# ─────────────────────────────────────────────────────────────────────────────

def smd_por_centro(s: pd.Series, centro: pd.Series) -> dict:
    """SMD máxima de la variable en un centro frente al resto.

    Cuantifica cuánto se aparta el centro más discrepante de los demás. Se calcula sobre
    casos observados; su lectura exige que el registro sea comparable entre centros
    (ver `daten.comparabilidad_registro`).
    """
    mejor, cual = np.nan, None
    for c in sorted(centro.dropna().astype(str).unique()):
        m = centro.astype(str).eq(c)
        v = abs(smd(s, m))
        if not np.isnan(v) and (np.isnan(mejor) or v > mejor):
            mejor, cual = v, c
    return {"smd_max": mejor, "centro_smd_max": cual,
            "heterogeneidad_relevante": bool(mejor >= config.SMD_UMBRAL) if not np.isnan(mejor) else False}


def prueba_global_centro(s: pd.Series, centro: pd.Series, tipo: str) -> dict:
    """Contraste global de la variable entre centros: χ² o Kruskal-Wallis."""
    from scipy import stats

    tipo = resolver_tipo(s, tipo)
    obs = s.notna() & centro.notna()
    if obs.sum() == 0 or s[obs].nunique() < 2:
        return {"prueba_global": None, "p_global": np.nan}
    if tipo in ("bin", "nom"):
        tabla = pd.crosstab(s[obs].astype(str), centro[obs].astype(str))
        if min(tabla.shape) < 2:
            return {"prueba_global": None, "p_global": np.nan}
        return {"prueba_global": "χ²", "p_global": float(stats.chi2_contingency(tabla)[1])}
    x = _rangos(s, tipo)[obs]
    grupos = [x[centro[obs].astype(str).eq(c)].dropna().to_numpy(float)
              for c in sorted(centro[obs].astype(str).unique())]
    grupos = [g for g in grupos if len(g) > 0]
    if len(grupos) < 2:
        return {"prueba_global": None, "p_global": np.nan}
    return {"prueba_global": "Kruskal-Wallis", "p_global": float(stats.kruskal(*grupos).pvalue)}


def heterogeneidad(cohorte: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Heterogeneidad entre centros de las variables del conjunto declarado.

    Devuelve, por variable, la SMD máxima centro contra resto, el centro responsable y el
    contraste global. El conjunto lo define `include_heterogeneidad`; la variable de centro
    se excluye por construcción.
    """
    tipos = dict(zip(cat["var_rename"], cat["conceptual_type"]))
    centro = cohorte[config.CENTER_VAR]
    filas = []
    for _, r in cat[cat["include_heterogeneidad"].fillna(False)].iterrows():
        v = r["var_rename"]
        if v not in cohorte.columns or v == config.CENTER_VAR:
            continue
        filas.append({"var_rename": v, "label": r["label"], "block": r["block"],
                      "capa": r["capa_heterogeneidad"],
                      **smd_por_centro(cohorte[v], centro),
                      **prueba_global_centro(cohorte[v], centro, tipos.get(v, "bin"))})
    return pd.DataFrame(filas)


def _es_binaria(s: pd.Series) -> bool:
    if str(s.dtype) in {"bool", "boolean"}:
        return True
    return s.dropna().nunique() <= 2


def _es_numerica(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not _es_binaria(s)


def smd_continua(a: pd.Series, b: pd.Series) -> float:
    """Diferencia de medias estandarizada entre dos grupos (Cohen)."""
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def smd_binaria(a: pd.Series, b: pd.Series) -> float:
    """SMD entre dos proporciones, estandarizada por la proporción combinada.

    Denominador `√(p̄(1−p̄))` con `p̄` la media de ambas proporciones. Es la variante
    empleada en el análisis original y la que reproduce las cifras publicadas; difiere de
    la formulación de Austin, que promedia las varianzas de cada grupo y arroja valores
    mayores ante proporciones extremas.
    """
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    p1, p2 = a.mean(), b.mean()
    s = np.sqrt(((p1 + p2) / 2) * (1 - (p1 + p2) / 2))
    return float((p1 - p2) / s) if s > 0 else 0.0


def smd_multinivel(a: pd.Series, b: pd.Series) -> float:
    """SMD máxima entre los niveles de una categórica, uno contra el resto."""
    a, b = a.astype("string").dropna(), b.astype("string").dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    niveles = sorted(set(a.unique()) | set(b.unique()))
    vals = [abs(v) for v in (smd_binaria(a.eq(n).astype(int), b.eq(n).astype(int)) for n in niveles)
            if not pd.isna(v)]
    return float(max(vals)) if vals else np.nan


def smd(s: pd.Series, grupo: pd.Series) -> float:
    """SMD de `s` entre los dos grupos definidos por la máscara booleana `grupo`.

    Elige la fórmula según la naturaleza de la variable. Opera sobre casos completos.
    """
    a, b = s[grupo], s[~grupo]
    if _es_numerica(s):
        return smd_continua(a, b)
    if str(s.dtype) in {"bool", "boolean"}:
        return smd_binaria(a.astype("boolean"), b.astype("boolean"))
    return smd_multinivel(a, b)   # cubre también las categóricas de dos niveles


def balance(s: pd.Series, grupo: pd.Series) -> dict:
    """Balance de una variable entre dos grupos: SMD, prueba y valor p.

    Usa Fisher en las tablas 2×2 con frecuencias esperadas bajas y χ² en el resto;
    Mann-Whitney (`method="auto"`) en las continuas. Los faltantes se excluyen.
    """
    from scipy import stats

    obs = s.notna()
    a, b = s[obs & grupo], s[obs & ~grupo]
    if len(a) == 0 or len(b) == 0:
        return {"smd": np.nan, "prueba": None, "p": np.nan}

    d = smd(s[obs], grupo[obs])
    if _es_numerica(s):
        p = stats.mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided", method="auto").pvalue
        return {"smd": d, "prueba": "Mann-Whitney", "p": float(p)}

    tabla = pd.crosstab(s[obs].astype(str), grupo[obs])   # sin categoría de faltantes
    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return {"smd": d, "prueba": None, "p": np.nan}
    esperadas = stats.chi2_contingency(tabla)[3]
    if tabla.shape == (2, 2) and (esperadas < 5).any():
        return {"smd": d, "prueba": "Fisher", "p": float(stats.fisher_exact(tabla)[1])}
    return {"smd": d, "prueba": "χ²", "p": float(stats.chi2_contingency(tabla)[1])}
