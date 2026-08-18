"""daten.py — capa de datos: curación, desenlace, cohorte y cribado.

Read-only sobre el catálogo manual (`repositorio/db/metadatos.xlsx`).
Por ahora implementa la puerta de entrada: cargar el catálogo y validar su
**contrato de esquema** (docs/estructura_metadatos.md §6). El resto de la API
(curar, construir_desenlace, construir_cohorte, cribar, ...) se implementa después.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


class ContratoError(ValueError):
    """El catálogo incumple el contrato de esquema (aborta el pipeline)."""


def cargar_catalogo(ruta=None, *, estricto: bool = True) -> pd.DataFrame:
    """Lee el catálogo único y valida su contrato de esquema.

    Aborta si hay errores de contrato (presencia, clave, obligatorias, dominio,
    tipos). Las advertencias (curación pendiente) se imprimen y no detienen.
    """
    ruta = ruta or config.CATALOGO
    cat = pd.read_excel(ruta)
    _validar_contrato(cat, estricto=estricto)
    return cat


def _es_vacio(serie: pd.Series) -> pd.Series:
    """True donde el valor es NA o cadena vacía/espacios."""
    return serie.isna() | (serie.astype(str).str.strip() == "")


def _validar_contrato(cat: pd.DataFrame, *, estricto: bool = True) -> None:
    errores: list[str] = []
    advertencias: list[str] = []

    # 1 · Presencia de las columnas de contrato
    faltan = [c for c in config.CONTRATO_COLUMNAS if c not in cat.columns]
    if faltan:
        errores.append(f"faltan columnas de contrato: {faltan}")

    # 2 · Clave: única y sin nulos
    if config.CLAVE in cat.columns:
        if cat[config.CLAVE].isna().any():
            errores.append(f"'{config.CLAVE}' tiene nulos")
        dup = cat.loc[cat[config.CLAVE].duplicated(), config.CLAVE].tolist()
        if dup:
            errores.append(f"'{config.CLAVE}' duplicado: {dup}")

    # 3 · Obligatorias sin nulos
    for c in config.OBLIGATORIAS:
        if c in cat.columns and _es_vacio(cat[c]).any():
            errores.append(f"'{c}' tiene valores vacíos (obligatoria)")

    # 4 · Dominio de las categóricas controladas
    for c, dominio in config.VOCAB.items():
        if c in cat.columns:
            observados = set(cat.loc[~_es_vacio(cat[c]), c].astype(str))
            fuera = observados - dominio
            if fuera:
                errores.append(f"'{c}' con valores fuera de vocabulario: {sorted(fuera)}")

    # 5 · Booleanas
    for c in config.BOOLEANAS:
        if c in cat.columns:
            vals = set(cat[c].dropna().tolist())
            if not vals <= {True, False, 0, 1}:
                errores.append(f"'{c}' no es booleana (valores: {sorted(map(str, vals))[:5]})")

    # 6 · Advertencias — curación pendiente entre las variables USADAS (incluidas y no excluidas).
    #     Las excluidas/administrativas pueden tener campos vacíos legítimamente.
    if not faltan:
        incluida = cat[config.BOOLEANAS].fillna(False).astype(bool).any(axis=1)
        usada = incluida & _es_vacio(cat["exclusion_causa"])

        def _falta(cond, msg):
            n = int((usada & cond).sum())
            if n:
                advertencias.append(f"{n} {msg}")

        _falta((cat["conceptual_type"] == "bin") & _es_vacio(cat["positive_label"]), "binarias sin 'positive_label'")
        _falta((cat["conceptual_type"] == "ord") & _es_vacio(cat["categories"]), "ordinales sin 'categories'")
        _falta((cat["conceptual_type"] == "num") & _es_vacio(cat["unit"]), "numéricas sin 'unit'")
        _falta(_es_vacio(cat["timepoint"]), "variables sin 'timepoint'")
        _falta(_es_vacio(cat["domain"]), "variables sin 'domain'")

    # Resolución
    if errores and estricto:
        raise ContratoError("Contrato del catálogo incumplido:\n  - " + "\n  - ".join(errores))
    for e in errores:
        print(f"✗ {e}")
    for a in advertencias:
        print(f"⚠ {a}")
    if not errores:
        print(f"✔ contrato válido: {len(cat)} variables · {len(config.CONTRATO_COLUMNAS)} columnas blindadas")


# ─────────────────────────────────────────────────────────────────────────────
# Ingesta y curación (tipado guiado por el catálogo)
# ─────────────────────────────────────────────────────────────────────────────

_BIN_MAP = {
    "1": True, 1: True, True: True, "sí": True, "si": True, "true": True,
    "verdadero": True, "yes": True, "y": True, "t": True, "s": True,
    "seleccionados": True,          # etiqueta de casilla de RedCAP
    "0": False, 0: False, False: False, "no": False, "false": False,
    "falso": False, "n": False, "f": False,
    "no seleccionados": False,      # etiqueta de casilla de RedCAP
}


def _parse_categories(cats):
    """Parsea 'categories' (lista, JSON o 'a,b,c') a lista de str, o None."""
    if cats is None or (isinstance(cats, float) and pd.isna(cats)):
        return None
    if isinstance(cats, (list, tuple, set)):
        return [str(c) for c in cats]
    txt = str(cats).strip()
    if not txt:
        return None
    if txt[0] in "[{":
        try:
            val = json.loads(txt)
            if isinstance(val, list):
                return [str(c) for c in val]
        except Exception:
            pass
    return [s.strip() for s in txt.split(",") if s.strip()]


def _to_boolean(s: pd.Series) -> pd.Series:
    if str(s.dtype) in {"bool", "boolean"}:
        return s.astype("boolean")
    m = s.astype("string").str.strip().str.lower().map(_BIN_MAP).astype("boolean")
    if m.isna().all():
        num = pd.to_numeric(s, errors="coerce")
        m = num.map(lambda v: True if v == 1 else (False if v == 0 else pd.NA)).astype("boolean")
    return m


def _to_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    nz = x.dropna()
    if len(nz) and np.all(np.isclose(nz, np.round(nz))):
        return x.round().astype("Int64")
    return x.astype("float64")


def _canon(v):
    """Token canónico de un valor: '3.0' y 3.0 → '3'; el resto, texto limpio."""
    if pd.isna(v):
        return pd.NA
    t = str(v).strip()
    if t == "":
        return pd.NA
    try:
        x = float(t.replace(",", "."))
        return str(int(round(x))) if np.isfinite(x) and np.isclose(x, round(x)) else str(x)
    except Exception:
        return t


def _to_ordinal(s: pd.Series, cats) -> pd.Series:
    """Categórica ordenada con el orden clínico del catálogo.

    Canoniza valores y categorías con el mismo criterio, de modo que un nivel almacenado
    como '3.0', 3.0 o '3' calce con la categoría '3' con independencia del dtype de origen.
    """
    cats_raw = _parse_categories(cats)
    if not cats_raw:
        return s.astype("string").astype("category")   # sin orden si falta
    norm = [_canon(c) for c in cats_raw]
    s_str = s.map(_canon).astype("string")
    unknown = ~s_str.isna() & ~s_str.isin(norm)
    return pd.Series(pd.Categorical(s_str.where(~unknown, other=pd.NA), categories=norm, ordered=True), name=s.name)


def _to_nominal(s: pd.Series, cats) -> pd.Series:
    if not cats:
        return s.astype("category")
    s_str = s.astype("string")
    unknown = ~s_str.isna() & ~s_str.isin(cats)
    return pd.Series(pd.Categorical(s_str.where(~unknown, other=pd.NA), categories=cats, ordered=False), name=s.name)


def _to_fecha(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _unico(target: str, usados: set) -> str:
    if target not in usados:
        usados.add(target)
        return target
    k = 1
    while f"{target}_{k}" in usados:
        k += 1
    usados.add(f"{target}_{k}")
    return f"{target}_{k}"


def cargar_raw(ruta=None) -> pd.DataFrame:
    """Lee los datos crudos de RedCAP y normaliza los encabezados.

    Los nombres de RedCAP traen espacios sobrantes (p. ej. 'Fecha de Nacimiento '), que
    impiden el calce con `var_original` del catálogo.
    """
    raw = pd.read_excel(ruta or config.RAW)
    raw.columns = [str(c).strip() for c in raw.columns]
    return raw


def verificar_calce_raw(raw: pd.DataFrame, cat: pd.DataFrame) -> list[str]:
    """Comprueba que cada `var_original` de origen crudo exista en los datos de RedCAP.

    `var_original` es la llave de unión con la fuente: debe conservar el nombre literal,
    erratas incluidas. Las correcciones de estilo se aplican en `label`/`description`.
    """
    cols = set(map(str, raw.columns))
    faltan = [f"{r['var_rename']} → {r['var_original']!r}"
              for _, r in cat[cat["source"] == "raw"].iterrows()
              if str(r["var_original"]).strip() not in cols]
    if faltan:
        print(f"✗ {len(faltan)} var_original sin calce en el crudo:")
        for f in faltan:
            print(f"    {f}")
    return faltan


def curar(raw: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Renombra (var_original -> var_rename) y tipa cada columna según el catálogo.

    bin -> boolean · num -> Int64/float · ord -> categórica ordenada (orden clínico del
    catálogo) · nom -> categórica · fecha -> datetime. El orden de las ordinales sale del
    catálogo, no del dato: mata el bug ordinal en su fuente conceptual.
    """
    faltan = verificar_calce_raw(raw, cat)
    if faltan:
        raise ContratoError(
            "Hay `var_original` sin correspondencia en los datos crudos. Esa columna es la "
            "llave de unión con la fuente y debe conservar el nombre literal, erratas "
            "incluidas; las correcciones de estilo van en `label`."
        )
    out = raw.copy()
    usados = set(map(str, out.columns))
    ren = {}
    for _, r in cat.iterrows():
        orig, new = str(r["var_original"]).strip(), str(r["var_rename"]).strip()
        if orig in out.columns and orig != new:
            ren[orig] = _unico(new, usados)
    out = out.rename(columns=ren)
    ahora = lambda o: ren.get(o, o)

    for _, r in cat.iterrows():
        col = ahora(str(r["var_original"]).strip())
        if col not in out.columns:
            continue
        ct = str(r["conceptual_type"]).strip().lower()
        cats = _parse_categories(r.get("categories"))
        s = out[col]
        if ct == "num":
            out[col] = _to_numeric(s)
        elif ct == "bin":
            out[col] = _to_boolean(s)
        elif ct == "ord":
            out[col] = _to_ordinal(s, cats)
        elif ct == "nom":
            out[col] = _to_nominal(s, cats)
        elif ct == "fecha":
            out[col] = _to_fecha(s)

    return curar_dolor_intraop(out)


def curar_dolor_intraop(df: pd.DataFrame) -> pd.DataFrame:
    """Completa el puntaje de dolor intraoperatorio cuando se registró ausencia de dolor.

    El puntaje (`d_io_nrs`) solo se consignaba ante dolor presente, de modo que las
    pacientes con `d_io = False` quedaban sin valor. La ausencia de dolor equivale a un
    puntaje de 0, que es lo que se imputa. El resto de las combinaciones se conserva.
    """
    if not {"d_io", "d_io_nrs"} <= set(df.columns):
        return df
    out = df.copy()
    nrs = out["d_io_nrs"]
    completar = out["d_io"].eq(False) & nrs.isna()
    if completar.any():
        if isinstance(nrs.dtype, pd.CategoricalDtype):
            if "0" not in nrs.cat.categories:
                out["d_io_nrs"] = nrs.cat.add_categories(["0"])
            out.loc[completar, "d_io_nrs"] = "0"
        else:
            out.loc[completar, "d_io_nrs"] = 0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Desenlace (END → binario por umbral, severidad, desenlaces por tiempo)
# ─────────────────────────────────────────────────────────────────────────────

UMBRAL_MG = 4    # dolor moderado-grave: END ≥ 4
UMBRAL_GRAVE = 7  # dolor intenso/grave: END ≥ 7
PAIN_COLS = ("d24", "d48", "d7")


def _limpia_end(s: pd.Series) -> pd.Series:
    """Normaliza un puntaje END a numérico 0–10; fuera de rango o no numérico → NA."""
    t = s.astype("string").str.strip().replace(
        {"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "<NA>": pd.NA, "NA": pd.NA}
    ).str.replace(",", ".", regex=False)
    x = pd.to_numeric(t, errors="coerce")
    x = x.mask(x.notna() & ~x.between(0, 10, inclusive="both"), np.nan)
    nz = x.dropna()
    if len(nz) and np.all(np.isclose(nz, np.round(nz))):
        return x.round().astype("Int64")
    return x.astype("float64")


def _severidad3(v):
    """END → ordinal 0/1/2 (leve / moderado / intenso)."""
    if pd.isna(v):
        return pd.NA
    v = float(v)
    return 0 if v < UMBRAL_MG else (1 if v < UMBRAL_GRAVE else 2)


def construir_desenlace(df: pd.DataFrame, pain_cols=PAIN_COLS) -> pd.DataFrame:
    """Construye los desenlaces de dolor a partir de los END de la primera semana.

    Produce: `dmg_24h/48h/7d` (END ≥ 4 por tiempo), `n_outcome_measures_available`,
    `desenlace_completo`, `d_ms_any`, `dmg_1s_bin` (primario: dolor moderado-grave en la
    semana, casos completos), `dmax_1s` (intensidad máxima), `severidad` (ordinal 0/1/2) y
    `dolor_grave` (END ≥ 7). Umbral moderado-grave = 4.
    """
    out = df.copy()
    pain_cols = list(pain_cols)

    for col in pain_cols:
        if col not in out.columns:
            raise KeyError(f"falta la columna de dolor '{col}'")
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(object)
        out[col] = _limpia_end(out[col])

    for src, dst in {"d24": "dmg_24h", "d48": "dmg_48h", "d7": "dmg_7d"}.items():
        out[dst] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        m = out[src].notna()
        out.loc[m, dst] = out.loc[m, src].ge(UMBRAL_MG)

    out["n_outcome_measures_available"] = out[pain_cols].notna().sum(axis=1)
    out["desenlace_completo"] = out[pain_cols].notna().all(axis=1)
    any_mg = out[pain_cols].ge(UMBRAL_MG).any(axis=1)

    out["d_ms_any"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[~out["desenlace_completo"], "d_ms_any"] = any_mg[~out["desenlace_completo"]]

    out["dmg_1s_bin"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[out["desenlace_completo"], "dmg_1s_bin"] = any_mg[out["desenlace_completo"]]

    out["dmax_1s"] = out[pain_cols].max(axis=1, skipna=True)
    out["severidad"] = out["dmax_1s"].map(_severidad3).astype("Int64")
    out["dolor_grave"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    m = out["dmax_1s"].notna()
    out.loc[m, "dolor_grave"] = out.loc[m, "dmax_1s"].ge(UMBRAL_GRAVE)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cohorte analítica (criterios de elegibilidad)
# ─────────────────────────────────────────────────────────────────────────────

EDAD_MIN = 18       # años
EG_MIN = 28         # semanas (se incluye desde 28 inclusive)
EDAD_IMPOSIBLE = 1  # bajo 1 año el registro es imposible en esta población → NA


def curar_edad(df: pd.DataFrame) -> pd.DataFrame:
    """Repara la edad de forma determinista: fecha_ces − fecha_nac.

    Sustituye a la versión previa basada en `Timestamp.today()`, que hacía depender la edad
    del día de ejecución. Completa los faltantes a partir de las fechas registradas y anula
    los valores imposibles (< 1 año), que corresponden a registros corruptos y no a una edad
    verificable. Las edades bajo el mínimo etario pero posibles se conservan: es el criterio
    de elegibilidad el que excluye a esas pacientes.
    """
    out = df.copy()
    edad = pd.to_numeric(out.get("edad"), errors="coerce")
    if {"fecha_nac", "fecha_ces"} <= set(out.columns):
        det = (pd.to_datetime(out["fecha_ces"], errors="coerce")
               - pd.to_datetime(out["fecha_nac"], errors="coerce")).dt.days / 365.25
        edad = edad.where(edad.notna(), det)
    out["edad"] = edad.mask(edad < EDAD_IMPOSIBLE)
    return out


def clasificar_elegibilidad(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega `motivo_exclusion` (NA en las incluidas) y `estrato` a cada paciente.

    Cada paciente se cuenta una sola vez, por el primer criterio que incumple. `estrato`
    resume el origen de la exclusión: protocolo (criterios FONIS) o seguimiento (criterio
    de la tesis), que son las dos vías que caracteriza el análisis de sesgo de selección.
    """
    d = curar_edad(df).copy()
    criterios = [
        ("Edad < 18 años", "protocolo",
         pd.to_numeric(d.get("edad"), errors="coerce").lt(EDAD_MIN)),
        ("Edad gestacional < 28 semanas", "protocolo",
         pd.to_numeric(d.get("eg"), errors="coerce").lt(EG_MIN)),
        ("Sin consentimiento informado", "protocolo",
         _to_boolean(d["consent"]).eq(False) if "consent" in d else pd.Series(False, index=d.index)),
        ("Anestesia general", "protocolo",
         d["tec_anest"].astype("string").str.strip().str.lower().eq("general")
         if "tec_anest" in d else pd.Series(False, index=d.index)),
        ("Seguimiento incompleto", "seguimiento", d["dmg_1s_bin"].isna()),
    ]
    d["motivo_exclusion"] = pd.Series(pd.NA, index=d.index, dtype="string")
    d["estrato"] = pd.Series("incluida", index=d.index, dtype="string")
    pendientes = pd.Series(True, index=d.index)
    for nombre, estrato, viola in criterios:
        viola = viola.fillna(False).astype(bool) & pendientes
        d.loc[viola, "motivo_exclusion"] = nombre
        d.loc[viola, "estrato"] = estrato
        pendientes &= ~viola
    return d


def construir_cohorte(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica los criterios de elegibilidad y devuelve (cohorte, flujo).

    Criterios de inclusión (FONIS): edad ≥ 18 años · edad gestacional ≥ 28 semanas ·
    consentimiento informado firmado · sin anestesia general.
    Criterio de la tesis: seguimiento completo (END a 24 h, 48 h y 7 d) que permita
    derivar el desenlace `dmg_1s_bin`.

    Un valor faltante no excluye por sí solo: se excluye únicamente cuando el criterio se
    incumple de forma comprobable. `flujo` cuenta las excluidas por criterio (insumo del
    diagrama de flujo).
    """
    d = clasificar_elegibilidad(df)
    n0 = len(d)
    orden = [("Edad < 18 años", "protocolo"),
             ("Edad gestacional < 28 semanas", "protocolo"),
             ("Sin consentimiento informado", "protocolo"),
             ("Anestesia general", "protocolo"),
             ("Seguimiento incompleto", "seguimiento")]
    conteo = d["motivo_exclusion"].value_counts()
    flujo = pd.DataFrame([{"criterio": c, "estrato": e, "n_excluidas": int(conteo.get(c, 0))}
                          for c, e in orden])
    flujo.loc[len(flujo)] = {"criterio": "Cohorte analítica", "n_excluidas": pd.NA}
    flujo["n_restante"] = list(n0 - flujo["n_excluidas"].fillna(0).cumsum().astype(int))
    return d.loc[d["motivo_exclusion"].isna()].copy(), flujo


# ─────────────────────────────────────────────────────────────────────────────
# Cribado de predictores (derivado; alimenta la selección para los modelos)
# ─────────────────────────────────────────────────────────────────────────────

def _metricas_categoricas(x: pd.Series) -> tuple[float, int, float]:
    c = x.value_counts(dropna=False)
    n = len(x)
    return float(c.max() / n), int(c.min()), float(c.min() / n)


def _puntajes_ordinales(x: pd.Series) -> pd.Series:
    """Códigos numéricos de una ordinal, respetando su orden."""
    if isinstance(x.dtype, pd.CategoricalDtype):
        s = pd.Series(x.cat.codes, index=x.index).replace(-1, np.nan)
        return s.dropna().astype(float)
    return pd.to_numeric(x, errors="coerce").dropna().astype(float)


def evaluar_variabilidad(s: pd.Series, tipo: str) -> dict:
    """Evalúa si una variable tiene baja variabilidad informativa.

    Replica los criterios de 01_EDA_screening: sin datos, muy pocos observados, un solo
    nivel, categoría dominante extrema, categoría rara extrema, y para ordinales/numéricas
    la ausencia de gradiente útil (sd/IQR/rango efectivo).
    """
    x = pd.Series(s).dropna()
    tipo = str(tipo).strip().lower()
    r = {"n_non_missing": len(x), "n_unique": int(x.nunique(dropna=True)),
         "dominant_pct": np.nan, "min_level_n": np.nan, "min_level_pct": np.nan,
         "sd": np.nan, "iqr": np.nan, "effective_range": np.nan,
         "baja_variabilidad": False, "motivo": None}

    if len(x) == 0:
        return {**r, "baja_variabilidad": True, "motivo": "sin_datos_observados"}
    if len(x) < config.MIN_NON_MISSING_VARIABILIDAD:
        return {**r, "baja_variabilidad": True, "motivo": "muy_pocos_datos_observados"}
    if x.nunique(dropna=True) <= 1:
        return {**r, "baja_variabilidad": True, "motivo": "un_solo_nivel_observado"}

    if tipo in {"bin", "nom", "ord"}:
        dom, mn_n, mn_p = _metricas_categoricas(x)
        r.update(dominant_pct=dom, min_level_n=mn_n, min_level_pct=mn_p)
        if dom >= config.DOMINANT_PCT_THRESHOLD:
            motivo = "ordinal_concentrada_en_un_nivel" if tipo == "ord" else "categoria_dominante_extrema"
            return {**r, "baja_variabilidad": True, "motivo": motivo}
        if tipo in {"bin", "nom"}:
            if mn_p < config.RARE_PCT_THRESHOLD and mn_n < config.RARE_N_THRESHOLD:
                return {**r, "baja_variabilidad": True, "motivo": "categoria_rara_extrema"}
            return r
        xo = _puntajes_ordinales(x)
        if len(xo) < config.MIN_NON_MISSING_VARIABILIDAD:
            return {**r, "baja_variabilidad": True, "motivo": "ordinal_no_convertible_o_con_orden_insuficiente"}
        q1, q3 = xo.quantile([0.25, 0.75])
        sd, iqr, rng = float(xo.std(ddof=1)), float(q3 - q1), float(xo.max() - xo.min())
        r.update(sd=sd, iqr=iqr, effective_range=rng)
        if rng < config.MIN_EFFECTIVE_RANGE_THRESHOLD and iqr < config.MIN_IQR_THRESHOLD and sd < config.MIN_SD_THRESHOLD:
            return {**r, "baja_variabilidad": True, "motivo": "ordinal_sin_gradiente_util"}
        return r

    xn = pd.to_numeric(x, errors="coerce").dropna()
    if len(xn) < config.MIN_NON_MISSING_VARIABILIDAD:
        return {**r, "baja_variabilidad": True, "motivo": "muy_pocos_datos_numericos"}
    q1, q3 = xn.quantile([0.25, 0.75])
    sd, iqr, rng = float(xn.std(ddof=1)), float(q3 - q1), float(xn.max() - xn.min())
    r.update(sd=sd, iqr=iqr, effective_range=rng)
    if sd < config.MIN_SD_THRESHOLD and iqr < config.MIN_IQR_THRESHOLD:
        return {**r, "baja_variabilidad": True, "motivo": "dispersion_minima"}
    if rng < config.MIN_EFFECTIVE_RANGE_THRESHOLD:
        return {**r, "baja_variabilidad": True, "motivo": "rango_efectivo_estrecho"}
    return r


def cribar(cohorte: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Calcula el cribado de predictores sobre la cohorte (tabla DERIVADA).

    Evalúa datos faltantes y variabilidad en las variables con rol evaluable
    (`pred`, `fr`, `proxy`) y marca `candidato` = cumple ambos criterios. No filtra ni
    modifica la cohorte: solo describe. El análisis descriptivo y bivariado se realiza
    sobre la cohorte completa, con independencia de este cribado.
    """
    filas = []
    for _, r in cat.iterrows():
        v, rol = r["var_rename"], str(r["prelim_role"]).strip().lower()
        evaluable = rol in config.ROLES_EVAL and v not in config.VARS_TEXTO_LIBRE
        fila = {"var_rename": v, "prelim_role": rol, "evaluable": evaluable,
                "en_cohorte": v in cohorte.columns}
        if v in cohorte.columns:
            s = cohorte[v]
            fila["missing_pct"] = round(100 * s.isna().mean(), 2)
            fila["missing_ok"] = fila["missing_pct"] <= config.MISSING_THRESHOLD
            fila.update(evaluar_variabilidad(s, r["conceptual_type"]))
            fila["variabilidad_ok"] = not fila["baja_variabilidad"]
        filas.append(fila)

    d = pd.DataFrame(filas)
    d["candidato"] = d["evaluable"] & d.get("missing_ok", False).fillna(False) & d.get("variabilidad_ok", False).fillna(False)
    return d


def comparabilidad_registro(cohorte: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Evalúa si cada variable se registró de forma comparable entre centros (DERIVADO).

    La heterogeneidad se estima sobre casos completos: si la completitud difiere entre
    centros, la comparación enfrenta la población de un centro con una submuestra
    seleccionada de otro, y la diferencia observada puede reflejar el registro y no a las
    pacientes. Marca como no comparable a las variables cuyo rango de datos faltantes entre
    centros supera el umbral declarado en `config`.
    """
    centro = cohorte[config.CENTER_VAR].astype(str)
    filas = []
    for v in cat.loc[cat["include_heterogeneidad"].fillna(False), "var_rename"]:
        if v not in cohorte.columns:
            continue
        por_centro = cohorte.groupby(centro)[v].apply(lambda s: 100 * s.isna().mean())
        rango = float(por_centro.max() - por_centro.min())
        filas.append({
            "var_rename": v,
            "missing_global_pct": round(100 * cohorte[v].isna().mean(), 2),
            "missing_rango_centros": round(rango, 2),
            "missing_max_centro": round(float(por_centro.max()), 2),
            "registro_comparable": rango <= config.MISSING_RANGO_CENTROS_MAX,
        })
    return pd.DataFrame(filas)


def causas_exclusion(cat: pd.DataFrame, cribado: pd.DataFrame) -> pd.DataFrame:
    """Tabla auditable de variables excluidas del análisis y su causa (insumo del anexo).

    Combina la causa curada del catálogo con la evidencia derivada del cribado (porcentaje
    de faltantes y motivo de baja variabilidad), de modo que cada exclusión quede
    justificada con su cifra y sea verificable contra los datos.
    """
    d = cat.loc[cat["exclusion_causa"].notna(),
                ["var_rename", "label", "block", "prelim_role",
                 "exclusion_causa", "exclusion_subcausa", "exclusion_nota"]].copy()
    cols = [c for c in ("var_rename", "missing_pct", "motivo", "variabilidad_ok") if c in cribado.columns]
    d = d.merge(cribado[cols], on="var_rename", how="left")
    return d.rename(columns={"motivo": "motivo_variabilidad"}).reset_index(drop=True)


def comparar_incluidas_excluidas(clasificado: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Compara a las incluidas con cada estrato de exclusión (sesgo de selección).

    Se restringe a las variables **basales preoperatorias** de rol clínico (características
    demográficas y predictores), excluyendo identificadores, fechas y campos de validación,
    conforme al reporte de características de participantes de TRIPOD+AI. Comparar el manejo
    intra o postoperatorio de quienes fueron excluidas por no tener seguimiento mezclaría la
    causa de exclusión con el objeto de comparación. La métrica de balance es la diferencia
    de medias estandarizada (SMD), que no depende del tamaño de los grupos.

    Los faltantes se enmascaran (comparación de casos completos) en lugar de tratarse como
    una categoría más: incluirlos hacía que la prueba detectara el desbalance de faltantes
    —tautológico, pues se excluyó por falta de mediciones— en vez del de la variable.
    """
    from . import statistik  # se importa aquí para no acoplar la carga del módulo

    basales = cat.loc[cat["block"].eq("preop")
                      & cat["prelim_role"].isin(config.ROLES_EVAL)
                      & ~cat["var_rename"].isin(config.VARS_TEXTO_LIBRE), "var_rename"]
    filas = []
    for estrato in ("protocolo", "seguimiento"):
        sub = clasificado[clasificado["estrato"].isin(["incluida", estrato])]
        grupo = sub["estrato"].eq("incluida")
        for v in basales:
            if v not in sub.columns or sub.loc[~grupo, v].notna().sum() == 0:
                continue
            filas.append({"estrato": estrato, "var_rename": v,
                          "n_incluidas": int(sub.loc[grupo, v].notna().sum()),
                          "n_excluidas": int(sub.loc[~grupo, v].notna().sum()),
                          **statistik.balance(sub[v], grupo)})
    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────
# Congelado de los datos analíticos
# ─────────────────────────────────────────────────────────────────────────────

def congelar(df: pd.DataFrame, ruta) -> str:
    """Escribe el parquet analítico y devuelve su hash (firma de reproducibilidad).

    Solo escribe dentro del espacio propio del flujo (`config.SALIDA`). Los artefactos del
    análisis original se conservan intactos como testigo para contrastar cifras.
    """
    import hashlib

    ruta = Path(ruta).resolve()
    if config.SALIDA.resolve() not in ruta.parents:
        raise PermissionError(
            f"'{ruta}' queda fuera de {config.SALIDA}. El flujo reestructurado no escribe "
            "sobre los artefactos del análisis original."
        )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta, index=False)
    h = hashlib.sha256(ruta.read_bytes()).hexdigest()[:16]
    print(f"✔ {ruta.name}: {len(df)} filas · {df.shape[1]} columnas · sha256 {h}")
    return h


# ═════════════════════════════════════════════════════════════════════════════
# Subcohorte espinal · recodificación del manejo anestésico
# ═════════════════════════════════════════════════════════════════════════════

# Analgésicos no opioides que componen el recuento multimodal. La morfina queda fuera por
# diseño: incluirla solapa la definición con la del propio término de morfina y deja la
# variable casi constante, por encima del umbral de variabilidad del cribado.
_NO_OPIOIDES = ["an_dipirona", "an_ketoprofeno", "an_ketorolaco", "an_paracetamol"]
_ANTIEMETICOS = ["dexa_pnv", "ondan_pnv", "drope_pnv"]

# Los nombres de nivel son texto publicable: aparecen en las tablas y en el manuscrito, de
# modo que se escriben sin abreviaturas ni signos. «Menos de dos» no es «ninguno»: agrupa a
# quienes recibieron cero y a quienes recibieron un solo analgésico no opioide, y llamarlo
# «No» invitaba a leerlo como ausencia de analgesia.
NIVELES_MORFINA = ["No", "50 a 80 mcg", "100 mcg o más"]
NIVELES_MULTIMODAL = ["Menos de dos", "2 agentes", "3 agentes"]
NIVELES_PNV_COMBO = ["Ninguno", "Sin dexametasona", "Dexametasona sola",
                     "Dexametasona y otros"]


def _presencia(s: pd.Series) -> pd.Series:
    """Convierte una categórica de dosis en presencia del fármaco, preservando faltantes."""
    return s.map(lambda x: pd.NA if pd.isna(x) else x != "No").astype("boolean")


def derivar_espinal(df: pd.DataFrame) -> pd.DataFrame:
    """Recodifica el manejo anestésico de la subcohorte espinal.

    Devuelve seis variables. `morf_bin` registra la administración de morfina intratecal.
    `an_multimodal_no` cuenta cuántos analgésicos no opioides se administraron de forma
    simultánea, lo que aporta una gradación de dosis-respuesta que el corte binario pierde.
    `pnv_multi` y `pnv_combo` describen la profilaxis antiemética, la primera por su
    intensidad y la segunda por su composición. `bundle_guia` resume la concordancia con la
    guía analgésica.

    Una fila con cualquier componente faltante queda como faltante: el recuento exacto es
    ambiguo si se desconoce si un fármaco se administró, y darlo por ausente sesgaría la
    gradación hacia abajo.
    """
    d = df.copy()
    for v in _NO_OPIOIDES + _ANTIEMETICOS + ["morf_it"]:
        if v not in d.columns:
            raise KeyError(f"derivar_espinal: falta la variable de origen {v!r}")

    d["morf_bin"] = _presencia(d["morf_it"])
    # Dosis agrupada en tres tramos: la escala original tiene niveles con muy pocos casos y
    # su gradación es lo que el arco predictivo aprovecha, mientras que la presencia basta
    # para el arco asociativo.
    def _tramo(x):
        if pd.isna(x):
            return np.nan
        t = str(x).strip()
        if t == "No":
            return "No"
        dosis = float("".join(c for c in t if c.isdigit() or c == "."))
        return NIVELES_MORFINA[1] if dosis < 100 else NIVELES_MORFINA[2]

    d["morf_rec"] = pd.Categorical([_tramo(x) for x in d["morf_it"].astype(object)],
                                   categories=NIVELES_MORFINA, ordered=True)

    no_op = pd.DataFrame({v: _presencia(d[v]) for v in _NO_OPIOIDES})
    for v in _NO_OPIOIDES:
        d[f"{v}_bin"] = no_op[v]
    incompleto = no_op.isna().any(axis=1)
    n_agentes = no_op.sum(axis=1, skipna=True).astype(int)
    menos, dos, tres = NIVELES_MULTIMODAL
    etiqueta = n_agentes.map({0: menos, 1: menos, 2: dos, 3: tres,
                              4: "4 agentes"}).astype(object)
    etiqueta[incompleto] = np.nan
    if (etiqueta == "4 agentes").any():
        raise ValueError("derivar_espinal: la categoría '4 agentes' se instanció y no está "
                         "declarada, revisar NIVELES_MULTIMODAL")
    d["an_multimodal_no"] = pd.Categorical(etiqueta, categories=NIVELES_MULTIMODAL, ordered=True)

    pnv = pd.DataFrame({v: _presencia(d[v]) for v in _ANTIEMETICOS})
    for v, nombre in zip(_ANTIEMETICOS, ("pnv_dexa", "pnv_ondan", "pnv_drope")):
        d[nombre] = pnv[v]
    n_pnv, na_pnv = pnv.sum(axis=1, skipna=True).astype(int), pnv.isna().sum(axis=1)
    d["pnv_multi"] = pd.array([True if t >= 2 else False if t + na < 2 else pd.NA
                               for t, na in zip(n_pnv, na_pnv)], dtype="boolean")

    dexa, otro = pnv["dexa_pnv"].fillna(False), (pnv["ondan_pnv"].fillna(False)
                                                 | pnv["drope_pnv"].fillna(False))
    combo = np.select([dexa & otro, dexa & ~otro, ~dexa & otro],
                      [NIVELES_PNV_COMBO[3], NIVELES_PNV_COMBO[2], NIVELES_PNV_COMBO[1]],
                      NIVELES_PNV_COMBO[0])
    combo = pd.Series(combo, index=d.index, dtype=object)
    combo[pnv.isna().any(axis=1)] = np.nan
    d["pnv_combo"] = pd.Categorical(combo, categories=NIVELES_PNV_COMBO, ordered=False)

    # Versión binaria del recuento: recibir dos o más analgésicos no opioides. Convive con
    # la gradación pero no en un mismo modelo, porque una es el corte de la otra.
    d["multimodal_bin"] = d["an_multimodal_no"].map(
        lambda x: pd.NA if pd.isna(x) else x != NIVELES_MULTIMODAL[0]).astype("boolean")
    d["bundle_guia"] = d["morf_bin"] & d["multimodal_bin"] & d["pnv_multi"]
    return d
