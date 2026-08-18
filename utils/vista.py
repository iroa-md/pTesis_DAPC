"""vista.py — capa de agregados de la presentación.

Decide **qué** se muestra; no dibuja ni escribe documentos. Las tres capas de salida
(`rendern` estático, reporte interactivo y `tabula`) consumen estos mismos agregados, de
modo que no puedan reportar cifras distintas para una misma variable.

Reglas que garantiza (docs/artefactos.md §4):
  · El orden de los niveles proviene del catálogo, nunca de la frecuencia observada.
  · El tipo se resuelve con `statistik.resolver_tipo` sobre la estructura observada.
  · El denominador es explícito: incluir o no los faltantes es un parámetro declarado.
  · En modo publicable, las celdas con menos de `config.SUPRESION_N_MIN` casos se suprimen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, daten, statistik


# ─────────────────────────────────────────────────────────────────────────────
# Metadatos de presentación de una variable
# ─────────────────────────────────────────────────────────────────────────────

def niveles(s: pd.Series, cat_fila: pd.Series | None = None) -> list[str]:
    """Niveles de una variable categórica en su orden de presentación.

    El orden proviene del catálogo (`categories`); si no está declarado, se usa el de la
    categoría del dato. Nunca se ordena por frecuencia ni alfabéticamente: en variables de
    dosis o de gradiente clínico ese orden es interpretativamente incorrecto.
    """
    if cat_fila is not None:
        declarados = daten._parse_categories(cat_fila.get("categories"))
        if declarados:
            observados = set(s.dropna().astype(str))
            return [n for n in map(str, declarados) if n in observados]
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [str(x) for x in s.cat.remove_unused_categories().cat.categories]
    return sorted(s.dropna().astype(str).unique())


def etiqueta(cat_fila: pd.Series, var: str) -> str:
    """Etiqueta legible de la variable; recurre al nombre técnico si falta."""
    lab = cat_fila.get("label") if cat_fila is not None else None
    return str(lab) if pd.notna(lab) and str(lab).strip() else var


# ─────────────────────────────────────────────────────────────────────────────
# Agregados
# ─────────────────────────────────────────────────────────────────────────────

def _suprimir(d: pd.DataFrame, col_n: str = "n", modo: str | None = None) -> pd.DataFrame:
    """Marca como suprimidas las celdas con recuento bajo el umbral (modo publicable)."""
    modo = modo or config.MODO_ARTEFACTO
    d = d.copy()
    d["suprimido"] = False
    if modo == "publicable" and col_n in d.columns:
        bajo = (d[col_n].fillna(0) < config.SUPRESION_N_MIN).to_numpy()
        d.loc[bajo, "suprimido"] = True
        for c in (col_n, "pct"):
            if c in d.columns:
                d.loc[bajo, c] = np.nan
    return d


def distribucion(cohorte: pd.DataFrame, var: str, cat: pd.DataFrame, *,
                 estrato: str | None = None, incluir_faltantes: bool = False,
                 modo: str | None = None) -> pd.DataFrame:
    """Distribución de una variable categórica, opcionalmente por estrato.

    Devuelve un agregado tidy con `nivel`, `n` y `pct`. El denominador excluye los
    faltantes salvo que `incluir_faltantes` sea verdadero, en cuyo caso se añaden como un
    nivel propio y el porcentaje se calcula sobre el total. Esa decisión debe coincidir con
    la de las tablas del manuscrito, que operan sobre casos completos.
    """
    fila = cat.loc[cat["var_rename"] == var]
    fila = fila.iloc[0] if len(fila) else None
    s = cohorte[var]
    orden = niveles(s, fila)

    d = pd.DataFrame({"nivel": s.astype("string")})
    if incluir_faltantes:
        d["nivel"] = d["nivel"].fillna("Sin registro")
        orden = orden + ["Sin registro"]
    else:
        d = d[d["nivel"].notna()]

    if estrato is not None:
        d[estrato] = cohorte.loc[d.index, estrato].astype("string")
        g = d.groupby([estrato, "nivel"], observed=True).size().reset_index(name="n")
        g["pct"] = 100 * g["n"] / g.groupby(estrato)["n"].transform("sum")
    else:
        g = d.groupby("nivel", observed=True).size().reset_index(name="n")
        g["pct"] = 100 * g["n"] / g["n"].sum()

    g["nivel"] = pd.Categorical(g["nivel"], categories=orden, ordered=True)
    g = g.sort_values([estrato, "nivel"] if estrato else "nivel").reset_index(drop=True)
    g.attrs.update(variable=var, etiqueta=etiqueta(fila, var) if fila is not None else var,
                   n_observado=int(s.notna().sum()), n_faltante=int(s.isna().sum()))
    return _suprimir(g, modo=modo)


def resumen_numerica(cohorte: pd.DataFrame, var: str, cat: pd.DataFrame, *,
                     estrato: str | None = None) -> pd.DataFrame:
    """Resumen de una variable numérica: mediana, cuartiles y extremos, por estrato.

    No expone valores individuales, de modo que es publicable sin transformación adicional.
    """
    fila = cat.loc[cat["var_rename"] == var]
    fila = fila.iloc[0] if len(fila) else None

    def _stats(x: pd.Series) -> dict:
        x = pd.to_numeric(x, errors="coerce").dropna()
        if x.empty:
            return {"n": 0, "mediana": np.nan, "p25": np.nan, "p75": np.nan,
                    "min": np.nan, "max": np.nan}
        return {"n": len(x), "mediana": x.median(), "p25": x.quantile(0.25),
                "p75": x.quantile(0.75), "min": x.min(), "max": x.max()}

    if estrato is not None:
        filas = [{estrato: k, **_stats(g)} for k, g in cohorte.groupby(cohorte[estrato].astype("string"))]
    else:
        filas = [_stats(cohorte[var])]
    d = pd.DataFrame(filas)
    d.attrs.update(variable=var, etiqueta=etiqueta(fila, var) if fila is not None else var,
                   unidad=(fila.get("unit") if fila is not None else None))
    return _suprimir(d, modo=None)


def perfil(cohorte: pd.DataFrame, var: str, cat: pd.DataFrame, *,
           estrato: str | None = None, **kw) -> pd.DataFrame:
    """Agregado de una variable, despachando por su tipo observado."""
    fila = cat.loc[cat["var_rename"] == var]
    tipo = statistik.resolver_tipo(
        cohorte[var], fila.iloc[0]["conceptual_type"] if len(fila) else "bin")
    if tipo == "num":
        return resumen_numerica(cohorte, var, cat, estrato=estrato)
    return distribucion(cohorte, var, cat, estrato=estrato, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Agregados de sección
# ─────────────────────────────────────────────────────────────────────────────

ETAPA_FLUJO = {"protocolo": ("Elegibles", "Criterios de elegibilidad del protocolo"),
               "seguimiento": ("Cohorte analítica", "Seguimiento incompleto")}


def flujo_participantes(flujo: pd.DataFrame) -> pd.DataFrame:
    """Cascada de selección agrupada por vía de exclusión.

    Reúne los criterios de un mismo origen en una sola etapa y conserva su desglose, que es
    la forma en que el manuscrito reporta el flujo: de la cohorte original a las elegibles y
    de estas a la cohorte analítica.
    """
    d = flujo[flujo["n_excluidas"].notna()].copy()
    n_actual = int(d["n_restante"].iloc[0] + d["n_excluidas"].iloc[0])
    etapas = []
    for estrato, g in d.groupby("estrato", sort=False):
        rotulo, motivo = ETAPA_FLUJO.get(estrato, (estrato, estrato))
        aplicados = g[g["n_excluidas"] > 0]
        detalle = ([f"{r['criterio'].lower()} (n = {int(r['n_excluidas'])})"
                    for _, r in aplicados.iterrows()] if len(aplicados) > 1 else [])
        n_ex = int(g["n_excluidas"].sum())
        n_actual -= n_ex
        etapas.append({"etapa": rotulo, "motivo": motivo, "n_excluidas": n_ex,
                       "detalle": " · ".join(detalle), "n_restante": n_actual})
    return pd.DataFrame(etapas)


def distribucion_desenlace(cohorte: pd.DataFrame) -> pd.DataFrame:
    """Intensidad del dolor por momento de medición, en categorías de severidad.

    Las tres mediciones de la primera semana se resumen en la escala clínica habitual
    (sin dolor o leve, moderado, intenso), que es la que da sentido al umbral del desenlace.
    """
    etiquetas = {0: "Sin dolor o leve", 1: "Moderado", 2: "Intenso"}
    filas = []
    for var, momento in [("d24", "24 horas"), ("d48", "48 horas"), ("d7", "7 días")]:
        s = pd.to_numeric(cohorte[var], errors="coerce").dropna()
        sev = s.map(daten._severidad3)
        for k, lab in etiquetas.items():
            n = int((sev == k).sum())
            filas.append({"momento": momento, "severidad": lab, "n": n,
                          "pct": 100 * n / len(s) if len(s) else np.nan})
    d = pd.DataFrame(filas)
    d["momento"] = pd.Categorical(d["momento"], ["24 horas", "48 horas", "7 días"], ordered=True)
    d["severidad"] = pd.Categorical(d["severidad"], list(etiquetas.values()), ordered=True)
    return _suprimir(d.sort_values(["momento", "severidad"]).reset_index(drop=True))


def incidencia_por_centro(cohorte: pd.DataFrame) -> pd.DataFrame:
    """Incidencia del desenlace por centro con IC 95 % de Wilson y contraste global."""
    d = statistik.frecuencia_por_centro(cohorte[config.OUTCOME], cohorte[config.CENTER_VAR])
    out = _suprimir(d, col_n="eventos")
    out.attrs["p_global"] = d.attrs.get("p_global")
    return out


def frecuencia_severidad_tiempo(cohorte: pd.DataFrame) -> pd.DataFrame:
    """Frecuencia de dolor por momento de medición, en las cuatro categorías clínicas.

    Extiende `distribucion_desenlace` (que colapsa sin dolor y leve en una sola categoría,
    la que da sentido al umbral del desenlace) a las cuatro por separado, excluyentes entre
    sí. La proporción con dolor (END > 0), que no lo es, se deriva aguas abajo como el
    complemento de «Sin dolor»: agregarla aquí como una quinta fila rompería esa
    exclusividad, que es lo que permite apilar la figura al 100 %. No se llama
    «incidencia»: ese término se reserva para el desenlace principal (`dmg_1s_bin`).
    """
    etiquetas = {0: "Sin dolor", 1: "Leve", 2: "Moderado", 3: "Intenso"}

    def _categoria(v: float) -> int:
        if v <= 0:
            return 0
        if v < daten.UMBRAL_MG:
            return 1
        if v < daten.UMBRAL_GRAVE:
            return 2
        return 3

    filas = []
    for var, momento in [("d24", "24 horas"), ("d48", "48 horas"), ("d7", "7 días")]:
        s = pd.to_numeric(cohorte[var], errors="coerce").dropna()
        cat = s.map(_categoria)
        for k, lab in etiquetas.items():
            n = int((cat == k).sum())
            filas.append({"momento": momento, "categoria": lab, "n": n,
                          "n_total": len(s), "pct": 100 * n / len(s) if len(s) else np.nan})
    d = pd.DataFrame(filas)
    d["momento"] = pd.Categorical(d["momento"], ["24 horas", "48 horas", "7 días"], ordered=True)
    d["categoria"] = pd.Categorical(d["categoria"], list(etiquetas.values()), ordered=True)
    return _suprimir(d.sort_values(["momento", "categoria"]).reset_index(drop=True))


def frecuencia_por_centro_tiempo(cohorte: pd.DataFrame) -> pd.DataFrame:
    """Frecuencia de dolor (END > 0) por centro y momento de medición, con IC 95 % de Wilson.

    Generaliza `incidencia_por_centro` (que opera sobre el desenlace compuesto de la
    semana, el único al que corresponde llamar «incidencia») a cada momento por separado;
    la fila GLOBAL es la frecuencia de la cohorte completa en ese momento, la referencia
    frente a la que se lee cada centro. El contraste χ² es propio de cada momento: se
    guarda en `attrs`, no en una celda.
    """
    centro = cohorte[config.CENTER_VAR]
    partes, p_global = [], {}
    for var, momento in [("d24", "24 horas"), ("d48", "48 horas"), ("d7", "7 días")]:
        raw = pd.to_numeric(cohorte[var], errors="coerce")
        # NA donde falta la medición, no False: de lo contrario un faltante cuenta
        # como "sin dolor" y la frecuencia queda subestimada.
        presencia = pd.Series(pd.NA, index=raw.index, dtype="boolean")
        presencia[raw.notna()] = raw[raw.notna()].gt(0)
        d = statistik.frecuencia_por_centro(presencia, centro)
        p_global[momento] = d.attrs["p_global"]
        d["momento"] = momento
        partes.append(d)
    out = pd.concat(partes, ignore_index=True)
    out.attrs["p_global"] = p_global
    return out


ORDEN_BLOQUE = {"preop": 1, "intraop": 2, "postop": 3}


def _tasa_exposicion(obs: pd.Series, cat_fila) -> str | None:
    """n/N (%) fuera del primer nivel, cuando ese nivel declarado es «No».

    Las categóricas de dosis -morfina intratecal, ketorolaco, dipirona- declaran «No»
    como su primer nivel: son exposiciones graduadas, y su categoría modal no permite
    recuperar qué proporción del centro estuvo expuesta. «No (67%)» obliga a derivar el
    33% en vez de leerlo, y con más de dos dosis la modal ni siquiera es el complemento.
    Se informa entonces la frecuencia del complemento de «No», en el mismo formato que
    las binarias.

    El criterio no se inventa aquí, ya está declarado en el catálogo. La comparación es
    contra la **cadena** «No» y no contra el primer nivel sin más: `d_io_nrs` declara
    [0, 1, …, 10] y su primer nivel, el 0, es un puntaje de dolor y no una ausencia de
    exposición. Devuelve None cuando no aplica, de modo que la variable conserve su
    categoría modal.
    """
    declarados = daten._parse_categories(cat_fila.get("categories")) if cat_fila is not None else None
    if not declarados:
        return None
    base = str(declarados[0]).strip().casefold()
    if base != "no":
        return None
    k = int((obs.astype(str).str.strip().str.casefold() != base).sum())
    return f"{k}/{len(obs)} ({100 * k / len(obs):.1f}%)".replace(".", ",")


def _resumen_celda(s: pd.Series, tipo: str, cat_fila, *,
                   tasa_exposicion: bool = False) -> str:
    """Resumen de una variable en una celda: mediana [RIC], n/N (%) o nivel modal.

    Con `tasa_exposicion`, las categóricas cuyo primer nivel declarado es «No» informan
    la frecuencia del complemento de ese nivel en vez de su categoría modal. Es opcional
    y no por omisión porque cambia la lectura de la celda: la tabla por centro contrasta
    exposición contra desenlace y necesita la tasa, mientras que la descriptiva y la de
    sesgo de selección despliegan los niveles y no la necesitan. Activarlo en todas
    alteraría tablas ya congeladas sin que nadie lo pidiera.
    """
    obs = s.dropna()
    if obs.empty:
        return "—"
    if tipo == "num":
        return statistik.mediana_ric(obs)
    if tipo == "bin":
        pos = str(cat_fila.get("positive_label") or "Sí")
        marcas = {pos, "True", "Sí", "1"}
        # En las binarias verdaderas se informa la frecuencia del nivel positivo, aunque
        # sea nula en ese estrato. En las categóricas de dos niveles sin nivel positivo
        # declarado (p. ej. ASA II/III) se informa el nivel modal, que es lo interpretable.
        if str(s.dtype) in {"bool", "boolean"} or (marcas & set(obs.astype(str))):
            k = int(obs.astype(str).isin(marcas).sum())
            return f"{k}/{len(obs)} ({100 * k / len(obs):.1f}%)".replace(".", ",")
    if tasa_exposicion:
        tasa = _tasa_exposicion(obs, cat_fila)
        if tasa is not None:
            return tasa
    modal = obs.astype(str).value_counts().idxmax()
    pct = 100 * (obs.astype(str) == modal).mean()
    return f"{modal} ({pct:.0f}%)"


def _faltantes(s: pd.Series) -> str:
    n = int(s.isna().sum())
    return f"{n} ({100 * n / len(s):.1f}%)".replace(".", ",")


def tabla_descriptiva(cohorte: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Características de la cohorte y asociación cruda con el desenlace.

    Las categóricas de más de dos niveles se despliegan en subfilas: el nombre de la
    variable aparece una vez y cada nivel ocupa su propia línea con su frecuencia.
    """
    y = cohorte[config.OUTCOME]
    filas = []
    # Las variables de texto libre quedan fuera: sus valores no son categorías
    # comparables y su tabulación no admite lectura descriptiva.
    sel = cat[cat["prelim_role"].isin(config.ROLES_EVAL)
              & ~cat["var_rename"].isin(config.VARS_TEXTO_LIBRE)].copy()
    sel["_o"] = sel["block"].map(ORDEN_BLOQUE).fillna(9)
    for _, r in sel.sort_values(["_o", "label"]).iterrows():
        v = r["var_rename"]
        if v not in cohorte.columns:
            continue
        s = cohorte[v]
        tipo = statistik.resolver_tipo(s, r["conceptual_type"])
        b = statistik.bivariado(s, y, r["conceptual_type"])
        base = {"bloque": r["block"], "Variable": etiqueta(r, v), "Nivel": "",
                "Faltantes": _faltantes(s), "n análisis": b["n"],
                "Prueba": b["prueba"], "p": b["p"], "Medida cruda": b["efecto"]}
        if tipo in ("nom", "ord") and s.dropna().nunique() > 2:
            filas.append({**base, "Global": ""})
            for niv in niveles(s, r):
                k = int((s.astype(str) == niv).sum())
                filas.append({"bloque": r["block"], "Variable": "", "Nivel": niv,
                              "Global": f"{k}/{s.notna().sum()} ({100 * k / s.notna().sum():.1f}%)".replace(".", ","),
                              "Faltantes": "", "n análisis": "", "Prueba": "", "p": None,
                              "Medida cruda": ""})
        else:
            filas.append({**base, "Global": _resumen_celda(s, tipo, r)})
    return pd.DataFrame(filas)


ETIQUETA_ESTRATO = {"protocolo": "Excluidas por criterios del protocolo",
                    "seguimiento": "Excluidas por seguimiento incompleto"}


def tabla_sesgo_seleccion(clasificada: pd.DataFrame, sesgo: pd.DataFrame,
                          cat: pd.DataFrame) -> pd.DataFrame:
    """Comparación de características basales entre incluidas y cada grupo de excluidas.

    Marca como frágil la SMD de las binarias con menos de cinco eventos en el grupo
    excluido: con recuentos tan bajos la estimación es inestable y su magnitud no admite
    lectura directa.
    """
    meta = cat.set_index("var_rename")
    incluidas = clasificada[clasificada["estrato"] == "incluida"]
    filas = []
    for estrato, g in sesgo.groupby("estrato", sort=False):
        exc = clasificada[clasificada["estrato"] == estrato]
        for _, r in g.iterrows():
            v = r["var_rename"]
            if v not in meta.index:
                continue
            fila_cat = meta.loc[v]
            tipo = statistik.resolver_tipo(clasificada[v], fila_cat["conceptual_type"])
            eventos_exc = (int(exc[v].astype(str).isin({"True", "Sí", "1"}).sum())
                           if tipo == "bin" else None)
            filas.append({
                "estrato": ETIQUETA_ESTRATO.get(estrato, estrato),
                "n_excluidas": len(exc),
                "Variable": etiqueta(fila_cat, v),
                f"Incluidas (n={len(incluidas)})": _resumen_celda(incluidas[v], tipo, fila_cat),
                "Excluidas": _resumen_celda(exc[v], tipo, fila_cat),
                "SMD": abs(r["smd"]) if pd.notna(r["smd"]) else np.nan,
                "Frágil": "Sí" if (eventos_exc is not None and eventos_exc < 5) else "",
                "Prueba": r["prueba"] or "—",
                "p": r["p"],
            })
    d = pd.DataFrame(filas)
    return d.sort_values(["estrato", "SMD"], ascending=[True, False]).reset_index(drop=True)


def _test_p(prueba, p) -> str:
    """Prueba global y su p en una celda, con el rótulo corto y coma decimal."""
    rotulo = {"χ²": "χ²", "Kruskal-Wallis": "K-W"}.get(prueba, prueba or "—")
    if pd.isna(p):
        texto = "—"
    else:
        texto = "< 0,001" if p < 0.001 else format(p, ".3f").replace(".", ",")
    return f"{rotulo} ({texto})"


BLOQUE_DESENLACE = "Desenlace"


def tabla_por_centro(cohorte: pd.DataFrame, cat: pd.DataFrame, het: pd.DataFrame) -> pd.DataFrame:
    """Características por centro y magnitud de la heterogeneidad entre hospitales.

    La primera fila es la frecuencia del desenlace por centro: es la referencia contra
    la que se lee la exposición de cada hospital, y sin ella la tabla obliga a buscarla
    en otra. Las categóricas de dosis informan su tasa de exposición, no su nivel modal
    (ver `_tasa_exposicion`), que es lo que permite ese contraste dentro de una sola
    tabla.
    """
    centro = cohorte[config.CENTER_VAR].astype(str)
    centros = sorted(centro.dropna().unique())
    n_centro = {c: int((centro == c).sum()) for c in centros}
    h = het.set_index("var_rename")

    # El desenlace no pertenece a ningún bloque temporal -es posterior a los tres- de
    # modo que abre su propia banda en vez de colarse en el postoperatorio, donde
    # quedaría al final y leído como una covariable más. La SMD queda en «—»: la de un
    # centro frente al resto mide heterogeneidad de covariables y no corresponde
    # aplicarla al desenlace.
    y = cohorte[config.OUTCOME]
    ry = cat.loc[cat["var_rename"] == config.OUTCOME].iloc[0]
    py = statistik.prueba_global_centro(y, cohorte[config.CENTER_VAR], "bin")
    fila_y = {"bloque": BLOQUE_DESENLACE, "Variable": etiqueta(ry, config.OUTCOME),
              "Faltantes": _faltantes(y)}
    for c in centros:
        fila_y[f"{c} (n={n_centro[c]})"] = _resumen_celda(y[centro == c], "bin", ry)
    fila_y["Test (p)"] = _test_p(py["prueba_global"], py["p_global"])
    fila_y["SMD máx. (centro)"] = "—"
    fila_y["Registro comparable"] = "Sí"
    fila_y["p"] = py["p_global"]

    filas = [fila_y]
    sel = cat[cat["include_heterogeneidad"].fillna(False)].copy()
    sel["_o"] = sel["block"].map(ORDEN_BLOQUE).fillna(9)
    for _, r in sel.sort_values(["_o", "label"]).iterrows():
        v = r["var_rename"]
        if v not in cohorte.columns or v not in h.index:
            continue
        s = cohorte[v]
        tipo = statistik.resolver_tipo(s, r["conceptual_type"])
        hr = h.loc[v]
        p = hr["p_global"]
        fila = {"bloque": r["block"], "Variable": etiqueta(r, v), "Faltantes": _faltantes(s)}
        for c in centros:
            fila[f"{c} (n={n_centro[c]})"] = _resumen_celda(s[centro == c], tipo, r,
                                                            tasa_exposicion=True)
        fila["Test (p)"] = _test_p(hr["prueba_global"], p)
        fila["SMD máx. (centro)"] = (f"{hr['smd_max']:.2f}".replace(".", ",") + f" ({hr['centro_smd_max']})"
                                     if pd.notna(hr["smd_max"]) else "—")
        fila["Registro comparable"] = "Sí" if hr.get("registro_comparable", True) else "No"
        fila["p"] = p
        filas.append(fila)
    return pd.DataFrame(filas)


MARCA_FRAGIL = "*"
MARCA_MINIMO = "**"
MARCA_SUPRIMIDO = "†"
COLUMNA_AGREGADA = "Cohorte"
ROTULO_DIFERENCIA = "Diferencia (pp)"


def niveles_de_especificacion(s: pd.Series) -> list[str]:
    """Niveles con que una variable entra a la especificación, en su orden clínico.

    Es la misma fuente que lee `modell.ClinicalOrdinalEncoder` al ajustarse -las categorías
    del dtype de la columna, que `daten` fijó desde el catálogo- y no una lista redefinida
    aquí. Si se redefiniera, la tabla podría mostrar una partición que el modelo nunca usó,
    que es justo lo que la tabla debe permitir contrastar. En las binarias son los dos
    valores que ve el codificador tras normalizar a No/Sí.
    """
    from . import modell

    if isinstance(s.dtype, pd.CategoricalDtype):
        return [str(x) for x in s.cat.categories]
    observados = set(modell._a_categoria(s).dropna().astype(str))
    return [n for n in ("No", "Sí") if n in observados] or sorted(observados)


def _texto_niveles(s: pd.Series) -> pd.Series:
    """La columna como texto de nivel, con la misma normalización de la especificación."""
    from . import modell

    if pd.api.types.is_bool_dtype(s):
        return modell._a_categoria(s).astype("string")
    return s.astype("string")


def _marca(n: int, n_fragil: int, n_minimo: int) -> str:
    """Marca de la celda: `**` si el nivel falta o es mínimo, `*` si es frágil."""
    if n < n_minimo:
        return MARCA_MINIMO
    return MARCA_FRAGIL if n <= n_fragil else ""


def _celda_n_pct(k: int, n: int, *, marca: str = "") -> str:
    """n/N (%) del desenlace en un estrato, con su marca si procede.

    Un nivel que el centro no observa se emite como `0/0` y no como raya: la raya es lo que
    marca una diferencia suprimida, y las dos cosas en la misma tabla se confundirían. `0/0`
    dice literalmente que no hubo pacientes en ese estrato.
    """
    cuerpo = "0/0" if not n else f"{k}/{n} ({100 * k / n:.1f}%)".replace(".", ",")
    return cuerpo + (f" {marca}" if marca else "")


def _celda_diferencia(ke: int, ne: int, kr: int, nr: int, *,
                      n_fragil: int, n_minimo: int) -> str:
    """Diferencia contraste menos referencia, en puntos porcentuales y con signo.

    Se suprime cuando cualquiera de los dos estratos queda bajo el mínimo: la celda de nivel
    sigue mostrándose, lo que desaparece es la cifra que la compararía con las demás.

    El negativo lleva el menos tipográfico (U+2212) y no el guion: esta tabla se lee por el
    signo, y un guion junto a un más se distingue mal a cuerpo pequeño. El preámbulo ya
    declara ese glifo con `\\newunicodechar`, de modo que no es un carácter sin cobertura.
    """
    if min(ne, nr) < n_minimo:
        return "—"
    d = 100 * ke / ne - 100 * kr / nr
    return (f"{d:+.1f}".replace(".", ",").replace("-", "−")
            + (f" {MARCA_FRAGIL}" if min(ne, nr) <= n_fragil else ""))


def desenlace_por_estrato_centro(cohorte: pd.DataFrame, bloques: list[dict],
                                 het: pd.DataFrame, *, n_fragil: int | None = None,
                                 n_minimo: int | None = None,
                                 n_suprimir: int | None = None) -> pd.DataFrame:
    """Frecuencia del desenlace **dentro** de cada nivel de exposición, por centro.

    Responde una pregunta distinta de `tabula.tabla_exposicion_desenlace_centro`, que pone
    lado a lado dos frecuencias marginales del centro, la de la exposición y la del
    desenlace. Aquí el desenlace se mide dentro de cada nivel, de modo que su gradiente se
    lea centro por centro y la diferencia se contraste con la de la cohorte completa: si el
    signo se conserva en los cuatro centros y se invierte al agregar, la asociación agregada
    es un artefacto de la composición y no una asociación del predictor.

    Se despliegan **todos** los niveles con que la variable entra a la especificación, no
    una dicotomía construida para la tabla: en una exposición graduada el gradiente y sus
    cambios de signo son el hallazgo, y dicotomizar los borra. Los niveles los da
    `niveles_de_especificacion`, no el bloque.

    El bloque declara solo qué niveles forman el contraste y cuáles la referencia de la
    **única** fila de diferencia de esa variable. Se comprueba que existan en el dato: un
    rótulo mal escrito -«> 7,5 mg» con coma, cuando el dato dice «> 7.5 mg»- vaciaría el
    estrato en silencio y la tabla saldría con ceros perfectamente creíbles. La diferencia
    va en puntos porcentuales y no como razón de ventajas: se verifica a ojo desde las filas
    de arriba y no se dispara con un solo caso.

    El orden de las variables es descendente por SMD máxima entre centros, tomada de `het`
    y no transcrita, de modo que la tabla exhiba la gradación de heterogeneidad.

    El denominador son los casos observados de la variable en ese centro, la convención de
    `tabla_por_centro`, no el total del centro que usa la tabla de morfina. Con faltantes
    las dos difieren, y esta es la que permite contrastar la tabla contra las tasas de
    exposición ya publicadas por centro.
    """
    n_fragil = config.ESTRATO_N_FRAGIL if n_fragil is None else n_fragil
    n_minimo = config.ESTRATO_N_MINIMO if n_minimo is None else n_minimo
    n_suprimir = config.ESTRATO_N_SUPRIMIR if n_suprimir is None else n_suprimir
    centro = cohorte[config.CENTER_VAR].astype(str)
    y = cohorte[config.OUTCOME]
    mascaras = {c: centro.eq(c) for c in sorted(centro.dropna().unique())}
    mascaras[COLUMNA_AGREGADA] = pd.Series(True, index=cohorte.index)
    smd = het.set_index("var_rename")["smd_max"]

    filas, marcados, faltantes, contrastes, suprimidos = [], [], {}, {}, []
    for b in sorted(bloques, key=lambda b: -float(smd.loc[b["var"]])):
        s = cohorte[b["var"]]
        texto, obs = _texto_niveles(s), s.notna()
        niveles = niveles_de_especificacion(s)
        declarados = set(b["contraste"]) | set(b["referencia"])
        if not declarados <= set(niveles):
            raise ValueError(
                f"{b['var']}: niveles declarados que la especificación no tiene: "
                f"{sorted(declarados - set(niveles))} · niveles: {niveles}")
        faltantes[b["predictor"]] = int((~obs).sum())
        contrastes[b["predictor"]] = (b["rotulo_contraste"], b["rotulo_referencia"])

        f_niv = {niv: {"predictor": b["predictor"], "estrato": niv} for niv in niveles}
        f_dif = {"predictor": b["predictor"], "estrato": ROTULO_DIFERENCIA}
        n_celda = {niv: {} for niv in niveles}
        for col, mc in mascaras.items():
            # Los niveles parten los casos observados: comprueba que la lista de niveles sea
            # la de la especificación y no un subconjunto con un resto invisible.
            cubiertos = 0
            for niv in niveles:
                m = texto.eq(niv) & obs & mc
                n, k = int(m.sum()), int(y[texto.eq(niv) & obs & mc].sum())
                cubiertos += n
                n_celda[niv][col] = n
                marca = _marca(n, n_fragil, n_minimo)
                if marca:
                    marcados.append({"predictor": b["predictor"], "nivel": niv,
                                     "columna": col, "n": n, "marca": marca})
                f_niv[niv][col] = _celda_n_pct(k, n, marca=marca)
            assert cubiertos == int((obs & mc).sum()), \
                f"{b['var']} en {col}: los niveles suman {cubiertos} y hay " \
                f"{int((obs & mc).sum())} observados"

            m_c = texto.isin(b["contraste"]) & obs & mc
            m_r = texto.isin(b["referencia"]) & obs & mc
            f_dif[col] = _celda_diferencia(
                int(y[m_c].sum()), int(m_c.sum()), int(y[m_r].sum()), int(m_r.sum()),
                n_fragil=n_fragil, n_minimo=n_minimo)
        # Control de divulgación. No se publica el contenido de un estrato de `n_suprimir`
        # pacientes o menos. La columna de la cohorte es la suma de los centros, de modo que
        # ocultar una sola celda la dejaría recuperable por resta: cuando en una fila queda
        # una sola oculta se oculta también la menor de las restantes.
        columnas_centro = [c for c in mascaras if c != COLUMNA_AGREGADA]
        for niv in niveles:
            ns = n_celda[niv]
            ocultas = {c for c in mascaras if 0 < ns[c] <= n_suprimir}
            if ocultas & set(columnas_centro) and COLUMNA_AGREGADA not in ocultas \
                    and len(ocultas & set(columnas_centro)) == 1:
                resto = [c for c in columnas_centro if c not in ocultas and ns[c] > 0]
                ocultas.add(min(resto, key=lambda c: ns[c]) if resto else COLUMNA_AGREGADA)
            for c in sorted(ocultas):
                suprimidos.append({"predictor": b["predictor"], "nivel": niv,
                                   "columna": c, "n": ns[c]})
                f_niv[niv][c] = MARCA_SUPRIMIDO
        filas += [f_niv[niv] for niv in niveles] + [f_dif]

    d = pd.DataFrame(filas)
    d.attrs.update(marcados=marcados, faltantes=faltantes, contrastes=contrastes,
                   n_fragil=n_fragil, n_minimo=n_minimo, n_suprimir=n_suprimir,
                   suprimidos=suprimidos, marca=MARCA_FRAGIL, marca_minimo=MARCA_MINIMO,
                   marca_suprimido=MARCA_SUPRIMIDO)
    return d


def matriz_heterogeneidad(cohorte: pd.DataFrame, cat: pd.DataFrame, bloque: str) -> pd.DataFrame:
    """Matriz |SMD| de variable por centro para un bloque temporal.

    Cada celda es la separación de ese centro frente al resto en esa variable. Alimenta el
    mapa de calor de heterogeneidad.
    """
    sel = cat[cat["include_heterogeneidad"].fillna(False) & cat["block"].eq(bloque)]
    centros = sorted(cohorte[config.CENTER_VAR].dropna().astype(str).unique())
    filas = []
    for _, r in sel.iterrows():
        v = r["var_rename"]
        if v not in cohorte.columns or v == config.CENTER_VAR:
            continue
        fila = {"variable": etiqueta(r, v)}
        for c in centros:
            fila[c] = abs(statistik.smd(cohorte[v], cohorte[config.CENTER_VAR].astype(str).eq(c)))
        filas.append(fila)
    d = pd.DataFrame(filas)
    if d.empty:
        return d
    return d.set_index("variable").reindex(
        d.set_index("variable").max(axis=1).sort_values(ascending=False).index)
