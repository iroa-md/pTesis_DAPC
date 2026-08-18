"""rendern.py — figuras del manuscrito.

Dibuja los agregados que produce `vista`; no calcula nada ni accede al parquet. Toda figura
se exporta en PNG y SVG con el estilo declarado en `config`.
"""
from __future__ import annotations

import re
import textwrap

import matplotlib

# Backend sin pantalla: los módulos deben poder exportar figuras en un entorno headless, donde
# el backend por defecto no siempre lo es. Un cuaderno que quiera verlas en línea lo revierte
# con `%matplotlib inline` después de importar este módulo.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config


def _fuente() -> str:
    """Primera tipografía disponible de las declaradas en `config`."""
    from matplotlib import font_manager
    disponibles = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in config.TIPOGRAFIA if f in disponibles), "DejaVu Sans")


# Ancho de la caja de texto del manuscrito, en pulgadas: carta menos márgenes de 2,5 cm.
# Toda figura se reduce por el cociente entre este ancho y el suyo, de modo que su texto
# llega a la página multiplicado por ese factor.
ANCHO_CAJA = 6.53


def cuerpo_compensado(ancho: float, objetivo: float = 10.0) -> float:
    """Tamaño de fuente que, tras el escalado a la caja, se lee al tamaño buscado.

    Fijar la fuente en un valor único hace que una figura ancha llegue a la página con el
    texto reducido a la mitad: la que mide 9,7 pulgadas se imprime al 0,67 y sus rótulos de
    10 pt quedan en 6,7. Aquí la fuente se define en función del ancho, de modo que el
    tamaño sobre el papel sea el mismo en todas.
    """
    return float(min(22.0, max(9.0, objetivo * ancho / ANCHO_CAJA)))


def estilo(cuerpo: float = 10.0) -> None:
    """Estilo de casa según la regla de proporción 60-30-10.

    Predomina el espacio limpio; la estructura (texto, ejes) usa el azul profundo y la
    cuadrícula el azul hielo atenuado; el acento se reserva para el dato que se destaca.

    `cuerpo` permite compensar el escalado de la figura, ver `cuerpo_compensado`.
    """
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": config.FIG_DPI,
        "font.family": _fuente(), "font.size": cuerpo,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": config.COLOR_MEDIO, "axes.linewidth": 1.0,
        "axes.labelcolor": config.COLOR_TEXTO, "axes.titlecolor": config.COLOR_TEXTO,
        "text.color": config.COLOR_TEXTO,
        "xtick.color": config.COLOR_TEXTO, "ytick.color": config.COLOR_TEXTO,
        "axes.grid": True, "grid.color": config.COLOR_MEDIO,
        "grid.alpha": 0.35, "grid.linewidth": 0.7,
        "legend.frameon": False,
        "axes.titlesize": "large", "axes.labelsize": "medium",
        "xtick.labelsize": "small", "ytick.labelsize": "small",
        "legend.fontsize": "small", "figure.titlesize": "large",
    })


def cmap_secuencial():
    """Escala continua de la marca: del blanco al azul profundo."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "bal", ["#FFFFFF", config.COLOR_FONDO, config.COLOR_MEDIO, config.COLOR_TEXTO])


def _texto_sobre(color: str) -> str:
    """Blanco o azul profundo, según el contraste que ofrezca el fondo."""
    r, g, b = (int(color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return "white" if 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b) < 0.35 else config.COLOR_TEXTO


def guardar(fig, stem: str, carpeta=None, *, dpi: int | None = None) -> None:
    """Exporta la figura en PNG y SVG.

    `dpi` sube la resolución del PNG sin tocar el tamaño en pulgadas, de modo que la
    tipografía impresa no cambie: son ejes independientes. Sirve para las figuras de trazo
    fino, donde 300 ppp deja escalonada una curva que el SVG sí resuelve.
    """
    carpeta = carpeta or config.FIGURAS
    carpeta.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(carpeta / f"{stem}.{ext}", dpi=dpi or config.FIG_DPI,
                    bbox_inches="tight", facecolor="white")
    print(f"✔ {stem}.png · {stem}.svg")


def _coma(x: float, dec: int = 1) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def eje_coma(ax, ejes: str = "xy", dec: int = 2) -> None:
    """Separador decimal de coma en las marcas de los ejes.

    El manuscrito usa coma en todo el texto y en las tablas; dejar punto en las figuras
    rompe esa unidad dentro de una misma página.
    """
    f = plt.FuncFormatter(lambda v, _: _coma(v, dec))
    if "x" in ejes:
        ax.xaxis.set_major_formatter(f)
    if "y" in ejes:
        ax.yaxis.set_major_formatter(f)


def _realinear_titulo(ax) -> None:
    """Corre el título -ya fijado con `loc="left"`- hasta el borde real de las etiquetas del
    eje Y, en vez del borde de los ejes.

    `loc="left"` alinea con el borde de los propios ejes, que con etiquetas largas a la
    izquierda (nombres de variable, de hospital) queda bien a la derecha del contenido
    visible. Sumado a que `guardar` exporta con `bbox_inches="tight"` -que recorta al
    contenido ya renderizado, etiquetas incluidas-, el título termina pareciendo flotar hacia
    la derecha en vez de encabezar la figura. Se mide la extensión ya renderizada, con la
    misma técnica que ya usa la leyenda de `consistencia`, y se llama después de
    `fig.tight_layout()`, cuando la posición de los ejes ya es la definitiva.
    """
    etiquetas = ax.get_yticklabels()
    if not ax.title.get_text() or not etiquetas:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    borde = min(lbl.get_window_extent(renderer).x0 for lbl in etiquetas)
    ax.title.set_x(ax.transAxes.inverted().transform((borde, 0))[0])


# ─────────────────────────────────────────────────────────────────────────────

def flujo_participantes(etapas: pd.DataFrame, n_inicial: int, n_espinal: int | None = None):
    """Diagrama de flujo de la selección de participantes.

    Recibe la cascada ya agrupada por vía de exclusión (`vista.flujo_participantes`): una
    caja por etapa, con el desglose de criterios al costado.
    """
    estilo()
    paso = 2.0
    alto = paso * (len(etapas) + 0.9)
    fig, ax = plt.subplots(figsize=(9.6, 1.25 * (len(etapas) + 1) + 1.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, alto); ax.axis("off")

    def caja(x, y, ancho, rotulo, n, principal=True):
        ax.add_patch(FancyBboxPatch((x, y - 0.34), ancho, 0.76,
                                    boxstyle="round,pad=0.02,rounding_size=0.14",
                                    fc="white" if principal else config.COLOR_FONDO,
                                    ec=config.COLOR_TEXTO if principal else config.COLOR_MEDIO,
                                    lw=1.4 if principal else 1.2))
        ax.text(x + ancho / 2, y + 0.04, rotulo, ha="center", fontsize="large")
        ax.text(x + ancho / 2, y - 0.22, f"(n = {n})", ha="center", fontsize="large", fontweight="bold")

    y = alto - 0.7
    caja(0.7, y, 4.0, "Cohorte FONIS", n_inicial)

    for _, r in etapas.iterrows():
        y_sig = y - paso
        ax.add_patch(FancyArrowPatch((2.7, y - 0.36), (2.7, y_sig + 0.44), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.4, color=config.COLOR_TEXTO))
        y_ram = y - paso / 2
        ax.add_patch(FancyArrowPatch((2.7, y_ram), (5.0, y_ram), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.2, ls=(0, (4, 3)), color=config.GRIS_TENUE))
        ax.text(5.2, y_ram + 0.16, f"{r['motivo']}  (n = {int(r['n_excluidas'])})",
                va="center", fontsize=9.5, color=config.COLOR_TEXTO)
        if r["detalle"]:
            ax.text(5.2, y_ram - 0.18, r["detalle"], va="center", fontsize=8.4, color=config.GRIS_TENUE)
        caja(0.7, y_sig, 4.0, r["etapa"], int(r["n_restante"]))
        y = y_sig

    if n_espinal is not None:
        ax.add_patch(FancyArrowPatch((4.8, y), (6.0, y), arrowstyle="-|>", mutation_scale=12,
                                     lw=1.2, ls=(0, (4, 3)), color=config.GRIS_TENUE))
        caja(6.1, y, 3.4, "Subcohorte espinal", n_espinal, principal=False)
    fig.tight_layout()
    return fig


def _barras_apiladas_severidad(d: pd.DataFrame, col: str, leyenda: str):
    """Cuerpo compartido de las figuras de severidad por momento (barras al 100 %)."""
    piv = d.pivot(index="momento", columns=col, values="pct")
    colores = [config.COLOR_SEVERIDAD.get(str(c), config.COLOR_MEDIO) for c in piv.columns]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    base = np.zeros(len(piv))
    for c, color in zip(piv.columns, colores):
        v = piv[c].to_numpy(float)
        ax.bar(piv.index.astype(str), v, bottom=base, color=color, label=str(c),
               edgecolor="white", linewidth=0.8)
        for x, (b, h) in enumerate(zip(base, v)):
            if h >= 4:
                ax.text(x, b + h / 2, f"{_coma(h)} %", ha="center", va="center", fontsize="small",
                        color=_texto_sobre(color))
        base += v
    ax.set_ylabel("% de la cohorte"); ax.set_xlabel("Momento de medición")
    ax.set_ylim(0, 100); ax.xaxis.grid(False)
    ax.legend(title=leyenda, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def distribucion_desenlace(d: pd.DataFrame):
    """Severidad del dolor por momento de medición (barras apiladas al 100 %)."""
    estilo()
    return _barras_apiladas_severidad(d, "severidad", "Intensidad")


def frecuencia_severidad_tiempo(d: pd.DataFrame):
    """Severidad del dolor por momento de medición, en las cuatro categorías clínicas.

    Mismo cuerpo que `distribucion_desenlace`, sobre `vista.frecuencia_severidad_tiempo`
    en vez de `vista.distribucion_desenlace`: separa «sin dolor» de «leve», que ahí van
    juntas porque es lo que da sentido al umbral del desenlace.
    """
    estilo()
    return _barras_apiladas_severidad(d, "categoria", "Intensidad")


def _panel_proporcion_centro(ax, sub: pd.DataFrame, *, p=None, prefijo: str = ""):
    """Cuerpo compartido de un panel de proporción por centro con IC 95 % y referencia global.

    Sirve tanto a `incidencia_por_centro` (el desenlace principal) como a
    `frecuencia_por_centro_tiempo` (frecuencia de dolor por momento): el panel no sabe ni
    le importa cuál de las dos proporciones dibuja.
    """
    centros = sub[sub["centro"] != "GLOBAL"].reset_index(drop=True)
    glob = sub[sub["centro"] == "GLOBAL"].iloc[0]
    x = np.arange(len(centros))
    inc = 100 * centros["incidencia"].to_numpy(float)
    lo = inc - 100 * centros["ic_low"].to_numpy(float)
    hi = 100 * centros["ic_high"].to_numpy(float) - inc

    ax.bar(x, inc, color=[config.PALETA_CENTROS.get(c, config.COLOR_TEXTO) for c in centros["centro"]],
           width=0.62)
    ax.errorbar(x, inc, yerr=[lo, hi], fmt="none", ecolor=config.COLOR_TEXTO, capsize=4, lw=1.2)
    ax.axhline(100 * glob["incidencia"], ls="--", lw=1.2, color=config.GRIS_TENUE)
    ax.text(len(centros) - 0.4, 100 * glob["incidencia"] + 2,
            f"Global {_coma(100 * glob['incidencia'])} %", fontsize="small", color=config.GRIS_TENUE, ha="right")
    for xi, (v, tope) in enumerate(zip(inc, 100 * centros["ic_high"].to_numpy(float))):
        ax.text(xi, tope + 2.5, f"{_coma(v)} %", ha="center", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(centros["centro"])
    ax.set_ylabel("%"); ax.set_ylim(0, 100); ax.xaxis.grid(False)
    if p is not None:
        texto_p = f"χ² global: p {'< 0,001' if p < 0.001 else '= ' + _coma(p, 3)}"
        ax.set_title(f"{prefijo}{texto_p}", loc="left", fontsize=10, color=config.COLOR_TEXTO)


def incidencia_por_centro(d: pd.DataFrame):
    """Incidencia del desenlace por centro con IC 95 % y referencia global."""
    estilo()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    _panel_proporcion_centro(ax, d, p=d.attrs.get("p_global"))
    ax.set_xlabel("Centro")
    fig.tight_layout()
    return fig


def frecuencia_por_centro_tiempo(d: pd.DataFrame):
    """Frecuencia de dolor (END > 0) por centro y momento de medición, barras agrupadas.

    Un eje único, con cada centro agrupando sus tres momentos: comparar la trayectoria
    temporal de un centro y comparar entre centros se leen en el mismo golpe de vista.
    Reemplaza un diseño anterior de un panel por momento apilado en columna, que obligaba
    a escanear verticalmente para lo primero. Global entra como una categoría más del eje
    X, no como línea de referencia: con tres series por centro, una línea horizontal
    cruzando los grupos se perdía. No se llama «incidencia»: ese término se reserva para
    el desenlace principal (`dmg_1s_bin`), que es lo que grafica `incidencia_por_centro`.
    """
    estilo()
    momentos = list(dict.fromkeys(d["momento"]))
    # Progresión clara → oscura para el orden temporal, la misma regla que ya usa
    # COLOR_BLOQUE para preop/intraop/postop.
    colores_momento = [config.COLOR_FONDO, config.COLOR_MEDIO, config.COLOR_TEXTO][:len(momentos)]
    centros = [c for c in config.PALETA_CENTROS if c in d["centro"].unique()]
    d = d.copy()
    d["centro"] = d["centro"].replace({"GLOBAL": "Global"})
    orden = [*centros, "Global"]

    x = np.arange(len(orden))
    ancho = 0.8 / len(momentos)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, momento in enumerate(momentos):
        sub = d[d["momento"].eq(momento)].set_index("centro").reindex(orden)
        inc = 100 * sub["incidencia"].to_numpy(float)
        lo = inc - 100 * sub["ic_low"].to_numpy(float)
        hi = 100 * sub["ic_high"].to_numpy(float) - inc
        pos = x - 0.4 + ancho * (i + 0.5)
        ax.bar(pos, inc, width=ancho * 0.92, color=colores_momento[i], label=momento,
               edgecolor="white", linewidth=0.6)
        ax.errorbar(pos, inc, yerr=[lo, hi], fmt="none", ecolor=config.COLOR_TEXTO,
                    capsize=2.5, lw=1.0)
    ax.axvline(len(centros) - 0.5, ls=":", lw=1.0, color=config.GRIS_TENUE)
    ax.set_xticks(x); ax.set_xticklabels(orden)
    ax.set_ylabel("%"); ax.set_xlabel("Centro"); ax.set_ylim(0, 100); ax.xaxis.grid(False)
    ax.legend(title="Momento", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def heatmap_heterogeneidad(m: pd.DataFrame, titulo: str = "", vmax: float | None = None):
    """Mapa de calor de |SMD| por variable y centro."""
    estilo()
    vmax = vmax or float(np.nanmax(m.to_numpy(float)))
    ancho = 1.05 * m.shape[1] + 3.4
    estilo(cuerpo_compensado(ancho))
    fig, ax = plt.subplots(figsize=(ancho, 0.40 * len(m) + 1.6))
    im = ax.imshow(m.to_numpy(float), aspect="auto", cmap=cmap_secuencial(), vmin=0, vmax=vmax)
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns)
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index, fontsize="small")
    ax.grid(False)
    for i in range(len(m)):
        for j in range(m.shape[1]):
            v = m.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, _coma(v, 2), ha="center", va="center", fontsize="small",
                        color="white" if v > 0.74 * vmax else config.COLOR_TEXTO)
    if titulo:
        ax.set_title(titulo, loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.75, label="|SMD| centro vs resto")
    fig.tight_layout()
    return fig


_REF_BINARIA = ": Sí (ref: No)"


def etiqueta_corta(s: str) -> str:
    """Etiqueta de término sin el nivel de referencia y con los signos del manuscrito.

    El sufijo de una binaria -«: Sí (ref: No)»- es constante en todas ellas y se declara
    una vez en la leyenda, no dieciséis veces en el eje. En las categóricas de varios
    niveles se conserva el nivel, que sí distingue la fila, y se retira su referencia, que
    es común a todas las filas de esa variable.

    Se normaliza además «>=» a «≥», que es como el manuscrito escribe el umbral. El origen
    de esa cadena es la columna `label` del catálogo, de modo que la corrección de fondo va
    ahí; aquí se cubre la presentación.
    """
    s = str(s).replace(">=", "≥ ").replace("≥  ", "≥ ")
    if s.endswith(_REF_BINARIA):
        return s[:-len(_REF_BINARIA)]
    return re.sub(r"\s*\(ref:[^)]*\)\s*$", "", s)


def forest(t: pd.DataFrame, titulo: str = "", *, log: bool = True, medida: str = "OR",
           altura_fila: float = 0.34, valores: bool = True, abreviar: bool = True,
           etiquetas: dict | None = None, escala_valores: float = 0.8):
    """Estimaciones de un modelo multivariable con su intervalo de confianza.

    Cada fila es un término del modelo. Se destacan en acento las estimaciones cuyo
    intervalo no cruza el valor nulo, y el resto se dibuja en azul pizarra: la figura
    muestra el modelo completo, no solo lo que alcanza significación. La escala es
    logarítmica cuando la medida es multiplicativa, de modo que un efecto y su recíproco
    queden a la misma distancia del valor nulo.

    La columna de la derecha repite la estimación en cifras, que es la convención del
    diagrama de bosque y lo que permite leerlo sin ir al anexo. Ocupa el blanco que de otro
    modo queda a la derecha del dato más extremo, y va a `escala_valores` del cuerpo de los
    rótulos: es información de apoyo y compite por el ancho con el área de datos, de modo
    que igualarla al rótulo la vuelve el elemento más pesado de la figura.

    El lienzo se declara del ancho de la caja de texto y la tipografía se deriva de él con
    `cuerpo_compensado`, de modo que el escalado a la página sea la identidad. Un título
    largo rompía justamente eso: `guardar` recorta al contenido, de modo que el título
    ensanchaba el lienzo y **encogía todo lo demás** al ajustarse a la caja. Por eso el
    título se envuelve y se centra sobre la figura completa, y por eso lo habitual es no
    pasarlo: la leyenda del manuscrito ya lo dice.
    """
    estilo()
    d = t[t["termino"].ne("Intercept") & ~t.get("umbral", False)].copy()
    d = d.iloc[::-1]                                   # primero declarado, arriba
    nulo = 1.0 if medida.upper() == "OR" else 0.0
    sig = (d["ic_low"] > nulo) & (d["ic_high"] > nulo) | (d["ic_low"] < nulo) & (d["ic_high"] < nulo)
    color = np.where(sig, config.COLOR_ACENTO, config.COLOR_FRIO)
    efecto = d["efecto"] if "efecto" in d else d[medida]

    # El cuerpo se fija explícito y no por los tamaños relativos del estilo: «small» es
    # 8,3 pt sobre una base de 10 y en el papel quedaría bajo el objetivo declarado.
    cuerpo = cuerpo_compensado(ANCHO_CAJA)
    estilo(cuerpo)
    fig, ax = plt.subplots(figsize=(ANCHO_CAJA, max(3.4, 1.22 * altura_fila * len(d))))
    y = np.arange(len(d))
    ax.hlines(y, d["ic_low"], d["ic_high"], color=color, linewidth=1.8, zorder=2)
    ax.scatter(efecto, y, s=34, color=color, zorder=3, edgecolor="white", linewidth=0.8)
    ax.axvline(nulo, color=config.GRIS_TENUE, linestyle="--", linewidth=1.0, zorder=1)

    if log:
        ax.set_xscale("log")
        # Marcas explícitas: con un rango estrecho el localizador logarítmico por defecto
        # deja el eje casi sin etiquetar.
        escala = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]   # escala duplicativa, simétrica en log
        lo, hi = d["ic_low"].min(), d["ic_high"].max()
        marcas = [v for v in escala if lo / 1.6 <= v <= hi * 1.6]
        ax.set_xticks(marcas)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: _coma(v, 2 if v < 1 else 1)))
        ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(y)
    # `etiquetas` sustituye el rótulo de un término por uno más breve. El ancho que ocupa
    # la columna de rótulos lo fija su cadena más larga, y con nombres clínicos completos a
    # cuerpo legible esa columna se come el área de datos: acortarlos es la única palanca,
    # porque bajar la tipografía contradice el objetivo de 10 pt. Qué abreviatura es
    # aceptable es decisión editorial, y por eso entra por parámetro y no vive aquí.
    etiquetas = etiquetas or {}
    rotulos = [etiquetas.get(term, etiqueta_corta(l) if abreviar else l)
               for term, l in zip(d["termino"], d["label"])]
    ax.set_yticklabels(rotulos, fontsize=cuerpo)
    ax.tick_params(axis="x", labelsize=cuerpo)
    ax.set_xlabel(f"{medida} ajustado (IC95%)", fontsize=cuerpo)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.grid(axis="y", visible=False)

    if valores:
        # La columna vive fuera del área de datos, anclada al borde derecho de los ejes:
        # `x` en fracción de eje y `y` en coordenadas de dato, para que siga a su fila.
        cuerpo_val = cuerpo * escala_valores
        for yi, e, lo_, hi_ in zip(y, efecto, d["ic_low"], d["ic_high"]):
            ax.annotate(f"{_coma(e, 2)} [{_coma(lo_, 2)}–{_coma(hi_, 2)}]",
                        xy=(1.02, yi), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=cuerpo_val, color=config.COLOR_TEXTO,
                        annotation_clip=False)
        ax.annotate(f"{medida} [IC95%]", xy=(1.02, 1.0),
                    xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                    va="bottom", ha="left", fontsize=cuerpo_val, weight="bold",
                    color=config.COLOR_TEXTO, annotation_clip=False)

    if titulo:
        # Envuelto al ancho del lienzo y centrado sobre la figura completa, no sobre los
        # ejes: de lo contrario vuelve a arrastrar el borde y a encoger la figura.
        import textwrap
        fig.suptitle("\n".join(textwrap.wrap(titulo, 62)), fontsize="large", x=0.5, ha="center")
    fig.tight_layout()
    return fig


def forest_comparado(izquierda: pd.DataFrame, derecha: pd.DataFrame, titulo: str = "", *,
                     nombres: tuple = ("A", "B"), medida: str = "OR",
                     abreviar: bool = True, etiquetas: dict | None = None):
    """Estimaciones de una misma configuración de modelo en dos cohortes, término a término.

    Variante de `forest` con dos series por término en vez de una. La cohorte se codifica dos
    veces -color y forma del marcador-, porque en varios términos las dos cohortes producen
    casi la misma estimación (p. ej. edad, IMC) y ahí el color solo no basta: los puntos caen
    tan cerca que una diferencia de tono se pierde, en especial si además no alcanzan
    significación y quedan huecos. La forma se mantiene distinguible aunque los puntos casi
    se superpongan. El relleno marca si esa serie alcanza significación.

    Comparte con `forest` el lienzo del ancho de la caja de texto, la tipografía derivada de
    él y el acortamiento de rótulos, por la misma razón: `guardar` recorta al contenido, de
    modo que un título largo o un rótulo largo ensanchan el lienzo y encogen la figura al
    ajustarse a la página. No lleva columna de cifras: con dos series por término serían dos
    columnas, y el ancho que piden deja el área de datos sin sitio.
    """
    estilo()
    nulo = 1.0 if medida.upper() == "OR" else 0.0
    a = izquierda[izquierda["termino"].ne("Intercept") & ~izquierda.get("umbral", False)].copy()
    b = derecha[derecha["termino"].ne("Intercept") & ~derecha.get("umbral", False)].copy()
    label = dict(zip(a["termino"], a["label"]))
    for t, l in zip(b["termino"], b["label"]):
        label.setdefault(t, l)
    orden = a["termino"].tolist()[::-1]                 # primero declarado, arriba
    ia, ib = a.set_index("termino"), b.set_index("termino")
    y = {t: i for i, t in enumerate(orden)}

    # Misma fórmula de altura que `forest`, con una fila algo más alta para que las dos
    # series dodged no se encimen.
    cuerpo = cuerpo_compensado(ANCHO_CAJA)
    estilo(cuerpo)
    fig, ax = plt.subplots(figsize=(ANCHO_CAJA, max(3.4, 1.22 * 0.40 * len(orden))))
    off = 0.17
    colores = {nombres[0]: config.COLOR_TEXTO, nombres[1]: config.COLOR_FRIO}
    marcadores = {nombres[0]: "o", nombres[1]: "^"}
    for nombre, d, dy in ((nombres[0], ia, -off), (nombres[1], ib, off)):
        d = d.reindex([t for t in orden if t in d.index])
        ys = np.array([y[t] for t in d.index]) + dy
        c, m = colores[nombre], marcadores[nombre]
        e = d[medida] if medida in d else d["efecto"]
        sig = ((d["ic_low"] > nulo) & (d["ic_high"] > nulo)) | \
              ((d["ic_low"] < nulo) & (d["ic_high"] < nulo))
        ax.hlines(ys, d["ic_low"], d["ic_high"], color=c, linewidth=1.6, zorder=2)
        ax.scatter(e, ys, s=np.where(sig, 44, 30), marker=m, zorder=3, linewidth=1.1,
                  edgecolor=c, facecolor=np.where(sig, c, "white"), label=nombre)
    ax.axvline(nulo, color=config.GRIS_TENUE, linestyle="--", linewidth=1.0, zorder=1)
    if medida.upper() == "OR":
        ax.set_xscale("log")
        escala = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
        lo = min(ia["ic_low"].min(), ib["ic_low"].min())
        hi = max(ia["ic_high"].max(), ib["ic_high"].max())
        marcas = [v for v in escala if lo / 1.6 <= v <= hi * 1.6]
        ax.set_xticks(marcas)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _coma(v, 2 if v < 1 else 1)))
        ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(list(y.values()))
    etiquetas = etiquetas or {}
    ax.set_yticklabels([etiquetas.get(t, etiqueta_corta(label.get(t, t)) if abreviar
                                      else label.get(t, t)) for t in orden], fontsize=cuerpo)
    ax.tick_params(axis="x", labelsize=cuerpo)
    ax.set_xlabel(f"{medida} ajustado (IC95%)", fontsize=cuerpo)
    ax.set_ylim(-0.7, len(orden) - 0.3)
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker=marcadores[n], ls="", color=colores[n], label=n)
                       for n in nombres],
             loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=cuerpo,
             frameon=False)
    if titulo:
        import textwrap
        fig.suptitle("\n".join(textwrap.wrap(titulo, 62)), fontsize="large", x=0.5, ha="center")
    fig.tight_layout()
    return fig


def desplazamiento_terminos(d: pd.DataFrame, titulo: str = "", *, umbral: float | None = None):
    """Cambio porcentual máximo de cada término entre los peldaños de una escalera y su base.

    Una barra por término, con el mayor |Δ%| observado en cualquiera de los peldaños
    contrastados (`modell.cambio_en_estimacion`, ya agregado por quien llama: esta función
    solo dibuja). El color distingue los coeficientes del centro del resto de los términos:
    cuánto se mueven unos frente a otros es lo que la figura hace evidente de un vistazo, en
    vez de obligar al lector a creerlo en una frase.

    `d` trae una fila por término con columnas `label`, `max_abs_pct` y `centro` (booleana).
    """
    estilo()
    d = d.sort_values("max_abs_pct", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    color = np.where(d["centro"], config.COLOR_ACENTO, config.COLOR_FRIO)

    fig, ax = plt.subplots(figsize=(6.8, 0.4 * len(d) + 1.4))
    ax.barh(y, d["max_abs_pct"], color=color, height=0.62, zorder=3)
    for yi, v in zip(y, d["max_abs_pct"]):
        ax.annotate(f"{_coma(v, 1)} %", (v, yi), xytext=(4, 0), textcoords="offset points",
                   va="center", fontsize="small", color=config.COLOR_TEXTO)
    if umbral:
        ax.axvline(umbral, color=config.GRIS_TENUE, linestyle="--", linewidth=1.0, zorder=1)
        ax.annotate(f"Umbral {_coma(umbral, 0)} %", (umbral, len(d) - 0.3), xytext=(4, 4),
                   textcoords="offset points", fontsize="small", color=config.GRIS_TENUE, ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize="small")
    ax.set_xlabel("|Δ%| máximo frente al modelo base (M0)")
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlim(0, d["max_abs_pct"].max() * 1.18)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=config.COLOR_ACENTO, label="Coeficiente del centro"),
                       Patch(color=config.COLOR_FRIO, label="Predictor clínico")],
             loc="lower right", fontsize="small", frameon=False)
    if titulo:
        ax.set_title(titulo, fontsize="large", loc="left", pad=12)
    fig.tight_layout()
    _realinear_titulo(ax)
    return fig


def transportabilidad_por_centro(iecv: pd.DataFrame, titulo: str = ""):
    """Discriminación en el centro retenido, resumida por centro sobre todos los algoritmos.

    Una fila por centro: el punto es el promedio de los algoritmos evaluados en él y la barra
    fina recorre del mínimo al máximo. **La barra no es un intervalo de confianza**, es el
    rango entre algoritmos, y está para mostrar que ninguno se salva: si el peor y el mejor
    caen igual de cerca del azar, el problema no es la elección del modelo.

    No lleva color ni leyenda por algoritmo. La pregunta de la figura no es cuál algoritmo
    transporta mejor sino cuánta de la variación es institucional, y una leyenda de siete
    entradas invita a leer lo contrario.

    La única referencia es el azar. El valor y el rango de cada centro se anotan al costado,
    de modo que la figura no dependa de leer posiciones sobre el eje.
    """
    estilo()
    centros = [c for c in config.PALETA_CENTROS if c in iecv["centro"].unique()]
    r = (iecv[iecv["centro"].isin(centros)].groupby("centro")["auc"]
         .agg(["mean", "min", "max"]).reindex(centros))
    y = np.arange(len(centros))

    cuerpo = cuerpo_compensado(7.0)
    estilo(cuerpo)
    fig, ax = plt.subplots(figsize=(7.0, 0.62 * len(centros) + 2.0))

    # Guía desde el rótulo hasta el dato. Con cuatro filas repartidas en el alto del lienzo
    # la separación es amplia y el ojo pierde la correspondencia entre el nombre del centro
    # y su barra; punteada y tenue para no competir con el dato. Arranca del límite que se
    # fija más abajo, no del que el eje tenga en este punto, que aún es el de un eje vacío.
    izq = min(0.46, float(r["min"].min()) - 0.015)
    ax.hlines(y, izq, r["min"], color=config.GRIS_TENUE, linewidth=0.8,
              linestyle=(0, (1, 3)), alpha=0.5, zorder=1)
    ax.hlines(y, r["min"], r["max"], color=config.COLOR_MEDIO, linewidth=4.2, zorder=2)
    ax.scatter(r["mean"], y, s=86, color=config.COLOR_TEXTO, zorder=3,
              edgecolor="white", linewidth=1.2)

    # Las cifras en columna, alineadas entre sí y fuera del área de datos. Ancladas al
    # máximo de cada fila quedaban a distinta altura de sangría, de modo que no se leían
    # como columna sino como cuatro anotaciones sueltas.
    for yi, (_, f) in zip(y, r.iterrows()):
        ax.annotate(f"{_coma(f['mean'], 3)}  [{_coma(f['min'], 3)} – {_coma(f['max'], 3)}]",
                   xy=(1.03, yi), xycoords=("axes fraction", "data"),
                   va="center", ha="left", fontsize=cuerpo * 0.85,
                   color=config.GRIS_TENUE, annotation_clip=False)
    ax.annotate("Promedio  [mín. – máx.]", xy=(1.03, 1.0), xycoords="axes fraction",
                xytext=(0, 8), textcoords="offset points", ha="left", va="bottom",
                fontsize=cuerpo * 0.72, color=config.GRIS_TENUE, annotation_clip=False)

    ax.axvline(0.5, color=config.COLOR_TEXTO, linewidth=1.3, zorder=1)
    # El rótulo del azar, arriba de su línea: abajo quedaba pegado a las marcas del eje.
    ax.annotate("Azar", xy=(0.5, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, 4), textcoords="offset points", fontsize=cuerpo * 0.85,
                color=config.COLOR_TEXTO, ha="left", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(centros, fontsize=cuerpo)
    ax.tick_params(axis="x", labelsize=cuerpo * 0.9)
    ax.set_xlabel("AUC-ROC en el centro retenido", fontsize=cuerpo)
    # El eje se ajusta al rango observado: la columna de cifras ya no vive dentro.
    ax.set_xlim(izq, r["max"].max() + 0.012)
    # Marcas cada cinco centésimas: con el eje ajustado al rango, el localizador automático
    # elegía pasos de 0,025 y producía una escala que no se lee de corrido.
    ax.set_xticks([t / 100 for t in range(40, 101, 5)
                   if izq <= t / 100 <= float(r["max"].max()) + 0.012])
    ax.set_ylim(-0.6, len(centros) - 0.4)
    ax.invert_yaxis()
    eje_coma(ax, "x", 2)
    ax.grid(axis="y", visible=False)
    # Sin título: lo enuncia la leyenda del manuscrito. `titulo` se conserva y se ignora.
    fig.tight_layout()
    return fig


def valor_anadido_por_escenario(pareada: pd.DataFrame, titulo: str = "", *,
                                escenarios: tuple = ("sin_centro", "con_centro"),
                                etiquetas: tuple = ("Sin el centro", "Con el centro"),
                                nombres: dict | None = None):
    """Valor añadido de la información perioperatoria, con y sin la variable centro.

    Una fila por algoritmo y dos puntos unidos por una línea: el mismo contraste pareado
    -perioperatoria frente a preoperatoria- estimado en los dos escenarios. Lo que la figura
    hace ver es **cuánto se acorta cada línea**, no si alguna deja de alcanzar significación:
    el colapso ocurre en los siete y una tabla de valores p lo diría de siete formas
    distintas en vez de una sola.

    Espera el marco de `evaluar.comparacion_pareada` con ambos escenarios, cuya clave viaja
    como «cohorte · escenario · especificación · algoritmo».
    """
    estilo()
    nombres = nombres or config.NOMBRE_ALGORITMO
    d = pareada.copy()
    partes = d["referencia"].str.split("·")
    d["_esc"] = partes.str[1].str.strip()
    d["_alg"] = partes.str[3].str.strip()
    ancho = d.pivot_table(index="_alg", columns="_esc", values="delta_auc")
    faltan = [e for e in escenarios if e not in ancho.columns]
    if faltan:
        raise KeyError(f"valor_anadido_por_escenario: falta el escenario {faltan} en el marco")
    ancho = ancho.sort_values(escenarios[0], ascending=True)      # el mayor, arriba
    y = np.arange(len(ancho))

    estilo(cuerpo_compensado(7.0))
    fig, ax = plt.subplots(figsize=(7.0, 0.52 * len(ancho) + 2.2))
    ax.hlines(y, ancho[escenarios[1]], ancho[escenarios[0]], color=config.COLOR_MEDIO,
             linewidth=2.2, zorder=2)
    ax.scatter(ancho[escenarios[0]], y, s=58, color=config.COLOR_TEXTO, zorder=3,
              edgecolor="white", linewidth=0.8, label=etiquetas[0])
    ax.scatter(ancho[escenarios[1]], y, s=58, color=config.COLOR_ACENTO, zorder=4,
              edgecolor="white", linewidth=0.8, label=etiquetas[1])
    ax.axvline(0, color=config.GRIS_TENUE, linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([nombres.get(a, a) for a in ancho.index], fontsize="small")
    ax.set_xlabel("ΔAUC-ROC de la especificación perioperatoria frente a la preoperatoria")
    ax.set_ylim(-0.6, len(ancho) - 0.4)
    ax.set_xlim(left=min(-0.01, ancho.to_numpy().min() - 0.01))
    ax.grid(axis="y", visible=False)
    eje_coma(ax, "x", 2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize="small",
             frameon=False)
    if titulo:
        ax.set_title(titulo, fontsize="large", loc="left", pad=12)
    fig.tight_layout()
    _realinear_titulo(ax)
    return fig


def consistencia(d: pd.DataFrame, titulo: str = "", *, etiquetas: dict | None = None,
                 etiquetas_var: dict | None = None):
    """Señal y dirección de cada predictor a lo largo de varias definiciones del desenlace.

    Una fila por predictor y una columna por definición. El punto lleno marca una asociación
    que alcanza significación y su color, la dirección; el punto vacío indica que la variable
    entró al modelo sin alcanzarla. Leer la fila completa responde si la asociación se
    sostiene al cambiar la definición del desenlace, que es lo que la figura busca mostrar.
    """
    # `etiquetas` renombra las columnas -las definiciones del desenlace- y `etiquetas_var`
    # las filas. El ancho de la columna de rótulos lo fija su cadena más larga y es lo que
    # decide cuánto se reduce la figura al ajustarla a la página, de modo que acortarlos es
    # la única palanca: bajar la tipografía encoge el lienzo en la misma proporción y el
    # tamaño impreso no cambia. Qué abreviatura es aceptable es decisión editorial.
    etiquetas = etiquetas or {}
    etiquetas_var = etiquetas_var or {}
    orden = (d.groupby("var_rename")["n_desenlaces_con_senal"].first()
             .sort_values(ascending=True).index.tolist())
    cols = list(dict.fromkeys(d["desenlace"]))
    etiqueta = {v: etiquetas_var.get(v, etiqueta_corta(l))
                for v, l in zip(d["var_rename"], d["label"].str.split(":").str[0])}

    # El lienzo se declara del ancho de la caja de texto y la tipografía se deriva de él, de
    # modo que el escalado a la página sea la identidad. Declararlo por el número de columnas
    # producía un lienzo de más del doble del ancho útil: `guardar` recorta al contenido, la
    # página lo reducía a menos de la mitad y el cuerpo terminaba en 4,8 pt, el más pequeño
    # del manuscrito.
    cuerpo = cuerpo_compensado(ANCHO_CAJA)
    estilo(cuerpo)
    fig, ax = plt.subplots(figsize=(ANCHO_CAJA, 0.40 * len(orden) + 1.8))
    for j, des in enumerate(cols):
        sub = d[d["desenlace"].eq(des)].set_index("var_rename")
        for i, v in enumerate(orden):
            if v not in sub.index:
                continue
            r = sub.loc[v]
            arriba = int(r["direccion"]) > 0
            c = config.COLOR_ACENTO if arriba else config.COLOR_TEXTO
            if bool(r["senal"]):
                ax.scatter(j, i, s=90, color=c, zorder=3, edgecolor="white", linewidth=0.9)
            else:
                ax.scatter(j, i, s=52, facecolor="white", edgecolor=config.COLOR_MEDIO,
                           linewidth=1.1, zorder=2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([etiquetas.get(c, c) for c in cols], fontsize=cuerpo * 0.9,
                       rotation=30, ha="right")
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels([etiqueta.get(v, v) for v in orden], fontsize=cuerpo * 0.9)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(-0.7, len(orden) - 0.3)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    # Sin título en figura: la leyenda del manuscrito lo enuncia, y uno largo ensancha el
    # lienzo, que `guardar` recorta al contenido, encogiendo la figura al ajustarla a la
    # caja. `titulo` se conserva en la firma por compatibilidad y se ignora.
    #
    # `tight_layout` va antes de construir la leyenda: acomoda el eje según las etiquetas,
    # todavía sin ella, de modo que la medición de abajo use la posición final.
    fig.tight_layout()
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    borde_inferior = min(lbl.get_window_extent(renderer).y0 for lbl in ax.get_xticklabels())
    y_leyenda = ax.transAxes.inverted().transform((0, borde_inferior))[1]
    alto_eje_pulgadas = ax.get_position().height * fig.get_size_inches()[1]
    y_leyenda -= 0.28 / alto_eje_pulgadas  # ~0,28" de aire bajo la etiqueta rotada más larga
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=config.COLOR_ACENTO, label="Señal, dirección de riesgo"),
        Line2D([], [], marker="o", ls="", color=config.COLOR_TEXTO, label="Señal, dirección protectora"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor=config.COLOR_MEDIO, color="none", label="Sin señal")],
        # Debajo y en una fila: al costado, `tight_layout` encoge el eje para dejarle
        # sitio y la matriz pierde un tercio del ancho disponible. El ancla vertical se
        # mide sobre el descenso real de las etiquetas rotadas (arriba), no una fracción
        # fija: con esa fracción una etiqueta larga como "Intensidad máx. (β)" invadía la
        # leyenda.
        loc="upper center", bbox_to_anchor=(0.5, y_leyenda), ncol=3, fontsize=cuerpo * 0.85,
        borderaxespad=0.0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Arco predictivo
# ─────────────────────────────────────────────────────────────────────────────

def _paleta_algoritmos(nombres) -> dict:
    """Un color por algoritmo, dentro de la rampa de la marca."""
    from matplotlib.colors import LinearSegmentedColormap
    rampa = LinearSegmentedColormap.from_list(
        "alg", [config.COLOR_MEDIO, config.COLOR_FRIO, config.COLOR_TEXTO])
    n = max(len(nombres) - 1, 1)
    return {a: rampa(i / n) for i, a in enumerate(nombres)}


def curvas_roc(oof: pd.DataFrame, titulos: dict | None = None, *, orden=None):
    """Curvas de sensibilidad frente a especificidad, un panel por especificación.

    Poner ambas especificaciones lado a lado hace visible de un vistazo la separación que
    la comparación pareada cuantifica: la pregunta del trabajo no es cuánto discrimina un
    modelo, sino cuánto se gana al disponer de la información perioperatoria.

    Los paneles van en fila y no en columna. Apilados, y siendo cuadrados por el aspecto
    igual que exige una curva ROC, la figura resulta el doble de alta que ancha: a ancho de
    texto excedería la altura de la página y `\setkeys{Gin}` la reduciría al 61 %, encogiendo
    con ella toda la tipografía. En fila, la figura queda apaisada, se imprime a ancho
    completo y el escalado es la identidad.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    estilo()
    titulos = titulos or {}
    especificaciones = orden or list(dict.fromkeys(oof["especificacion"]))
    algoritmos = list(dict.fromkeys(oof["algoritmo"]))
    color = _paleta_algoritmos(algoritmos)

    n = len(especificaciones)
    cuerpo = cuerpo_compensado(ANCHO_CAJA)
    estilo(cuerpo)
    fig, ejes = plt.subplots(1, n, figsize=(ANCHO_CAJA, ANCHO_CAJA / n + 1.15),
                             sharey=True)
    ejes = np.atleast_1d(ejes)
    for ax, esp in zip(ejes, especificaciones):
        sub = oof[oof["especificacion"].eq(esp)]
        curvas = {}
        for alg in algoritmos:
            g = sub[sub["algoritmo"].eq(alg)].sort_values("indice")
            if g.empty:
                continue
            fpr, tpr, _ = roc_curve(g["y"], g["p"])
            curvas[alg] = (fpr, tpr, roc_auc_score(g["y"], g["p"]))
        # Solo el de mayor discriminación lleva color. El resto queda atenuado, porque la
        # lectura de la figura es cualitativa: cuánto se despega el conjunto de la diagonal.
        mejor = max(curvas, key=lambda a: curvas[a][2]) if curvas else None
        for alg, (fpr, tpr, auc) in curvas.items():
            destacado = alg == mejor
            ax.plot(fpr, tpr,
                    color=config.COLOR_ACENTO if destacado else config.COLOR_MEDIO,
                    linewidth=2.4 if destacado else 1.1,
                    alpha=1.0 if destacado else 0.75,
                    zorder=3 if destacado else 2,
                    label=(f"{alg} · AUC {_coma(auc, 3)}" if destacado
                           else ("resto de los algoritmos" if alg == next(iter(curvas)) else None)))
        ax.plot([0, 1], [0, 1], color=config.GRIS_TENUE, linestyle="--", linewidth=1.0,
                zorder=1, label="Azar")
        ax.set_title(titulos.get(esp, esp), fontsize=cuerpo, loc="left", pad=8)
        ax.set_xlabel("1 − especificidad", fontsize=cuerpo)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=cuerpo * 0.9)
        # La leyenda dentro del cuadro, bajo la diagonal, que es la zona que ninguna curva
        # ocupa: al costado obligaría a encoger el panel y a perder el aspecto cuadrado.
        ax.legend(loc="lower right", fontsize=cuerpo * 0.85, frameon=False,
                  borderaxespad=0.4, handlelength=1.4)
        eje_coma(ax, "xy", 1)
    ejes[0].set_ylabel("Sensibilidad", fontsize=cuerpo)
    fig.tight_layout()
    return fig


def _p_unilateral(v) -> str:
    """Valor p con coma decimal, acotado por debajo cuando ninguna remuestra lo alcanza."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return "p < 0,001" if v < 0.001 else f"p = {_coma(v, 3)}"


def valor_anadido(pareada: pd.DataFrame, titulo: str = "", *, etiqueta=None,
                  umbral: float = 0.05, orden=None, nombres: dict | None = None):
    """Desempeño de ambas especificaciones y diferencia entre ellas, por algoritmo.

    El punto hueco marca el desempeño con información preoperatoria y el sólido con
    información perioperatoria, ambos sobre la escala del área bajo la curva. La diferencia,
    su intervalo de confianza y el valor p se consignan como texto al costado, no como trazo:
    el intervalo corresponde a la diferencia y no al desempeño perioperatorio, de modo que
    dibujarlo sobre este eje invitaría a leerlo como el intervalo de esa área.

    El asterisco antecede a las diferencias que alcanzan significación.
    """
    estilo()
    nombres = nombres or config.NOMBRE_ALGORITMO
    d = pareada.copy()
    d["alg"] = (d["referencia"].str.split(" · ").str[-1] if etiqueta is None else d[etiqueta])
    orden = list(orden or config.ALGORITMOS)
    d["_o"] = d["alg"].map({a: i for i, a in enumerate(orden)}).fillna(len(orden))
    d = d.sort_values("_o", ascending=False)          # el primero declarado, arriba
    hay_ic = {"delta_auc_ic_low", "delta_auc_ic_high"} <= set(d.columns)
    hay_p = "p_unilateral" in d
    y = np.arange(len(d))

    # El lienzo se declara del ancho de la caja de texto, de modo que el escalado a la
    # página sea la identidad y el cuerpo llegue al papel al tamaño declarado. La altura da
    # una relación cercana a 1,6: más achatada, los datos quedan en una franja estrecha al
    # centro; más alta, la sección deja de caber en su página.
    cuerpo = cuerpo_compensado(ANCHO_CAJA)
    estilo(cuerpo)
    fig, ax = plt.subplots(figsize=(ANCHO_CAJA, 0.37 * len(d) + 1.52))
    ax.hlines(y, d["auc_referencia"], d["auc_alternativa"],
              color=config.COLOR_FONDO, linewidth=3.4, zorder=1)
    ax.scatter(d["auc_referencia"], y, s=58, facecolor="white",
               edgecolor=config.COLOR_FRIO, linewidth=1.6, zorder=3, label="Preoperatoria")
    ax.scatter(d["auc_alternativa"], y, s=58, color=config.COLOR_ACENTO, zorder=4,
               edgecolor="white", linewidth=0.9, label="Perioperatoria")

    # Valor de cada punto, sobre él
    for i, (_, r) in enumerate(d.iterrows()):
        for x in (r["auc_referencia"], r["auc_alternativa"]):
            ax.annotate(_coma(x, 3), (x, i), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=cuerpo * 0.72,
                        color=config.GRIS_TENUE)

    # Columna de ΔAUC, fuera del área de datos: anclada al borde derecho de los ejes en
    # fracción de eje y a la fila en coordenadas de dato. Dentro obligaba a estirar el eje
    # más allá del máximo observado, dejando un margen derecho vacío que no era dato.
    for i, (_, r) in enumerate(d.iterrows()):
        # Una sola línea por fila. Tres líneas por algoritmo saturaban la figura, y el
        # intervalo y el valor p se reportan completos en su anexo.
        marca = "*" if (hay_p and r["p_unilateral"] < umbral) else " "
        ax.annotate(f"+{_coma(r['delta_auc'], 3)}{marca}", xy=(1.02, i),
                    xycoords=("axes fraction", "data"), ha="left", va="center",
                    fontsize=cuerpo * 0.8, color=config.COLOR_TEXTO, annotation_clip=False)
    ax.annotate("ΔAUC", xy=(1.02, 1.0), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", ha="left", va="bottom", weight="bold",
                fontsize=cuerpo * 0.8, color=config.COLOR_TEXTO, annotation_clip=False)

    ax.set_yticks(y)
    # Los nombres de algoritmo van algo por debajo del cuerpo: son la columna más ancha de
    # la figura y a tamaño pleno empujan el área de datos hacia la derecha.
    ax.set_yticklabels([nombres.get(a, a) for a in d["alg"]], fontsize=cuerpo * 0.9)
    ax.tick_params(axis="x", labelsize=cuerpo)
    ax.set_xlabel("AUC-ROC sobre las predicciones fuera de pliegue", fontsize=cuerpo)
    # El eje se ajusta al rango observado y no al bloque de cifras, que ahora vive fuera.
    ax.set_xlim(min(d["auc_referencia"].min() - 0.02, 0.50),
                d["auc_alternativa"].max() + 0.015)
    ax.set_ylim(-0.62, len(d) - 0.38)
    ax.grid(axis="y", visible=False)
    eje_coma(ax, "x", 2)
    # La leyenda sale del área de datos: dentro colisiona con el bloque de ΔAUC del costado.
    # La leyenda ocupa la banda superior, fuera del área de datos. El título en figura se
    # omite a propósito: colisiona con ella y el pie de LaTeX ya lo dice.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
              fontsize=cuerpo * 0.9, markerscale=1.0, borderaxespad=0.0, frameon=False)
    # Sin título ni nota al pie: el pie de figura del manuscrito enuncia ambos, y un título
    # largo ensancharía el lienzo. `titulo` se conserva en la firma y se ignora.
    fig.tight_layout()
    return fig


def consenso_permutacion(consenso: pd.DataFrame, titulo: str = "", *, top: int = 15,
                         etiquetas: dict | None = None, columna: str = "n_modelos_top10"):
    """Ordenamiento de las variables por acuerdo entre algoritmos.

    El punto es el puesto medio que la variable ocupa entre los modelos y la banda, su
    desviación entre ellos. Se muestra el consenso y no un solo modelo porque el orden que
    produce cualquiera de ellos depende de su forma funcional, no solo de la información
    disponible.

    Un puesto es una posición, no una cantidad, y no tiene origen en cero: por eso se dibuja
    como punto sobre un eje que corre de 1 hacia arriba y no como barra. La versión anterior
    codificaba el puesto como largo de barra, de modo que la variable **menos** informativa
    era la de barra más larga, exactamente al revés de lo que el lector espera.

    `top` y `columna` deben referirse al mismo umbral. Dibujar quince filas mientras la
    columna de recuento informa la pertenencia al top 10 hace que las cinco últimas reporten
    su presencia en un conjunto del que ellas mismas están fuera, que es lo que esta figura
    afirmaba antes de la revisión. El umbral del rótulo se lee del nombre de la columna y no
    se escribe aparte, de modo que no pueda contradecirla.
    """
    estilo()
    etiquetas = etiquetas or {}
    d = consenso.nsmallest(top, "rango_medio").iloc[::-1]
    # Ausente la columna, la figura salía sin su recuento y nada lo advertía: es la forma en
    # que un cambio de umbral en `evaluar.consenso` -que renombra la columna- se perdería.
    if columna not in d:
        raise ValueError(f"el consenso no trae la columna {columna!r}; "
                         f"tiene {[c for c in d.columns if c.startswith('n_modelos')]}")
    umbral = "".join(ch for ch in columna if ch.isdigit()) or str(top)
    n_alg = int(d[columna].max()) if d[columna].max() else 7

    de = d["rango_de"] if "rango_de" in d else pd.Series(0.0, index=d.index)
    lo = np.maximum(d["rango_medio"] - de, 1.0)      # un puesto por debajo de 1 no existe
    hi = d["rango_medio"] + de

    # El alto por fila se conserva: con una variable por fila es lo que sostiene la
    # legibilidad, y encogerlo por ganar página dejaría los rótulos ilegibles. Lo que se
    # recupera es el espacio del título, que la leyenda del manuscrito ya enuncia.
    cuerpo = cuerpo_compensado(ANCHO_CAJA)
    estilo(cuerpo)
    fig, ax = plt.subplots(figsize=(ANCHO_CAJA, 0.46 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.hlines(y, 1, d["rango_medio"], color=config.GRIS_TENUE, linewidth=0.8,
              linestyle=(0, (1, 3)), alpha=0.55, zorder=1)   # guía, punteada para no competir
    ax.hlines(y, lo, hi, color=config.COLOR_MEDIO, linewidth=3.4, zorder=2)
    ax.scatter(d["rango_medio"], y, s=52, color=config.COLOR_TEXTO, zorder=3,
               edgecolor="white", linewidth=0.9)

    # La columna de recuento vive fuera del área de datos, anclada al borde derecho de los
    # ejes: dentro obligaba a estirar el eje un 15 % más allá del puesto máximo observado,
    # y ese tramo no es dato sino sitio para el texto.
    for i, (_, r) in enumerate(d.iterrows()):
        ax.annotate(f"{int(r[columna])}/{n_alg}", xy=(1.03, i),
                    xycoords=("axes fraction", "data"), ha="left", va="center",
                    fontsize=cuerpo * 0.8, color=config.GRIS_TENUE,
                    annotation_clip=False)
    ax.annotate(f"modelos que\nla sitúan en\nel top {umbral}", xy=(1.03, 1.0),
                xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                ha="left", va="bottom", fontsize=cuerpo * 0.68,
                color=config.GRIS_TENUE, linespacing=1.3, annotation_clip=False)

    ax.set_yticks(y)
    # Los rótulos van algo por debajo del cuerpo: son la columna más ancha y a tamaño pleno
    # dejan al eje sin sitio.
    ax.set_yticklabels([etiquetas.get(v, v) for v in d["var_rename"]], fontsize=cuerpo * 0.9)
    ax.tick_params(axis="x", labelsize=cuerpo)
    ax.set_xlabel(f"Puesto medio entre los {n_alg} algoritmos (1 = mayor importancia)",
                  fontsize=cuerpo)
    marcas = [m for m in (1, 5, 10, 15, 20, 25, 30, 35) if m <= float(hi.max()) * 1.02]
    ax.set_xticks(marcas)
    ax.set_xlim(0.2, float(hi.max()) * 1.02)
    # Sin margen superior: el que había reservaba sitio al rótulo de la columna de recuento
    # cuando vivía dentro del área de datos, y ahora dejaría una franja vacía.
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.grid(axis="y", visible=False)
    # Sin título en figura: la leyenda del manuscrito lo enuncia, y un título largo ensancha
    # el lienzo, que `guardar` recorta al contenido, encogiendo la figura al ajustarla a la
    # caja. `titulo` se conserva en la firma por compatibilidad y se ignora.
    fig.tight_layout()
    return fig


def reordenamiento(izquierda: pd.DataFrame, derecha: pd.DataFrame, titulo: str = "", *,
                   top: int = 12, etiquetas: dict | None = None,
                   nombres: tuple = ("Sin centro", "Con centro"),
                   equivalencias: dict | None = None):
    """Cambio de posición de cada variable entre dos escenarios.

    Cada columna ordena las variables por su puesto en el consenso y las líneas unen a cada
    una entre ambos escenarios: la pendiente muestra cuánto se desplaza. Se destaca en
    acento la variable que aparece solo en uno de los dos, porque su entrada es lo que
    reordena al resto.

    `equivalencias` declara, para una variable de `derecha` que es una recodificación de una
    de `izquierda` (p. ej. la dosis de morfina agrupada en tramos frente a la variable
    original), a cuál nombre de `izquierda` corresponde: `{nombre_en_derecha:
    nombre_en_izquierda}`. Sin la equivalencia, ambas se grafican como variables distintas y
    no se traza la línea entre ellas, aunque midan lo mismo.
    """
    estilo()
    etiquetas = etiquetas or {}
    equivalencias = equivalencias or {}
    a = izquierda.nsmallest(top, "rango_medio").reset_index(drop=True)
    b = derecha.nsmallest(top, "rango_medio").reset_index(drop=True)
    pos_a = {v: i for i, v in enumerate(a["var_rename"])}
    # Las posiciones de la derecha se indexan por el nombre canónico (el de la izquierda),
    # de modo que una variable y su recodificación comparten nodo; el nombre real de cada
    # lado se conserva aparte para que la etiqueta mostrada no cambie.
    nombre_real_b = {equivalencias.get(v, v): v for v in b["var_rename"]}
    pos_b = {equivalencias.get(v, v): i for i, v in enumerate(b["var_rename"])}
    nuevas = set(pos_b) - set(pos_a)

    cuerpo = cuerpo_compensado(ANCHO_CAJA)
    estilo(cuerpo)
    fig, ax = plt.subplots(figsize=(ANCHO_CAJA, 0.50 * top + 1.1))
    # El orden de dibujo se declara y no se toma de un conjunto: la iteración de un `set` de
    # cadenas depende del sembrado de hash del proceso, de modo que dos corridas del mismo
    # código apilaban las líneas superpuestas al revés y producían PNG distintos. La figura
    # se veía igual y aun así no era reproducible byte a byte.
    for v in sorted(set(pos_a) | set(pos_b),
                    key=lambda v: (pos_a.get(v, len(pos_a)), pos_b.get(v, len(pos_b)), v)):
        ia, ib = pos_a.get(v), pos_b.get(v)
        destaca = v in nuevas
        c = config.COLOR_ACENTO if destaca else config.COLOR_MEDIO
        if ia is not None and ib is not None:
            ax.plot([0, 1], [ia, ib], color=c, linewidth=1.8 if destaca else 1.2,
                    zorder=3 if destaca else 2)
        for x, i in ((0, ia), (1, ib)):
            if i is not None:
                ax.scatter(x, i, s=42, color=c, zorder=4, edgecolor="white", linewidth=0.8)
    for x, mapa, nombres_reales, ha in ((0, pos_a, {v: v for v in pos_a}, "right"),
                                        (1, pos_b, nombre_real_b, "left")):
        for v, i in mapa.items():
            real = nombres_reales.get(v, v)
            ax.annotate(etiquetas.get(real, real), (x, i),
                        xytext=(-10 if ha == "right" else 10, 0), textcoords="offset points",
                        ha=ha, va="center", fontsize=cuerpo,
                        color=config.COLOR_ACENTO if v in nuevas else config.COLOR_TEXTO)
    ax.set_xticks([0, 1]); ax.set_xticklabels(nombres, fontsize=cuerpo)
    ax.set_xlim(-0.62, 1.62)
    ax.invert_yaxis()
    ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    # Sin título en figura, por la misma razón que en `consenso_permutacion`: lo enuncia la
    # leyenda y ensancharía el lienzo. `titulo` se conserva en la firma y se ignora.
    fig.tight_layout()
    return fig


def _tramos_numericos(x: np.ndarray, bins: int) -> tuple[np.ndarray, list[str]]:
    """Cortes de percentil de un vector numérico y un rótulo legible por tramo.

    Miles de puntos sobre una escala continua no se leen con precisión clínica: agrupar en
    tramos de percentil, con su rango como etiqueta, es lo que hace comprensible el eje.
    """
    cortes = np.unique(np.nanpercentile(x, np.linspace(0, 100, bins + 1)))
    if len(cortes) < 3:
        cortes = np.array([np.nanmin(x), np.nanmax(x)])
    dec = 0 if np.allclose(x, np.round(x), atol=0.05) else 1
    # En dash con espacios: el mismo separador de rango que `tabula._ic` usa para el IC 95 %.
    rotulos = [f"{_coma(cortes[i], dec)} – {_coma(cortes[i + 1], dec)}"
              for i in range(len(cortes) - 1)]
    return cortes, rotulos


def _niveles_categoricos(x: pd.Series, orden: list | None) -> list[str]:
    """Niveles observados de una variable categórica, en el orden declarado si se entrega."""
    obs = x.dropna().unique().tolist()
    if orden:
        ordenados = [str(v) for v in orden if str(v) in obs]
        ordenados += sorted(v for v in obs if v not in ordenados)
        return ordenados
    return sorted(obs)


def _caja_por_nivel(ax, valores: list[np.ndarray], rotulos: list[str]) -> None:
    """Diagrama de caja compacto, un nivel por posición, en la paleta de la casa."""
    ax.boxplot(valores, positions=range(len(rotulos)), widths=0.5, patch_artist=True,
              medianprops=dict(color="white", linewidth=1.6),
              boxprops=dict(facecolor=config.COLOR_FRIO, edgecolor=config.COLOR_TEXTO,
                           linewidth=0.8),
              whiskerprops=dict(linewidth=0.8, color=config.COLOR_TEXTO),
              capprops=dict(linewidth=0.8, color=config.COLOR_TEXTO),
              flierprops=dict(marker="o", markersize=2.5, alpha=0.35,
                             markerfacecolor=config.COLOR_FRIO, markeredgewidth=0))
    ax.set_xticks(range(len(rotulos)))
    ax.set_xticklabels(rotulos, fontsize="x-small",
                       rotation=22 if len(rotulos) > 2 else 0, ha="right" if len(rotulos) > 2 else "center")


def _rotulo_con_puesto(etiqueta: str, rangos: dict | None, var: str, ancho: int = 30) -> str:
    """Rótulo del panel con el puesto de consenso, si se declaró alguno.

    El puesto alarga el título lo bastante como para que, alineado a la izquierda, invada el
    panel vecino: por eso se pliega. Sin `rangos` el rótulo sale intacto, que es lo que
    mantiene idénticas a las figuras que no declaran puesto.
    """
    if rangos is None:
        return etiqueta
    puesto = rangos.get(var)
    cola = ("· fuera del consenso" if puesto is None or pd.isna(puesto)
            else f"· consenso {int(puesto)}")
    # El puesto se pliega entero: dejar que el ajuste automático lo parta abandona el número
    # solo en la última línea, que es donde deja de leerse como puesto.
    lineas = textwrap.wrap(etiqueta, ancho) or [etiqueta]
    if len(lineas[-1]) + 1 + len(cola) <= ancho:
        lineas[-1] += f" {cola}"
    else:
        lineas.append(cola)
    return "\n".join(lineas)


def dependencia_shap(shap: pd.DataFrame, columnas, titulo: str = "", *,
                     etiquetas: dict | None = None, tipos: dict | None = None,
                     niveles: dict | None = None, rangos: dict | None = None,
                     ncols: int = 2, bins: int = 4):
    """Contribución de cada variable a la predicción, frente a su valor.

    A diferencia de la permutación, que resume el peso en un número, muestra la forma de la
    relación: si es monótona, si tiene umbral, o si cambia de signo. El eje horizontal
    respeta el tipo de la variable en vez de graficar la escala preprocesada: las
    categóricas -binarias, ordinales, nominales, `tipos` no distingue entre ellas porque acá
    todas son niveles discretos- muestran un diagrama de caja por nivel con su etiqueta
    original, en el orden clínico que declare `niveles`; las numéricas se agrupan en `bins`
    tramos de percentil.

    `tipos` declara, por variable, `"num"` o cualquier otro valor para categórica; sin
    declaración se asume numérica. `niveles` declara, por variable categórica, el orden de
    sus niveles; sin declaración se ordena alfabéticamente.

    `rangos` declara el puesto que el consenso de importancia por permutación asigna a cada
    variable, ver `evaluar.paneles_shap`. Sin él, los paneles no dicen de dónde salen y el
    lector los toma por el ordenamiento del consenso, que no es el mismo; con él, cada
    título lo declara, y el que no tiene puesto se rotula como tal en vez de aparentar uno.
    """
    estilo()
    etiquetas, tipos, niveles = etiquetas or {}, tipos or {}, niveles or {}
    columnas = list(columnas)
    nfil = int(np.ceil(len(columnas) / ncols))
    estilo(cuerpo_compensado(3.9 * ncols))
    fig, ejes = plt.subplots(nfil, ncols, figsize=(3.9 * ncols, 3.4 * nfil), squeeze=False)
    for ax, c in zip(ejes.ravel(), columnas):
        g = shap[shap["columna"].eq(c)]
        # El valor original de la variable, no el preprocesado: la matriz que alimenta al
        # modelo lleva las numéricas estandarizadas y las ordinales codificadas, de modo que
        # graficarla mostraría una escala que no es la de la variable.
        x = g["valor_original"] if "valor_original" in g else g["valor"]
        s = g["shap"].to_numpy(float)
        if tipos.get(c, "num") == "num":
            xv = pd.to_numeric(x, errors="coerce").to_numpy(float)
            ok = np.isfinite(xv) & np.isfinite(s)
            cortes, rotulos = _tramos_numericos(xv[ok], bins)
            tramo = pd.cut(xv[ok], bins=cortes, labels=range(len(rotulos)), include_lowest=True)
            valores = [s[ok][tramo == i] for i in range(len(rotulos))]
        else:
            xs = pd.Series(x).astype(object).where(pd.notna(x)).astype(str).where(pd.notna(x))
            ok = xs.notna().to_numpy() & np.isfinite(s)
            rotulos = _niveles_categoricos(xs[ok], niveles.get(c))
            valores = [s[ok][xs[ok].to_numpy() == niv] for niv in rotulos]
        _caja_por_nivel(ax, valores, rotulos)
        ax.axhline(0, color=config.GRIS_TENUE, linestyle="--", linewidth=0.9, zorder=1)
        ax.set_title(_rotulo_con_puesto(etiquetas.get(c, c), rangos, c),
                     fontsize="small", loc="left")
        ax.set_ylabel("Contribución SHAP", fontsize="small")
        ax.tick_params(labelsize="x-small")
        eje_coma(ax, "y", 2)
    for ax in ejes.ravel()[len(columnas):]:
        ax.set_visible(False)
    if titulo:
        fig.suptitle(titulo, fontsize="large", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def _asteriscos(p, umbral: float = 0.05) -> str:
    """Convención habitual de marcas de significación."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < umbral else "n. s."


def barras_comparadas(desempeno: pd.DataFrame, pareada: pd.DataFrame, titulo: str = "", *,
                      orden=None, nombres: dict | None = None, base: float = 0.5,
                      etiquetas: tuple = ("Preoperatoria", "Perioperatoria")):
    """Discriminación de ambas especificaciones por algoritmo, con su contraste.

    El eje arranca en 0,5 y no en cero: en un área bajo la curva ese valor es la ausencia de
    discriminación, de modo que la altura de cada barra representa cuánto mejora el modelo
    sobre el azar. Recortar el eje en un punto arbitrario exageraría las diferencias; hacerlo
    en el valor nulo de la medida es lo que las vuelve legibles.

    La marca sobre cada par corresponde a la comparación pareada, estimada sobre las mismas
    remuestras para ambas especificaciones. El intervalo bajo cada barra es la desviación del
    área entre los pliegues externos, que describe la variación del desempeño de un pliegue a
    otro y no la incertidumbre de la diferencia.
    """
    estilo()
    nombres = nombres or config.NOMBRE_ALGORITMO
    orden = list(orden or config.ALGORITMOS)
    especificaciones = list(config.ESPECIFICACIONES)

    d = desempeno.set_index(["algoritmo", "especificacion"])
    p = pareada.copy()
    p["alg"] = p["referencia"].str.split(" · ").str[-1]
    p = p.set_index("alg")

    algos = [a for a in orden if (a, especificaciones[0]) in d.index]
    x = np.arange(len(algos))
    ancho = 0.36
    color = {especificaciones[0]: config.COLOR_MEDIO, especificaciones[1]: config.COLOR_TEXTO}

    fig, ax = plt.subplots(figsize=(1.55 * len(algos) + 2.2, 5.6))
    for k, esp in enumerate(especificaciones):
        alturas = [d.loc[(a, esp), "auc"] for a in algos]
        de = [d.loc[(a, esp), "auc_pliegues_de"] if "auc_pliegues_de" in d else np.nan
              for a in algos]
        pos = x + (k - 0.5) * ancho
        ax.bar(pos, np.array(alturas) - base, bottom=base, width=ancho,
               color=color[esp], label=etiquetas[k], zorder=2)
        for xi, h, s in zip(pos, alturas, de):
            texto = _coma(h, 3) + (f" ± {_coma(s, 3)}" if np.isfinite(s) else "")
            ax.annotate(texto, (xi, h), xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize="x-small", color=config.GRIS_TENUE)

    # Corchete de significación sobre cada par
    tope = max(d.loc[(a, e), "auc"] for a in algos for e in especificaciones)
    alto = tope + 0.055
    for xi, a in zip(x, algos):
        if a not in p.index:
            continue
        izq, der = xi - 0.5 * ancho, xi + 0.5 * ancho
        ax.plot([izq, izq, der, der], [alto - 0.008, alto, alto, alto - 0.008],
                color=config.GRIS_TENUE, linewidth=1.0, zorder=3)
        ax.annotate(_asteriscos(p.loc[a, "p_unilateral"]) if "p_unilateral" in p else "",
                    (xi, alto), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color=config.COLOR_TEXTO)

    ax.set_xticks(x)
    ax.set_xticklabels([nombres.get(a, a) for a in algos], fontsize="small",
                       rotation=22, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(base, alto + 0.045)
    ax.grid(axis="x", visible=False)
    eje_coma(ax, "y", 2)
    ax.legend(loc="upper left", fontsize="small", ncol=2)
    ax.annotate("*** p < 0,001 en la comparación pareada · la barra parte del valor nulo "
                "(AUC 0,5) · ± desviación entre los diez pliegues externos",
                (0.0, -0.30), xycoords="axes fraction", ha="left", va="top",
                fontsize="x-small", color=config.GRIS_TENUE)
    if titulo:
        ax.set_title(titulo, fontsize="large", loc="left", pad=14)
    fig.tight_layout()
    return fig
