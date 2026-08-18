"""config.py — constantes del pipeline DAPC (flat constants).

Única fuente de parámetros: rutas, semilla, vocabularios controlados y el contrato
del catálogo (ver docs/estructura_metadatos.md). Sin dependencias de sklearn/pandas.
"""
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent          # repositorio/
ROOT = REPO.parent                                     # analisis_DAPC/
CATALOGO = REPO / "db" / "metadatos.xlsx"              # catálogo único (manual)
DATA = ROOT / "data"

# Diccionario de las variables originales del proyecto, insumo del anexo homónimo. Lo
# mantiene el autor a mano, igual que el catálogo, de modo que su lugar es `db/` y no
# `salida/`, que es donde el código escribe y de donde una limpieza puede borrarlo. Mientras
# el archivo siga en su ubicación actual, esta línea apunta ahí: al moverlo, se cambia por
# `REPO / "db" / "anexo_variables_fonis.xlsx"` y nada más.
VARIABLES_FONIS = REPO / "salida" / "outputs" / "tables" / "anexo_variables_fonis.xlsx"

# Única lectura del análisis original: los datos crudos de RedCAP.
RAW = DATA / "raw" / "dataFONIS_raw.xlsx"

# Espacio propio del flujo reestructurado. Los artefactos del análisis original
# (data/processed/, data/outputs/) son de solo lectura mientras dure la migración:
# se conservan intactos como testigo para contrastar cifras.
SALIDA = REPO / "salida"
PROCESSED = SALIDA / "processed"
OUTPUTS = SALIDA / "outputs"
CRIBADO = OUTPUTS / "cribado.parquet"                  # enriquecimiento derivado (runtime)

# ── Reproducibilidad ─────────────────────────────────────────────────────────
SEED = 42

# ── Presentación ─────────────────────────────────────────────────────────────
DECIMAL = "coma"
ORDEN_BLOQUE = {"preop": 1, "intraop": 2, "postop": 3}

# ── Vocabularios controlados (estructura_metadatos.md §3) ────────────────────
VOCAB = {
    "conceptual_type": {"bin", "nom", "ord", "num", "fecha"},
    "block":           {"preop", "intraop", "postop"},
    "timepoint":       {"basal", "intraop", "24h", "48h", "7d", "90d", "1s"},
    "source":          {"raw", "derivada", "espinal"},
    "prelim_role":     {"pred", "fr", "proxy", "ex", "target"},
    "capa_heterogeneidad": {"predictor", "contexto"},
    "rol_explicativo": {"literatura", "centro", "bivariado_estabilidad", "ajustador"},
}

# ── Contrato del catálogo (estructura_metadatos.md §5) ───────────────────────
# Columnas que el código consume (se blindan). El resto del catálogo es
# documentación libre (description, unit, nota_curacion).
CONTRATO_COLUMNAS = [
    "var_rename", "var_original", "label", "conceptual_type", "dtype_original",
    "categories", "positive_label", "block", "timepoint", "source", "domain",
    "prelim_role", "include_oe1", "include_model", "include_preop",
    "include_espinal", "include_heterogeneidad", "capa_heterogeneidad",
    "include_explicativo", "rol_explicativo",
    "plausibilidad_clinica",
    "exclusion_causa", "exclusion_subcausa", "exclusion_nota",
]
CLAVE = "var_rename"
# columnas que no admiten nulos en ninguna variable del universo
OBLIGATORIAS = ["var_rename", "var_original", "conceptual_type", "block", "source", "prelim_role"]
BOOLEANAS = ["include_oe1", "include_model", "include_preop", "include_espinal",
             "include_heterogeneidad", "include_explicativo"]

# ── Heterogeneidad entre centros ─────────────────────────────────────────────
# El cribado de predictores (variabilidad) NO aplica aquí: una variable de baja
# variabilidad global puede serlo por concentrarse en un centro, que es
# justamente la heterogeneidad de interés. Lo que sí importa es la
# COMPARABILIDAD DE MEDICIÓN: si la completitud difiere entre centros, la SMD
# puede reflejar quién fue medido y no cómo son las pacientes.
CENTER_VAR = "hosp"
SMD_UMBRAL = 0.20                  # magnitud relevante de heterogeneidad

# Cribado de normalidad (Shapiro-Wilk) que decide entre la prueba paramétrica y la de
# rangos en las variables numéricas. Réplica de los valores del análisis original.
NORMALIDAD_ALPHA = 0.05            # se rechaza la normalidad si p < alpha en algún grupo
SHAPIRO_MAX_N = 5000               # sobre este tamaño, Shapiro se evalúa en una submuestra
# Umbral en puntos porcentuales del rango de faltantes entre centros. Se sitúa en el
# hueco de la distribución observada (cuatro variables entre 17 y 25 puntos; el resto,
# 5,6 o menos), de modo que cualquier valor entre 6 y 17 arroja la misma clasificación.
MISSING_RANGO_CENTROS_MAX = 10.0

# ── Cribado de predictores (criterios de 01_EDA_screening) ───────────────────
# Alcance: el cribado define QUÉ VARIABLES ENTRAN A LOS MODELOS (explicativo y
# predictivo). El análisis descriptivo y bivariado se realiza sobre la cohorte
# completa, sin aplicar estos filtros.
OUTCOME = "dmg_1s_bin"
ROLES_EVAL = {"pred", "fr", "proxy"}          # roles que se someten a cribado
ROLES_EXCL = {"ex", "target", "desenlace"}    # no evaluables como predictores
VARS_TEXTO_LIBRE = ["otras_enf"]

MISSING_THRESHOLD = 5.0            # % máximo de datos faltantes
DOMINANT_PCT_THRESHOLD = 0.95      # categoría dominante ≥ 95 % → baja variabilidad
RARE_PCT_THRESHOLD = 0.05          # categoría rara < 5 % ...
RARE_N_THRESHOLD = 10              # ... y < 10 casos
MIN_NON_MISSING_VARIABILIDAD = 20  # mínimo de observaciones para evaluar
MIN_SD_THRESHOLD = 0.10
MIN_IQR_THRESHOLD = 1.0
MIN_EFFECTIVE_RANGE_THRESHOLD = 2.0

# ── Capa de presentación (docs/artefactos.md) ────────────────────────────────
FIGURAS = OUTPUTS / "figures"
TABLAS = OUTPUTS / "tables"

# Privacidad: en modo publicable no se emite ninguna celda con menos casos que
# el umbral, porque un cruce de estratos con recuentos mínimos permite reidentificar.
MODO_ARTEFACTO = "publicable"       # {"interno", "publicable"}
SUPRESION_N_MIN = 5

# Un estrato con este número de casos o menos se marca como frágil, no se omite: con 20
# casos un solo paciente desplaza la proporción 5 puntos porcentuales o más, de modo que la
# diferencia que se lee sobre él no admite la misma lectura que la de un estrato grande.
# Marcarlo y conservarlo es preferible a suprimirlo, que dejaría el hueco sin explicación.
ESTRATO_N_FRAGIL = 20

# Por debajo de este número, o si el nivel no se observa en el centro, la celda se muestra
# igual -omitirla dejaría un hueco sin explicación- pero la diferencia que la usa se suprime:
# con menos de diez casos la proporción del estrato mueve la diferencia más que el efecto que
# se quiere leer, y publicarla invitaría a compararla con las demás.
# Control de divulgación: un estrato de este tamaño o menor no publica su contenido.
ESTRATO_N_SUPRIMIR = 5

ESTRATO_N_MINIMO = 10

# Ejes de estratificación admitidos (máximo dos simultáneos: el cubo de agregados
# crece de forma combinatoria).
EJES_ESTRATIFICACION = ("desenlace", "centro", "bloque", "espinal")

# ── Identidad gráfica ────────────────────────────────────────────────────────
# Paleta derivada del manual de marca (azul petróleo #033F5F, turquesa mate
# #5EA7AE) y extendida con tonos desaturados para estratificar sin perder
# sobriedad. Se aplica bajo la regla de proporción 60-30-10: espacio limpio,
# estructura y acento.

COLOR_FONDO = "#D7E9EB"    # cian claro   · fondos, sombreado de filas
COLOR_MEDIO = "#9ECACE"    # azul hielo   · elementos secundarios, divisiones
COLOR_TEXTO = "#295C77"    # azul profundo· texto, títulos, ejes, serie principal
COLOR_ACENTO = "#D48C70"   # coral opaco  · significación, desenlace crítico
COLOR_NEUTRO = "#D9C3A0"   # beige arena  · subpoblaciones secundarias
COLOR_FRIO = "#7097A8"     # azul pizarra · cuarta categoría

# Bloques temporales: rampa clara→oscura, que codifica la progresión.
COLOR_BLOQUE = {"preop": COLOR_FONDO, "intraop": COLOR_MEDIO, "postop": COLOR_TEXTO}

# Series categóricas, en orden de asignación (máxima separación primero).
PALETA_CATEGORICA = (COLOR_TEXTO, COLOR_FRIO, COLOR_ACENTO, COLOR_NEUTRO)
PALETA_CENTROS = {"HCUCH": COLOR_TEXTO, "HCVB": COLOR_FRIO,
                  "HLTB": COLOR_ACENTO, "HSJD": COLOR_NEUTRO}

# Severidad del dolor: la categoría crítica lleva el acento. Para la variante de cuatro
# categorías, «Leve» es el punto medio entre COLOR_FONDO y COLOR_MEDIO: un blanco puro
# quedaba invisible sobre el fondo de la figura. «Sin dolor o leve» se conserva para la
# variante de tres.
COLOR_SEVERIDAD = {"Sin dolor o leve": COLOR_FONDO, "Moderado": COLOR_MEDIO,
                   "Intenso": COLOR_ACENTO, "Sin dolor": COLOR_FONDO, "Leve": "#BADADC"}

# Nombres canónicos de las tablas del manuscrito (un archivo por nombre).
TABLAS_MANUSCRITO = {
    # Introducción — literatura, sin dependencia de la cohorte
    "L1_factores_riesgo":       "tabla_L1_factores_riesgo",
    "L2_modelos_literatura":    "tabla_L2_modelos_literatura",
    # 01_cohorte
    "DM_descriptiva_maestra":   "tabla_DM_descriptiva_maestra",
    "HET_por_centro":           "tabla_HET_heterogeneidad_por_centro",
    "variables_excluidas":      "tabla_variables_excluidas",
    "sesgo_seleccion":          "tabla_SS_sesgo_seleccion",
    "FS_frecuencia_severidad":  "tabla_FS_frecuencia_severidad",
    "FC_frecuencia_centro":     "tabla_FC_frecuencia_centro_tiempo",
    # 02_modelamiento
    "EC_escalera":              "tabla_EC_escalera_modelos",
    "DA_descomposicion":        "tabla_DA_descomposicion_ajuste",
    # Una tabla logística por cohorte, cada una con el top 10 de su propio consenso. La
    # versión cruzada `RL_periop` tomaba la unión de ambas y dejaba media tabla en «no
    # incluido», porque las especificaciones no comparten predictores.
    "RL_periop_principal":      "tabla_RL_logistica_periop_principal",
    "RL_periop_espinal":        "tabla_RL_logistica_periop_espinal",
    # ⚠ obsoleta desde 2026-08-15, reemplazada por las dos anteriores. Se conserva la clave
    # por trazabilidad del artefacto ya emitido: ningún notebook la vuelve a generar.
    "RL_periop":                "tabla_RL_logistica_periop",
    "AS_apoyo_seleccion":       "tabla_AS_apoyo_seleccion",
    "EP_eventos_por_parametro": "tabla_EP_eventos_por_parametro",
    "DS_desenlaces_secundarios":"tabla_DS_desenlaces_secundarios",
    "RC_recodificaciones":      "tabla_RC_recodificaciones_espinal",
    "ME_modelo_explicativo":    "tabla_ME_modelo_explicativo",
    "VIF_explicativo":          "tabla_VIF_explicativo",
    "ES_estabilidad":           "tabla_ES_estabilidad_seleccion",
    "PM_desempeno":             "tabla_PM_desempeno",
    "PareadaMaestra":           "tabla_PareadaMaestra",
    "T3_predictores":           "tabla_T3_predictores_asociaciones",
    "T4_especificaciones":      "tabla_T4_especificaciones",
    "AX_especificaciones_esp":  "tabla_AX_especificaciones_espinal",
    "AX_hiperparametros":       "tabla_AX_hiperparametros",
    "AX_transportabilidad":     "tabla_AX_transportabilidad",
    "M0_espinal":               "tabla_M0_espinal",
    "MC_morfina_centro":        "tabla_MC_morfina_centro",
    # Compañera de MC_morfina_centro y su recíproca: aquella reporta exposición y desenlace
    # como dos frecuencias marginales del centro, esta mide el desenlace **dentro** de cada
    # estrato de exposición, que es lo que permite ver si la asociación sobrevive a la
    # estratificación o se invierte al agregar.
    "DE_desenlace_estrato":     "tabla_DE_desenlace_por_estrato_centro",
    "AN_variables_fonis":       "tabla_AN_variables_fonis",
}

GRIS_TENUE = "#8A9BA5"     # anotaciones y referencias
FIG_DPI = 300
TIPOGRAFIA = ("Lato", "DejaVu Sans")   # la primera disponible

# ── Modelamiento (docs/cleanse.md §4b y §4c) ─────────────────────────────────

# Validación cruzada anidada: el bucle externo estima el desempeño y el interno
# selecciona los hiperparámetros, de modo que la selección no vea los datos con
# que se evalúa.
CV_EXTERNO = 10
CV_INTERNO = 5
METRICA = "roc_auc"

BOOTSTRAP_B = 1000     # remuestras estratificadas para los intervalos
PERMUT_REPS = 30       # repeticiones de la importancia por permutación
EPV_MIN = 10           # eventos por variable: mínimo orientador

# Especificaciones del arco predictivo. La preoperatoria se restringe a la
# información disponible antes del procedimiento; la perioperatoria incorpora
# además el intra y el postoperatorio temprano.
ESPECIFICACIONES = ("preop", "periop")
ESCENARIOS = ("sin_centro", "con_centro")

# Grillas de hiperparámetros por algoritmo. La regresión logística clásica no
# tiene: omite el paso de selección.
GRILLAS = {
    "LR":         None,
    "LASSO":      {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "ElasticNet": {"C": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.2, 0.5, 0.8]},
    "RF":         {"n_estimators": [200, 400], "max_depth": [4, 6, None],
                   "min_samples_leaf": [10, 20]},
    "GBM":        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                   "max_depth": [3, 4]},
    "XGB":        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                   "max_depth": [3, 5]},
    "SVM":        {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
}

ALGORITMOS = tuple(GRILLAS)

# Etiquetas para tablas y figuras.
NOMBRE_ALGORITMO = {
    "LR": "Regresión logística", "LASSO": "Regresión logística LASSO",
    "ElasticNet": "Regresión logística ElasticNet", "RF": "Bosque aleatorio",
    "GBM": "Potenciación del gradiente", "XGB": "XGBoost",
    "SVM": "Máquina de soporte vectorial",
}
NOMBRE_ESPECIFICACION = {"preop": "Preoperatoria", "periop": "Perioperatoria"}

# Interpretabilidad: los valores SHAP se calculan sobre el bosque aleatorio en
# la especificación perioperatoria del análisis principal.
SHAP_ALGORITMO = "RF"

# Selección por estabilidad (arco explicativo): frecuencia con que cada variable
# sobrevive al remuestreo bajo penalización LASSO. El remuestreo es una validación
# cruzada estratificada repetida, y la penalización se elige por validación interna
# dentro de cada remuestra, de modo que la frecuencia no dependa de un C fijado a mano.
ESTABILIDAD_PARTICIONES = 5
ESTABILIDAD_REPETICIONES = 10
ESTABILIDAD_C = [10 ** x for x in
                 [-3 + 4 * i / 14 for i in range(15)]]   # 15 valores entre 1e-3 y 10
ESTABILIDAD_UMBRAL = 0.60          # frecuencia mínima para considerarla estable
ESTABILIDAD_UMBRAL_FUERTE = 0.75   # frecuencia que califica de estabilidad fuerte
ESTABILIDAD_SIGNO_MIN = 0.80       # consistencia mínima del signo del coeficiente

# Escalera de modelos anidados: desplazamiento del odds ratio que se considera indicio de
# confusión, en porcentaje respecto del modelo base.
CAMBIO_ESTIMACION_UMBRAL = 10.0
