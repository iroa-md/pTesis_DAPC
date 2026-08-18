# Dolor agudo postcesárea · código del análisis

Código del análisis secundario del proyecto **FONIS SA21I0036**, desarrollado como tesis del
Magíster en Informática Médica de la Facultad de Medicina de la Universidad de Chile.

El trabajo estudia, sobre 714 pacientes de cuatro hospitales públicos chilenos reclutadas entre 2022
y 2024, qué factores perioperatorios se asocian al dolor moderado a intenso durante la primera semana
postcesárea, y si la información intraoperatoria y postoperatoria temprana mejora su predicción
frente a la exclusivamente preoperatoria.

El desenlace es una variable binaria compuesta: una intensidad de 4 o más en la escala numérica del
dolor (END) en al menos una de las mediciones de 24 horas, 48 horas y séptimo día.

---

## ⚠ Este repositorio no distribuye datos

**No incluye la base del proyecto FONIS ni ningún archivo derivado de ella.** Quedan fuera la base
original, las cohortes analíticas en parquet, las predicciones fuera de pliegue y los valores SHAP,
porque todos ellos están a nivel de paciente.

En consecuencia, **los cuadernos no son ejecutables de extremo a extremo** sin acceso autorizado a los
datos del proyecto original. Eso es deliberado y no un defecto del repositorio.

Lo que sí se publica es **el análisis completo y su resultado**: los cuadernos conservan sus salidas
ejecutadas, de modo que cada figura, cada tabla y cada cifra del manuscrito pueden leerse y
contrastarse aquí sin ejecutar nada.

---

## Contenido

```
├── 01_cohorte.ipynb       construcción y caracterización de la cohorte
├── 02_modelamiento.ipynb  modelo de asociaciones y arco predictivo
├── db/metadatos.xlsx      catálogo de las 144 variables del estudio
└── utils/                 los ocho módulos del análisis
```

### Los cuadernos

**`01_cohorte.ipynb`** va de la ingesta a la cohorte analítica: curación y tipado según el catálogo,
definición del desenlace, cribado de predictores, sesgo de selección, análisis bivariado, frecuencia
del desenlace y heterogeneidad entre centros.

**`02_modelamiento.ipynb`** cubre los dos arcos. El **de asociaciones**, un modelo logístico
multivariable sobre 12 predictores, y el **predictivo**, que contrasta una especificación
preoperatoria de 16 variables contra una perioperatoria de 38, entrenando siete algoritmos bajo un
mismo esquema de validación cruzada anidada de diez pliegues externos y cinco internos. Cierra con
sensibilidad y robustez: consistencia entre operacionalizaciones del desenlace, subcohorte espinal,
incorporación del centro y transportabilidad entre hospitales.

Cada sección construye sus propias figuras y tablas, junto al análisis que las produce.

### Los módulos

Ninguno accede a los datos por su cuenta: reciben marcos ya construidos y devuelven agregados.

| Módulo | Qué hace |
|---|---|
| `config` | rutas, constantes, umbrales y el contrato del catálogo |
| `daten` | ingesta, curación, cohorte, cribado y congelado de artefactos |
| `statistik` | pruebas, medidas de efecto, SMD y heterogeneidad |
| `modell` | ajuste, especificaciones, validación cruzada anidada y estabilidad |
| `evaluar` | desempeño, remuestreo, comparación pareada, permutación, SHAP y transportabilidad |
| `vista` | agregados de presentación, entre el análisis y la salida |
| `rendern` | figuras, en PNG y SVG |
| `tabula` | tablas, en `.docx` y en fragmento LaTeX desde una sola fuente |

`modell` no contiene ningún nombre de variable del estudio: su único acoplamiento son las constantes
de `config`. Para reusarlo en otra cohorte se cambian el catálogo y `config`, no el módulo.

---

## Entorno

Python 3.10.16 en un entorno conda propio del proyecto, declarado en `environment.yml`:

```bash
conda env create -f environment.yml
conda activate DAPC
```

Las versiones fijadas son las que reporta el manuscrito: pandas 2.2.3, numpy 2.2.4, scipy 1.15.2,
statsmodels 0.14.4, scikit-learn 1.6.1, xgboost 3.2.0, shap 0.47.2 y matplotlib 3.10.0.

Los cuadernos importan los módulos desde la raíz del repositorio, de modo que deben ejecutarse desde
ella. `rendern` fija el backend `Agg` al importarse, para que los módulos exporten figuras sin
pantalla, y los cuadernos lo revierten con `%matplotlib inline` justo después del import para poder
verlas en línea.

## Reproducibilidad

- Semilla única declarada en remuestreos, particiones y ajuste.
- Todo el preprocesamiento ocurre **dentro** del pliegue de entrenamiento de cada partición, de modo
  que la selección de hiperparámetros no vea los datos con que se evalúa.
- Los cálculos largos se persisten, de modo que una interrupción no obligue a repetir lo anterior.
- El catálogo se valida contra un contrato de columnas al cargarse, y `02` cierra con un **contrato de
  reproducibilidad**: si el n, los eventos, los términos del modelo, la composición de las
  especificaciones o las combinaciones de la validación cruzada dejan de coincidir con lo publicado,
  el cuaderno falla en vez de emitir cifras distintas en silencio.

## Ética

El proyecto FONIS SA21I0036 contó con la aprobación del comité de ética **OAIC N°04/2022** y con el
consentimiento informado de las pacientes, declarado compatible con la realización de una tesis.

Este análisis se realizó sobre una base anonimizada, sin identificadores directos, bajo acceso
restringido y sin contacto adicional con las participantes. Ninguna salida de este repositorio
contiene información a nivel de paciente.

## Financiamiento

El estudio original fue financiado por el proyecto FONIS SA21I0036, adjudicado en la Agencia Nacional
de Investigación y Desarrollo de Chile (ANID). El presente trabajo no contó con financiamiento
adicional.

## Licencia

Este trabajo se publica bajo **Creative Commons Atribución 4.0 Internacional (CC BY 4.0)**.

Se permite compartir y adaptar el material, para cualquier finalidad incluso comercial, siempre que
se dé el crédito correspondiente, se enlace la licencia y se indique si hubo cambios. El texto legal
completo está en [`LICENSE`](LICENSE), y su resumen legible en
<https://creativecommons.org/licenses/by/4.0/deed.es>.

La licencia cubre el código y la documentación de este repositorio. **No alcanza a los datos del
proyecto FONIS SA21I0036**, que no se distribuyen aquí y se rigen por sus propias condiciones de
acceso.
