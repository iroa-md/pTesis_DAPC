# Dolor agudo postcesárea · código del análisis

Tesis del Magíster en Informática Médica de la Facultad de Medicina de la Universidad de Chile.
Análisis secundario del proyecto **FONIS SA21I0036** 

---

## Este repositorio no contiene los datos del estudio

---

## Contenido

```
├── 01_cohorte.ipynb       construcción y caracterización de la cohorte
├── 02_modelamiento.ipynb  modelo de asociaciones y modelos predictivos
├── db/metadatos.xlsx      catálogo de las variables del estudio
└── utils/                 módulos con el código del análisis
```

### Los cuadernos

**`01_cohorte.ipynb`** importación de la base de datos de la cohorte FONIS: curación y tipado según el catálogo,
definición del desenlace, cribado de predictores, análisis de sesgo de selección, análisis descriptiov bivariado.

**`02_modelamiento.ipynb`** presenta el **análisis de asociaciones**, un modelo logístico
multivariable sobre 12 predictores, y el **predictivo**, que contrasta una especificación
preoperatoria de 16 variables contra una perioperatoria de 38, entrenando siete algoritmos bajo un
mismo esquema de validación cruzada anidada de diez pliegues externos y cinco internos. Cierra con
sensibilidad y robustez: consistencia entre operacionalizaciones del desenlace, subcohorte espinal,
incorporación del centro y transportabilidad entre hospitales.

### Los módulos

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

---

## Entorno

Python 3.10.16 en un entorno conda propio del proyecto, declarado en `environment.yml`:

```bash
conda env create -f environment.yml
conda activate DAPC
```

Las versiones fijadas son las que reporta el manuscrito: pandas 2.2.3, numpy 2.2.4, scipy 1.15.2, statsmodels 0.14.4, scikit-learn 1.6.1, xgboost 3.2.0, shap 0.47.2 y matplotlib 3.10.0.

## Ética

El proyecto FONIS SA21I0036 contó con la aprobación del comité de ética **OAIC N°04/2022** y con el consentimiento informado de las pacientes, declarado compatible con la realización de una tesis.

Este análisis se realizó sobre una base anonimizada, sin identificadores directos, bajo acceso restringido y sin contacto adicional con las participantes. Ninguna salida de este repositorio contiene información a nivel de paciente.

## Financiamiento

El estudio original fue financiado por el proyecto FONIS SA21I0036, adjudicado en la Agencia Nacional de Investigación y Desarrollo de Chile (ANID). El presente trabajo no contó con financiamiento adicional.

## Licencia

Este trabajo se publica bajo **Creative Commons Atribución 4.0 Internacional (CC BY 4.0)**. La licencia cubre el código y la documentación de este repositorio. **No alcanza a los datos del proyecto FONIS SA21I0036**, que no se distribuyen aquí y se rigen por sus propias condiciones de acceso.
