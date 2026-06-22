# Entrenamiento del modelo cross-subject (EEGNet pooled)

Guía operativa para **reentrenar el modelo con muchos datos** y alcanzar el máximo
rendimiento honesto posible, **sin que el proceso consuma tokens de Claude**: los
comandos se ejecutan en *tu terminal*, no a través del asistente.

> **Resumen en una frase:** entrenar un solo EEGNet con **muchos sujetos juntos**
> (PhysioNet, ~105 sujetos) + **aumentación de datos**, evaluarlo de forma honesta con
> **LOSO** (leave-one-subject-out) y guardar el modelo base para un futuro *fine-tuning*.

---

## 1. Qué se entrena y por qué

Hay dos regímenes de clasificación en el proyecto:

| Régimen | Qué mide | Quién lo cubre |
|---|---|---|
| **within-subject** | el modelo se entrena y evalúa en el MISMO sujeto (calibrado) | CSP+LDA (sujeto-específico) y EEGNet |
| **cross-subject (LOSO)** | el modelo se entrena con OTROS sujetos y se evalúa en uno NUEVO sin calibrar | **EEGNet pooled** (este documento) |

CSP+LDA es **sujeto-específico** por diseño (sus filtros espaciales se ajustan a la
anatomía de cada persona), así que generaliza mal a un usuario nuevo. La pregunta que
responde este entrenamiento es: **¿puede una red generalizar a alguien que no vio?**

El script entrena **una sola red** con todos los sujetos juntos (*pooled*) y reporta:

- **LOSO**: para cada sujeto evaluado, se entrena con TODOS los demás y se prueba en él.
  Es la estimación honesta de "ponérselo a un usuario nuevo sin calibrar".
- **Modelo base**: el modelo final se entrena con TODOS los sujetos. Es la base de la
  que partiría un *fine-tuning* con una calibración corta (siguiente fase, ver §6).

## 2. Filosofía de datos: "bastantes datos para el máximo"

> **Realidad (techo del estado del arte):** imaginación motora de 2 clases, no invasiva,
> ~**70–85 %**. Objetivos honestos: cross-subject **sin calibrar** ≈ 0.65–0.70 (NO 0.80);
> con calibración/fine-tuning ≈ 0.75–0.85 en buenos sujetos. Siempre se reporta el **rango
> entre sujetos**, no solo la media.

Lo que más mueve la aguja en cross-subject es **MÁS SUJETOS**, no más trials por sujeto:

- **BCI IV 2a** (`default.yaml`): 9 sujetos, 22 canales, 250 Hz. → LOSO ≈ **0.60**.
- **PhysioNet MI** (`physionet.yaml`): **109 sujetos**, 64 canales, 160 Hz. → es el gran salto.

Además aplicamos **aumentación de datos** (`--augment`), que crea variantes plausibles de
cada trial (ruido leve, desplazamiento temporal, escalado de amplitud) **solo en la
partición de entrenamiento** (nunca en test, para no falsear la precisión). Código:
`backend/src/bci/datasets/augment.py`.

## 3. El comando "máximo" (PhysioNet, ~105 sujetos)

Todo se corre **desde `backend/` con el entorno activado**:

```bash
cd backend
source .venv/bin/activate
```

### Paso 0 — (opcional) descargar los datos primero

La primera carga descarga ~**3 GB** a la caché de MNE. El propio entrenamiento lo descarga
solo si falta, pero conviene hacerlo aparte para separar "descarga" de "entrenamiento":

```zsh
# Excluimos 88, 89, 92, 100: en PhysioNet tienen fs/anotaciones inconsistentes.
# OJO (zsh): se usa un ARRAY; con un string "1 2 3..." zsh NO lo separa en argumentos.
SUBS=($(seq 1 109 | grep -vxE '88|89|92|100'))
python scripts/download_data.py --config ../configs/physionet.yaml --subjects $SUBS --no-validate
```
*(En bash sería `--subjects "${SUBS[@]}"`. En zsh `$SUBS` ya expande todos los elementos.)*

### Paso 1 — lanzar el entrenamiento en segundo plano

Usamos `nohup ... &` para que siga aunque cierres la terminal, y redirigimos a un log:

```zsh
SUBS=($(seq 1 109 | grep -vxE '88|89|92|100'))   # array (ver nota zsh arriba)
nohup python scripts/train_eegnet_pooled.py \
    --config ../configs/physionet.yaml \
    --subjects $SUBS \
    --epochs 300 \
    --augment \
    --loso-subset 25 \
    > /tmp/train_pooled_physionet.log 2>&1 &
```

Y para ver el progreso (Ctrl-C solo deja de mirar el log, NO detiene el entrenamiento):

```bash
tail -f /tmp/train_pooled_physionet.log
```

### Qué hace cada flag

| Flag | Significado |
|---|---|
| `--config ../configs/physionet.yaml` | dataset de 109 sujetos (el que da el salto cross-subject) |
| `--subjects $SUBS` | sujetos a agrupar (todos menos los inconsistentes) |
| `--epochs 300` | épocas de entrenamiento por red |
| `--augment` | aumentación de datos (triplica el set de entrenamiento) |
| `--loso-subset 25` | el **LOSO** evalúa solo los primeros 25 sujetos (cada uno entrenado con TODOS los demás). El **modelo base final usa los ~105**. Evita esperar horas evaluando los 109. |
| `--device cuda\|cpu` | opcional; por defecto detecta GPU automáticamente |
| `--no-save` | opcional; NO guarda (para pruebas, no pisa un modelo bueno) |

## 4. Tiempos y recursos

- Se entrena en **GPU** automáticamente si hay CUDA (la hay en este equipo).
- Coste ≈ (`--loso-subset` + 1) entrenamientos completos. Con `--loso-subset 25` son **26
  redes** sobre ~105 sujetos × aumentación → del orden de **1–3 horas** en GPU.
- Para una prueba rápida antes del run largo (minutos): pocos sujetos y pocas épocas:
  ```bash
  python scripts/train_eegnet_pooled.py --config ../configs/physionet.yaml \
      --subjects 1 2 3 4 5 --epochs 50 --augment --loso-subset 3 --no-save
  ```
- Para subir más el LOSO (más lento pero más representativo): `--loso-subset 40`.

## 5. Qué produce y cómo se usa

Al terminar guarda en `backend/data/processed/`:

- `model_PhysionetMI_s0_eegnet_pooled.pkl` — el modelo base (red entrenada con todos).
- `model_PhysionetMI_s0_eegnet_pooled.json` — la **ficha** (`ModelCard`), que incluye en
  `extra`: `loso_per_subject`, `loso_mean`, `n_subjects`, `augment`, `epochs`, `device`.

El número clave es `loso_mean` (cross-subject honesto). El log también lo imprime al final:

```
  -> media LOSO = 0.6xx
```

Estos artefactos son los que el frontend mostrará en la futura **página de comparación
estadística** (accuracy + κ por sujeto/método). El `s0` indica "pooled" (no pertenece a un
sujeto concreto).

> **Nota:** este modelo NO se usa en la demo en vivo (la inferencia en vivo sigue siendo
> CSP+LDA sujeto-específico, según `CLAUDE.md`). Es para la comparación cross-subject y como
> base del fine-tuning.

## 6. Siguiente fase (aún no implementada): fine-tuning con calibración

El paso que de verdad lleva a un usuario concreto hacia 0.75–0.85 es el *transfer learning*:
partir del modelo base pooled, **congelar las capas tempranas** (el banco de filtros FIR/CSP
aprendido) y reentrenar solo las últimas con **20–40 trials** del usuario nuevo (1–2 runs de
calibración). Está descrito como "Paso 4" en `docs/roadmap.md`; cuando se implemente se añadirá
aquí el comando correspondiente.

## 7. Referencia rápida de otros entrenamientos

```bash
# CSP+LDA + EEGNet sujeto-específicos (within-subject), para la demo y visualización:
python scripts/train_model.py --config ../configs/default.yaml --subjects 1 3 8
python scripts/train_eegnet.py --config ../configs/default.yaml --subjects 1

# Comparación 2×2 (CSP+LDA vs EEGNet × within vs cross), genera CSV:
python scripts/compare_methods.py --config ../configs/default.yaml --subjects 1 2 3 4 5 6 7 8 9

# Pooled cross-subject en el 2a (rápido, 9 sujetos) — la línea base ya medida (LOSO ≈ 0.60):
python scripts/train_eegnet_pooled.py --config ../configs/default.yaml --subjects 1 2 3 4 5 6 7 8 9 --epochs 250

# Pooled cross-subject en Dreyer2023 (el dataset homogéneo MÁS GRANDE: 87 sujetos):
# {1..87} = expansión de llaves (zsh y bash la separan en argumentos; un string NO en zsh).
nohup python scripts/train_eegnet_pooled.py --config ../configs/dreyer2023.yaml \
    --subjects {1..87} --epochs 300 --augment --loso-subset 25 \
    > /tmp/train_pooled_dreyer.log 2>&1 &

# Cross-dataset (PhysioNet + Dreyer + Cho, remuestreo a fs común) — ver docs/remuestreo.md:
python scripts/train_eegnet_crossdataset.py --max-subjects 20 --epochs 300 --augment
```

> **Dos tipos de pooled:** dentro de UN dataset (`train_eegnet_pooled.py`, sin remuestreo) y
> ENTRE datasets (`train_eegnet_crossdataset.py`, con remuestreo a fs común + armonización de
> canales). El segundo es el **Paso 3** del roadmap; su proceso y parámetros están en
> `docs/remuestreo.md`.

---

**Archivos relevantes:** `backend/scripts/train_eegnet_pooled.py` (CLI),
`backend/src/bci/pipeline/training.py` (`train_eegnet_pooled`, `_eegnet_features`),
`backend/src/bci/datasets/augment.py` (aumentación),
`backend/src/bci/models/eegnet.py` (red + `pick_device`).
