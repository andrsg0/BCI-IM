# Datasets: criterios, usos y plan de adquisición

Decisión registrada sobre **qué datasets usamos, para qué, y por qué**. Complementa a
`docs/entrenamiento.md` (cómo se entrena) y al roadmap (qué falta).

> **Idea central:** hay **dos usos con requisitos opuestos**. No existe "el mejor dataset":
> existe el mejor dataset *para cada uso*. Por eso un mismo criterio (p. ej. "muchos sujetos")
> sirve para entrenar pero es irrelevante para la demo en vivo.

---

## 1. Los dos usos y sus requisitos

| Uso | Qué necesita | Por qué |
|---|---|---|
| **ENTRENAR** (pool cross-subject) | **muchos sujetos diversos**, buena calidad | lo que más sube el cross-subject es MÁS SUJETOS, no más trials por sujeto |
| **EN VIVO** (calibrar ≠ probar) | **≥ 2 sesiones** (días distintos), grabaciones largas, alta calidad | calibras en una sesión y transmites OTRA → estimación honesta de uso real |

## 2. Cómo se elige un dataset (no basta con ordenar por nº de sujetos)

El catálogo de MOABB (la web de datasets) tiene columnas que mapean casi 1:1 con nuestros
criterios. El procedimiento:

1. **Filtrar por `Imagery`** (paradigma correcto) — obligatorio. Descarta SSVEP, P300/ERP.
2. **Leer las columnas según el uso:**

   | Uso | Columnas clave |
   |---|---|
   | Entrenar | **#SUBJ** alto · **#CLASSES** = 2 (o subset a izq./der.) · **HEALTH** = healthy · **#CHAN** con C3/Cz/C4 · **#TRIALS** decente |
   | En vivo | **#SESS ≥ 2** · **TRIAL(S)** largo · **HEALTH** = healthy |

3. **Verificar lo que la tabla NO dice:** que las clases sean mapeables a **mano izq./der.**
   (nuestro pipeline es binario L/R). Se comprueba con:
   ```python
   import moabb.datasets as mds
   ds = mds.Dreyer2023()
   print(ds.event_id, ds.n_sessions)
   ```
4. **Que sea cargable** vía `moabb.datasets.<Nombre>` (todos los de abajo lo son en moabb 1.5.0).

### Distinción crítica: `#SESS` vs `#RUNS`

- **`#SESS`** = sesiones en **días distintos** → es lo que da el "otro día" honesto para EN VIVO.
- **`#RUNS`** = bloques dentro de la MISMA sesión (mismo día) → útil, pero menos honesto que
  cambiar de día.

Para la demo en vivo priorizamos `#SESS ≥ 2`, no `#RUNS`.

### Criterios de CALIDAD (cuándo un dataset "sirve")

- Canales motores presentes (al menos C3/Cz/C4).
- `fs` conocido y consistente; clases balanceadas; bajo artefacto; sujetos `healthy`.
- Paradigma MI claro basado en *cue* (no continuo sin etiquetas).
- 2-clases mapeable a mano izq./der. (o subseteable desde un set multiclase).
- Licencia pública / acceso vía MOABB.

## 3. Datasets integrados hoy

| Dataset | #SUBJ | #SESS | #CHAN | fs | Estado / uso |
|---|---|---|---|---|---|
| **BCI IV 2a** (`BNCI2014_001`) | 9 | **2** (días distintos) | 22 | 250 | ✅ en vivo (calibrar→probar) + entrenar |
| **PhysioNet MMI** (`PhysionetMI`) | 109 | 1 | 64 | 160 | ✅ entrenar (pool). ❌ mal para en vivo (1 sesión, ~45 trials L/R) |
| **Liu2024** | 50 | 1 | 29 | 500 | ⚠️ calidad baja en nuestras medidas (~0.54), `patients`. Candidato a descartar |
| **Dreyer2023** | 87 | 1 | 27 | 512 | ✅ **integrado** (config + REGISTRY). Entrenar (pool). k-fold suj.1 = **0.696** |
| **Cho2017** | 52 | 1 | 64 | 512 | ✅ **integrado** — reemplaza a Lee2019_MI. Entrenar (pool). k-fold suj.1 = **0.755** |

> Integrar un dataset = `configs/<nombre>.yaml` + entrada en `REGISTRY` (`server/app.py`) +
> cargable vía `moabb.datasets`. Dreyer2023 y Cho2017 cumplen los tres.

### 3.1 Por qué DESCARTAMOS Lee2019_MI

Lee2019_MI prometía mucho (54 sujetos × 2 sesiones), pero al integrarlo se descartó por:

1. **Pesa muchísimo:** ~609 MB por sesión → **~1.2 GB por sujeto** (los 54 = decenas de GB).
   Eso obliga a usar un subconjunto pequeño de sujetos.
2. **Su 2ª sesión NO es accesible:** `get_data()` en MOABB 1.5.0 devuelve solo la sesión `1`
   (verificado pasando `sessions=[1,2]`, `train_run`/`test_run`, y con ambos `.mat` en disco —
   siempre 1 sesión). Encima el `test_run` de Lee **no tiene etiquetas** (inservible para
   clasificar, según la propia descripción del dataset).
3. **El argumento decisivo:** su única ventaja era *muchos sujetos* (y *2 sesiones*); pero por
   el tamaño solo podríamos usar pocos sujetos, y la 2ª sesión no carga. Pocos sujetos = lo
   mismo que un dataset pequeño → no aporta nada que no tengamos ya. **Descartado.**

Lo reemplaza **Cho2017** (52 sujetos, 2-clases nativas, 64 canales, mucho más ligero).

> Las accuracies de §3 son **provisionales (1 sujeto)**. La media real se mide con
> `scripts/evaluate_all.py --config ../configs/<nombre>.yaml`.

## 4. Candidatos verificados (clases y sesiones REALES)

Instanciados con `event_id`/`n_sessions` para confirmar lo que la tabla no muestra:

| Dataset | #SUBJ | #SESS | Clases reales | Veredicto |
|---|---|---|---|---|
| **Dreyer2023** | 87 | 1 | `left_hand / right_hand` ✓ | ✅ **integrado** — 2-clases nativo, **240 trials/sujeto** (vs ~45 de PhysioNet) |
| **Cho2017** | 52 | 1 | `left_hand / right_hand` ✓ | ✅ **integrado** — ligero, k-fold suj.1 = 0.755 |
| **Lee2019_MI** | 54 | 2* | `left_hand / right_hand` ✓ | ❌ **descartado** — pesa ~1.2 GB/suj y la 2ª sesión no carga (§3.1) |
| **Stieger2021** | 62 | **11** | 4-clases (incluye L/R) | **En vivo** longitudinal (subset a L/R) |
| **Zhou2020** | 20 | **7** | 4-clases (incluye L/R) | En vivo alternativo (subset) |
| **Ma2020** | 25 | 15 | `right_hand / right_elbow` ✗ | ❌ **descartar** — no es izq./der., es otro paradigma |

> **Lección:** Ma2020 parecía perfecto para en vivo por sus 15 sesiones, pero sus clases son
> *mano vs codo* — no encaja en el pipeline binario izq./der. Por eso el paso 3 (verificar
> clases) es obligatorio: la tabla de MOABB **no** dice qué son las clases concretas.

## 5. Plan de adquisición propuesto

### Pool de ENTRENAMIENTO (cross-subject)

2-clases L/R nativo, sano, muchos sujetos:

- PhysionetMI (109) — *ya integrado*
- **Dreyer2023 (87)** — *ya integrado*, ligero, muchísimos trials/sujeto
- **Cho2017 (52)** — *ya integrado*, ligero, k-fold suj.1 = 0.755
- → **~248 sujetos** disponibles para el pool cross-subject
- (~~Lee2019_MI~~ descartado, §3.1)

### Pool de EN VIVO (calibrar día 1 → probar día 2)

- BCI IV 2a (2 sess) — *ya integrado* (la opción sólida hoy)
- Stieger2021 (11 sess, subset a L/R) — candidato fuerte para longitudinal (pendiente de evaluar tamaño)

### Caveat de Sistemas Lineales y Señales (Paso 3 del roadmap)

Mezclar estos datasets en un solo pool requiere **remuestrear a un `fs` común**, porque
difieren: 160 / 250 / 512 / 1000 Hz. Eso es interpolación/diezmado + **filtro anti-aliasing**,
más armonización de montajes (subconjunto de canales motores común). Es justamente el
contenido de la asignatura y el motivo de dejar el cross-dataset pooling "al final".

## 6. Por dónde seguir (integración)

Para meter un dataset nuevo al proyecto hacen falta tres cosas (ver `CLAUDE.md`):

1. Un **YAML** en `configs/` (copiar `physionet.yaml`/`dreyer2023.yaml` y ajustar `name`,
   `classes`, ventana de epoch según el `interval` real del dataset).
2. Una entrada en **`REGISTRY`** (`backend/src/bci/server/app.py`).
3. Que sea cargable vía `moabb.datasets.<Nombre>`.

Hecho: **Dreyer2023** (✅) y **Cho2017** (✅). Lee2019_MI **descartado** (§3.1). Próximos:

- **Medir la media real** con `evaluate_all.py` (las accuracies de §3 son de 1 sujeto).
- **Entrenar el pooled** con el pool ampliado (PhysioNet + Dreyer2023 + Cho2017) — pero esto
  es **Paso 3** (cross-dataset): difieren en fs (160/512) y montaje, hay que remuestrear a un
  fs común + anti-aliasing antes de juntarlos (ver §5 y roadmap).
- **Frontend:** los nuevos NO están en el selector de la UI (son datasets de *entrenamiento*,
  sin modelos de demo por sujeto). Exponerlos requeriría medir accuracy + entrenar modelos.

---

**Refs:** `docs/entrenamiento.md` (cómo entrenar el pooled), `docs/roadmap.md` (Paso 2/3),
`backend/src/bci/datasets/moabb_loader.py` (loader), `configs/` (YAMLs por dataset).
