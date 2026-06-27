# Mejoras de UI

Listado de mejoras pequeñas a aplicar en la interfaz, organizadas por sección.
La idea es ir resolviéndolas fase por fase. Los puntos ambiguos se marcarán como
**(pendiente)** para decidirlos más adelante.

---

## Mejoras generales (aplican a todas las secciones)

- [ ] Quitar el texto descriptivo que va debajo del título de cada sección y moverlo a
  la zona de explicación (icono `?`). Aprovechar para redactar mejor esa explicación,
  una por una.
- [ ] Quitar los textos explicativos sueltos de estos paneles (su contenido debería vivir
  en la zona de explicación):
  - [ ] Mapas topográficos
  - [ ] Separabilidad de clases (espacio de características)
- [ ] Evaluar añadir una `x` (cerrar) en los widgets.

---

## Inicio

- [x] En el hero, hacer que el texto principal diga lo mismo que el logo del sitio
  ("BCI·MI") y mejorar el subtítulo siguiendo buenas prácticas de diseño de hero.
- [x] Revisar la utilidad de los botones del hero: quitar los que no aportan
  (relacionado con las buenas prácticas de hero). → Se mantienen los 2 CTA
  ("Explorar el laboratorio" primario + "Ver demo en vivo" secundario) por
  representar los dos mundos (offline/online); es el patrón estándar primario+secundario.
- [x] Quitar el botón de explicación: esta sección no necesita explicación.
- [x] Quitar el tag superior que dice "Interfaz Cerebro-Computadora · Imaginación Motora".

---

## Dashboard

- [x] En el menú de "añadir panel", alinear los tags con los colores de la barra superior:
  - [x] Tag "estático" → renombrar a "modelo" y usar color naranja.
  - [x] Tag "en vivo" → color verde.
- [x] Quitar el tag "detenido — pulsa play". → Ahora el badge solo aparece cuando está
  EN VIVO (en verde, antes rojo) y desaparece al estar detenido.

---

## Entrenamiento

- [x] Renombrar título: "Datos del dataset y preprocesamiento" → "Información del dataset
  y preprocesamiento".
- [x] Renombrar títulos de secciones:
  - [x] "Mapas topográficos (un filtro por componente)" → "Mapas topográficos"
  - [x] "Ecuación de proyección Z = W·X" → "Ecuación de proyección"
  - [x] "Cálculo: log-varianza por componente" → "Fórmula log-varianza" → N/A: ese título
    desapareció en el rewrite (la fórmula va dentro de la caja "Log-var (Características)").
  - [x] "Regla del clasificador y = w·F + b" → "Regla del clasificador" → N/A: ese título
    desapareció en el rewrite (la regla va dentro de la caja "LDA (Clasificación)").

### Filtro espacial (CSP)

- [x] Renombrar título: → "CSP (Filtro espacial)".
- [x] Documentar en algún lugar cómo se obtienen los autovalores del filtro topográfico.
  → Explicado en la caja CSP y en el `?` (problema de autovalores generalizados).
- [x] Mejorar la cronología de las subsecciones. → Resuelto con el rewrite: diagrama de
  recorrido (FIR → CSP → log-var → LDA) arriba y cajas en orden.
- [x] Cambiar textos de la barra selectora de dataset:
  - [x] "Datos (offline)" → borrar.
  - [x] "No depende del selector lateral (ese controla la demo en vivo)." → borrar.
  - [x] Unificar las etiquetas de Sujeto. → Unificado a números planos (el sidebar ya usa
    números): selector y ficha muestran "1, 2, …".
- [x] En la ecuación de proyección: añadir en `?` una explicación de cómo se obtuvo.
  → El `?` explica W (mapas de pesos) y el origen de λ.
- [x] Mejorar el gráfico "Señal: cruda vs filtrada por CSP":
  - [x] La señal cruda se ve demasiado opaca. → Trazo más oscuro (#94a3b8) y algo más grueso.
  - [x] Aclarar por qué se muestra ese electrodo. → Caption: "el de mayor peso en este
    componente" (por eso cambia entre componentes).
  - [x] (Mismo punto: por qué P7/F7/etc.) → Mismo motivo.
  - [x] Evaluar si este gráfico es relevante. → Se mantiene: muestra el efecto de limpieza del CSP.

### Características (log-var)

- [x] Renombrar título: → "Log-var (Características)".
- [x] En "Separabilidad de clases", explicar por qué el eje X usa el comp 0 y no el comp 1.
  → Caption: usamos los dos componentes más discriminativos (los extremos del espectro).
- [x] Revisar las etiquetas de los ejes (X "log-var comp 0", Y solo "comp N"). → Corregido:
  ambos ejes dicen "log-var comp N".

### Clasificación (LDA)

- [x] Renombrar título: → "LDA (Clasificación)".
- [x] Truncar los números de los ejes de la frontera de decisión. → `tickFormatter` a 1 decimal.
- [x] Revisar las etiquetas de los ejes. → Corregido igual que en log-var.
- [x] Aclarar la matriz de confusión / held-out / cuándo se obtiene. → Explicado en el `?`
  (validación honesta) y nota: las cifras se ven en Resultados.
- [x] Eliminar las métricas de entrenamiento y la matriz de confusión. → Caja "Validación
  (held-out)" eliminada (ValidationCards + ConfusionMatrix). Las métricas viven en Resultados.

> Pendiente (2ª pasada general): explicar conceptos como ventana de clasificación, comp CSP,
> shrinkage CSP y validación cruzada con enlaces desde el `?`. (Hoy ya hay hints en la ficha
> y auto-enlace al glosario.)

### EEGNet

- [x] Unificar el formato con el resto de la página. → Resuelto por el rewrite: se eligió la
  alternativa "sin widgets/canvas". Toda la sección "El Modelo" (CSP+LDA y EEGNet) usa el
  mismo formato de cajas en flujo cronológico.

---

## Resultados

- [ ] Cambiar textos:
  - [ ] "Demo en vivo (calibrar un día → probar otro)" → "Demo en vivo"
  - [ ] "Requisito: ≥ 2 sesiones (días distintos) para una estimación honesta inter-sesión."
    → borrar.
  - [ ] "Aparecen también en el benchmark de al lado (cada dataset es autosuficiente)."
    → borrar.
  - [ ] "Comparación general de métodos · toda la población" → "Comparación general de métodos"
  - [ ] "Resumen por dataset · CSP+LDA within-subject (media y rango)" → "Resumen por dataset"
  - [ ] "Comparación de métodos (2×2)" → "Comparación de métodos"
- [ ] Eliminar la sección "Benchmark de población (todos)": ya no aporta, porque todos los
  datasets se usan en el benchmark.
- [ ] En "Comparación general de métodos": extraer la métrica "Wilcoxon p" y darle más
  énfasis (es importante). Alternativa: llevarla al final, junto con más métricas de
  comparación y rendimiento.
- [ ] En "Resumen por dataset": quitar las etiquetas "Medido (2×2 completo)" y cambiar el
  texto "demo en vivo" por "demo".
- [ ] En "Comparación de métodos (2×2)": borrar la etiqueta "Medido (2×2)" y los datos que
  aparecen debajo. Mover todo eso a una sección aparte que reúna las métricas de
  comparación y rendimiento. Los datos a mover son:
  - Gini · variabilidad inter-sujeto
  - Azar = 50.0% · left_hand vs right_hand · T = 2s
  - Within: diferencia CSP+LDA vs EEGNet significativa (Wilcoxon p = 0.020, n = 9)
  - Cross: diferencia CSP+LDA vs EEGNet no significativa (Wilcoxon p = 0.496, n = 9)
- [ ] En el gráfico "Accuracy vs κ (kappa de Cohen)", la leyenda del eje X se solapa con la
  leyenda de colores de los puntos. Separarlas.
- [ ] Aclarar por qué solo el dataset 2a tiene "pooled" (¿es por tiempo de cómputo en el
  resto?). **(pendiente: definir)**.
- [ ] **Falta definir:** qué cambios hacer a la sección desplegable "cómo interpretar estos
  números" (¿debería ir en el icono `?`?). **(pendiente)**.

---

## Laboratorio

- [ ] Cambiar texto: "detenido — pulsa Play" → "detenido".

---

## Clasificación

- [ ] Cambiar textos:
  - [ ] "Modelo ya entrenado (antes del streaming)" → "Modelo"
  - [ ] "esperando primer trial…" → borrar.
- [ ] Cambiar el color del tag "Modelo" a azul.
- [ ] Eliminar el texto: "Los paneles ya están montados (el espacio CSP y la frontera LDA
  muestran el modelo entrenado). Pulsa Play en el panel lateral para que empiece a llegar
  la señal."
- [ ] Aclarar si el "umbral de decisión" es necesario aquí (el clasificador ya está
  entrenado). **(pendiente: revisar)**.
- [ ] **Falta definir:** qué textos de los gráficos se quieren eliminar y mover a la zona
  superior. Listar aquí los nombres de los gráficos afectados. **(pendiente)**.

---

## Demo en vivo

- [ ] Renombrar título a "Benchmark".
- [ ] Eliminar la sección "Benchmark de población (todos)".
- [ ] Añadir la matriz de confusión.
- [ ] Permitir ver también los resultados de cross, y de EEGNet (cross y within).
- [ ] Eliminar la sección "lectura honesta" y, si no está ya, mover algo equivalente al
  botón de explicación (`?`).
- [ ] **(pendiente: evaluar qué más añadir)**.

---

## Cerebro 3D

- [ ] Entender qué está pasando en esta sección y documentarlo. **(pendiente)**.

---

## Ideas

- [ ] Reutilizar el sistema de tags de la sección de Clasificación (me gusta) para explicar
  el pipeline mediante tags en lugar de un diagrama grande. Considerar implementarlo
  también en Laboratorio.
- [x] Reescribir la sección de Entrenamiento: en lugar de 3 subsecciones (CSP – log-var – LDA),
  unificarlas en una sola "CSP+LDA" con un formato parecido al de EEGNet (sin canvas ni
  widgets), con cronología clara de cómo se procesan los datos, intercalando gráficos,
  mapas topográficos y textos explicativos en orden. → Hecho: 2 pestañas (CSP+LDA · EEGNet),
  flujo en cajas `<Widget>` con diagrama de recorrido arriba, explicaciones concisas
  auto-enlazadas al glosario (`GlossaryText`) y un solo botón `?` por pestaña. Las mejoras
  de texto puntuales (renombres, quitar matriz de confusión, etc.) quedan para la 2ª pasada.
