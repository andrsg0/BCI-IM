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

- [ ] Renombrar título: "Datos del dataset y preprocesamiento" → "Información del dataset
  y preprocesamiento".
- [ ] Renombrar títulos de secciones:
  - [ ] "Mapas topográficos (un filtro por componente)" → "Mapas topográficos"
  - [ ] "Ecuación de proyección Z = W·X" → "Ecuación de proyección"
  - [ ] "Cálculo: log-varianza por componente" → "Fórmula log-varianza"
  - [ ] "Regla del clasificador y = w·F + b" → "Regla del clasificador"

### Filtro espacial (CSP)

- [ ] Renombrar título: → "CSP (Filtro espacial)".
- [ ] Documentar en algún lugar cómo se obtienen los autovalores del filtro topográfico.
- [ ] Mejorar la cronología de las subsecciones. Esta es la primera etapa del pipeline:
  partimos de muchas señales crudas y hay que explicar cómo se convierten en la matriz
  / mapa topográfico. Posible solución: un diagrama dentro de una sección desplegable,
  igual que "Información del dataset y preprocesamiento". **(pendiente: definir el
  enfoque antes de implementar)**.
- [ ] Cambiar textos de la barra selectora de dataset:
  - [ ] "Datos (offline)" → borrar.
  - [ ] "No depende del selector lateral (ese controla la demo en vivo)." → borrar.
  - [ ] Unificar las etiquetas de Sujeto: o bien cambiar "S1, S2, …" por "1, 2, …", o bien
    cambiar las etiquetas de la barra lateral ("1, 2, …") por "S1, S2, …".
- [ ] En la ecuación de proyección: el texto actual está bien, pero en la zona de `?` añadir
  una explicación de cómo se obtuvo.
- [ ] Mejorar el gráfico "Señal: cruda vs filtrada por CSP":
  - [ ] La señal cruda se ve demasiado opaca.
  - [ ] Aclarar por qué solo se muestra un electrodo crudo (por ejemplo F7) y por qué ese en concreto.
  - [ ] Aclarar por qué se elige un electrodo específico (P7) para mostrar.
  - [ ] Evaluar si este gráfico es relevante en esta sección.

### Características (log-var)

- [ ] Renombrar título: → "Log-var (Características)".
- [ ] En el gráfico "Separabilidad de clases (espacio de características)", explicar por qué
  el eje X usa solo el componente 0 y no también el componente 1.
- [ ] Revisar las etiquetas de los ejes: en X dice "log-var comp 0" y en Y solo "comp 3".
  Corregir si es inconsistencia.

### Clasificación (LDA)

- [ ] Renombrar título: → "LDA (Clasificación)".
- [ ] En el gráfico "Frontera de decisión sobre el espacio de características", los números
  de los ejes son demasiado largos: truncarlos para que se vean mejor.
- [ ] Revisar las etiquetas de los ejes: en X dice "log-var comp 0" y en Y solo "comp 3".
  Corregir si es inconsistencia.
- [ ] Aclarar la presencia de la matriz de confusión en el LDA: ¿se prueban los datos al
  finalizar para reportar los resultados del entrenamiento? ¿Qué es el "held out"? Esto
  se relaciona con la cronología: debería explicarse mejor cómo y cuándo se obtienen
  estos resultados.
- [ ] Eliminar las métricas de entrenamiento y la matriz de confusión.

> Explicar los conceptos: ventana de clasificación, componentes CSP, shrinkage CSP y
> validación cruzada. Posible solución: un icono `?` que lleve a la zona específica de
> la explicación principal donde se describe el concepto de cada gráfico.

### EEGNet

- [ ] Unificar el formato con el resto de la página: poner el canvas y convertir los gráficos
  en widgets como en las demás secciones. Alternativa: quitar el formato de widgets de
  toda la sección "El Modelo" y dejar páginas normales, lo que quizás encaje mejor con la
  cronología. **(pendiente: decidir entre ambas opciones)**.

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
