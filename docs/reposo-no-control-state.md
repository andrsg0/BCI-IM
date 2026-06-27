# El problema del reposo (*no-control state*) y cómo lo abordamos

> **Para quién es este documento:** para cualquiera, aunque no sepa nada de cerebros,
> señales ni programación. Empezamos desde lo más básico y vamos subiendo hasta el
> problema técnico y su solución. Si solo quieres la idea central, lee el «Resumen en
> tres frases» y la analogía del semáforo; el resto es el detalle.

---

## Resumen en tres frases

1. Nuestra interfaz cerebro-computadora (BCI) lee la actividad eléctrica del cerebro e
   intenta adivinar si la persona **imagina mover la mano izquierda o la derecha**.
2. Durante mucho tiempo la demo «hacía trampa»: sabía de antemano *en qué momento exacto*
   la persona estaba imaginando, y solo se fijaba en ese momento; un casco real **nunca**
   sabe eso, porque la señal llega sin avisar cuándo empieza ni termina.
3. Al quitar la trampa para que fuera realista, salió a la luz un problema clásico de las
   BCI —el **reposo no se distingue bien de la acción**— y añadimos una mejora parcial
   (un «detector de reposo») que lo reduce, aunque no lo elimina.

---

## 1. Lo más básico: ¿qué hace este proyecto?

Imagina un gorro con sensores que se apoya en el cuero cabelludo. Esos sensores
(**electrodos**) miden las pequeñísimas corrientes eléctricas que produce el cerebro: es
un **electroencefalograma (EEG)**. No «leen pensamientos»; solo registran ondas eléctricas,
como un micrófono registra sonido sin entender las palabras.

Resulta que cuando una persona **imagina que mueve una mano** (sin moverla de verdad), la
actividad eléctrica sobre cierta zona del cerebro cambia de una forma medible. Y —esto es
lo clave— cambia de forma **distinta** según imagine la mano **izquierda** o la **derecha**.

El objetivo del proyecto es construir el sistema que, mirando esas ondas, diga:

> «esta persona está imaginando la mano **izquierda**» … o «… la **derecha**».

Si eso funciona, podrías controlar algo (un cursor, una silla de ruedas, un videojuego)
solo con la imaginación. Esa es la promesa de una BCI.

### El clasificador: una máquina que solo conoce dos respuestas

El componente que toma la decisión se llama **clasificador**. El nuestro es **binario**:
solo sabe responder **«izquierda»** o **«derecha»**. No tiene una tercera respuesta como
«no está haciendo nada». Recuerda este detalle: es el origen de todo el problema que
explicamos más abajo.

---

## 2. Cómo decide «en vivo»: ventana a ventana

La señal del cerebro es un chorro continuo, como el agua de un grifo abierto. El sistema no
espera a tener «todo»; va cogiendo **trocitos** de señal de 2 segundos, uno tras otro,
solapados, y clasifica cada trocito. A cada trocito lo llamamos **ventana**.

```
señal continua:  ───────────────────────────────────────────►  (tiempo)
ventanas:        [──2s──]
                    [──2s──]
                       [──2s──]   ← cada una se clasifica por separado
                          [──2s──]
```

Por cada ventana, el clasificador no da un sí/no tajante, sino una **confianza**: por
ejemplo «80% izquierda, 20% derecha». Para no dar bandazos ventana a ventana, suavizamos
esas confianzas en el tiempo (una media que da más peso a lo reciente). Cuando la confianza
suavizada de una clase supera un **umbral** (un listón que tú puedes subir o bajar), el
sistema **se compromete** con esa respuesta. Si nunca llega al umbral, **se abstiene**.

Esto ya es razonable: «solo actúa si estás bastante seguro». El problema es **qué pasa
cuando la persona no está imaginando nada** (está en **reposo**). Y para entenderlo, antes
hay que contar la trampa que teníamos.

---

## 3. La trampa que teníamos (y por qué era trampa)

Los datos con los que probamos el sistema vienen de experimentos de laboratorio. En esos
experimentos, todo está cronometrado: a la persona se le muestra una flecha en un momento
**conocido** y se le pide imaginar la mano durante un intervalo **conocido**. Es decir,
en los datos sabemos *exactamente* el segundo en que empieza y termina la imaginación. A
ese intervalo lo llamábamos internamente `[alo, ahi]` (el inicio y el fin de la franja
activa).

La demo usaba ese dato para decidir: **solo contaba las ventanas que caían dentro de la
franja conocida** y descartaba el resto. Es como un examen tipo test en el que alguien te
sopla: «las preguntas que cuentan son de la 5 a la 12; el resto ni las mires».

**¿Por qué es trampa?** Porque un **casco real no tiene ese soplo**. Cuando alguien se pone
la BCI en casa, la señal llega sin etiquetas: nadie le dice al sistema «atención, AHORA la
persona va a imaginar». El sistema tiene que **descubrirlo solo**. Apoyarse en `[alo, ahi]`
hacía que la demo pareciera mejor de lo que sería en la vida real.

---

## 4. Quitar la trampa (lo que llamamos «B.2»)

Para que la demo fuera honesta hicimos dos cosas:

1. **Darle reposo de verdad.** Antes, la señal de la demo era *solo* el trozo en que la
   persona imaginaba (todo «acción», sin pausas). Ahora la demo **incluye los segundos de
   reposo previos**, cuando la persona aún no imagina nada. Así la señal se parece a la
   real: a veces hay intención, a veces no, y el sistema no sabe cuándo.

2. **Decidir de forma continua, sin mirar `[alo, ahi]`.** El sistema ya no usa la franja
   conocida para decidir. Clasifica todo el rato y se guía **solo** por su propia confianza
   y el umbral. La franja `[alo, ahi]` se sigue usando, pero **únicamente para dos cosas
   honestas**: (a) para *puntuar* a posteriori si acertó (necesitamos saber la verdad para
   calcular la precisión, igual que un profesor necesita el solucionario para corregir), y
   (b) para mover el muñeco demostrativo. **Nunca** para decidir.

Resultado: durante la imaginación, el sistema sigue acertando muy bien (~88–90% en el
sujeto de ejemplo). **Pero apareció un problema nuevo.**

---

## 5. El problema que destapó: el *no-control state* (estado de no-control)

Recuerda el detalle del principio: nuestro clasificador es **binario**, solo conoce
«izquierda» y «derecha». **No tiene una opción «reposo».**

Entonces, ¿qué hace cuando la persona está en reposo (sin imaginar nada)? No puede decir
«nada», porque esa respuesta no existe para él. Así que… **igualmente responde
«izquierda» o «derecha»**, y encima lo hace **con mucha seguridad**.

> **Analogía del adivino tramposo.** Imagina a alguien que presume de adivinar si tienes
> una moneda en la mano izquierda o la derecha. Le enseñas las dos manos **vacías** y, en
> vez de decir «no tienes ninguna moneda», te suelta con total aplomo: «¡la izquierda!».
> No tiene la opción «ninguna», así que se inventa una con seguridad. Eso es exactamente lo
> que hace nuestro clasificador en el reposo.

Lo medimos y es contundente: en el reposo, la «seguridad» media del sistema es **0.97**
(altísima), **incluso más alta que durante la imaginación real (0.92)**. Por eso subir el
umbral **no arregla** el reposo: el sistema está sobradísimo de confianza también cuando no
debería decir nada.

A esas decisiones equivocadas durante el reposo las llamamos **falsas alarmas**: el sistema
«dispara» una orden (mueve el cursor, etc.) cuando la persona no quería nada. En una BCI
real, las falsas alarmas son peligrosas (imagina una silla de ruedas que arranca sola).

Este problema tiene nombre propio en el campo de las BCI: el **no-control state** o
**idle state** (el estado en que el usuario *no* quiere controlar nada). Es uno de los
retos clásicos y difíciles del área. **La trampa de `[alo, ahi]` lo ocultaba**; al quitarla
para ser realistas, quedó a la vista. Que se vea no es un fallo: es el sistema mostrando un
límite **real**.

---

## 6. La solución parcial: un «detector de reposo» (lo que llamamos «B.4»)

Si el clasificador izquierda/derecha no sabe distinguir el reposo, le ponemos al lado un
**segundo** vigilante cuyo único trabajo es responder otra pregunta: **«¿esta persona está
imaginando algo, o está en reposo?»**. Es un **detector de reposo**.

### ¿En qué se fija el detector?

En la **potencia de las ondas** en una banda de frecuencias concreta (las llamadas ondas
**µ/β**, las que cambian con el movimiento imaginado). La intuición física:

> Cuando imaginas mover una mano, la potencia de esas ondas sobre la zona motora del
> cerebro **baja** (el cerebro «se pone en marcha» y esas ondas en reposo se atenúan; en la
> jerga se llama *desincronización*). En reposo, esas ondas están más «llenas». Esa
> diferencia de energía es la pista que usa el detector.

El detector es deliberadamente **simple y lineal** (una *regresión logística*), en línea con
el espíritu del proyecto (preferimos métodos transparentes y explicables a cajas negras).
Se **calibra** mostrándole ejemplos de ventanas de reposo y ventanas de imaginación **de
los datos de entrenamiento** (nunca de los datos reservados para la demo, para no hacer
trampa otra vez). Calibrarlo es instantáneo y no necesita tarjeta gráfica.

### ¿Cómo se usa? Una compuerta

El detector actúa como una **compuerta** (un guardia en la puerta). El sistema solo se
compromete con «izquierda/derecha» si **se cumplen dos condiciones a la vez**:

1. El clasificador está seguro (confianza ≥ umbral), **y**
2. el detector de reposo cree que **sí hay imaginación** (no es reposo).

Si el guardia dice «esto es reposo», la decisión se bloquea aunque el clasificador esté muy
seguro. En la interfaz es un **interruptor** («detector de reposo: ON/OFF»). Está **apagado
por defecto** a propósito: así puedes encenderlo y **ver con tus ojos** cómo bajan las
falsas alarmas.

> **Analogía del semáforo con sensor.** El clasificador izquierda/derecha es el semáforo:
> siempre te dice «ve por la izquierda» o «ve por la derecha». El detector de reposo es un
> sensor de presencia que primero comprueba **si hay alguien** que quiera cruzar. Si no hay
> nadie (reposo), el semáforo no se activa, por muy convencido que esté de su color.

---

## 7. Seamos honestos: cuánto ayuda y cuánto no

Esto es importante y está dicho sin maquillaje, tanto aquí como en la propia interfaz:

| Métrica (sujeto de ejemplo, 2a s1) | Sin detector | Con detector |
|---|---|---|
| Precisión durante la imaginación | 0.86 | 0.86 *(intacta)* |
| Falsas alarmas en reposo | muchas | **~45% menos** |
| Trials reales que se detectan | 134 de 141 | 104 de 141 *(pierde ~20%)* |

En una frase: **el detector reduce las falsas alarmas a casi la mitad, sin perder
precisión, pero a cambio se vuelve más tímido y deja escapar uno de cada cinco intentos
reales.** No es magia.

¿Por qué no es perfecto? Porque distinguir reposo de imaginación con estas señales es
**intrínsecamente difícil**: medimos su «separabilidad» y es modesta (en jerga, un AUC de
~0.71, donde 1.0 sería perfecto y 0.5 sería tirar una moneda). El reposo de laboratorio
(estar mirando una pantalla quieto) se parece bastante, eléctricamente, a imaginar algo de
forma tenue.

**¿Qué haría falta para resolverlo de verdad?** Un detector más potente y dedicado: por
ejemplo, un clasificador entrenado explícitamente con **tres** respuestas (reposo /
izquierda / derecha) en lugar de dos, o un detector de «novedad» que aprenda a fondo cómo es
el reposo. Eso es una mejora futura; el detector lineal actual es el primer escalón honesto.

---

## 8. Cómo probarlo tú mismo

1. Arranca el servidor y la interfaz (ver el README / `CLAUDE.md`).
2. Entra en la sección **«Clasificación»** y pulsa **Play**.
3. Observa el contador de **«falsas alarmas (reposo)»**: verás que sube durante las pausas.
4. Pulsa el interruptor **«detector de reposo»** para ponerlo en **ON**. Verás que las
   falsas alarmas bajan notablemente… y que de vez en cuando el sistema «se calla» también
   en algún intento real. Ese es justo el compromiso que explica la tabla de arriba.
5. Juega también con el **umbral de decisión**: comprobarás que subirlo **no** elimina las
   falsas alarmas (porque el problema no es de confianza, es de no tener clase «reposo»).

---

## 9. Dónde vive esto en el código (para quien programe)

- **Backend** (`backend/src/bci/server/app.py`):
  - `_get_demo_data()` — carga la señal de la demo con un trozo **más ancho** que incluye el
    reposo previo (sección B.2).
  - `_ensure_rest_detector()` — entrena y cachea el detector de reposo lineal (sección B.4).
  - `_make_sim()` — crea el simulador de streaming (un solo sitio, compartido).
  - `ws_stream()` — el canal en vivo (`/ws/stream`); añade `p_act` (la probabilidad de
    «activo» que da el detector) a cada mensaje.
- **Frontend** (`frontend/src/pages/LiveStream.tsx`): la decisión continua (media móvil +
  umbral), el interruptor del detector de reposo y los contadores de aciertos y falsas
  alarmas.
- **Contexto e historia**: `docs/roadmap.md` (entradas **B.2** y **B.4**).

---

## 10. Mini-glosario

| Término | En cristiano |
|---|---|
| **EEG** | Medir la electricidad del cerebro con sensores en la cabeza. |
| **Imaginación motora** | Imaginar un movimiento sin hacerlo; cambia el EEG de forma medible. |
| **Clasificador binario** | Programa que solo sabe responder una de **dos** opciones. |
| **Ventana** | Un trocito de señal (aquí, 2 segundos) que se clasifica de una vez. |
| **Confianza / umbral** | Cómo de seguro está el sistema, y el listón mínimo para actuar. |
| **Abstención** | Que el sistema decida **no** decidir (callarse). |
| **Reposo / no-control state** | El usuario no quiere controlar nada; no hay intención. |
| **Falsa alarma** | El sistema «dispara» una orden cuando el usuario no quería nada. |
| **Detector de reposo** | Vigilante extra que distingue «hay intención» de «reposo». |
| **Potencia µ/β** | La energía de unas ondas cerebrales que cambian al imaginar movimiento. |
| **AUC** | Nota de 0.5 (azar) a 1.0 (perfecto) de lo bien que separa dos cosas. |
