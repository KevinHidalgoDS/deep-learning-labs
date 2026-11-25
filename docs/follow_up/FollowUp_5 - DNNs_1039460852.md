---
title: "FollowUp 5"
author: "_Kevin Ferney Hidalgo Higuita_"
date: "20 de Noviembre de 2026"
institute: "Universidad Nacional de Colombia, Sede Medellín"
description: ""

# --- Configuraciones para Pandoc (PDF) ---
lang: es-CO
bibliography: docs/referencias.bib
csl: docs/ieee.csl

# Formato visual del PDF (LaTeX)
geometry:
  - top=2.5cm
  - bottom=2.5cm
  - left=3cm
  - right=2.5cm
fontsize: 12pt
linestretch: 1.15
papersize: letter
toc: true
toc-depth: 2
---

<hr style="border:2px solid #CC00FF;">

<div style="display:flex; justify-content:center; align-items:center; gap:25px; flex-wrap:nowrap; margin:5px 0;">
    <a href="https://medellin.unal.edu.co" target="_blank">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR9ItLTT_F-3Q30cu7ZCCoKmuFGBt22pe7pNA"
             style="height:60px; width:auto; max-width:150px; object-fit:contain;" alt="Logo UNAL">
    </a>
    <a href="https://minaslap.net/?redirect=0" target="_blank">
        <img src="https://minaslap.net/pluginfile.php/1/core_admin/logo/0x200/1770226238/Logo%20MinasLAP-3%20%281%29.png"
             style="height:60px; width:auto; max-width:180px; object-fit:contain;" alt="Logo MinasLAP">
    </a>
    <a href="https://micampus.unal.edu.co" target="_blank">
        <img src="https://micampus.unal.edu.co/pluginfile.php/1/theme_almondb/sliderimage1/1788871931/Slider_1.png"
             style="height:60px; width:auto; max-width:200px; object-fit:contain;" alt="Logo MiCampus">
    </a>
</div>

<h2 align="center" style="font-size:1.4em; margin:6px 0;">Deep Learning 2026-2S</h2>
<h3 align="center" style="font-size:1.05em; margin:4px 0;">Follow Up 5 </h3>
<h4 align="center" style="font-size:0.9em; margin:4px 0; color:#555;">Semana 5</h4>

<p align="center" style="font-size:0.82em; line-height:1.3; margin:6px 0;">
Departamento de Ciencias de la Computación y de la Decisión<br>
Facultad de Minas · Universidad Nacional de Colombia<br>
Docente: John R. Ballesteros
</p>

<hr style="border:1px solid #CC00FF; margin:8px 0;">

<p style="font-size:0.78em; line-height:1.35; margin:4px 0;"><b>Equipo de trabajo:</b></p>

<table style="width:100%; font-size:0.72em; line-height:1.25; border-collapse:collapse;">
  <tr>
    <td style="padding:2px 6px;">• Kevin F. Hidalgo Higuita</td>
    <td style="padding:2px 6px;">📧 <i>kfhidalgoh@unal.edu.co</i></td>
  </tr>
</table>

<p align="center" style="font-size:0.75em; margin:8px 0 0 0;">
📅 20 de Noviembre de 2026 ·
<a href="https://github.com/KevinHidalgoDS/deep-learning-labs">
  <img src="https://img.shields.io/badge/repository-GitHub-181717.svg" alt="GitHub Repo" style="vertical-align:middle;">
</a>
</p>

<p align="center" style="font-size:0.68em; color:#777; margin:4px 0 0 0;">
<i>Nota: el tema del navegador puede afectar la visualización de la presentación.</i>
</p>

---

# Parte 1

LSTMs are good for time-series analysis and sequence modeling due to that they can predict the
current value based on its own past values, the model output depends on a sequence of prior outputs
or inputs ht=LSTM(xt, ht−1). Use the code companion “LSTMs for Forecasting.ipynb” to forecast a
wave (a basic example of time-series prediction), predicting future values using an autoregressive
approach (feeding the model's predictions back as inputs) and graph the original sine wave, the
training predictions, and the autoregressive forecast.

<!-- Las redes LSTM son adecuadas para el análisis de series temporales y el modelado de secuencias,
ya que permiten predecir el valor actual basándose en sus propios valores pasados; la salida del
modelo depende de una secuencia de salidas o entradas anteriores: $h_t = \text{LSTM}(x_t, h_{t-1})$.
Utiliza el archivo de código complementario «LSTMs for Forecasting.ipynb» para realizar una
predicción sobre una onda (un ejemplo básico de predicción de series temporales). Para ello,
predice valores futuros mediante un enfoque autorregresivo —reintroduciendo las predicciones del
modelo como entradas— y genera un gráfico que muestre la onda sinusoidal original, las predicciones
obtenidas durante el entrenamiento y la predicción autorregresiva. -->

## 1. What does it mean that LSTMs are autoregressive?

<!-- ¿Qué significa que las LSTM sean autorregresivas? -->

> Que una LSTM es autorregresiva significa que el modelo utiliza sus propias predicciones pasadas
> como entradas para generar predicciones futuras. En otras palabras, la salida de la LSTM en un
> momento dado depende no solo de las entradas actuales, sino también de las salidas anteriores, lo
> que le permite capturar dependencias temporales y patrones en los datos secuenciales. Esto es
> especialmente útil en tareas de series temporales, donde el valor futuro puede estar influenciado
> por los valores anteriores.

## 2. Change the dataset values: samples, time_step, step, and train size and comment the results in the graph.

<!-- Modifica los valores del conjunto de datos: muestras (samples), paso temporal (time_step), paso
(step) y tamaño del conjunto de entrenamiento (train size)— y comenta los resultados en el gráfico. -->

Se realizaron dos iteraciones con diferentes configuraciones de parámetros para observar cómo
afectan el rendimiento del modelo LSTM en la predicción de la onda sinusoidal. A continuación, se
presentan los resultados obtenidos en cada iteración:

<!-- prettier-ignore -->
| _Parámetro_ | _Profesor_ | _1_ | _2_ |
|:---:|:---:|:---:|:---:|
| Sample | 1000 | 2000 |2000|
| Time_step | 10 | 10 |5|
| Step | 0.25 | 0.25 |0.5|
| Train size | $0.9\times len(X)$ | $0.8\times len(X)$ |$0.75\times len(X)$|
| Epochs | 3 | 5 |5|
| LSTM Units | 5 | 5 |10|

### Iteracción 1

Estos parámetros determinan cuánta información recibe la LSTM, su capacidad y el tiempo que tiene
para aprender:

<!-- prettier-ignore -->
| Parámetro | Influencia en este experimento |
|:---:|:---:|
| `samples=2000` | Genera 2000 puntos de la onda seno. Con ventanas de 10 puntos se forman **1990 secuencias**. Más puntos pueden aportar más ejemplos, aunque aquí la señal es periódica y muchos son similares. |
| `time_step=10` | Usa los **10 valores anteriores** para predecir el siguiente. Una ventana mayor aporta más contexto, pero también aumenta el costo y no garantiza mejores resultados. |
| `step=0.25` | Controla el espaciado temporal aproximado entre puntos. Con este valor hay unos **25 puntos por ciclo** de la onda; la ventana de 10 cubre alrededor del 40 % de un ciclo. Un paso menor muestrea la onda con mayor detalle; uno mayor deja menos puntos por ciclo. |
| `train_size=0.8` | Usa el 80 % de las secuencias para entrenamiento: **1592 para entrenar y 398 para evaluar**. Más datos de entrenamiento pueden ayudar al aprendizaje, pero dejan menos para evaluar. |
| `epochs=5` | El modelo recorre los datos de entrenamiento cinco veces. La gráfica de pérdida muestra que en este caso fueron suficientes para reducir mucho el error; más épocas podrían no mejorar el resultado y, en otros problemas, provocar sobreajuste. |
| `lstm_units=5` | Define cinco unidades LSTM, es decir, una capacidad relativamente pequeña. Bastaron para aprender esta señal sencilla; más unidades podrían ayudar con patrones complejos, pero aumentarían el costo y el riesgo de sobreajuste. |

En conjunto, la señal es **regular, sin ruido y repetitiva**, por lo que un modelo pequeño consigue
un ajuste excelente. No se puede atribuir la precisión a un parámetro concreto sin cambiarlo y
comparar los resultados bajo las mismas condiciones.

![pto 2 iter 1 prediccion][pto_2_lstm_iter_1_p] ![pto 2 iter 1 loss][pto_2_lstm_iter_1_l]

<!-- prettier-ignore -->
| _Métrica_   | _Valor_ | _Interpretación_                   |
|:---:|:---:|:---:|
| MAE       | 0.010968 | Menor es mejor                   |
| MSE       | 0.000172 | Menor es mejor                   |
| RMSE      | 0.013132 | Menor es mejor                   |
| R²        | 0.999653 | Mejor: 1; puede ser negativo     |
| MAPE (%)  | 5.659896 | Menor es mejor; excluidos: 0/398 |
| sMAPE (%) | 5.076742 | Menor es mejor; rango: 0 a 200   |

### Iteracción 2

![pto 2 iter 2 prediccion][pto_2_lstm_iter_2_p] ![pto 2 iter 2 loss][pto_2_lstm_iter_2_l]

<!-- prettier-ignore -->
| _Métrica_   | _Valor_ | _Interpretación_                   |
|:---:|:---:|:---:|
| MAE       | 0.014290 | Menor es mejor                   |
| MSE       | 0.000266 | Menor es mejor                   |
| RMSE      | 0.016309 | Menor es mejor                   |
| R²        | 0.999467 | Mejor: 1; puede ser negativo     |
| MAPE (%)  | 5.514197 | Menor es mejor; excluidos: 0/499 |
| sMAPE (%) | 5.455340 | Menor es mejor; rango: 0 a 200   |

Prácticamente, no hay diferencia en el desempeño del modelo entre las dos iteraciones, aunque la
segunda tiene un error ligeramente mayor. Esto se debe a que la señal es muy regular y el modelo es
capaz de aprenderla bien con diferentes configuraciones de parámetros. La elección de parámetros
puede afectar la eficiencia del entrenamiento y la capacidad del modelo para generalizar, pero en
este caso, ambos conjuntos de parámetros producen resultados muy similares.

## 3. Change the Size of the LSTM layer (from 50 to 30 to 10) see and comment the results in the graph.

<!-- Modifica el tamaño de la capa LSTM (de 50 a 30 y a 10), observa los resultados en el gráfico y comenta al respecto. -->

**La LSTM reproduce bien la onda seno**, aunque presenta pequeñas diferencias en la amplitud y la
posición de algunos picos y valles.

### Gráficas

- **Predicciones:** la curva naranja sigue de cerca los valores reales en verde. En los máximos y
  mínimos, la predicción sobrepasa ligeramente la amplitud real. El **MAE de 0.021** y el **R² de
  0.9989** confirman un ajuste muy cercano.
- **Pérdida:** el MSE de entrenamiento baja rápidamente entre las épocas 1 y 2 y permanece muy bajo
  en la 3. La pérdida de validación también es baja y disminuye; no se observa una divergencia que
  indique sobreajuste en estas tres épocas. La pérdida de entrenamiento inicial puede ser mayor
  porque se promedia mientras los pesos aún se están actualizando, mientras la validación se
  calcula al finalizar la época.

![pto 3 prediccion][pto_3_lstm_p] ![pto 3 loss][pto_3_lstm_l]

### Parámetros

<!-- prettier-ignore -->
| Parámetro | Influencia en este experimento |
|:---:|:---:|
| `samples=1000` | Se generan 1000 puntos y, con ventanas de 10, se obtienen 990 secuencias. |
| `time_step=10` | Se usan los 10 valores anteriores para predecir el siguiente. |
| `step=0.25` | El intervalo aproximado entre puntos produce unos 25 puntos por ciclo. |
| `train=891`, `test=99` | Aproximadamente el 90 % de las secuencias se usa para entrenamiento y el 10 % para evaluación. |
| `units=30` | La capa tiene 30 unidades LSTM, lo que le da mayor capacidad que la configuración anterior de 5 unidades. |
| `epochs=3` | El modelo recorre tres veces los datos de entrenamiento; en esta ejecución aprende rápidamente. |

<!-- prettier-ignore -->
| _Métrica_   | _Valor_ | _Interpretación_                   |
|:---:|:---:|:---:|
| MAE       |  0.021008 | Menor es mejor |
| MSE       |  0.000565 | Menor es mejor |
| RMSE      |  0.023761 | Menor es mejor |
| R²        |  0.998863 | Mejor: 1 |
| MAPE (%)  | 12.343703 | Menor es mejor |
| sMAPE (%) |  9.731429 | Menor es mejor |

El **MAPE de 12.34 %** debe interpretarse con cautela: los valores reales cercanos a cero pueden
inflar los errores porcentuales, aunque el error absoluto sea pequeño.

Frente al experimento anterior, el MAE fue mayor (**0.021 frente a 0.011**), pero también cambiaron
el número de muestras, la división y las épocas. Por tanto, no se puede atribuir esa diferencia
solamente al aumento de unidades. Además, el mismo conjunto de prueba se usa como validación
durante el entrenamiento.

**La LSTM de 10 unidades aprendió la forma de la onda seno**, aunque todavía presenta pequeñas
diferencias respecto a los valores reales.

### Gráficas

- **Predicciones:** la curva naranja sigue la periodicidad de la curva verde. Se observa una ligera
  sobreestimación en los máximos y diferencias en algunos tramos de subida y bajada.
- **Pérdida:** tanto el MSE de entrenamiento como el de validación disminuyen durante las tres
  épocas. No se observa una divergencia indicativa de sobreajuste en esta ejecución. La pérdida de
  validación es menor, en parte porque se calcula al finalizar cada época, mientras la de
  entrenamiento se promedia durante las actualizaciones.

![pto 3-2 prediccion][pto_3_2_lstm_p] ![pto 3-2 loss][pto_3_2_lstm_l]

### Parámetros

<!-- prettier-ignore -->
| Parámetro | Influencia en este experimento |
|:---:|:---:|
| `samples=1000` | Se generan 1000 puntos y, con ventanas de 10, se obtienen 990 secuencias. |
| `time_step=10` | Se emplean los 10 valores anteriores para predecir el siguiente. |
| `step=0.25` | Espaciado aproximado entre puntos; hay unos 25 puntos por ciclo. |
| `train=891`, `test=99` | Se usa el 90 % de las secuencias para entrenamiento y el 10 % para evaluación. |
| `units=10` | La capa LSTM tiene 10 unidades. |
| `epochs=3` | El modelo recorre tres veces los datos de entrenamiento. |

### Métricas

<!-- prettier-ignore -->
| _Métrica_   | _Valor_ | _Interpretación_                   |
|:---:|:---:|:---:|
| MAE       |  0.035890 | Menor es mejor                  |
| MSE       |  0.001988 | Menor es mejor                  |
| RMSE      |  0.044583 | Menor es mejor                  |
| R²        |  0.995997 | Mejor: 1    |
| MAPE (%)  | 16.210677 | Menor es mejor |
| sMAPE (%) | 12.859084 | Menor es mejor  |

El **MAE de 0.0359** indica el error absoluto medio; el **RMSE de 0.0446** penaliza más los errores
grandes. El **MSE es 0.001988** y el **R² es 0.9960**, por lo que las predicciones explican
aproximadamente el 99.6 % de la variabilidad observada.

El **MAPE es 16.21 %** y el **sMAPE 12.86 %**. Conviene interpretarlos con cautela: los valores de
la onda cercanos a cero pueden inflar los errores porcentuales.

En comparación con la ejecución anterior de **30 unidades**, que mantuvo los demás parámetros
indicados, esta tuvo mayor MAE: **0.0359 frente a 0.0210**. Es una comparación de una sola
ejecución; se necesitarían varias semillas para valorar si la diferencia es consistente. Además, el
conjunto de prueba se usa también como validación durante el entrenamiento.

# Parte 2

Using an LSTM to predict the next word in a text sequence is a classic application of recurrent
neural networks (RNNs), showcasing the ability of LSTM layer to process sequences and captures
temporal and sequential dependencies in language data. Run the code companion “LSTM.ipynb” to train
an LSTM on a small corpus to predict the next word based on the sequence of preceding words useful
in Autocomplete (e.g., in search engines or chat applications).

<!-- El uso de una red LSTM para predecir la siguiente palabra en una secuencia de texto constituye
una aplicación clásica de las redes neuronales recurrentes (RNN, por sus siglas en inglés). Esto
pone de manifiesto la capacidad de una capa LSTM para procesar secuencias y capturar dependencias
temporales y secuenciales en datos lingüísticos. Ejecuta el archivo de código «LSTM.ipynb» para
entrenar una red LSTM con un corpus reducido, con el fin de predecir la siguiente palabra a partir
de la secuencia de palabras anteriores; una funcionalidad útil para el autocompletado (por ejemplo,
en motores de búsqueda o aplicaciones de chat). -->

## 1. Try different phrases and see the results

<!-- Prueba diferentes frases y observa los resultados -->

Al probar distintas frases, el modelo a veces predice una continuación coherente con el corpus de
entrenamiento. Por ejemplo, para “the sun” y “the weather” predice “is”, que coincide con las
frases originales. Sin embargo, para “make you” predice “you” en lugar de “want”, y para “I love”
predice “a” en lugar de “hot”. Esto indica que el modelo ha aprendido patrones del corpus, pero su
capacidad de generalización es limitada debido al tamaño reducido del conjunto de datos y la
complejidad del lenguaje.

Al generar varias palabras aparecen repeticiones y secuencias poco naturales, como “the sun is a
there there”. Esto muestra que una primera predicción acertada no garantiza una continuación
coherente: cada palabra generada pasa a formar parte del contexto de la siguiente predicción.

También se observa una limitación con palabras fuera del vocabulario. En “the moon is glowing”, el
tokenizer ignora “moon” y “glowing”, por lo que el modelo solo recibe “the is”. Como conclusión, el
modelo reconoce algunos patrones del corpus pequeño, pero sus respuestas son limitadas y no
demuestran una comprensión general del significado.

## 2. Create a longer list of phrases and see the results

<!-- Crea una lista más larga de frases y observa los resultados -->

Se probaron 11 frases nuevas y se observó que el modelo produce continuaciones similares incluso
para contextos diferentes. Por ejemplo, varias frases terminan en “is a there in”, mientras que
otras generan “feet there there there”. Estas continuaciones presentan repeticiones y no conservan
el sentido de las frases originales.

La principal limitación es el vocabulario: aunque las frases tienen entre 15 y 16 palabras, el
modelo utilizó en promedio solo 3,6 tokens, porque el tokenizer ignora las palabras que no aparecen
en el corpus de entrenamiento. Por ejemplo, en la frase sobre el búho se descartaron palabras como
“owl”, “moon” y “tree”.

En conclusión, ampliar la lista de prueba permite observar más casos, pero no amplía el
conocimiento del modelo. Para mejorar su capacidad de generar continuaciones coherentes sería
necesario entrenarlo con un corpus más amplio y representativo, y evaluar los resultados con
ejemplos no utilizados durante el entrenamiento.

## 3. Compare the results on short vs long phrases

<!-- Compara los resultados en frases cortas vs largas -->

En la prueba inicial, las 8 frases cortas aportaron dos tokens cada una y ninguna fue truncada.
Algunas obtuvieron una siguiente palabra coherente con el corpus, como “the sun” → “is”. Sin
embargo, también hubo predicciones incorrectas y repeticiones. Las 8 frases largas aportaron en
promedio 4,9 tokens, pero esto no garantizó mejores continuaciones: “make you want to move” generó
“move move move move”. Además, una frase superó la capacidad de seis tokens del modelo, por lo que
se descartó su primer token. En las 11 frases largas nuevas, muchas palabras eran desconocidas y se
ignoraron; por eso el modelo utilizó solo 3,6 tokens en promedio. Una frase más larga no
necesariamente proporciona más contexto útil: depende de que sus palabras estén en el vocabulario y
de que quepan en la ventana del modelo. Esta comparación es descriptiva y no permite concluir cuál
grupo tiene mayor precisión sin respuestas esperadas.

## 4. Why do you think Transformers are better for the same task?

<!-- ¿Por qué crees que los Transformers son mejores para la misma tarea? -->

Los Transformers pueden ser más eficaces para predecir la siguiente palabra porque usan
autoatención para relacionar palabras de distintas posiciones dentro del contexto. Esto les permite
capturar dependencias lejanas sin procesar la entrada palabra por palabra, como lo hace una LSTM.
Además, pueden paralelizar buena parte del entrenamiento. En este experimento, el modelo LSTM
produce repeticiones y continuaciones poco coherentes. No obstante, esas dificultades también están
relacionadas con el corpus pequeño, el vocabulario limitado y la ventana de contexto, no solamente
con la arquitectura. Un Transformer tendría limitaciones similares si recibiera los mismos datos y
un tratamiento equivalente de las palabras desconocidas. Por tanto, no se puede afirmar que sería
mejor en este caso sin entrenar y evaluar ambos modelos con los mismos datos de prueba y métricas
comparables.

# Parte 3

Now use the code companion notebook: “LSTM CSV.ipynb” to try to predict the next word in the name
of a point of interest saved in the poi.csv file. Run the following experiments and report

<!-- Ahora, utilice el cuaderno de código complementario: “LSTM CSV.ipynb” para intentar predecir la
siguiente palabra en el nombre de un punto de interés guardado en el archivo poi.csv. Ejecute los
siguientes experimentos e informe los resultados. -->

<!-- prettier-ignore -->
| _Contexto (palabras previas)_ | _Unidades LSTM_ | _Accuracy Inicial_ | _Accuracy Final_ | _Cambio Accuracy (p.p.)_ | _Loss Inicial_ | _Loss Final_ | _Cambio Loss_ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 100 | 0.0893 | 0.2879 | +19.86 p.p. | 7.3983 | 3.0837 | -4.3145 |
| 1 | 150 | 0.0893 | 0.2877 | +19.83 p.p. | 7.3472 | 3.0606 | -4.2866 |
| 1 | 200 | 0.0897 | 0.2895 | +19.98 p.p. | 7.3204 | 3.0411 | -4.2793 |
| 3 | 100 | 0.0890 | 0.5360 | +44.70 p.p. | 7.2846 | 1.6596 | -5.6251 |
| 3 | 150 | 0.0892 | 0.6303 | +54.11 p.p. | 7.2735 | 1.0699 | -6.2036 |
| 3 | 200 | 0.0895 | 0.6184 | +52.89 p.p. | 7.2712 | 1.1200 | -6.1512 |
| 5 | 100 | 0.0890 | 0.6432 | +55.42 p.p. | 7.2285 | 1.1201 | -6.1084 |
| 5 | 150 | 0.0891 | 0.6813 | +59.22 p.p. | 7.2173 | 0.9402 | -6.2771 |
| 5 | 200 | 0.0895 | 0.6434 | +55.40 p.p. | 7.2239 | 1.1243 | -6.0996 |

![pto 4][pto_4_lstm]

## 1. Train during 500 epochs, graph the results and observe how if the metric improves.

<!-- Entrena durante 500 épocas, grafica los resultados y observa si la métrica mejora. -->

- Evolución general de las métricas: En comparación con el entrenamiento base de 20 épocas (donde
  el modelo apenas alcanzaba un 22.24% de precisión y una pérdida de 3.81), extender el
  entrenamiento a 500 épocas permite una mejora sustancial. La precisión global pasa de un valor
  inicial aleatorio (~8.9%) hasta un pico de 68.13% y la pérdida se reduce de ~7.22 a 0.94 en la
  mejor configuración (Contexto=5, LSTM=150).
- Dinámica de convergencia y meseta (plateau):
  - Fase de aprendizaje acelerado (épocas 1 a 100–150): Prácticamente todo el aprendizaje útil se
    concentra en las primeras 100 a 150 épocas. En este rango, las curvas de pérdida caen de forma
    casi exponencial y el accuracy escala rápidamente.
  - Estancamiento temprano en contexto corto: En las secuencias de longitud 1, el modelo alcanza su
    límite predictivo hacia la época 40 (~28.8% de precisión) y permanece completamente plano
    durante las 460 épocas restantes. Entrenar 500 épocas en este escenario representa un consumo
    innecesario de cómputo.
  - Fase tardía (épocas 200 a 500): Para contextos de 3 y 5 palabras, las curvas entran en una
    meseta asintótica con oscilaciones debidas a la tasa de aprendizaje fija de Adam. La mejora
    marginal después de la época 200 es inferior al 2-3% de precisión.
- Riesgo de sobreajuste (overfitting): Al no contar con capas de regularización (Dropout o
  Recurrent Dropout) ni un conjunto de validación separado, entrenar durante 500 épocas favorece la
  memorización exacta de los nombres de los lugares del corpus más frecuentes, en lugar de aprender
  patrones morfosintácticos generalizables.

## 2. Use different sentence lengths (1, 3, 5) and report.

<!-- Prueba diferentes longitudes de frases (1, 3, 5) y reporta los resultados. -->

La longitud del contexto de entrada es el factor más determinante en el desempeño del modelo:

- Longitud 1 (Modelo de Bigramas, $P(w_t \mid w_{t-1})$):
  - Resultados: Precisión estancada en ~28.8% y pérdida alta en ~3.06.
  - Interpretación: Al alimentar la red con una única palabra previa, se produce una severa
    ambigüedad condicional. Palabras muy frecuentes en nombres de lugares (como "de", "san",
    "restaurante", "hotel", "la") tienen decenas o cientos de continuaciones posibles en el
    vocabulario. Matemáticamente, la entropía condicional es muy alta y existe un techo teórico
    insuperable: ninguna capacidad de red neuronal puede adivinar la siguiente palabra sin más
    información de contexto.
- Longitud 3 (Contexto medio, Trigramas/4-gramas):
  - Resultados: La precisión salta a un rango de 53.60% a 63.03% (+25 a +34 p.p. respecto a
    longitud 1) y la pérdida desciende a 1.07 – 1.66.
  - Interpretación: Disponer de 3 palabras previas permite desambiguar la entidad (por ejemplo,
    reconocer secuencias como "parque recreativo y..." o "centro comercial santa..."), reduciendo
    drásticamente el espacio de búsqueda del softmax.
- Longitud 5 (Contexto largo):
  - Resultados: Alcanza el mejor rendimiento global con 68.13% de precisión y 0.9402 de pérdida
    (con LSTM=150).
  - Interpretación: Proporciona a las compuertas de memoria (forget, input, cell gates) de la LSTM
    la información estructural necesaria para capturar dependencias a mediano plazo típicas de
    nombres compuestos de establecimientos comerciales y geográficos.

## 3. Try different sizes of lstm modules # LSTM(50), 100, 200 and report.

<!-- Modifica el tamaño de la capa LSTM (de 50 a 100 y a 200), observa los resultados en el gráfico
y comenta al respecto. -->

- Independencia de la capacidad en contexto 1: Con longitud 1, variar de 100 a 150 y a 200 unidades
  LSTM produce exactamente la misma precisión (~28.8%) y pérdida (~3.06). Esto evidencia que
  aumentar la capacidad del modelo no compensa la carencia de datos o contexto en la entrada.
- Comparativa entre 100, 150 y 200 unidades:
  - 100 unidades: Con contextos de 3 y 5, 100 unidades resultan en un rendimiento respetable (53.6%
    y 64.3%), pero la red sufre de un leve subajuste (underfitting) de capacidad frente a la
    diversidad del vocabulario total. Un tamaño menor (como 50 unidades) acentuaría este cuello de
    botella representacional.
  - 150 unidades (Punto óptimo): Logró los mejores resultados tanto en contexto 3 (63.03%) como en
    contexto 5 (68.13%), con las curvas de convergencia más estables y la menor pérdida final
    (0.9402).
  - 200 unidades (Saturación y sobreparametrización): A pesar de contar con mayor número de
    parámetros, 200 unidades obtuvo resultados ligeramente inferiores a 150 unidades (61.84% en
    contexto 3 y 64.34% en contexto 5). En el gráfico se aprecia mayor ruido y fluctuaciones en sus
    curvas. Esto ocurre porque un espacio paramétrico excesivo incrementa la varianza, dificulta la
    convergencia del optimizador y favorece mínimos locales sin aportar mejoras representacionales
    para este volumen de datos.
- Compromiso Costo vs. Rendimiento: El tiempo de entrenamiento escaló significativamente con 200
  unidades (tardando casi el doble que con 100 unidades por cada 500 épocas) sin retribución en
  métricas, confirmando que 150 unidades de memoria interna es la dimensión más eficiente para este
  dataset.

---

<!-- comentario -->

[pto_2_lstm_iter_1_p]: D:\deep-learning-labs\reports\figures\lstm_20261123_144006_371814_5b89b542_predictions.png
[pto_2_lstm_iter_2_p]: D:\deep-learning-labs\reports\figures\lstm_20261124_162339_727340_5932939d_predictions.png
[pto_2_lstm_iter_1_l]: D:\deep-learning-labs\reports\figures\lstm_20261123_144006_371814_5b89b542_loss.png
[pto_2_lstm_iter_2_l]: D:\deep-learning-labs\reports\figures\lstm_20261124_162339_727340_5932939d_loss.png
[pto_3_lstm_p]: D:\deep-learning-labs\reports\figures\lstm_20261124_170636_822840_bc64bbf4_predictions.png
[pto_3_lstm_l]: D:\deep-learning-labs\reports\figures\lstm_20261124_170636_822840_bc64bbf4_loss.png
[pto_3_2_lstm_p]: D:\deep-learning-labs\reports\figures\lstm_20261124_173018_077529_72b788cb_predictions.png
[pto_3_2_lstm_l]: D:\deep-learning-labs\reports\figures\lstm_20261124_173018_077529_72b788cb_loss.png
[pto_4_lstm]: D:\deep-learning-labs\reports\figures\comparacion_lstm_351658fc82a8464a882177b9cebb672f.png


<!-- [pto_2_lstm_iter_1_p]: reports/figures/lstm_20261123_144006_371814_5b89b542_predictions.png
[pto_2_lstm_iter_2_p]: reports/figures/lstm_20261124_162339_727340_5932939d_predictions.png
[pto_2_lstm_iter_1_l]: reports/figures/lstm_20261123_144006_371814_5b89b542_loss.png
[pto_2_lstm_iter_2_l]: reports/figures/lstm_20261124_162339_727340_5932939d_loss.png
[pto_3_lstm_p]: reports/figures/lstm_20261124_170636_822840_bc64bbf4_predictions.png
[pto_3_lstm_l]: reports/figures/lstm_20261124_170636_822840_bc64bbf4_loss.png
[pto_3_2_lstm_p]: reports/figures/lstm_20261124_173018_077529_72b788cb_predictions.png
[pto_3_2_lstm_l]: reports/figures/lstm_20261124_173018_077529_72b788cb_loss.png
[pto_4_lstm]: reports/figures/comparacion_lstm_351658fc82a8464a882177b9cebb672f.png -->
