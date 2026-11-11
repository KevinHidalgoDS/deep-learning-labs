---
title: "FollowUp 3"
author: "_Kevin Ferney Hidalgo Higuita_"
date: "10 de Noviembre de 2026"
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
<h3 align="center" style="font-size:1.05em; margin:4px 0;">Follow Up 3 </h3>
<h4 align="center" style="font-size:0.9em; margin:4px 0; color:#555;">Semana 3</h4>

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
📅 10 de Noviembre de 2026 ·
<a href="https://github.com/KevinHidalgoDS/deep-learning-labs">
  <img src="https://img.shields.io/badge/repository-GitHub-181717.svg" alt="GitHub Repo" style="vertical-align:middle;">
</a>
</p>

<p align="center" style="font-size:0.68em; color:#777; margin:4px 0 0 0;">
<i>Nota: el tema del navegador puede afectar la visualización de la presentación.</i>
</p>

---

- [A. Regression and Classification with WEKA](#a-regression-and-classification-with-weka)
  - [1. Descargue WEKA](#1-descargue-weka)
- [B. Intriguing properties of neural networks](#b-intriguing-properties-of-neural-networks)
- [C. Regression and Classification with WEKA](#c-regression-and-classification-with-weka)
  - [1. Create your own dataset for image classification. Use your dataset for making an image classification model](#1-create-your-own-dataset-for-image-classification-use-your-dataset-for-making-an-image-classification-model)
  - [2. Create your own dataset for detection or semantic segmentation using open source and stand alone software:](#2-create-your-own-dataset-for-detection-or-semantic-segmentation-using-open-source-and-stand-alone-software)
- [Bonus](#bonus)

# A. Regression and Classification with WEKA

## 1. Descargue [WEKA](https://waikato.github.io/weka-wiki/downloading_weka/)

- Cree y compare al menos dos algoritmos disponibles para la **clasificación** del dataset de
  diabetes.

Modelo 1 _BayesNet_

- Es un clasificador probabilístico basado en el Teorema de Bayes que utiliza un Grafo Acíclico
  Dirigido (DAG). Los nodos representan los atributos y las aristas representan las dependencias
  condicionales entre ellos. Para predecir, calcula la probabilidad posterior de cada clase dadas
  las características de la instancia y selecciona la más probable.
- Supuestos: A diferencia del clasificador Naive Bayes (que asume independencia total entre todos
  los atributos dada la clase), BayesNet relaja esta restricción y permite modelar dependencias
  explícitas entre ciertos atributos, asumiendo independencia condicional solo respecto a los nodos
  que no son sus descendientes en el grafo.
- Fortalezas: Es altamente interpretable (el grafo muestra las relaciones entre variables), maneja
  bien la incertidumbre, es robusto ante conjuntos de datos pequeños y permite integrar
  conocimiento experto previo.
- Debilidades: El aprendizaje de la estructura óptima del grafo a partir de los datos es
  computacionalmente costoso (problema NP-duro), por lo que se recurre a heurísticas de búsqueda
  local (como el algoritmo K2 usado en este experimento).
- Mejor desempeño: Problemas donde la explicabilidad es crítica (como diagnósticos médicos), datos
  con ruido o cuando se busca entender las relaciones causales o de influencia entre variables.

<!-- prettier-ignore -->
```text
=== Run information ===

Scheme:       weka.classifiers.bayes.BayesNet -D -Q weka.classifiers.bayes.net.search.local.K2 -- -P 1 -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5
Relation:     pima_diabetes
Instances:    768
Attributes:   9
              preg
              plas
              pres
              skin
              insu
              mass
              pedi
              age
              class
Test mode:    10-fold cross-validation

=== Classifier model (full training set) ===

Bayes Network Classifier
not using ADTree
#attributes=9 #classindex=8
Network structure (nodes followed by parents)
preg(2): class 
plas(4): class 
pres(1): class 
skin(1): class 
insu(3): class 
mass(2): class 
pedi(2): class 
age(2): class 
class(2): 
LogScore Bayes: -4030.960854130862
LogScore BDeu: -4053.744069918824
LogScore MDL: -4054.1531320488457
LogScore ENTROPY: -3991.037129583943
LogScore AIC: -4010.037129583943


Time taken to build model: 0.06 seconds

=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances         571               74.349  %
Incorrectly Classified Instances       197               25.651  %
Kappa statistic                          0.429 
Mean absolute error                      0.2987
Root mean squared error                  0.4208
Relative absolute error                 65.7116 %
Root relative squared error             88.28   %
Total Number of Instances              768     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,816    0,392    0,795      0,816    0,806      0,429    0,806     0,882     tested_negative
                 0,608    0,184    0,639      0,608    0,623      0,429    0,806     0,677     tested_positive
Weighted Avg.    0,743    0,319    0,741      0,743    0,742      0,429    0,806     0,811     

=== Confusion Matrix ===

   a   b   <-- classified as
 408  92 |   a = tested_negative
 105 163 |   b = tested_positive
```

Modelo 2 _AdaBoostM1_

- Es un método de ensamble (ensemble learning) predictivo. Su principio es construir un
  clasificador "fuerte" combinando secuencialmente múltiples clasificadores "débiles" (en este
  caso, Decision Stumps o árboles de un solo nivel). En cada iteración, el algoritmo incrementa el
  peso de las instancias que fueron clasificadas incorrectamente por el modelo anterior, forzando
  al siguiente modelo a enfocarse en los casos más difíciles.
- Supuestos: Asume que el clasificador base es al menos ligeramente mejor que adivinar al azar
  (tasa de error < 50%).
- Fortalezas: Reduce significativamente tanto el sesgo como la varianza, logrando altas tasas de
  exactitud. Es resistente al sobreajuste (overfitting) en muchos escenarios.
- Debilidades: Es sensible a datos atípicos (outliers) y ruido, ya que el algoritmo les asignará
  cada vez más peso al no poder clasificarlos, arruinando el modelo general. Su interpretabilidad
  es baja comparada con un solo árbol o red.
- Mejor desempeño: Problemas de clasificación estructurada donde el objetivo principal es maximizar
  la exactitud predictiva y no importa tanto la interpretabilidad del modelo final.

<!-- prettier-ignore -->
```text
=== Run information ===

Scheme:       weka.classifiers.meta.AdaBoostM1 -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump
Relation:     pima_diabetes
Instances:    768
Attributes:   9
              preg
              plas
              pres
              skin
              insu
              mass
              pedi
              age
              class
Test mode:    10-fold cross-validation

=== Classifier model (full training set) ===

AdaBoostM1: Base classifiers and their weights: 

Decision Stump

Classifications

plas <= 127.5 : tested_negative
plas > 127.5 : tested_positive
plas is missing : tested_negative

Class distributions

plas <= 127.5
tested_negative	tested_positive	
0.8061855670103093	0.19381443298969073	
plas > 127.5
tested_negative	tested_positive	
0.38515901060070673	0.6148409893992933	
plas is missing
tested_negative	tested_positive	
0.6510416666666666	0.3489583333333333	


Weight: 1.02

Decision Stump

Classifications

mass <= 26.95 : tested_negative
mass > 26.95 : tested_negative
mass is missing : tested_negative

Class distributions

mass <= 26.95
tested_negative	tested_positive	
0.8960790304511662	0.1039209695488339	
mass > 26.95
tested_negative	tested_positive	
0.5378130286474941	0.46218697135250586	
mass is missing
tested_negative	tested_positive	
0.6144906055189836	0.3855093944810163	


Weight: 0.47

Decision Stump

Classifications

mass <= 26.95 : tested_negative
mass > 26.95 : tested_positive
mass is missing : tested_positive

Class distributions

mass <= 26.95
tested_negative	tested_positive	
0.84398332333169	0.15601667666830984	
mass > 26.95
tested_negative	tested_positive	
0.4219712026437726	0.5780287973562274	
mass is missing
tested_negative	tested_positive	
0.4999999999999976	0.5000000000000023	


Weight: 0.52

Decision Stump

Classifications

age <= 28.5 : tested_negative
age > 28.5 : tested_positive
age is missing : tested_negative

Class distributions

age <= 28.5
tested_negative	tested_positive	
0.7482468134383282	0.2517531865616718	
age > 28.5
tested_negative	tested_positive	
0.46730609450883326	0.5326939054911668	
age is missing
tested_negative	tested_positive	
0.5857115420616531	0.4142884579383468	


Weight: 0.5

Decision Stump

Classifications

mass <= 29.95 : tested_negative
mass > 29.95 : tested_negative
mass is missing : tested_negative

Class distributions

mass <= 29.95
tested_negative	tested_positive	
0.7408795682760606	0.2591204317239394	
mass > 29.95
tested_negative	tested_positive	
0.5453819429356981	0.45461805706430186	
mass is missing
tested_negative	tested_positive	
0.6119518052602652	0.3880481947397348	


Weight: 0.46

Decision Stump

Classifications

mass <= 29.95 : tested_negative
mass > 29.95 : tested_positive
mass is missing : tested_positive

Class distributions

mass <= 29.95
tested_negative	tested_positive	
0.6445164019060797	0.3554835980939202	
mass > 29.95
tested_negative	tested_positive	
0.43204921728785944	0.5679507827121406	
mass is missing
tested_negative	tested_positive	
0.49999999999999667	0.5000000000000033	


Weight: 0.37

Decision Stump

Classifications

plas <= 99.5 : tested_negative
plas > 99.5 : tested_positive
plas is missing : tested_negative

Class distributions

plas <= 99.5
tested_negative	tested_positive	
0.7284244173333062	0.2715755826666938	
plas > 99.5
tested_negative	tested_positive	
0.4935912140159397	0.5064087859840604	
plas is missing
tested_negative	tested_positive	
0.5344901459495953	0.46550985405040474	


Weight: 0.18

Decision Stump

Classifications

pedi <= 0.5015000000000001 : tested_negative
pedi > 0.5015000000000001 : tested_positive
pedi is missing : tested_negative

Class distributions

pedi <= 0.5015000000000001
tested_negative	tested_positive	
0.6266877293757988	0.37331227062420125	
pedi > 0.5015000000000001
tested_negative	tested_positive	
0.4620861682696193	0.5379138317303807	
pedi is missing
tested_negative	tested_positive	
0.5643880208706795	0.43561197912932037	


Weight: 0.38

Decision Stump

Classifications

plas <= 165.5 : tested_negative
plas > 165.5 : tested_positive
plas is missing : tested_negative

Class distributions

plas <= 165.5
tested_negative	tested_positive	
0.5678524052856765	0.4321475947143235	
plas > 165.5
tested_negative	tested_positive	
0.3339980031468084	0.6660019968531916	
plas is missing
tested_negative	tested_positive	
0.543265052059565	0.456734947940435	


Weight: 0.32

Decision Stump

Classifications

age <= 56.5 : tested_positive
age > 56.5 : tested_negative
age is missing : tested_positive

Class distributions

age <= 56.5
tested_negative	tested_positive	
0.4609074246674672	0.5390925753325329	
age > 56.5
tested_negative	tested_positive	
0.7576996838229271	0.2423003161770729	
age is missing
tested_negative	tested_positive	
0.4810683912581745	0.5189316087418255	


Weight: 0.22

Number of performed Iterations: 10


Time taken to build model: 0.06 seconds

=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances         571               74.349  %
Incorrectly Classified Instances       197               25.651  %
Kappa statistic                          0.4136
Mean absolute error                      0.3127
Root mean squared error                  0.4178
Relative absolute error                 68.8    %
Root relative squared error             87.6631 %
Total Number of Instances              768     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0,846    0,448    0,779      0,846    0,811      0,417    0,801     0,882     tested_negative
                 0,552    0,154    0,658      0,552    0,600      0,417    0,801     0,647     tested_positive
Weighted Avg.    0,743    0,345    0,737      0,743    0,738      0,417    0,801     0,800     

=== Confusion Matrix ===

   a   b   <-- classified as
 423  77 |   a = tested_negative
 120 148 |   b = tested_positive
```

**Comparación de los resultados**

> - Exactitud (Accuracy): Ambos modelos logran exactamente la misma exactitud global: 74.349% (571
>   instancias clasificadas correctamente y 197 incorrectamente). En términos de precisión pura y
>   global, hay un empate absoluto.
> - Tiempo de construcción: Ambos modelos tardaron 0.06 segundos en construirse. No hay diferencia
>   significativa en costo computacional, lo cual es esperable en un conjunto de datos pequeño (768
>   instancias).
> - Estadístico Kappa: BayesNet (0.429) supera ligeramente a AdaBoostM1 (0.4136). Esto indica que
>   BayesNet tiene una leve ventaja en la concordancia de sus predicciones más allá del azar,
>   aunque ambas cifras representan una concordancia "moderada".
> - Errores (MAE y RMSE):
>
> > - Error Absoluto Medio (MAE): BayesNet (0.2987) es superior a AdaBoostM1 (0.3127), indicando
> >   que, en promedio, las probabilidades predichas por BayesNet están más cerca del valor real.
> > - Raíz del Error Cuadrático Medio (RMSE): AdaBoostM1 (0.4178) supera de forma casi
> >   imperceptible a BayesNet (0.4208). Como el RMSE penaliza más los errores grandes, esto
> >   sugiere que cuando BayesNet se equivoca, lo hace con mayor "confianza" (peor calibración en
> >   los extremos).
>
> - Precisión, Recall y F-Measure (Promedio Ponderado):
>
> > - BayesNet muestra una ventaja marginal en Precisión (0.741 vs 0.737) y F-Measure (0.742 vs
> >   0.738). El Recall ponderado es idéntico en ambos (0.743).
>
> - Área ROC (AUC): BayesNet (0.806) supera a AdaBoostM1 (0.801). Ambos modelos tienen un buen
>   poder de discriminación, pero BayesNet es ligeramente mejor separando las clases a lo largo de
>   diferentes umbrales.
> - Análisis de la Matriz de Confusión (Desempeño por clase): A pesar de tener la misma exactitud
>   global, los modelos cometen errores muy distintos, lo que revela un comportamiento asimétrico:
>
> > - Clase `tested_negative` (mayoritaria): AdaBoostM1 es mejor. Logra identificar correctamente
> >   423 casos frente a los 408 de BayesNet, obteniendo un Recall de 0.846 vs 0.816.
> > - Clase `tested_positive` (minoritaria): BayesNet es mejor. Identifica correctamente 163 casos
> >   frente a los 148 de AdaBoostM1, con un Recall de 0.608 vs 0.552.
> > - Confusiones: AdaBoostM1 confunde más casos positivos asumiendo que son negativos (120 Falsos
> >   Negativos frente a los 105 de BayesNet). Por el contrario, BayesNet confunde más casos
> >   negativos asumiendo que son positivos (92 Falsos Positivos frente a los 77 de AdaBoostM1).

Para este conjunto de datos particular (`diabetes`), se recomienda el clasificador BayesNet. Aunque
ambos modelos tienen una exactitud global idéntica, el contexto del problema (diagnóstico médico de
diabetes) dicta la decisión. En salud clínica, un Falso Negativo (decirle a un paciente con
diabetes que está sano) tiene un costo humano y médico mucho mayor que un Falso Positivo (hacerle
más pruebas a un paciente sano). BayesNet detecta significativamente mejor la clase positiva (menor
tasa de Falsos Negativos: 105 vs 120 de AdaBoostM1). Además, su naturaleza probabilística ofrece
una interpretabilidad gráfica útil para los médicos y presenta mejores métricas agregadas como el
Área ROC y el estadístico Kappa.

- Cree y compare varios algoritmos de **regresión** del precio de casas usando el archivo .csv
  disponible en esta carpeta.

Modelo 1 _LinearRegression_

- Es un método estadístico clásico que asume que el valor objetivo (el precio) puede expresarse
  como una combinación lineal (una suma ponderada) de los atributos de entrada. El algoritmo de
  WEKA utiliza el método de Mínimos Cuadrados Ordinarios (OLS) para encontrar los coeficientes que
  minimicen la suma de los errores al cuadrado entre la predicción y el valor real.

- Supuestos: Asume una relación lineal entre las variables predictoras y la variable objetivo,
  ausencia de multicolinealidad perfecta (variables que midan lo mismo), y que los errores tienen
  una varianza constante (homocedasticidad).

- Fortalezas: Es extremadamente rápido de entrenar y altamente interpretable. La ecuación generada
  permite ver directamente el peso de cada atributo; por ejemplo, por cada pie cuadrado adicional
  de área habitable (sqft_living), el modelo suma $81.41 al precio final.

- Debilidades: Es incapaz de modelar relaciones no lineales (por ejemplo, si el precio sube
  exponencialmente y no de forma lineal) a menos que se transformen los datos previamente. Es
  sensible a valores atípicos (outliers).

- Mejor desempeño: Problemas donde las variables tienen una correlación fuerte, directa y
  proporcional con la variable objetivo.

<!-- prettier-ignore -->
```text
=== Run information ===

Scheme:       weka.classifiers.functions.LinearRegression -S 0 -R 1.0E-8 -num-decimal-places 4
Relation:     kc_house_data_yr_weka-weka.filters.unsupervised.attribute.Remove-R1-2-weka.filters.unsupervised.attribute.Remove-R15
Instances:    21613
Attributes:   18
              price
              bedrooms
              bathrooms
              sqft_living
              sqft_lot
              floors
              waterfront
              view
              condition
              grade
              sqft_above
              sqft_basement
              yr_built
              yr_renovated
              lat
              long
              sqft_living15
              sqft_lot15
Test mode:    split 66.0% train, remainder test

=== Classifier model (full training set) ===


Linear Regression Model

price =

 -34195.388  * bedrooms +
  42364.7977 * bathrooms +
     81.4154 * sqft_living +
      0.127  * sqft_lot +
 587912.0719 * waterfront +
  49506.3459 * view +
  31026.5128 * condition +
  97354.2202 * grade +
     98.3389 * sqft_above +
     65.0549 * sqft_basement +
  -2455.2128 * yr_built +
     21.5841 * yr_renovated +
 561364.634  * lat +
-117578.2856 * long +
     27.3383 * sqft_living15 +
     -0.3936 * sqft_lot15 +
-36947572.9974

Time taken to build model: 0.41 seconds

=== Evaluation on test split ===

Time taken to test model on test split: 0.11 seconds

=== Summary ===

Correlation coefficient                  0.8304
Mean absolute error                 127606.693 
Root mean squared error             215719.9068
Relative absolute error                 53.8636 %
Root relative squared error             55.8832 %
Total Number of Instances             7348
```

Modelo 2 _AdditiveRegression_

- Es un método de ensamble (boosting) para regresión, similar a lo que hace AdaBoost en
  clasificación. Comienza con una predicción inicial (la media de los precios) y luego ajusta
  secuencialmente modelos débiles (en este caso, Decision Stumps o árboles de un solo nivel). Cada
  nuevo modelo no intenta predecir el precio en sí, sino los residuos (el error o la diferencia
  entre el precio real y lo que el ensamble ha predicho hasta el momento).

- Supuestos: Asume que la suma de múltiples predictores simples puede aproximar funciones complejas
  y no lineales mediante la corrección iterativa de errores.

- Fortalezas: Puede capturar relaciones no lineales y es robusto. Al enfocarse en los residuos,
  corrige activamente las deficiencias de los modelos anteriores de la secuencia.

- Debilidades: La interpretabilidad global se pierde en favor de la suma de múltiples pasos
  lógicos. Su capacidad depende fuertemente del clasificador base y del número de iteraciones.

- Mejor desempeño: Problemas con datos complejos, relaciones no lineales e interacciones entre
  variables, siempre y cuando se le dé la complejidad suficiente (iteraciones y profundidad) para
  aprender.

<!-- prettier-ignore -->
```text
=== Run information ===

Scheme:       weka.classifiers.meta.AdditiveRegression -S 1.0 -I 10 -W weka.classifiers.trees.DecisionStump
Relation:     kc_house_data_yr_weka-weka.filters.unsupervised.attribute.Remove-R1-2-weka.filters.unsupervised.attribute.Remove-R15
Instances:    21613
Attributes:   18
              price
              bedrooms
              bathrooms
              sqft_living
              sqft_lot
              floors
              waterfront
              view
              condition
              grade
              sqft_above
              sqft_basement
              yr_built
              yr_renovated
              lat
              long
              sqft_living15
              sqft_lot15
Test mode:    split 66.0% train, remainder test

=== Classifier model (full training set) ===

Additive Regression

Initial prediction: 540182.1587933188

Base classifier weka.classifiers.trees.DecisionStump

10 models generated.

Model number 0

Decision Stump

Classifications

grade <= 8.5 : -102881.99982538608
grade > 8.5 : 420192.2561675605
grade is missing : -2.466800433430111E-9


Model number 1

Decision Stump

Classifications

sqft_living <= 4185.0 : -20211.671096610098
sqft_living > 4185.0 : 694738.9793310437
sqft_living is missing : 1.2142053968907615E-8


Model number 2

Decision Stump

Classifications

lat <= 47.533249999999995 : -136265.32823764466
lat > 47.533249999999995 : 83338.17181453359
lat is missing : -2.0892734034394135E-8


Model number 3

Decision Stump

Classifications

waterfront <= 0.5 : -6381.095969983234
waterfront > 0.5 : 839720.9113889459
waterfront is missing : 1.1928635753878383E-8


Model number 4

Decision Stump

Classifications

sqft_living <= 7940.0 : -1433.1800803917915
sqft_living > 7940.0 : 3096098.9276678897
sqft_living is missing : -1.1558690020582795E-9


Model number 5

Decision Stump

Classifications

grade <= 8.5 : 31136.846219036666
grade > 8.5 : -127169.58928604097
grade is missing : -2.1039131836072752E-9


Model number 6

Decision Stump

Classifications

sqft_living <= 1993.5 : -56986.63796781678
sqft_living > 1993.5 : 66067.8336475879
sqft_living is missing : 7.47100566055996E-9


Model number 7

Decision Stump

Classifications

lat <= 47.703050000000005 : 18616.167305337003
lat > 47.703050000000005 : -97536.03360989234
lat is missing : -4.894363953342973E-9


Model number 8

Decision Stump

Classifications

view <= 0.5 : -12487.873426485
view > 0.5 : 114583.88192503667
view is missing : 5.384653008902376E-10


Model number 9

Decision Stump

Classifications

grade <= 8.5 : 19328.70253358051
grade > 8.5 : -78942.58607105161
grade is missing : -7.224241859374244E-10



Time taken to build model: 1.11 seconds

=== Evaluation on test split ===

Time taken to test model on test split: 0.01 seconds

=== Summary ===

Correlation coefficient                  0.8099
Mean absolute error                 128227.5828
Root mean squared error             226712.3043
Relative absolute error                 54.1257 %
Root relative squared error             58.7308 %
Total Number of Instances             7348
```

**Comparación de los resultados**

> - Coeficiente de correlación: Este indicador (que va de 0 a 1, donde 1 es una correlación
>   perfecta entre la predicción y el valor real) muestra que LinearRegression (0.8304) es superior
>   a AdditiveRegression (0.8099). El modelo lineal logra capturar mejor la varianza de los
>   precios.
> - Error Absoluto Medio (MAE): LinearRegression obtiene un MAE de 127,606.69, frente a los
>   128,227.58 de AdditiveRegression. Esto significa que, en promedio, las predicciones del modelo
>   lineal están más cerca del precio real de la vivienda por una diferencia de ~$620 a su favor.
> - Raíz del Error Cuadrático Medio (RMSE): El modelo LinearRegression (215,719.90) vuelve a
>   superar a AdditiveRegression (226,712.30). Dado que el RMSE penaliza severamente los errores de
>   gran magnitud (al elevarlos al cuadrado), la diferencia de ~$11,000 indica que la regresión
>   aditiva está cometiendo errores mucho más extremos (probablemente subestimando propiedades de
>   muy alto valor).
> - Tiempo de construcción: LinearRegression fue más de un 50% más rápido en entrenar (0.41
>   segundos vs 1.11 segundos), debido a que AdditiveRegression debe construir y evaluar
>   iterativamente los modelos sobre los residuos.
> - Análisis estructural (Los modelos generados):
>
> > - La Regresión Lineal generó una ecuación coherente: atributos como vistas al agua
> >   (`waterfront`) o un alto grado de construcción (`grade`) suman un valor masivo al precio,
> >   mientras que el año de construcción (`yr_built`) resta (las casas más viejas sin renovar
> >   pierden valor).
> > - La Regresión Aditiva muestra claramente su comportamiento corrector: el Modelo 0 suma hasta
> >   $420,192 si el `grade > 8.5`. Sin embargo, el Modelo 5 nota que esto generó una
> >   sobreestimación para algunas casas y resta $127,169 para ese mismo rango de `grade`.

Para este escenario exacto, se recomienda LinearRegression. Supera a AdditiveRegression en
absolutamente todas las métricas de evaluación (mejor correlación, menor MAE, menor RMSE) y es más
rápido de entrenar. La naturaleza del problema de bienes raíces sugiere que existe una fuerte
relación lineal de base (a más metros cuadrados y más habitaciones, mayor precio de forma
relativamente predecible), la cual es capturada eficientemente por la ecuación paramétrica
generada.

# B. Intriguing properties of neural networks

- Read the following paper and make some conclusions:
  [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)
- Visit this web page:
  [https://transcranial.github.io/keras-js/#/inception-v3](https://transcranial.github.io/keras-js/#/inception-v3)
  Upload your own image and its attack to https://imgur.com/ or to Github. Enter the provided
  images: cat.png and cat_attack.png using different pre-trained models and report the results.
  [https://github.com/jrballesteros/adversarial_attacks](https://github.com/jrballesteros/adversarial_attacks)
  ![pig confuso][pig]

# C. Regression and Classification with WEKA

## 1. Create your own dataset for image classification. Use your dataset for making an image classification model

- Visit [Teachable Machine](https://teachablemachine.withgoogle.com/) and create a simple image
  Project to classify images of angry and happy faces using your webcam. Report the results by
  screen shots.
- Do the same but this time create images of two or three type of sounds like seen in class. Report
  by screen shots.

## 2. Create your own dataset for detection or semantic segmentation using open source and stand alone software:

- A Detection Dataset: Run the open source software called LabelImg.exe shown during the class and
  label different images to produce a detection dataset. Use at least 5 image examples.
- A Segmentation Dataset: Run the open source software called PixelAnnotation.exe shown during the
  class and label different images to produce a Segmentation dataset. Use at least 5 image
  examples. See a tutorial of the PixelAnnotation tool in the following video:
  [https://www.youtube.com/watch?v=tX-xcg5wY4U](https://www.youtube.com/watch?v=tX-xcg5wY4U)

# Bonus

Produce and show your own adversarial attacks by running the attached notebook:
produce_adversarial_images.ipynb

---

<!-- [pig]: D:\deep-learning-labs\reports\figures\pig.png -->

[pig]: reports/figures/pig.png
