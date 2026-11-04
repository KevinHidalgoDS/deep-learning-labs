---
title: "FollowUp 2"
author: "_Kevin Ferney Hidalgo Higuita_"
date: "03 de Octubre de 2026"
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
<h3 align="center" style="font-size:1.05em; margin:4px 0;">Follow Up 2 </h3>
<h4 align="center" style="font-size:0.9em; margin:4px 0; color:#555;">Semana 2</h4>

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
📅 03 de Noviembre de 2026 ·
<a href="https://github.com/KevinHidalgoDS/deep-learning-labs">
  <img src="https://img.shields.io/badge/repository-GitHub-181717.svg" alt="GitHub Repo" style="vertical-align:middle;">
</a>
</p>

<p align="center" style="font-size:0.68em; color:#777; margin:4px 0 0 0;">
<i>Nota: el tema del navegador puede afectar la visualización de la presentación.</i>
</p>

---

- [1. Lleve a cabo una operación de Hyperparameter Tuning](#1-lleve-a-cabo-una-operación-de-hyperparameter-tuning)
- [2. Use teacheble machines (https://teachablemachine.com)](#2-use-teacheble-machines-httpsteachablemachinecom)
  - [Modelo 1 Clasificación balanceada](#modelo-1-clasificación-balanceada)
  - [Modelo 2 Clasificación desbalanceada](#modelo-2-clasificación-desbalanceada)
  - [Modelo 3 Desbalanceada con cambios en hiperparámetros](#modelo-3-desbalanceada-con-cambios-en-hiperparámetros)
- [3. Create your own dataset using images from the Internet](#3-create-your-own-dataset-using-images-from-the-internet)
- [4. Responda las siguientes preguntas según la clase](#4-responda-las-siguientes-preguntas-según-la-clase)
- [5. Use a Python Library for image handling i.e. PIL or CV2 to carry out the following exercise.](#5-use-a-python-library-for-image-handling-ie-pil-or-cv2-to-carry-out-the-following-exercise)


## 1. Lleve a cabo una operación de Hyperparameter Tuning

Para el ejercicio de regresión de la clase pasada, encontrar los mejores hyperparametros tales
como: learning rate, dropout, y arquitectura de la red en términos del número de capas. Ver la
descripción en el notebook: Hyperparameter Fine Tunning.ipynb

![Model Performance Comparison][model_performance_comparison]
![Validation MAE by Number of Trials][validation_mae_by_number_of_trials]

<!-- prettier-ignore -->
| Number of trials | Best validation MAE | Test Mae |
|:---:|:---:|:---:|
| 3 | 86245.910156 | 74765.437500 |
| 5 | 79044.945312 | 77648.914062 |
| 8 | 71460.683594 | 77439.265625 |

> Los resultados obtenidos evidencian que aumentar el número de pruebas (trials) durante el proceso
> de ajuste de hiperparámetros permitió mejorar progresivamente el desempeño del modelo. Al
> comparar los experimentos de 3, 5 y 8 trials, se observa una disminución del error absoluto medio
> (MAE), tanto en los datos de validación como en los de prueba.
>
> El mejor desempeño se obtuvo con 8 trials, alcanzando un MAE de validación de 71460.683594 y un
> MAE de prueba de 77439.265625. Esto indica que, al explorar un mayor número de configuraciones,
> aumentó la posibilidad de identificar una combinación de hiperparámetros que favoreciera la
> capacidad predictiva del modelo.
>
> Asimismo, las tablas y gráficas permiten observar que existe una variabilidad importante entre
> las configuraciones evaluadas, lo que demuestra que la selección de hiperparámetros influye
> directamente en los resultados.
>
> En conclusión, el escenario de 8 trials presentó el mejor desempeño entre los tres experimentos
> realizados, aunque la mejora respecto a los 5 trials fue menor que la observada al pasar de 3
> a 5. Esto sugiere que ampliar la búsqueda puede mejorar los resultados, pero también que las
> ganancias adicionales pueden ser cada vez menores frente al mayor costo computacional.

## 2. Use teacheble machines (https://teachablemachine.com)

Visite ese sitio de Google y entrene varios modelos desbalanceados y no usando las imagenes de
perros y gatos provistas en el drive, cree varias clases, juegue con los hiperparámetros, saque
conclusiones, haga pantallazos y reporte los hallazgos de al menos 3 modelos diferentes.

### Modelo 1 Clasificación balanceada

### Modelo 2 Clasificación desbalanceada

### Modelo 3 Desbalanceada con cambios en hiperparámetros

Se ha intentado abrir la url con los navegadores de google chrome, opera y edge, pero no se ha
podido acceder a la página de teachable machine. Por lo tanto, no se han podido entrenar los
modelos ni realizar los experimentos solicitados.

## 3. Create your own dataset using images from the Internet

1. Open Google Chrome or any browser.
2. Go to Google Images and search for the images you need.
3. Scroll down to load more images (as many as you want).
4. Open the Developer Console (Ctrl + Shift + J or Cmd + Option + J on Mac).
5. Paste the script Function to download a .csv file…. below into the console and hit Enter.
6. A CSV file named image_urls.csv will be downloaded, containing all the image URLs found on the
   page. You can then manually or programmatically download the images using these URLs.
7. To programmatically download the images listed in the .csv file to a folder, you can use the
   python script “Download_url_images.ipynb” that reads the .csv file, extracts the URLs, and
   downloads each image to a specified folder.

> Para desarrollar el ejercicio, se realizó una búsqueda de imágenes mediante Google Images,
> seleccionando imágenes relacionadas con aves. Posteriormente, se utilizó un script en la consola
> del navegador para extraer las URL de las imágenes disponibles y generar el archivo
> image_urls.csv.
>
> A continuación, mediante el script Download_url_images.py se procesaron las URL recopiladas para
> descargar las imágenes y almacenarlas en una carpeta local.
>
> Como resultado, se obtuvo un conjunto de imágenes recopiladas de Internet que conforman un
> dataset propio, el cual puede utilizarse posteriormente en ejercicios de entrenamiento y
> evaluación de modelos de aprendizaje automático.

## 4. Responda las siguientes preguntas según la clase

- ¿Por qué la salida de una convolución es equivariante a la traslación y la de una red
  completamente conectada no? Ayuda: pesos compartidos en la CNN.
  > La traslación equivale a un desplazamiento de la imagen, y la convolución es una operación que
  > aplica un filtro a la imagen de manera local, es decir, el mismo filtro se aplica a todas las
  > posiciones de la imagen. Esto significa que si se traslada la imagen, la salida de la
  > convolución también se trasladará de manera correspondiente, manteniendo la relación entre la
  > entrada y la salida. En cambio, una red completamente conectada no tiene esta propiedad, ya que
  > cada neurona está conectada a todas las entradas, y un desplazamiento de la imagen no garantiza
  > que la salida se traslade de manera coherente, ya que los pesos no están compartidos y la
  > relación entre la entrada y la salida puede cambiar drásticamente.
- ¿Qué pasa con el campo receptivo si se reemplaza un pooling por una convolución con paso 2?
  Ayuda: casi nada pero…
  > Reemplazar un pooling por una convolución con paso 2 tiene un efecto similar en términos de
  > reducción de la resolución espacial de la imagen, ya que ambos métodos disminuyen el tamaño de
  > la representación de la imagen. Sin embargo, la convolución con paso 2 introduce parámetros
  > adicionales (los pesos del filtro) y puede aprender características más complejas, mientras que
  > el pooling simplemente selecciona valores máximos o promedios sin aprender características
  > adicionales. En términos de campo receptivo, ambos métodos aumentan el campo receptivo de las
  > neuronas en la capa siguiente, pero la convolución con paso 2 puede capturar patrones más
  > complejos debido a la capacidad de aprendizaje de los filtros.
- ¿Por qué dos capas convolucionales sin activación entre ellas equivalen a una? Ayuda: convolución
  es una función lineal.
  > Dos capas convolucionales sin activación entre ellas equivalen a una sola capa convolucional
  > porque la convolución es una operación lineal. Cuando se aplican dos convoluciones consecutivas
  > sin una función de activación no lineal entre ellas, el resultado final es simplemente otra
  > convolución que combina los efectos de ambas capas. Esto se debe a que la composición de dos
  > funciones lineales sigue siendo una función lineal, y por lo tanto, no se introduce ninguna no
  > linealidad que pueda cambiar la naturaleza de la operación.
- Una capa 1×1 "combina canales" eso significa que, en cada píxel, toma ese vector y lo multiplica
  por sus pesos, uno por canal, y suma todo. El resultado es un único número nuevo. Para el Ejemplo
  con 3 canales de una imagen RGB y pesos 0,5, 1 y -0,2, calcule el valor del pixel (1,1) con
  R=0,3, G=0,1, B=0.2 Ayuda: salida = $0,5\cdot\left(\text{valor en
  rojo}\right)+1\cdot\left(\text{valor en verde}\right)-0,2\cdot\left(\text{valor en azul}\right)$
  > Calculando el valor del píxel (1,1) utilizando los pesos proporcionados y los valores de los
  > canales de la imagen RGB: Canal rojo (R) = $0,3\cdot0,5=0,15$ Canal verde (G) =
  > $0,1\cdot1=0,10$ Canal azul (B) = $0,2\cdot(-0,2)=-0,04$ Valor del píxel (1,1) =
  > $0,15+0,10-0,04=0,21$

## 5. Use a Python Library for image handling i.e. PIL or CV2 to carry out the following exercise.

Masks are geometric filters on an image. For instance, if we want to extract a region of an image,
we may do it by multiplying the matrix of the original image by a matrix of equal size containing
1's in the region we want to keep and 0's otherwise. Extract a circular region of radious 150 for
the attached image lena_gray_512.tif by following next steps:

- Read the image and convert it to double.
- Create a matrix of the same dimensions filled with zeros.
- Modify the above matrix to contain 1's in a circle of radious 150, i.e. if
  $\left(j-cx\right)2+\left(i-cy\right)<150\exp^{2}$, where (cx,cy) is the center of the image.
- Multiply the image by the mask (they are matrices!)
- Show the result.

When multiplying by zero, you set to black the pixels out of the circle. Modify the program to make
visible those pixels with half the intensity. Hint. a.shape[0] is the number of rows of a and
a.shape[1] the number of columns.

> Resultado del ejercicio de enmascaramiento circular aplicado a la imagen `lena_gray_512.tif`. Se
> ha creado una máscara circular de radio 150 centrada en la imagen, y se ha multiplicado la imagen
> original por esta máscara. Los píxeles fuera del círculo se han establecido a negro (intensidad
> cero) en la primera versión, y en la segunda versión se han reducido a la mitad su intensidad
> para hacerlos visibles. ![mascara con exterior negro][circle_mask]
> ![mascara con intensidad reducida][circle_mask_2]

---

<!-- [model_performance_comparison]: D:\deep-learning-labs\reports\figures\model_performance_comparison.png
[validation_mae_by_number_of_trials]: D:\deep-learning-labs\reports\figures\validation_mae_by_number_of_trials.png
[circle_mask]: D:\deep-learning-labs\reports\figures\circle_mask.png
[circle_mask_2]: D:\deep-learning-labs\reports\figures\circle_mask_2.png -->

[model_performance_comparison]: reports/figures/model_performance_comparison.png
[validation_mae_by_number_of_trials]: reports/figures/validation_mae_by_number_of_trials.png
[circle_mask]: reports/figures/circle_mask.png
[circle_mask_2]: reports/figures/circle_mask_2.png
