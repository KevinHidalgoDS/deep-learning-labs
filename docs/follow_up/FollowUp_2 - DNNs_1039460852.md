---
title: "FollowUp 2"
author: "_Kevin Hidalgo_"
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

# FollowUp 2

## 1. Lleve a cabo una operación de Hyperparameter Tuning

Para el ejercicio de regresión de la clase pasada, encontrar los mejores hyperparametros tales
como: learning rate, dropout, y arquitectura de la red en términos del número de capas. Ver la
descripción en el notebook: Hyperparameter Fine Tunning.ipynb

## 2. Use teacheble machines (https://teachablemachine.com)

Visite ese sitio de Google y entrene varios modelos desbalanceados y no usando las imagenes de
perros y gatos provistas en el drive, cree varias clases, juegue con los hiperparámetros, saque
conclusiones, haga pantallazos y reporte los hallazgos de al menos 3 modelos diferentes.

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

## 4. Responda las siguientes preguntas según la clase

- ¿Por qué la salida de una convolución es equivariante a la traslación y la de una red
  completamente conectada no? Ayuda: pesos compartidos en la CNN.
- ¿Qué pasa con el campo receptivo si se reemplaza un pooling por una convolución con paso 2?
  Ayuda: casi nada pero…
- ¿Por qué dos capas convolucionales sin activación entre ellas equivalen a una? Ayuda: convolución
  es una función lineal.
- Una capa 1×1 "combina canales" eso significa que, en cada píxel, toma ese vector y lo multiplica
  por sus pesos, uno por canal, y suma todo. El resultado es un único número nuevo. Para el Ejemplo
  con 3 canales de una imagen RGB y pesos 0,5, 1 y -0,2, calcule el valor del pixel (1,1) con
  R=0,3, G=0,1, B=0.2 Ayuda: salida = $0,5\cdot\left(\text{valor en
  rojo}\right)+1\cdot\left(\text{valor en verde}\right)-0,2\cdot\left(\text{valor en azul}\right)$

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
