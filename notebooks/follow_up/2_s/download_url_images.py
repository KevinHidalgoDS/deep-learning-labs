#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cliente HTTP para la orquestación y consumo de la API IA OCR BCT.

Proporciona métodos tipados para interactuar con los endpoints de registro,
análisis de malware y procesamiento de documentos. Implementa inyección
segura de credenciales y políticas de reintento transitorias.

Programa: download_url_images.py

Soporte: kfhidalgoh@unal.edu.co

Versión: 1.0.0

Lenguaje: Python 3.14.5

CD: 20261104

LUD: 20261104

Comentarios:
    - 20261104 Kevin Hidalgo -> creacion inicial.
"""
#%%
__authors__ = ["Kevin Hidalgo"]
__contact__ = "kfhidalgoh@unal.edu.co"
__copyright__ = "Copyright 2026, Universidad Nacional de Colombia"
__credits__ = ["Kevin Hidalgo"]
__email__ = "kfhidalgoh@unal.edu.co"
__status__ = "Desarrollo"
__version__ = "1.0.0"
__date__ = "2026-11-04"

import os
import sys

import pandas as pd
import requests
from src.utils import logger as log
from src.config import PROJECT_PATH
from pathlib import Path
from tqdm import tqdm

logger = log.LOGGER
config = log.CONFIG

logger.info("Python %s en %s", sys.version, sys.platform)

#%%
def _leer_urls_desde_csv(csv_file: str) -> list[str]:
    """Lee y valida las URLs contenidas en un archivo CSV.

    Args:
        csv_file (str): Ruta al archivo CSV que contiene una URL por línea.
            Se admite una cabecera opcional con el texto ``"Image URL"``.

    Returns:
        List[str]: Lista de URLs válidas (no vacías) encontradas en el archivo.

    Raises:
        FileNotFoundError: Si el archivo especificado no existe.
        PermissionError: Si no hay permisos para leer el archivo.
        UnicodeDecodeError: Si el archivo no puede decodificarse como UTF-8.
    """
    # ------------------------------------------------------------------
    # Bloque 1: Validación de la ruta del archivo de entrada
    # ------------------------------------------------------------------
    if not os.path.isfile(csv_file):
        logger.error("El archivo CSV no existe o no es un archivo válido: %s", csv_file)
        raise FileNotFoundError(f"No se encontró el archivo CSV: {csv_file}")

    logger.debug("Leyendo URLs desde el archivo CSV: %s", csv_file)

    # ------------------------------------------------------------------
    # Bloque 2: Lectura de líneas preservando comas internas
    # ------------------------------------------------------------------
    try:
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            urls = [line.strip() for line in f if line.strip()]
    except UnicodeDecodeError as exc:
        logger.exception("Error de codificación al leer el CSV: %s", csv_file)
        raise UnicodeDecodeError(exc.encoding, exc.object, exc.start, exc.end, str(exc)) from exc
    except PermissionError:
        logger.exception("Permiso denegado para leer el archivo: %s", csv_file)
        raise
    except OSError as exc:
        logger.exception("Error de E/S al leer el archivo CSV: %s", exc)
        raise

    # ------------------------------------------------------------------
    # Bloque 3: Eliminación de la cabecera si está presente
    # ------------------------------------------------------------------
    if urls and urls[0].lower() == "image url":
        logger.debug("Cabecera 'Image URL' detectada; se elimina de la lista.")
        urls = urls[1:]

    if not urls:
        logger.warning("El archivo CSV no contiene URLs válidas: %s", csv_file)
    else:
        logger.info("Se han leído %d URLs desde el archivo CSV.", len(urls))

    return urls


def _descargar_imagen(url: str, filename: str) -> bool:
    """Descarga una imagen individual desde una URL y la guarda en disco.

    Args:
        url (str): URL de la imagen a descargar.
        filename (str): Ruta de destino donde se guardará la imagen.

    Returns:
        bool: ``True`` si la descarga se completó con éxito, ``False`` en
        caso contrario (contenido no válido o error recuperable).

    Raises:
        requests.exceptions.RequestException: Si ocurre un error de red no
            recuperable tras los reintentos internos de ``requests``.
    """
    # ------------------------------------------------------------------
    # Bloque 1: Petición HTTP en modo streaming
    # ------------------------------------------------------------------
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Timeout al descargar la imagen: %s", url)
        return False
    except requests.exceptions.HTTPError as exc:
        logger.error("Error HTTP al descargar %s: %s", url, exc)
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("Error de red al descargar %s: %s", url, exc)
        return False

    # ------------------------------------------------------------------
    # Bloque 2: Validación del tipo de contenido devuelto
    # ------------------------------------------------------------------
    content_type = response.headers.get("Content-Type", "").lower()
    if "image" not in content_type:
        logger.warning("El contenido descargado no es una imagen (%s): %s", content_type, url)
        return False

    # ------------------------------------------------------------------
    # Bloque 3: Escritura del contenido en disco por chunks
    # ------------------------------------------------------------------
    try:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)
    except OSError as exc:
        logger.error("Error al escribir la imagen en disco (%s): %s", filename, exc)
        # Se elimina el archivo parcial para evitar artefactos corruptos.
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except OSError:
            logger.warning("No se pudo eliminar el archivo parcial: %s", filename)
        return False

    logger.debug("Imagen descargada correctamente: %s", filename)
    return True


def download_images(csv_file: str, download_folder: str | Path) -> None:
    """Descarga imágenes a partir de las URLs contenidas en un archivo CSV.

    Lee un archivo CSV con una URL por línea (con cabecera opcional
    ``"Image URL"``), descarga cada imagen y la almacena en la carpeta
    indicada con el nombre ``image_<índice>.jpg``. Los archivos ya
    existentes se omiten para permitir reanudar descargas interrumpidas.

    Args:
        csv_file (str): Ruta al archivo CSV con las URLs de las imágenes.
        download_folder (str | Path): Carpeta de destino donde se guardarán las
            imágenes descargadas. Se crea si no existe.

    Returns:
        None: La función no retorna valor; los resultados se registran en
        el log y se reflejan en el sistema de archivos.

    Raises:
        FileNotFoundError: Si ``csv_file`` no existe.
        ValueError: Si ``download_folder`` está vacío o no es una ruta válida.
        PermissionError: Si no se puede crear la carpeta de destino.

    Examples:
        >>> download_images("urls.csv", "imagenes/")
        2024-01-15 10:00:00 - INFO - Se han leído 42 URLs desde el archivo CSV.
        ...
    """
    logger.info("Iniciando proceso de descarga de imágenes.")
    logger.debug("Parámetros recibidos: csv_file=%s, download_folder=%s", csv_file, download_folder)

    # ------------------------------------------------------------------
    # Bloque 1: Validación de parámetros de entrada
    # ------------------------------------------------------------------
    if not download_folder or not isinstance(download_folder, (str, Path)):
        logger.error("La carpeta de descarga no es válida: %r", download_folder)
        raise ValueError("El parámetro 'download_folder' debe ser una cadena o Path no vacío.")

    # ------------------------------------------------------------------
    # Bloque 2: Creación de la carpeta de destino
    # ------------------------------------------------------------------
    try:
        os.makedirs(download_folder, exist_ok=True)
        logger.info("Carpeta de descarga lista: %s", download_folder)
    except PermissionError:
        logger.exception("Permisos insuficientes para crear la carpeta: %s", download_folder)
        raise
    except OSError as exc:
        logger.exception("No se pudo crear la carpeta de descarga: %s", exc)
        raise

    # ------------------------------------------------------------------
    # Bloque 3: Lectura y validación de URLs
    # ------------------------------------------------------------------
    try:
        urls = _leer_urls_desde_csv(csv_file)
    except FileNotFoundError:
        logger.exception("Archivo CSV no encontrado; se aborta la ejecución.")
        raise
    except Exception:
        logger.exception("Error inesperado al leer el CSV; se aborta la ejecución.")
        raise

    if not urls:
        logger.warning("No hay URLs para procesar; finaliza la ejecución.")
        return

    # ------------------------------------------------------------------
    # Bloque 4: Construcción del DataFrame de seguimiento
    # ------------------------------------------------------------------
    try:
        df = pd.DataFrame({"Image URL": urls})
        logger.debug("DataFrame creado con columnas: %s", df.columns.tolist())
        logger.info("Total de URLs a procesar: %d", len(df))
    except Exception:
        logger.exception("Error al construir el DataFrame de URLs.")
        raise

    # ------------------------------------------------------------------
    # Bloque 5: Iteración y descarga de cada imagen
    # ------------------------------------------------------------------
    exitos = 0
    omitidas = 0
    fallidas = 0

    for i, url in tqdm(
        enumerate(df["Image URL"]),
        total=len(df),
        desc="Descargando imágenes",
    ):
        filename = os.path.join(download_folder, f"image_{i}.jpg")

        # Sub-bloque 5.1: Omitir archivos existentes
        if os.path.exists(filename):
            logger.debug("Archivo existente, se omite: %s", filename)
            omitidas += 1
            continue

        # Sub-bloque 5.2: Descarga con manejo robusto de errores
        try:
            if _descargar_imagen(url, filename):
                exitos += 1
            else:
                fallidas += 1
        except Exception:
            # Captura de salvaguarda: errores inesperados no deben detener el bucle.
            logger.exception("Error inesperado al procesar la URL %d: %s", i, url)
            fallidas += 1

    # ------------------------------------------------------------------
    # Bloque 6: Resumen final del proceso
    # ------------------------------------------------------------------
    logger.info(
        "Proceso de descarga finalizado. Éxitos: %d | Omitidas: %d | Fallidas: %d | Total: %d",
        exitos,
        omitidas,
        fallidas,
        len(df),
    )

#%%

if __name__ == "__main__":

    csv_file = (
        Path(config.paths.data.raw) / "image_urls.csv"
    )  # Ruta al archivo CSV con las URLs
    if not csv_file.exists():
        alternate_path = PROJECT_PATH / config.paths.data.raw / "image_urls.csv"
        if alternate_path.exists():
            logger.warning(
                "No se encontró el archivo en la ruta configurada; se usará %s", alternate_path
            )
            csv_file = alternate_path

    download_folder = PROJECT_PATH / config.paths.reports.figures / "aves"  # Carpeta donde se guardarán las imágenes
    # sonar.issue.ignore.start # NOSONAR
    # if not download_folder.exists():
    #     alternate_path = Path(config.paths.reports.figures) / "aves"
    #     if alternate_path.exists():
    #         logger.warning(
    #             "No se encontró el archivo en la ruta configurada; se usará %s", alternate_path
    #         )
    #         download_folder = alternate_path
    # sonar.issue.ignore.end

    try:
        download_images(csv_file, download_folder)
    except Exception as e:
        logger.error("Se produjo un error durante la descarga de imágenes: %s", e)
