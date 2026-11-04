from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.config import PROJECT_PATH
from src.utils import logger as log

logger = log.LOGGER
config = log.CONFIG


def graficar_coordenadas_paralelas(busquedas):
    """Genera gráficos interactivos de coordenadas paralelas para cada búsqueda.

    Args:
        busquedas: Diccionario de búsquedas ajustadas con hiperparámetros y puntuaciones.

    Returns:
        None.

    Raises:
        TypeError: Si ``busquedas`` no es un diccionario.
        ValueError: Si una búsqueda no contiene resultados graficables.
        OSError: Si no se puede guardar un archivo HTML.

    Examples:
        >>> graficar_coordenadas_paralelas(busquedas_ajustadas)
    """
    # Validar la colección antes de recorrer las búsquedas.
    if not isinstance(busquedas, dict):
        raise TypeError("busquedas debe ser un diccionario.")
    if not busquedas:
        raise ValueError("Se requiere al menos una búsqueda para graficar.")

    # Preparar y guardar un gráfico con los resultados de cada búsqueda.
    for nombre, searcher in busquedas.items():
        try:
            cv_res = searcher.cv_results_
            df_cv = pd.DataFrame(cv_res["params"])
            df_cv["mean_test_score"] = cv_res["mean_test_score"]
        except (AttributeError, KeyError, ValueError) as error:
            logger.exception("No se pudieron obtener resultados graficables para %s.", nombre)
            raise ValueError(f"Resultados incompletos para la búsqueda {nombre}.") from error
        if df_cv.empty or df_cv.shape[1] < 2:
            raise ValueError(f"No hay dimensiones suficientes para graficar la búsqueda {nombre}.")
        if "criterion" in df_cv.columns:
            df_cv["criterion_num"] = df_cv["criterion"].map({"gini": 0, "entropy": 1})
            df_cv = df_cv.drop(columns=["criterion"])
        logger.debug("Generando coordenadas paralelas para %s.", nombre)
        fig = px.parallel_coordinates(
            df_cv,
            color="mean_test_score",
            dimensions=df_cv.columns,
            title=f"Coordenadas paralelas de hiperparámetros - {nombre}",
            color_continuous_scale=px.colors.diverging.Tealrose,
        )
        file_figures = (
            PROJECT_PATH / config.paths.reports.figures / (f"coordenadas_paralelas_{nombre}.html")
        )
        try:
            file_figures.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(file_figures)
        except OSError:
            logger.exception("No se pudo guardar el gráfico de coordenadas en %s.", file_figures)
            raise
        logger.info("Gráfico de coordenadas paralelas guardado en %s.", file_figures)
        fig.show()


def graficar_matriz_confusion(y_test, y_pred, path_salida, params):
    """Genera y guarda la matriz de confusión del modelo evaluado.

    Args:
        y_test: Etiquetas verdaderas.
        y_pred: Etiquetas predichas, alineadas con ``y_test``.
        path_salida: Ruta del archivo de imagen de salida.
        params: Diccionario de configuración que contiene ``CLASES``.

    Returns:
        None.

    Raises:
        ValueError: Si las etiquetas no tienen igual longitud o están vacías.
        ValueError: Si ``params`` no contiene clases válidas.
        OSError: Si no se puede guardar la imagen.

    Examples:
        >>> graficar_matriz_confusion(
        ...     etiquetas, predicciones, 'reports/matriz.png', PARAMS
        ... )
    """
    # Comprobar los datos antes de calcular la matriz.
    if len(y_test) == 0 or len(y_test) != len(y_pred):
        raise ValueError("y_test e y_pred deben tener la misma longitud no vacía.")
    if not isinstance(params, dict) or "CLASES" not in params:
        raise ValueError("params debe ser un diccionario que contenga la clave 'CLASES'.")
    clases = params["CLASES"]
    if len(clases) == 0:
        raise ValueError("params['CLASES'] debe contener al menos una clase.")
    cm = confusion_matrix(y_test, y_pred, labels=range(len(clases)))
    fig, ax = plt.subplots(figsize=(8, 6))

    # Renderizar y guardar la figura, cerrándola también ante un error.
    try:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=clases,
            yticklabels=clases,
            linewidths=0.5,
            square=True,
            ax=ax,
        )
        ax.set_title("Matriz de confusión - BayesSearchCV", size=15)
        ax.set_ylabel("Etiqueta real (verdad de terreno)", size=12)
        ax.set_xlabel("Predicción del modelo", size=12)
        fig.tight_layout()
        destino = Path(path_salida)
        destino.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino, dpi=150, bbox_inches="tight")
        logger.info("Matriz de confusión guardada en %s.", destino)
    except OSError:
        logger.exception("No se pudo guardar la matriz de confusión en %s.", path_salida)
        raise
    finally:
        plt.close(fig)
