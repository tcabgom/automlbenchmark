#!/usr/bin/env python3
# frameworks/TFMAutoML_Test/exec.py

import os
import logging
import pickle
import numpy as np
import json
import time

log = logging.getLogger(__name__)

# COMANDO: python runbenchmark.py TFMAutoML_Test test -f 0 -m local

def run(dataset, config):
    """
    Ejecutar TFM_AutoML con los datos y configuracion dados

    Args:
        dataset: objeto Dataset de AMLB con los datos de train/test
        config: objeto Config de AMLB con la configuracion del benchmark (tipo de problema, metricas, etc)
    """
    from basicautoml.main import TFM_AutoML
    from basicautoml.config import AutoMLConfig

    # result: Helper del bechmark para recoger/gardar resultados
    # output_subdir: Helper para crear subdirectorios de salida
    # measure_inference_times: Helper para medir tiempos de inferencia
    # Timer: Context manager para medir tiempos
    from frameworks.shared.callee import result, output_subdir, measure_inference_times
    from frameworks.shared.utils import Timer

    log.info("**** Running AutoML ****")

    # Detectar si es clasificacion o regresion e intentar sacar el nombre de la etiqueta
    is_classification = getattr(config, "type", None) == "classification"
    label = getattr(dataset, "target", None)
    if label is not None:
        label = label.name if hasattr(label, "name") else label

    # Extraer los datos que AMLB pasa
    X_train, y_train = dataset.train.X, dataset.train.y
    X_test, y_test = dataset.test.X, dataset.test.y

    # Asegurar que las etiquetas son str en clasificacion para asegurar que no las interpreta como continuas, sino categoricas
    if is_classification:
        try:
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)
        except Exception:
            # si no tiene astyp, forzamos con list comprehension
            y_train = [str(v) for v in y_train]
            y_test = [str(v) for v in y_test]

    # Configurar AutoML
    automl_config = AutoMLConfig(
        test_size=0.2,
        random_state=42,
        search_type="bayesian",
        scoring="roc_auc",
        verbose=True,
        n_trials=5,
        timeout=20
    )

    # Crear instancia de AutoML a partir de la configuracion
    automl = TFM_AutoML(automl_config)

    # Entrenar con el conjunto de entrenamiento
    with Timer() as training:
        automl.fit(X_train, y_train)

    # Predecir con el conjunto de prueba
    with Timer() as predict:
        preds_proba = automl.predict_proba(X_test) # Obtenemos probabilidades
        preds = np.argmax(preds_proba, axis=1)     # Convertimos a etiquetas TODO esto se podria mejorar

    # Guardar artefactos si se pide
    save_artifacts(automl, config, output_subdir)

    # Devolver resultados
    return result(
        # Ruta donde guardar las predicciones. None significa que no se guardan
        output_file=getattr(config, "output_predictions_file", None),
        # Predicciones de clase (n_samples,) o None si no se tienen
        predictions=preds,
        # Predicciones de probabilidad (n_samples, n_classes) o None si no se tienen
        probabilities=preds_proba,
        # Valores reales (n_samples,) o None si no se tienen
        truth=y_test,
        # Tiempos de entrenamiento (automl.fit)
        training_duration=training.duration,
        # Tiempos de prediccion (automl.predict)
        predict_duration=predict.duration,
    )


def save_artifacts(automl, config, output_subdir_fn):
    try:
        artifacts = getattr(config, "framework_params", {}).get("_save_artifacts", [])
        models_dir = output_subdir_fn("models", config)

        if "models" in artifacts:
            with open(os.path.join(models_dir, "automl_model.pkl"), "wb") as f:
                pickle.dump(automl, f)

    except Exception:
        log.warning("Error when saving artifacts", exc_info=True)


if __name__ == "__main__":
    from frameworks.shared.callee import call_run
    call_run(run)
