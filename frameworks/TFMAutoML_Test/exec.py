#!/usr/bin/env python3
# frameworks/TFMAutoML_Test/exec.py
from sklearn.preprocessing import LabelEncoder

import os
import logging
import pickle
import numpy as np
import json
import time

log = logging.getLogger(__name__)

# COMANDO: python runbenchmark.py TFMAutoML_Test test -f 0 -m local -t kc2 iris

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

    # Preprocesar etiquetas si es clasificacion (convertir a enteros 0,1,...,n_classes-1)
    if is_classification:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    # Configurar AutoML
    automl_config = AutoMLConfig(
        test_size=0.2,
        random_state=42,
        search_type="bayesian",
        scoring="roc_auc" if len(set(y_train)) == 2 else "accuracy",
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
        preds = np.array([np.unique(y_train)[i] for i in np.argmax(preds_proba, axis=1)])     # Convertimos a etiquetas TODO esto se podria mejorar
        print("   - Predicciones de probabilidad:", preds_proba)
        print("   - Predicciones de clase:", preds)

    # Guardar artefactos si se pide
    save_artifacts(automl, config, output_subdir)

    # Guardar predicciones/probabilidades en CSV (evita "predictions file missing")
    out_file = getattr(config, "output_predictions_file", None)
    try:
        if out_file:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            import pandas as _pd
            # Preparar dataframe de salida: predicción + columnas de probabilidades por clase
            cols = {"prediction": preds.tolist()}
            # si preds_proba es 2D, anyadir columnas prob_0, prob_1, ...
            try:
                for i in range(preds_proba.shape[1]):
                    cols[f"prob_class_{i}"] = preds_proba[:, i].astype(float).tolist()
            except Exception:
                # En caso de que preds_proba no sea array 2D, intentar convertir a lista
                cols["probabilities"] = preds_proba.tolist() if hasattr(preds_proba, "tolist") else list(preds_proba)

            df_out = _pd.DataFrame(cols)
            df_out.to_csv(out_file, index=False)

            # Guardar metadata auxiliar
            meta_path = os.path.join(os.path.dirname(out_file), "metadata.json")
            metadata = {
                # Atributos generales
                "framework": "TFMAutoML_Test",
                "training_duration": training.duration,
                "predict_duration": predict.duration,
                "n_samples": int(len(preds)),
                "framework_version": "0.0.1", 
                "utc": time.strftime("%Y-%m-%dT%H:%M:%S"),
                
                # Atributos del benchmark
                "type_": getattr(config, "type", None),
                "seed": getattr(config, "seed", None),
                "metric": getattr(config, "metric", None),
                "metrics": getattr(config, "metrics", None),
                "fold": getattr(config, "fold", None),
                "task": getattr(config, "name", None),
                "openml_task_id": getattr(config, "openml_task_id", None),

                # Convertimos a JSON strings los dicts para evitar problemas de serializacion
                "framework_params": json.dumps(getattr(config, "framework_params", {})),
                "git_info": json.dumps(getattr(config, "git_info", {})),
            }
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(metadata, mf, indent=2)
    except Exception:
        log.warning("Could not write predictions file or metadata", exc_info=True)

    # --- Preparar versiones 'hashables' de predictions/probabilities para devolver en result() ---
    # pandas puede fallar con listas/ndarray/dict (no-hasheables). Para mantener las probabilidades
    # y al mismo tiempo evitar 'unhashable type' convertimos a tuplas (hashables).
    try:
        preds_hashable = tuple(preds.tolist())
    except Exception:
        preds_hashable = None

    try:
        # convertir cada fila de probabilidades a una tupla de floats
        probs_hashable = tuple(tuple(float(x) for x in row) for row in preds_proba)
    except Exception:
        probs_hashable = None

    # Convertir truth a lista simple (no hay problema en dejarla como lista)
    try:
        truth_list = list(y_test)
    except Exception:
        truth_list = [v for v in y_test]

    """
    print(">>> DEBUG CONFIG ATTRS <<<")
    for attr in dir(config):
        if not attr.startswith("_"):
            try:
                val = getattr(config, attr)
                print(f"{attr}: {type(val)} -> ejemplo: {str(val)[:200]}")
            except Exception as e:
                print(f"{attr}: ERROR al acceder ({e})")
    print(">>> END CONFIG DEBUG <<<")

    print(">>> DEBUG RESULT ARGS <<<")
    res_args = dict(
        output_file=getattr(config, "output_predictions_file", None),
        predictions=preds_hashable,
        probabilities=probs_hashable,
        truth=truth_list,
        training_duration=training.duration,
        predict_duration=predict.duration,
    )

    for k, v in res_args.items():
        print(f"{k}: type={type(v)}")
        if isinstance(v, (list, tuple)):
            print(f"   first element type: {type(v[0]) if len(v) > 0 else None}")
    print(">>> END RESULT ARGS <<<")
    """

    def sanitize(obj):
        # Si es un dict o Namespace → convertir a JSON string
        if isinstance(obj, (dict,)):
            return json.dumps(obj, default=str)
        # Si es lista/tuple → procesar elementos
        if isinstance(obj, (list, tuple)):
            return type(obj)(sanitize(x) for x in obj)
        return obj

    output_file = getattr(config, "output_predictions_file", None)

    def sanitize_for_result(obj):
        import json
        if isinstance(obj, (dict, list, tuple)):
            try:
                return json.dumps(obj, default=str)
            except Exception:
                return str(obj)
        return obj

    # Preparar lo estándar
    preds_hashable = tuple(preds.tolist()) if preds is not None else None
    probs_hashable = tuple(tuple(float(x) for x in row) for row in preds_proba) if preds_proba is not None else None
    truth_list = [str(v) for v in y_test]

    # Sanitizar
    preds_hashable = sanitize_for_result(preds_hashable)
    probs_hashable = sanitize_for_result(probs_hashable)
    truth_list    = sanitize_for_result(truth_list)

    others_dict = {
        "framework": "TFMAutoML_Test",
        "framework_version": "0.0.1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "type_": getattr(config, "type", None),
        "seed": getattr(config, "seed", None),
        "metric": getattr(config, "metric", None),
        "metrics": getattr(config, "metrics", None),
        "fold": getattr(config, "fold", None),
        "task": getattr(config, "name", None),
        "openml_task_id": getattr(config, "openml_task_id", None),
        "framework_params": sanitize_for_result(getattr(config, "framework_params", {})),
        "git_info": sanitize_for_result(getattr(config, "git_info", {})),
    }

    # Devolver resultados
    return result(
        output_file=getattr(config, "output_predictions_file", None), # Ruta donde guardar las predicciones. None significa que no se guardan
        predictions=preds_hashable,                                   # Predicciones de clase (n_samples,) o None si no se tienen
        probabilities=probs_hashable,                                 # Probabilidades (n_samples, n_classes) o None si no se tienen
        truth=truth_list,                                             # Valores reales (n_samples,) o None si no se tienen
        training_duration=float(training.duration),                   # Tiempos de entrenamiento (automl.fit)
        predict_duration=float(predict.duration),                     # Tiempos de prediccion (automl.predict)
        others=json.dumps(others_dict, default=str)                   # Otros datos adicionales que se quieran guardar
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
