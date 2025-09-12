#!/usr/bin/env python3
# frameworks/TFMAutoML_Test/exec.py

import os
import logging
import pickle
import numpy as np

log = logging.getLogger(__name__)

# COMANDO: python runbenchmark.py TFMAutoML_Test test -f 0 -m local

def run(dataset, config):

    # Imports dentro de run() para evitar problemas de paquete???
    from frameworks.TFMAutoML_Test.TFM.BasicAutoML.src.main import TFM_AutoML
    from frameworks.TFMAutoML_Test.TFM.BasicAutoML.src.config import AutoMLConfig
    # from frameworks.TFMAutoML_Test.TFM.BasicAutoML.src.data_loader import DataLoader

    # AMLB helpers
    from frameworks.shared.callee import result, output_subdir, measure_inference_times
    from frameworks.shared.utils import Timer

    log.info("**** Running AutoML ****")

    # Necesario para AutoMLBenchmark
    is_classification = getattr(config, "type", None) == "classification"
    label = getattr(dataset, "target", None)
    if label is not None:
        label = label.name if hasattr(label, "name") else label

    X_train = dataset.train.X
    y_train = dataset.train.y
    X_test = dataset.test.X
    y_test = dataset.test.y

    if is_classification:
        # Asegurarnos que no hay NaNs y convertir a str si hace falta
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)

    automl_config = AutoMLConfig(
        test_size=0.2, # TODO Deberia ponerlo a 0, el benchmark creo que lo divide ya
        random_state=42,
        search_type="bayesian",
        scoring="roc_auc",
        verbose=True,
        n_trials=5,
        timeout=20
    )

    automl = TFM_AutoML(automl_config)

    print("Training...")
    with Timer() as training:
        automl.fit(X_train, y_train)
    print(f"Training finished in {training.duration:.2f}s")

    print("Predicting...")
    with Timer() as predict:
        # Intentar predict_proba, si no existe usar predict
        if hasattr(automl, "predict_proba"):
            preds_proba = automl.predict_proba(X_test)
            if is_classification:
                # Si devuelve probas por clase, tomar argmax
                if preds_proba is None:
                    preds = [None] * len(X_test)
                else:
                    preds_idx = np.argmax(preds_proba, axis=1)
                    # mapear clases si existe label encoder
                    if hasattr(automl, "label_encoder_") and hasattr(automl.label_encoder_, "classes_"):
                        classes = automl.label_encoder_.classes_
                        preds = [classes[i] for i in preds_idx]
                    else:
                        preds = preds_idx.tolist()
            else:
                preds = preds_proba # Regresion
        else: # No existe predict_proba
            preds = automl.predict(X_test) 
            preds_proba = None
    print(f"Prediction finished in {predict.duration:.2f}s")

    # Measure inference times si esta habilitado
    inference_times = {}
    if getattr(config, "measure_inference_time", False):
        def infer(data):
            # same guard: prefer predict_proba si existe
            if hasattr(automl, "predict_proba"):
                return automl.predict_proba(data)
            return automl.predict(data)

    # Guardar artefactos si se solicita
    save_artifacts(automl, config, output_subdir)

    return result(
        output_file=getattr(config, "output_predictions_file", None),
        predictions=preds,
        probabilities=preds_proba,
        truth=y_test,
        training_duration=training.duration,
        predict_duration=predict.duration,
        inference_times=inference_times,
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
    # Importar call_run solo en ejecucion directa, evita imports top-level de frameworks
    from frameworks.shared.callee import call_run
    call_run(run)
