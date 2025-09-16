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
    from basicautoml.main import TFM_AutoML
    from basicautoml.config import AutoMLConfig
    from frameworks.shared.callee import result, output_subdir
    from frameworks.shared.utils import Timer

    log.info("**** Running AutoML ****")

    # === PASO 1: Datos ===
    X_train, y_train = dataset.train.X, dataset.train.y
    X_test, y_test   = dataset.test.X, dataset.test.y

    # === PASO 2: Preprocesamiento ===
    is_classification = getattr(config, "type", None) in ("classification", "multiclass", "binary")
    if is_classification:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test  = le.transform(y_test)

    # === PASO 3: Configuración AutoML ===
    automl_config = AutoMLConfig(
        test_size=0.2,
        random_state=42,
        search_type="bayesian",
        scoring="roc_auc" if len(set(y_train)) == 2 else "accuracy",
        verbose=True,
        n_trials=5,
        timeout=20
    )
    automl = TFM_AutoML(automl_config)

    with Timer() as training:
        automl.fit(X_train, y_train)

    # === PASO 4: Predicciones ===
    with Timer() as predict:
        preds_proba = automl.predict_proba(X_test)
        preds_idx = np.argmax(preds_proba, axis=1)
        preds = le.inverse_transform(preds_idx) if is_classification else preds_idx
        y_test = le.inverse_transform(y_test) if is_classification else y_test

    # === PASO 5: Guardado de resultados en archivos ===
    save_predictions_and_metadata(preds, preds_proba, y_test, training, predict, config)

    # === PASO 6: Devolver resultados al benchmark ===
    return result(
        output_file=getattr(config, "output_predictions_file", None),
        predictions=tuple(preds.tolist()),
        probabilities=tuple(tuple(float(x) for x in row) for row in preds_proba),
        truth=[str(v) for v in y_test],
        training_duration=float(training.duration),
        predict_duration=float(predict.duration),
        others=json.dumps(collect_metadata(config), default=str)
    )


def save_predictions_and_metadata(preds, preds_proba, y_test, training, predict, config):
    """Guarda predicciones y metadatos en CSV/JSON si se especifica output_predictions_file"""
    out_file = getattr(config, "output_predictions_file", None)
    if not out_file:
        return

    try:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        import pandas as pd

        # CSV con predicciones + probabilidades
        cols = {"prediction": preds.tolist()}
        if preds_proba.ndim == 2:
            for i in range(preds_proba.shape[1]):
                cols[f"prob_class_{i}"] = preds_proba[:, i].astype(float).tolist()
        else:
            cols["probabilities"] = preds_proba.tolist()
        pd.DataFrame(cols).to_csv(out_file, index=False)

        # Metadata auxiliar
        metadata = collect_metadata(config)
        metadata.update({
            "training_duration": training.duration,
            "predict_duration": predict.duration,
            "n_samples": len(preds),
        })
        meta_path = os.path.join(os.path.dirname(out_file), "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=2)

    except Exception:
        log.warning("Could not write predictions file or metadata", exc_info=True)


def collect_metadata(config):
    """Extrae metadatos relevantes del objeto config."""
    return {
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
        "framework_params": json.dumps(getattr(config, "framework_params", {}), default=str),
        "git_info": json.dumps(getattr(config, "git_info", {}), default=str),
    }


def save_artifacts(automl, config, output_subdir_fn):
    try:
        artifacts = getattr(config, "framework_params", {}).get("_save_artifacts", [])
        if "models" in artifacts:
            models_dir = output_subdir_fn("models", config)
            with open(os.path.join(models_dir, "automl_model.pkl"), "wb") as f:
                pickle.dump(automl, f)
    except Exception:
        log.warning("Error when saving artifacts", exc_info=True)


if __name__ == "__main__":
    from frameworks.shared.callee import call_run
    call_run(run)
