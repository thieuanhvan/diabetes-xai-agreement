import logging
import time

from sklearn.pipeline import Pipeline

from models.baselines import get_baseline_estimators
from evaluation.classification import evaluate_classification_pipeline


def run_step03_baseline_models(data):
    """
    STEP 03 - Train baseline models.
    Returns (results, trained_models) tuple.
    trained_models: dict {model_name: fitted Pipeline}
    """

    logging.info("============================================================")
    logging.info("STEP 03 - BASELINE MODELS")
    logging.info("============================================================")

    estimators = get_baseline_estimators()

    results        = []
    trained_models = {}

    for name, model in estimators.items():

        logging.info(f"Training baseline model: {name}")

        pipeline = Pipeline(steps=[
            ("preprocessor", data["preprocessor"]),
            ("model", model)
        ])

        t_start = time.perf_counter()
        pipeline.fit(data["X_train"], data["y_train"])
        training_time_s = round(time.perf_counter() - t_start, 2)

        logging.info(f"Training time for {name}: {training_time_s}s")

        metrics = evaluate_classification_pipeline(
            pipeline, data["X_test"], data["y_test"], name
        )
        metrics["training_time_s"] = training_time_s

        logging.info(f"Metrics for {name}: {metrics}")

        results.append(metrics)
        trained_models[name] = pipeline

    logging.info("STEP 03 COMPLETED")

    return results, trained_models