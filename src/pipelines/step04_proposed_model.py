import logging
import time
from sklearn.pipeline import Pipeline

from models.proposal import get_proposed_estimators
from evaluation.classification import evaluate_classification_pipeline


def select_best_model(results, trained_models):
    """
    Select best model based on ROC-AUC,
    but PRIORITIZE tree-based models (for explainability).
    """

    logging.info("Selecting best model...")

    # -------------------------------------------------------
    # Priority list (for explainability)
    # -------------------------------------------------------
    preferred_models = [
        "random_forest",
        "xgboost",
        "gradient_boosting"
    ]

    # -------------------------------------------------------
    # Filter candidates
    # -------------------------------------------------------
    candidates = []

    for r in results:
        model_name = r["model"]

        if model_name in preferred_models:
            candidates.append(r)

    # -------------------------------------------------------
    # If no preferred model → fallback all
    # -------------------------------------------------------
    if len(candidates) == 0:
        logging.warning("No tree-based model found → fallback to all models")
        candidates = results

    # -------------------------------------------------------
    # Sort by ROC-AUC
    # -------------------------------------------------------
    candidates = sorted(candidates, key=lambda x: x["roc_auc"], reverse=True)

    best = candidates[0]
    best_model_name = best["model"]

    logging.info(f"Best model selected: {best_model_name}")
    logging.info(f"ROC-AUC: {best['roc_auc']:.4f}")

    best_model = trained_models[best_model_name]

    return best_model, best_model_name


def run_step04_proposed_models(data):
    """
    STEP 04
    Train proposed models + select best model
    """

    logging.info("============================================================")
    logging.info("STEP 04 - PROPOSED MODELS")
    logging.info("============================================================")

    estimators = get_proposed_estimators()

    results = []
    trained_models = {}

    # -------------------------------------------------------
    # Train all models
    # -------------------------------------------------------
    for name, model in estimators.items():

        logging.info(f"Training proposed model: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", data["preprocessor"]),
                ("model", model)
            ]
        )

        t_start = time.perf_counter()
        pipeline.fit(data["X_train"], data["y_train"])
        training_time_s = round(time.perf_counter() - t_start, 2)

        logging.info(f"Training time for {name}: {training_time_s}s")

        metrics = evaluate_classification_pipeline(
            pipeline,
            data["X_test"],
            data["y_test"],
            name
        )

        metrics["training_time_s"] = training_time_s

        results.append(metrics)
        trained_models[name] = pipeline

    # -------------------------------------------------------
    # Select best model (IMPORTANT)
    # -------------------------------------------------------
    best_model, best_model_name = select_best_model(results, trained_models)

    logging.info("STEP 04 COMPLETED")

    return {
        "metrics": results,
        "models": trained_models,
        "best_model": best_model,
        "best_model_name": best_model_name
    }