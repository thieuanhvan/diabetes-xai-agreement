from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def generate_model_comparison_plots(
    results_df: pd.DataFrame,
    output_path: Path,
):
    """
    Generate bar chart comparing model performance.
    """

    metrics = ["accuracy", "f1", "roc_auc"]

    results_df.set_index("model")[metrics].plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()