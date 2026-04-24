from pathlib import Path
import pandas as pd


def generate_paper_tables(
    shap_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
    model_name: str = "xgboost",
):
    """
    Generate tables used in the research paper.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    shap_path = shap_dir / f"{model_name}_shap_feature_importance.csv"
    fp_path = analysis_dir / "false_positive_samples.csv"
    fn_path = analysis_dir / "false_negative_samples.csv"

    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        shap_df.head(10).to_csv(output_dir / "table_top_features.csv", index=False)

    if fp_path.exists():
        pd.read_csv(fp_path).head(20).to_csv(
            output_dir / "table_false_positives.csv",
            index=False,
        )

    if fn_path.exists():
        pd.read_csv(fn_path).head(20).to_csv(
            output_dir / "table_false_negatives.csv",
            index=False,
        )