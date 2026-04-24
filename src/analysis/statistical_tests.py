from pathlib import Path
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


def run_mcnemar_test(
    y_true,
    model_a_pred,
    model_b_pred,
    output_path: Path,
):
    """
    Run McNemar statistical test to compare two classifiers.
    """

    b01 = ((model_a_pred == y_true) & (model_b_pred != y_true)).sum()
    b10 = ((model_a_pred != y_true) & (model_b_pred == y_true)).sum()

    table = [[0, b01], [b10, 0]]

    result = mcnemar(table, exact=False, correction=True)

    df = pd.DataFrame({
        "b01": [b01],
        "b10": [b10],
        "statistic": [result.statistic],
        "p_value": [result.pvalue],
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df