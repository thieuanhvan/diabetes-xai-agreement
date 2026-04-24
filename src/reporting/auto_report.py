from pathlib import Path
import pandas as pd


def generate_auto_report(
    tables_dir: Path,
    output_path: Path,
):
    """
    Generate markdown report summarizing experiment results.
    """

    report_lines = ["# Experiment Report\n"]

    tables = sorted(tables_dir.glob("*.csv"))

    for table in tables:
        df = pd.read_csv(table)

        report_lines.append(f"## {table.stem}\n")
        report_lines.append(df.head().to_markdown())
        report_lines.append("\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return output_path