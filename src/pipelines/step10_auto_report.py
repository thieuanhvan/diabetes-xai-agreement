import logging
from pathlib import Path

from utils.project_paths import get_outputs_dir
from reporting.auto_report import generate_auto_report


def run_step10_auto_report():

    logging.info("============================================================")
    logging.info("STEP 10 - AUTO REPORT GENERATION")
    logging.info("============================================================")

    # Folder chứa outputs của pipeline
    report_dir = get_outputs_dir()

    # File report output
    output_path = Path(report_dir) / "auto_report.md"

    report_path = generate_auto_report(report_dir, output_path)

    logging.info(f"Report generated at: {report_path}")
    logging.info("STEP 10 COMPLETED")

    return report_path