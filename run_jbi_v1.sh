#!/usr/bin/env bash
# Runs every analysis of the journal article (release jbi-v1) in order.
# Usage: bash run_jbi_v1.sh [path/to/manuscript.tex]
set -euo pipefail
cd "$(dirname "$0")"
run() { echo "== $*"; python -m "$@"; }
run src.analysis.run_metric_comparison
run src.analysis.run_cabc_bootstrap
run src.analysis.run_brfss_multiseed --stage all --seeds 5
run src.analysis.run_brfss_multiseed --stage all --seeds 3 --class-weighting none
run src.analysis.run_brfss_multiseed --stage all --seeds 3 --class-weighting balanced
run src.analysis.run_brfss_multiseed --stage cw-compare
run src.analysis.run_label_axis --stage all --seeds 10
run src.analysis.run_label_axis --stage all --seeds 10 --train-weighted
run src.analysis.run_label_axis --outcome hypertension --class-weighting pipeline --seeds 5
run src.analysis.run_label_axis --outcome hypertension --class-weighting none --seeds 5
run src.analysis.run_label_fairness --seeds 10
run src.analysis.run_variance_decomposition
run src.analysis.run_instance_vs_population
run src.analysis.run_brfss_income_harmonised
run src.analysis.run_sensitivity_checks --n-perm 999
run src.analysis.run_dependence_checks
run src.analysis.run_soa_comparison
run src.analysis.build_reproducibility_table
run src.analysis.manuscript_numbers
run src.analysis.make_supplementary_tables
run src.analysis.make_journal_figures
if [ $# -ge 1 ]; then run src.analysis.check_manuscript_numbers "$1"; fi
run tests.test_agreement_metrics
run tests.test_journal_analyses
