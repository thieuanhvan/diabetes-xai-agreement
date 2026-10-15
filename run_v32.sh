#!/usr/bin/env bash
# v32 outcome-definition analysis. Usage: bash run_v32.sh [manuscript.tex supplementary.tex]
set -euo pipefail
cd "$(dirname "$0")"
run() { echo "== $*"; python -m "$@"; }
run src.analysis.run_detection_power
run src.analysis.run_label_controls --stage all --outcome both
run src.analysis.run_label_null --outcome diabetes --R 50
run src.analysis.run_label_null --outcome diabetes --R 30 --gamma 0.3
run src.analysis.run_label_null --outcome hypertension --R 30
run src.analysis.run_label_null --outcome hypertension --R 30 --gamma 0.35
run src.analysis.run_label_null --outcome diabetes --R 30 --gamma 0.4
run src.analysis.run_label_null --outcome hypertension --R 30 --gamma 0.25
run src.analysis.run_null_validation --stage all
run src.analysis.run_reviewer_checks
run src.analysis.run_holdout_cycle --outcome diabetes --stage all
run src.analysis.run_holdout_cycle --outcome hypertension --stage all
run src.analysis.make_supplementary_tables
run src.analysis.make_manuscript_outputs
if [ $# -ge 1 ]; then run src.analysis.check_manuscript_numbers "$@"; fi
run tests.test_v32
