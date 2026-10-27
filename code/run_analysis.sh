#!/usr/bin/env bash
# Rebuild everything downstream of the simulation run files in data/runs.
#
#   bash code/run_analysis.sh            compile, threshold curve, all figures (uses the released fits)
#   bash code/run_analysis.sh --refit    additionally refit all 36 surfaces (several minutes; writes results/fits)
#
# Run from any directory; paths are resolved from the repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python code/02_compile_results/compile_results.py data/runs data/processed/qaoa_unified_dataset_new.csv --expect-pairs 36
python code/03_fit_thresholds/extract_threshold_curve.py

if [[ "${1:-}" == "--refit" ]]; then
    mkdir -p results
    python code/03_fit_thresholds/fit_logistic_thresholds.py > results/refit.log 2>&1
fi

for f in code/05_figures/main/*.py code/05_figures/supp/*.py; do
    echo "figure: $f"
    python "$f" > /dev/null
done
if python -c "import qiskit_ibm_runtime" 2>/dev/null; then
    echo "hardware analysis: code/04_hardware/hardware_analysis.py"
    python code/04_hardware/hardware_analysis.py > /dev/null
else
    echo "skipped code/04_hardware/hardware_analysis.py (Figure 8): install requirements-hardware.txt"
fi
echo "done: figures/main, figures/supp"
