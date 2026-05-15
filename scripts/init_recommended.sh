#!/bin/bash
# Initialise the recommended forward sessions for jusik.
# Run once after cloning the repo or after the results/ directory is cleared.

set -euo pipefail

BUDGET="${BUDGET:-10000000}"
START="${START:-2024-05-15}"

run_init() {
    local name="$1"; shift
    if [ -f "results/forward/$name/config.json" ]; then
        echo "skip (exists): $name"
        return
    fi
    jusik forward init "$name" "$@"
}

# Stable across regimes (OOS Sharpe +3.23):
run_init lowvol_multi5 \
    --strategy low_vol --params top_n=5 window=20 \
    --trade-mode multiday5 --budget-mode compound \
    --budget "$BUDGET" --start "$START"

# Diversifies lookback window (OOS Sharpe +3.18):
run_init lowvol_multi5_w60 \
    --strategy low_vol --params top_n=5 window=60 \
    --trade-mode multiday5 --budget-mode compound \
    --budget "$BUDGET" --start "$START"

# Recent regime winner — keep position size small (OOS Sharpe +6.06 but regime-dependent):
run_init volspike_overnight \
    --strategy volume_spike --params top_n=3 window=20 require_green=false \
    --trade-mode overnight --budget-mode compound \
    --budget "$BUDGET" --start "$START"

echo
echo "next: jusik forward run-all"
