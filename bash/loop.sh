#!/usr/bin/env bash
set -euo pipefail

# ── 1. Build two Bash arrays with 25 points in [0, 1] ──────────────────────────
readarray -t batch_compression_rates  < <(
  python - <<'PY'
import numpy as np
for v in np.linspace(0.1, 0.5, 10):
    print(f'{v:.6f}')
PY
)

readarray -t token_compression_rates  < <(
  python - <<'PY'
import numpy as np
for v in np.linspace(0.1, 1, 25):
    print(f'{v:.6f}')
PY
)

# ── 2. Grid-search the two hyper-parameters ────────────────────────────────────
for batch_compression_rate in "${batch_compression_rates[@]}"; do
  for token_compression_rate in "${token_compression_rates[@]}"; do

    # numeric value that main.py expects
    k_over_n=$(python - <<PY
batch  = $batch_compression_rate
token  = $token_compression_rate
print((192*197*0.5 * batch * token) / (224*224*3))
PY
)

    python main.py \
      hyperparameters.k_over_n="$k_over_n" \
      method.parameters.token_compression_rate="$token_compression_rate" \
      method.parameters.batch_compression_rate="$batch_compression_rate" \
    || {
      echo "⚠️  Skipping combination: batch=$batch_compression_rate, token=$token_compression_rate"
      continue
    }

  done
done
