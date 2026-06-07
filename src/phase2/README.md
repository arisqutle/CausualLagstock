# Phase 2: STACD Causal Discovery

Phase 2 trains the STACD event-sequence model used by Phase 3. It consumes
Phase 1 embeddings plus event metadata, builds ticker-local chronological
event windows, labels each horizon as up/down/flat from the next trading-day
close-to-close return, and saves both the model checkpoint and the learned
causal/lag matrices.

## Current Training Entry Point

Run from `DirectionA`:

```powershell
C:\Users\fer20\anaconda3\python.exe src\phase2\train.py `
  --epochs 12 `
  --batch-size 64 `
  --output-dir data\stage2_repaired `
  --device cpu
```

The script is CPU/CUDA safe. If `--device cuda` is requested without CUDA, it
falls back to CPU.

## Data Handling Fixes

The repaired dataset loader:

- reads the actual lowercase price schema: `date`, `open`, `high`, `low`,
  `close`, `adj close`, `volume`;
- filters to valid 20-class event labels only;
- groups and sorts events chronologically within each ticker;
- skips windows without a valid next-trading-day label;
- now uses a three-class direction target (`down`, `up`, `flat`) and still
  ignores only windows that cannot be labeled.

With the current local artifacts this builds `9527` valid Stage 2 examples.

## Outputs

The trainer writes:

- `best_model.pt`: model state dict selected by validation accuracy;
- `last_model.pt`: final model state dict;
- `best_graph.pt`: learned `causal_matrix` and `lag_matrix` from the best epoch;
- `causal_epoch_<n>.pt`: per-epoch graph snapshots;
- `metrics.csv`: train/validation losses, accuracy, and graph diagnostics;
- `run_meta.json`: run arguments and summary metrics.

The repaired 12-epoch run selected epoch 2:

```text
best_val_accuracy = 0.5362
causal_matrix mean/std = 0.2672 / 0.0479
causal_matrix min/max = 0.1228 / 0.4060
lag_matrix mean/std = 1.3256 / 0.4199
```

This fixes the previous graph-collapse symptom where the Phase 3 graph had
near-zero off-diagonal variation.

## Downstream Check

The repaired graph and repaired Stage 2 event representations were precomputed
to:

```text
data/stage3/phase3_dataset_ticker_repaired_precomputed.pt
```

The first full-data Phase 3 check with weighted CE reached:

```text
best validation macro-F1 = 0.4185
test macro-F1 = 0.4167
test accuracy = 0.4242
```

So the Stage 2 graph-collapse bug is fixed, but this repaired Stage 2 checkpoint
does not yet improve Phase 3. The next useful work is to improve the Stage 2
training objective or labels, not just run more Phase 3 hyperparameter sweeps.
