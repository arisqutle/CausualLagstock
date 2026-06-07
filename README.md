# CausalLagStock

CausalLagStock is an end-to-end pipeline for chronological three-class stock
movement prediction from financial news. It extracts structured events with an
LLM, learns an event-type causal-lag graph, and trains a Neural ODE stock
predictor using the learned graph.

The three direction classes are:

- `0`: Up
- `1`: Down
- `2`: Flat

## Pipeline

1. **Stage 1: LLM event extraction**
   - Reads financial-news CSV or JSONL files.
   - Extracts event type, polarity, magnitude, affected tickers, surprise,
     scope, novelty, and credibility.
   - Writes structured event labels as JSONL.
2. **Stage 2: causal-lag graph learning**
   - Encodes headlines with FinBERT.
   - Trains the STACD causal attention model with multi-horizon supervision.
   - Produces a causal-strength matrix \(A\) and lag matrix \(T\).
3. **Stage 3: stock movement prediction**
   - Builds chronological stock-event sequences and price windows.
   - Trains a Neural ODE model with `full`, `no_graph`, `a_only`, `t_only`, or
     `random` graph mode.
   - Evaluates the best checkpoint on the held-out test partition.

## Project Layout

```text
remote/
|-- run_pipeline.py                 # End-to-end pipeline entry point
|-- evaluate.py                     # Standalone Neural ODE evaluation
|-- requirement.txt
|-- data/
|   `-- meta_prices/                # Bundled per-ticker OHLCV price files
`-- src/
    |-- phase1/
    |   `-- llm_event_labeling.py   # Stage 1 LLM extraction
    |-- phase2/
    |   |-- preprocess.py           # FinBERT headline embeddings
    |   `-- train.py                # Causal-lag graph training
    `-- phase3/
        |-- build_dataset.py        # Stage 3 tensor construction
        |-- train_neural_ode.py     # Neural ODE training
        |-- neural_ode_model.py
        `-- tabular_baseline*.py    # HGB/RF/ExtraTrees baselines
```

## Requirements

- Python 3.10 or later
- NVIDIA GPU and CUDA are recommended for Stage 2 and Stage 3
- Sufficient disk space for embeddings and Stage 3 tensor files
- An OpenAI-compatible API endpoint for Stage 1

Create an environment and install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

For CUDA systems, install the PyTorch build matching the installed CUDA version
before installing the remaining requirements. See the official PyTorch
installation instructions for the appropriate command.

## Data Preparation

The repository includes the price data required by Stage 2 and Stage 3 under
`data/meta_prices`. The bundled snapshot contains:

- 825 valid per-ticker CSV files.
- Daily observations from January 2, 2018 through December 29, 2023.
- Approximately 123 MB of price data.
- `all_tickers.txt`, `selected_tickers.txt`, and `failed_tickers.txt` inventory
  files.

The ticker inventory files use tab-separated lines containing the ticker and
its matched-news frequency. `failed_tickers.txt` records tickers for which a
usable price CSV was not produced.

The complete pipeline requires:

- A raw financial-news CSV or JSONL file for Stage 1.
- The bundled per-ticker price CSV files for Stage 2 and Stage 3.
- A FinBERT model available locally or from Hugging Face.

Each bundled price CSV is named after its ticker, such as `AAPL.csv`, and
contains:

```text
Date,Open,High,Low,Close,Adj Close,Volume
```

The included data layout is:

```text
remote/
`-- data/
    `-- meta_prices/
        |-- AAPL.csv
        |-- MSFT.csv
        |-- all_tickers.txt
        |-- selected_tickers.txt
        |-- failed_tickers.txt
        `-- ...
```

The bundled directory can be selected with:

```bash
--price-dir data/meta_prices
```

The repository does not include the raw news dataset, LLM credentials,
FinBERT weights, generated labels, embeddings, learned graphs, Stage 3 tensor
datasets, or trained checkpoints.

## API Configuration

Store one API key per line in a text file outside version control:

```text
key-1
key-2
```

Pass the file explicitly:

```bash
--api-key-file /path/to/keys.txt
```

Alternatively, configure:

```bash
LLM_API_KEY_FILE=/path/to/keys.txt
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5.4
```

Do not commit API keys, private endpoints, or credential files.

## End-to-End Execution

Run the complete pipeline from the `remote` directory:

```bash
python run_pipeline.py \
  --news-input /path/to/news.csv \
  --api-key-file /path/to/keys.txt \
  --price-dir data/meta_prices \
  --run-dir /path/to/pipeline_run \
  --stage2-device cuda \
  --stage3-device cuda
```

On PowerShell:

```powershell
python .\run_pipeline.py `
  --news-input "C:\path\to\news.csv" `
  --api-key-file "C:\path\to\keys.txt" `
  --price-dir ".\data\meta_prices" `
  --run-dir "C:\path\to\pipeline_run" `
  --stage2-device cuda `
  --stage3-device cuda
```

The default configuration uses:

- Sequence length: `32`
- Prediction horizons: `1,3,5,7,10`
- Direction threshold: `0.005`
- Stage 2 epochs: `40`
- Stage 3 epochs: `20`
- Chronological split: `70%/15%/15%`
- Graph mode: `full`

Use `--dry-run` to inspect every generated command without running the
pipeline:

```bash
python run_pipeline.py \
  --news-input /path/to/news.csv \
  --api-key-file /path/to/keys.txt \
  --dry-run
```

## Resume from Existing Artifacts

Individual stages can be skipped when their outputs already exist:

```bash
python run_pipeline.py \
  --skip-llm \
  --labels /path/to/labels.jsonl \
  --price-dir data/meta_prices \
  --run-dir /path/to/pipeline_run
```

Available controls are:

```text
--skip-llm
--skip-embeddings
--skip-stage2
--skip-stage3-build
--skip-stage3-train
--skip-evaluate
```

When skipping a stage, supply its existing output path if it is not located
under the selected run directory.

## Output Files

A normal run produces:

```text
pipeline_run/
|-- labels.jsonl
|-- embeddings/
|   |-- emb.pt
|   `-- meta.json
|-- stage2/
|   |-- best_model.pt
|   |-- best_graph.pt
|   |-- last_model.pt
|   |-- metrics.csv
|   `-- run_meta.json
|-- stage3/
|   `-- phase3_dataset.pt
`-- stage3_model/
    |-- best_model.pt
    |-- last_model.pt
    |-- run_meta.json
    |-- concise_metrics.json
    |-- test_predictions.csv
    |-- confusion.csv
    |-- evaluation.json
    `-- evaluation_predictions.csv
```

## Standalone Evaluation

Evaluate a trained Neural ODE checkpoint:

```bash
python evaluate.py \
  --run-dir /path/to/stage3_model \
  --data /path/to/phase3_dataset.pt \
  --device cuda \
  --output-json /path/to/evaluation.json \
  --predictions-csv /path/to/evaluation_predictions.csv
```

The evaluator reports:

- Accuracy
- Macro-Precision
- Macro-Recall
- Macro-F1
- Weighted-F1
- Matthews correlation coefficient (MCC)
- Per-class precision, recall, F1, and support
- Confusion matrix
- Magnitude MSE and MAE

By default, `evaluate.py` reconstructs the held-out test partition from
`run_meta.json`. It also supports `train`, `val`, `test`, and `all`
partitions.

## Graph Ablations

Select a Stage 3 graph condition with:

```bash
--graph-mode full
--graph-mode no_graph
--graph-mode a_only
--graph-mode t_only
--graph-mode random
```

`no_graph`, `a_only`, and `t_only` remove the corresponding graph inputs from
the Neural ODE model rather than replacing them with zero-valued matrices.

## Runtime Reference

Observed runtime depends on the dataset, GPU, batch size, and early stopping.
For the previously tested configuration:

- Stage 2 training: approximately 14 minutes and 32 seconds.
- Stage 3 dataset construction: approximately 23 minutes.
- Stage 3 Neural ODE training: approximately 13-19 minutes per seed.

## Reproducibility

Use `--stage2-seed` and `--stage3-seed` to control model initialization,
shuffling, random graph construction, and other stochastic operations.
Chronological splitting should be used for the reported stock-prediction
experiments to prevent future observations from entering earlier partitions.

## Notes

- Stage 2 embeds the news `headline`, not the full article.
- Large `.pt` datasets should never be overwritten in place. Write a new file,
  verify it with `torch.load`, and then replace the original artifact.
- If `torchdiffeq` is unavailable, the Neural ODE implementation has an Euler
  fallback, but installing `torchdiffeq` is recommended.
- If CUDA is unavailable, pass `--stage2-device cpu --stage3-device cpu`;
  training will be substantially slower.
