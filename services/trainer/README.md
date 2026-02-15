# Trainer Service (Detector v2)

This service trains and exports the AI-content detector used by the API.

## What improved

- Stronger base model default: `microsoft/deberta-v3-large`
- Weighted focal BCE training for harder positives/negatives
- Temperature scaling + optimal threshold search on validation logits
- Richer evaluation metrics (F1/precision/recall/ROC-AUC/Brier)
- Runtime bundle export (`model.onnx` + tokenizer + calibration + manifest)

## Quick start

From `services/trainer`:

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Run full pipeline:

```bash
python -m src --csv ../../train_data/balanced_ai_human_prompts.csv --output-dir ../artifacts/latest --quantize
```

## Step-by-step commands

Train:

```bash
python -m src.train --csv ../../train_data/balanced_ai_human_prompts.csv --output-dir ../artifacts/latest --model-name microsoft/deberta-v3-large
```

Calibrate:

```bash
python -m src.calibration --val-logits ../artifacts/latest/val_logits.npz --output ../artifacts/latest/calibration.json
```

Evaluate:

```bash
python -m src.evaluate --test-logits ../artifacts/latest/test_logits.npz --calibration ../artifacts/latest/calibration.json --output ../artifacts/latest/eval_metrics.json
```

Export ONNX + runtime bundle:

```bash
python -m src.export_onnx --model-dir ../artifacts/latest/model --output ../artifacts/latest/model.onnx --bundle-dir ../artifacts/latest/runtime_bundle --calibration ../artifacts/latest/calibration.json --metrics ../artifacts/latest/eval_metrics.json --training-config ../artifacts/latest/training_config.json --quantize
```

Create model card:

```bash
python -m src.model_card --artifact-dir ../artifacts/latest
```

## Output artifacts

- `artifacts/latest/model/` - fine-tuned HF checkpoint + tokenizer files
- `artifacts/latest/model.onnx` - ONNX graph for API inference
- `artifacts/latest/calibration.json` - temperature + reliability + optimal threshold
- `artifacts/latest/eval_metrics.json` - test metrics at `0.5` and tuned threshold
- `artifacts/latest/runtime_bundle/` - deployable runtime bundle
- `artifacts/latest/model_card.json` - summary metadata for audit/deploy

## Deploy handoff to API

The API should have access to:

- ONNX file path (default `DETECTOR_ONNX_PATH`)
- tokenizer directory (`runtime_bundle/model`)
- calibration file (`runtime_bundle/calibration.json`)

If these are missing, API falls back to heuristic detection.
