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
python -m src --data-path ../../train_data/data.parquet --data-path ../../train_data/train_data_1.csv --output-dir ../artifacts/latest --label-col auto --unlabeled-default-label 1 --quantize
```

## Step-by-step commands

Train:

```bash
python -m src.train --data-path ../../train_data/data.parquet --data-path ../../train_data/train_data_1.csv --output-dir ../artifacts/latest --model-name microsoft/deberta-v3-large --label-col auto --unlabeled-default-label 1
```

Label convention used by this trainer: `0 = human`, `1 = ai`.
If a file has no label column, set `--unlabeled-default-label` to assign all rows in that file to one class.

Train from scratch with an 8-layer Transformer and enforce 100M+ parameters:

```bash
python -m src.train --data-path ../../train_data/data.parquet --data-path ../../train_data/train_data_1.csv --output-dir ../artifacts/latest --model-name bert-base-uncased --from-scratch --layers 8 --hidden-size 1024 --attention-heads 16 --ffn-size 4096 --max-position-embeddings 514 --min-params 100000000 --label-col auto --unlabeled-default-label 1
```

TPU training (auto-detected in code, 8 epochs):
Install `torch_xla` matching your TPU VM/PyTorch version before running this command.

```bash
python -m torch_xla.distributed.xla_run --num_cores 8 -m src.train --data-path ../../train_data/data.parquet --data-path ../../train_data/train_data_1.csv --output-dir ../artifacts/latest --model-name bert-base-uncased --from-scratch --layers 8 --hidden-size 1024 --attention-heads 16 --ffn-size 4096 --max-position-embeddings 514 --min-params 100000000 --label-col auto --unlabeled-default-label 1 --epochs 8 --tpu-bf16
```

GPU training (single GPU, auto-detected in code, 8 epochs):

```bash
python -m src.train --data-path ../../train_data/data.parquet --data-path ../../train_data/train_data_1.csv --output-dir ../artifacts/latest --model-name bert-base-uncased --from-scratch --layers 8 --hidden-size 1024 --attention-heads 16 --ffn-size 4096 --max-position-embeddings 514 --min-params 100000000 --label-col auto --unlabeled-default-label 1 --epochs 8 --gpu-bf16 --gpu-fp16 --gpu-tf32
```

GPU training (multi-GPU, example with 4 GPUs, 8 epochs):

```bash
torchrun --nproc_per_node 4 -m src.train --data-path ../../train_data/data.parquet --data-path ../../train_data/train_data_1.csv --output-dir ../artifacts/latest --model-name bert-base-uncased --from-scratch --layers 8 --hidden-size 1024 --attention-heads 16 --ffn-size 4096 --max-position-embeddings 514 --min-params 100000000 --label-col auto --unlabeled-default-label 1 --epochs 8 --gpu-bf16 --gpu-fp16 --gpu-tf32
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
