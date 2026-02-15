# Trainer Service

## Commands

1. Train
```bash
python -m src.train --csv ../../train_data/balanced_ai_human_prompts.csv
```

2. Calibrate
```bash
python -m src.calibration --val-logits ../artifacts/latest/val_logits.npz --output ../artifacts/latest/calibration.json
```

3. Export ONNX
```bash
python -m src.export_onnx --model-dir ../artifacts/latest/model --output ../artifacts/latest/model.onnx
```

4. Evaluate
```bash
python -m src.evaluate --test-logits ../artifacts/latest/test_logits.npz --calibration ../artifacts/latest/calibration.json --output ../artifacts/latest/eval_metrics.json
```

5. Model card
```bash
python -m src.model_card
```

