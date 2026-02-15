from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "latest"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

