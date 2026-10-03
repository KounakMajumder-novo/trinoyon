"""Create a clean source ZIP for the standalone Windows webcam demo."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "trishul.x.zip"
SKIP = {".venv", "__pycache__", ".git", "logs", "config"}
SKIP_NAMES = {"trishul.x.zip", "final_runable.zip", "final_project.zip"}

with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.relative_to(ROOT).parts):
            continue
        if path.name in SKIP_NAMES:
            continue
        if path.name.endswith((".pyc", ".pyo")):
            continue
        archive.write(path, path.relative_to(ROOT))
print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
