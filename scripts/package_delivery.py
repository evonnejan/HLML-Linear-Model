"""Build the customer delivery package: dist/project/ and dist/project.zip.

Copies only the whitelisted files from project_stage1/ and renames the package
project_stage1 -> project inside the shipped text files. The repo keeps project_stage1
because internal code (full_fit/, tests, scripts, docs) imports it. Run from any cwd.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "project_stage1"
DIST = ROOT / "dist"
PACKAGE = "project"
FILES = [
    "README.md",
    "README.html",
    "INPUT_MANUAL.md",
    "INPUT_MANUAL.html",
    "examples/example_minute.csv",
    "inference/README.md",
    "inference/README.html",
    "inference/__init__.py",
    "inference/predictor.py",
    "inference/validation.py",
    "inference/input_schema.json",
    "inference/model.py",
    "inference/run_forecast.py",
    "inference/forecast_from_table.py",
    "inference/model_bundle/config.json",
    "inference/model_bundle/scaler.npz",
    "inference/model_bundle/checkpoint.pth",
]
RENAME_SUFFIXES = {".py", ".md", ".html", ".json"}


def main() -> None:
    missing = [name for name in FILES if not (SRC / name).is_file()]
    if missing:
        raise SystemExit(f"Missing delivery files: {missing}")

    target = DIST / PACKAGE
    if target.exists():
        shutil.rmtree(target)  # only ever our own build output
    for name in FILES:
        src, dst = SRC / name, target / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in RENAME_SUFFIXES:
            dst.write_text(src.read_text(encoding="utf-8").replace("project_stage1", PACKAGE), encoding="utf-8")
        else:
            shutil.copyfile(src, dst)

    leftovers = [p for p in target.rglob("*") if p.is_file() and p.suffix in RENAME_SUFFIXES
                 and "project_stage1" in p.read_text(encoding="utf-8")]
    if leftovers:
        raise SystemExit(f"project_stage1 still present in: {leftovers}")
    shipped = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
    if shipped != sorted(FILES):
        raise SystemExit(f"Unexpected package contents: {shipped}")

    archive = shutil.make_archive(str(DIST / PACKAGE), "zip", root_dir=DIST, base_dir=PACKAGE)
    print(f"{len(shipped)} files -> {target}")
    print(f"zip -> {archive}")


if __name__ == "__main__":
    main()
