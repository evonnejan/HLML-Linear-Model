"""Evaluate one predefined equal-weight 3-seed ensemble on saved validation only.

No training, test loading, seed subset search, or weight fitting. Run from any cwd.
Outputs are reproducible from the frozen prediction/checkpoint/scaler artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = [
    (42, "runs/delivery_lr_20260926/lr0.001_exp"),
    (43, "runs/delivery_followup_20260929/01_stability/lr0.001_exp_s43"),
    (44, "runs/delivery_followup_20260929/01_stability/lr0.001_exp_s44"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(pred, true):
    # Same definition as run_delivery_sweep.score: mean horizon-wise Pearson r.
    if pred.shape != true.shape or not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Invalid arrays")
    error = pred.astype(np.float64) - true
    corrs = [float(np.corrcoef(pred[:, h, 0], true[:, h, 0])[0, 1])
             for h in range(pred.shape[1])
             if pred[:, h, 0].std() > 0 and true[:, h, 0].std() > 0]
    if len(corrs) != pred.shape[1] or not np.isfinite(corrs).all():
        raise ValueError("Undefined horizon correlation")
    return dict(corr=float(np.mean(corrs)), mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.mean(error ** 2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/delivery_ensemble_20261001")
    args = parser.parse_args()
    arrays, provenance, reference_args, reference_scaler = [], [], None, None
    for seed, folder in MEMBERS:
        path = ROOT / folder
        cfg = json.loads((path / "run_args.json").read_text())
        result = json.loads((path / "result.json").read_text())
        if cfg["seed"] != seed or (cfg["learning_rate"], cfg["lradj"], cfg["criterion"],
                                   cfg.get("exp_decay_gamma", .85)) != (.001, "exp", "mse", .85):
            raise ValueError("Unexpected member configuration")
        comparable = {k: v for k, v in cfg.items() if k not in
                      {"seed", "run_dir", "setting", "exp_decay_gamma"}}
        if reference_args is not None and comparable != reference_args:
            raise ValueError("Member configurations differ beyond seed/output paths")
        reference_args = comparable
        checkpoint = path / "checkpoints/checkpoint.pth"
        if sha(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("Checkpoint fingerprint mismatch")
        with np.load(path / "scaler.npz", allow_pickle=False) as saved:
            scaler = {k: saved[k] for k in saved.files}
        if reference_scaler is not None:
            if scaler.keys() != reference_scaler.keys() or any(
                    not np.array_equal(scaler[k], reference_scaler[k]) for k in scaler):
                raise ValueError("Member scalers or feature orders differ")
        reference_scaler = scaler
        with np.load(path / "validation_predictions.npz", allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        if data["raw"].shape != (5418, 15, 1) or data["anchor"].shape != (5418, 1, 1):
            raise ValueError("Unexpected validation shape")
        if arrays:
            for key in ("origins", "true", "anchor"):
                # Compare positions exactly; overlapping events can repeat timestamps.
                if not np.array_equal(data[key], arrays[0][key]):
                    raise ValueError(f"Validation alignment mismatch: {key}")
        np.testing.assert_allclose(data["anchored"], data["raw"] -
                                   (data["raw"][:, :1] - data["anchor"]), rtol=0, atol=0)
        np.testing.assert_allclose(data["anchored"][:, :1], data["anchor"], rtol=0, atol=1e-4)
        for kind in ("raw", "anchored"):
            actual = score(data[kind], data["true"])
            for key, value in actual.items():
                np.testing.assert_allclose(value, result[kind][key], rtol=1e-12, atol=1e-12)
        arrays.append(data)
        provenance.append(dict(seed=seed, path=folder, best_epoch=result["best_epoch"],
            hashes={name: sha(path / name) for name in
                    ("validation_predictions.npz", "scaler.npz", "run_args.json",
                     "result.json", "checkpoints/checkpoint.pth")}))

    raw_mean = np.mean([d["raw"].astype(np.float64) for d in arrays], axis=0)
    anchor, true = arrays[0]["anchor"], arrays[0]["true"]
    ensemble = anchor + raw_mean - raw_mean[:, :1]
    # Float32 member postprocessing has tiny rounding differences from float64 mean.
    commutation_error = float(np.max(np.abs(ensemble -
        np.mean([d["anchored"].astype(np.float64) for d in arrays], axis=0))))
    np.testing.assert_allclose(ensemble[:, :1], anchor, rtol=0, atol=1e-10)
    if commutation_error > 2e-4:
        raise ValueError("Anchoring/averaging commutation check failed")
    predictions = {f"seed_{seed}": data["anchored"] for (seed, _), data in zip(MEMBERS, arrays)}
    predictions["equal_mean_42_43_44"] = ensemble
    rows, horizons = [], []
    for name, pred in predictions.items():
        for label, sl in (("h1_15", slice(None)), ("h2_15", slice(1, None)),
                          ("h15", slice(14, 15))):
            rows.append(dict(model=name, horizon_group=label, **score(pred[:, sl], true[:, sl])))
        for h in range(15):
            horizons.append(dict(model=name, horizon=h + 1, **score(pred[:, h:h+1], true[:, h:h+1])))
    report = dict(
        protocol="Prespecified equal weights across all 3 existing seeds; frozen validation only",
        selection="Each member is its existing best raw-validation-correlation checkpoint",
        n_windows=len(true), unique_origins=len(np.unique(arrays[0]["origins"])),
        members=provenance, metrics=rows, ensemble_raw=score(raw_mean, true),
        max_anchor_mean_commutation_error=commutation_error,
        limitations=["Not an independent test estimate; configuration was selected on validation",
                     "Overlapping windows are not independent statistical samples",
                     "No new preprocessing, all-data retraining, or live availability replay",
                     "Three model forwards; wall-clock latency and memory not benchmarked",
                     "Physical water-level unit pending confirmation of source _Alt table"],
        evaluator_sha256=sha(Path(__file__)),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                            allow_nan=False) + "\n")
    for filename, records in (("metrics.csv", rows), ("per_horizon.csv", horizons)):
        with (args.output / filename).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    np.savez_compressed(args.output / "ensemble_validation_predictions.npz", raw=raw_mean,
        anchored=ensemble, true=true, anchor=anchor, origins=arrays[0]["origins"])
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
