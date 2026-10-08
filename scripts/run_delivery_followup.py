"""Staged validation-only follow-up, with frozen rules and per-stage plans.

Run after phase-one LR study. No architecture/data/split changes or test scoring.
Each adaptive stage is frozen before training; choices use anchored val corr.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import torch
from scripts.run_delivery_sweep import configuration, plan as source_plan, run_one, save_json, sha

FIRST = ROOT / "runs/delivery_lr_20260926"


def item(key, lr, schedule="exp", seed=42, **kw):
    return dict(key=key, learning_rate=lr, lradj=schedule, seed=seed, **kw)


def original(key):
    p = FIRST / key
    r = json.loads((p / "result.json").read_text())
    if sha(p / "checkpoints/checkpoint.pth") != r["checkpoint_sha256"]:
        raise ValueError(f"Changed original checkpoint: {key}")
    return r | {"seed": 42, "source": str(p), "criterion": "mse", "exp_decay_gamma": 0.85}


def read_stage(root, name):
    stage = root / name
    p = json.loads((stage / "plan.json").read_text())
    results = []
    for run in p["runs"]:
        folder = stage / run["key"]
        result = json.loads((folder / "result.json").read_text())
        if sha(folder / "checkpoints/checkpoint.pth") != result["checkpoint_sha256"]:
            raise ValueError(f"Changed checkpoint: {folder}")
        results.append(result | {"source": str(folder)})
    return results


def flat(result):
    out = {k: v for k, v in result.items() if not isinstance(v, dict)}
    out.update({f"anchored_val_{k}": v for k, v in result["anchored"].items()})
    out.update({f"raw_val_{k}": v for k, v in result["raw"].items()})
    return out


def best(results):
    return max(results, key=lambda r: (r["anchored"]["corr"], -r["anchored"]["rmse"], r["key"]))


def config_from(result):
    return {k: result[k] for k in ("learning_rate", "lradj", "exp_decay_gamma", "lr_decay_gamma",
            "warmup_epochs", "criterion", "huber_beta") if k in result}


def snapshot():
    evidence = source_plan()
    evidence["hashes"]["scripts/run_delivery_followup.py"] = sha(__file__)
    evidence["original_results"] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(FIRST.glob("*/result.json"))}
    evidence["first_plan_sha256"] = sha(FIRST / "plan.json")
    return evidence


def run_stage(root, name, items, parents):
    frozen = json.loads((root / "plan.json").read_text())
    if frozen["sources"] != snapshot():
        raise ValueError("Sources/config/environment changed; do not mix studies")
    folder = root / name
    folder.mkdir(exist_ok=True)
    stage_plan = {"runs": items, "parents": parents, "selection": frozen["selection"]}
    if (folder / "plan.json").exists():
        if json.loads((folder / "plan.json").read_text()) != stage_plan:
            raise ValueError("Stage selection/plan changed")
    else:
        save_json(folder / "plan.json", stage_plan)
    for run in items:
        if (folder / run["key"] / "result.json").exists():
            continue
        print(f"START {name}/{run['key']}", flush=True)
        try:
            result = run_one(frozen["base_config"], run, folder)
        except Exception as exc:
            save_json(folder / "failure.json", {"run": run["key"], "error": repr(exc)})
            raise
        print(f"DONE {name}/{run['key']} {result['anchored']}", flush=True)
        if run["key"] == "replay_lr0.001_type1_s42":
            if result["checkpoint_sha256"] != original("lr0.001_type1")["checkpoint_sha256"]:
                raise ValueError("Baseline replay differs; stop before remaining runs")
        done = [json.loads(p.read_text()) | {"source": str(p.parent)} for p in sorted(folder.glob("*/result.json"))]
        pd.DataFrame([flat(r) for r in done]).sort_values("anchored_val_corr", ascending=False).to_csv(folder / "summary.csv", index=False)
    return read_stage(root, name)


def execute(root):
    # 1: reproduce original baseline after scheduler parameterization, then 8 seed repeats.
    bases = [(0.001, "type1"), (0.001, "exp"), (0.003, "exp"), (0.003, "warmup_exp")]
    seeds = [item("replay_lr0.001_type1_s42", .001, "type1")]
    seeds += [item(f"lr{lr:g}_{sc}_s{seed}", lr, sc, seed) for lr, sc in bases for seed in (43, 44)]
    stability = run_stage(root, "01_stability", seeds, [str(FIRST)])
    baseline = original("lr0.001_type1")
    if stability[0]["checkpoint_sha256"] != baseline["checkpoint_sha256"]:
        raise ValueError("Baseline replay is not bitwise identical; stop before adaptive stages")
    compared = stability[1:] + [original(f"lr{lr:g}_{sc}") for lr, sc in bases]
    frame = pd.DataFrame([flat(r) for r in compared])
    frame.to_csv(root / "stability_runs.csv", index=False)
    frame.groupby(["learning_rate", "lradj"])[["anchored_val_corr", "anchored_val_mae", "anchored_val_rmse"]].agg(["mean", "std"]).to_csv(root / "stability_summary.csv")

    # 2: local LR exploration; select two rates from 6 candidates by seed42 validation.
    refined = run_stage(root, "02_lr_refine", [item(f"lr{lr:g}_exp_s42", lr) for lr in (.0015, .0025, .004)],
                        [r["source"] for r in compared])
    pool = refined + [original(f"lr{lr:g}_exp") for lr in (.001, .002, .003)]
    top2 = sorted(pool, key=lambda r: (-r["anchored"]["corr"], r["anchored"]["rmse"], r["key"]))[:2]

    # 3: tune decay on the two candidate rates; keep 0.85 controls.
    decays = run_stage(root, "03_decay", [item(f"lr{r['learning_rate']:g}_g{g:g}_s42", r["learning_rate"], exp_decay_gamma=g)
        for r in top2 for g in (.80, .92)], [r["source"] for r in top2])
    selected = best(top2 + decays)
    gamma = selected.get("exp_decay_gamma", .85)

    # 4: match gamma, floor, optimizer and base LR; vary warmup length only.
    warms = run_stage(root, "04_warmup", [item(f"lr{selected['learning_rate']:g}_g{gamma:g}_w{w}_s42",
        selected["learning_rate"], "warmup_exp", warmup_epochs=w, lr_decay_gamma=gamma) for w in (3, 8)],
        [selected["source"]])
    selected = best([selected, *warms])

    # 5: compare MSE with existing SmoothL1 beta in scaled target units.
    loss_items = []
    for beta in (.5, 1.0):
        loss_items.append({"key": f"huber_b{beta:g}_s42", "seed": 42, **config_from(selected),
                           "criterion": "huber", "huber_beta": beta})
    losses = run_stage(root, "05_loss", loss_items, [selected["source"]])
    huber = best(losses)

    # 6: same-seed paired confirmation, MSE and chosen SmoothL1 candidate.
    confirmation_items = []
    for label, source in (("mse", selected), ("huber", huber)):
        for seed in (43, 44):
            confirmation_items.append({"key": f"{label}_s{seed}", **config_from(source), "criterion": label, "seed": seed})
    confirmation = run_stage(root, "06_confirmation", confirmation_items, [selected["source"], huber["source"]])
    final = [selected | {"criterion": "mse"}, huber, *confirmation]
    frame = pd.DataFrame([flat(r) for r in final])
    frame.to_csv(root / "final_candidates.csv", index=False)
    frame.groupby("criterion")[["anchored_val_corr", "anchored_val_mae", "anchored_val_rmse"]].agg(["mean", "std"]).to_csv(root / "final_summary.csv")
    save_json(root / "completion.json", {"new_runs": 24, "final_mse": selected["source"], "final_huber": huber["source"],
        "scope": "adaptive validation exploration; no untouched test evaluation or full-data retraining",
        "limitations": "3 seeds share the same validation period; adaptive search is not independent validation. No final weight chosen by a lucky seed."})
    print("COMPLETED all 24 new runs", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["plan", "run"])
    ap.add_argument("--output", default="runs/delivery_followup_20260929")
    args = ap.parse_args()
    torch.set_num_threads(1)
    root = Path(args.output).resolve()
    if args.mode == "plan":
        root.mkdir(parents=True, exist_ok=True)
        if (root / "plan.json").exists():
            raise ValueError("Plan already exists")
        save_json(root / "plan.json", {"sources": snapshot(), "base_config": configuration(),
            "selection": "raw val corr checkpoint; rank anchored val corr, tie break lower RMSE; retain MAE tradeoffs",
            "stages": {"stability": "baseline replay + 4 configs x seeds43/44, reuse original seed42",
                       "lr_refine": [.0015, .0025, .004], "decay": "top2 LR x gamma0.8/0.92 + existing0.85",
                       "warmup": "best LR/gamma: no-warmup control versus3/8 epochs",
                       "loss": "selected MSE versus SmoothL1 beta0.5/1.0", "confirmation": "MSE and best SmoothL1 x seeds43/44"},
            "total_new_runs": 24})
        print(f"Saved plan: {root}", flush=True)
    else:
        execute(root)


if __name__ == "__main__":
    main()
