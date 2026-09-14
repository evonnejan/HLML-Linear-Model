"""Read-only repository checks and entirely mocked matrix execution/collection.

Run from repository root with .HLML_Linear_venv/bin/python -B <this file>.
No training, checkpoint access, real model predictions, or fixture disk writes.
Only stdout is produced; caller may save it inside this review's evidence/.
"""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch, mock_open
import warnings

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import pandas as pd
import run_matrix as rm
import collect_matrix as cm
from build_splits import expand_cols
from utils.metrics import CORR


def capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = fn()
    return result, buf.getvalue()


result = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
paths = ["run_matrix.py", "collect_matrix.py", "run.py", "exp/exp_Main2.py",
         "data_provider/Data_Factory.py", "data_provider/Data_Loader.py", "utils/metrics.py", "build_splits.py",
         "experiments/manifest.csv", "dataset/train_old.csv", "dataset/train_drycut_L3h_buf60.csv",
         "dataset/splits_train_old.csv", "dataset/splits_train_old.json",
         "dataset/splits_train_drycut_L3h_buf60.csv", "dataset/splits_train_drycut_L3h_buf60.json"]
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
result["sha256"] = {p: digest(p) for p in paths}
plan = rm.build_manifest()
assert len(plan) == 24 and plan.run_key.nunique() == 24
result["manifest_status"] = plan.status.value_counts().to_dict()
actual_manifest = pd.read_csv(rm.MANIFEST, keep_default_na=False)
result["environment"] = {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
                         "actual_manifest_dtypes": {c: str(t) for c, t in actual_manifest.dtypes.items()}}
commands = [rm.build_cmd(row) for _, row in plan.iterrows()]
parser_tree = ast.parse(Path("run.py").read_text())
cli = {a.value for node in ast.walk(parser_tree) if isinstance(node, ast.Call)
       and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
       for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
for (_, row), cmd in zip(plan.iterrows(), commands):
    assert all(x in cli for x in cmd if x.startswith("--"))
    assert cmd[cmd.index("--input_col") + 1] == "HL02,HL03,HL04,HL05,HL06"
    assert "--segment_col" in cmd and "--split_file" in cmd
    assert ("--exog_col" in cmd) == (row.exog == "full")
    assert not any("isRain" in x or "min_since_rain" in x for x in cmd)
result["commands"] = {"count": len(commands), "all_flags_in_run_parser": True,
                      "example_none": commands[0], "example_full": commands[6]}
_, result["preflight_actual"] = capture(rm.preflight)
with patch.dict(rm.DATASETS["old"], split_file=rm.DATASETS["drycut"]["split_file"]):
    _, result["preflight_wrong_dataset_pair"] = capture(rm.preflight)

# Independently reproduce the loader's segment/window NaN rule from actual CSVs.
# Only selected columns loaded; no wide observation source loaded.
result["data"] = {}
for name, ds in rm.DATASETS.items():
    csv = Path("dataset") / ds["data_path"]
    header = pd.read_csv(csv, nrows=0).columns.tolist()
    inp = expand_cols(rm.CONST["input_col"], header)
    exog = expand_cols(rm.EXOGS["full"], header)
    cols = list(dict.fromkeys(["segment_id", "SegmentStart", "date", "HL01", *inp, *exog]))
    data = pd.read_csv(csv, usecols=cols)
    split = pd.read_csv(ds["split_file"]).set_index("segment_id")
    ids = data[["segment_id", "SegmentStart"]].drop_duplicates().sort_values("SegmentStart").segment_id.tolist()
    assert set(ids) == set(split.index)
    train70 = data[data.segment_id.isin(ids[:int(len(ids) * .7)])]
    dropped = [c for c in inp + exog if train70[c].dropna().nunique() <= 1]
    modes = {}
    for mode in ("none", "full"):
        check = [c for c in inp + (exog if mode == "full" else []) if c not in dropped] + ["HL01"]
        counts = {}
        test_keys = []
        for sid, group in data.groupby("segment_id", sort=False):
            group = group.sort_values("date").reset_index(drop=True)
            bad = group[check].isna().any(axis=1).to_numpy()
            csum = np.r_[0, bad.cumsum()]
            starts = np.arange(max(0, len(group) - 75 + 1))
            good = starts[(csum[starts + 75] - csum[starts]) == 0]
            counts[sid] = len(good)
            if split.loc[sid, "split"] == "test":
                test_keys.extend((int(sid), str(group.loc[int(i) + 59, "date"])) for i in good)
        per_fold = {}
        for fold in rm.FOLDS:
            per_fold[str(fold)] = {
                part: int(sum(counts[sid] for sid, row in split.iterrows()
                              if (row["split"] == "test" if part == "test" else
                                  row["split"] != "test" and row[f"fold_{fold}"] == part)))
                for part in ("train", "val", "test")}
        modes[mode] = {"fold_counts": per_fold, "test_key_hash": hashlib.sha256(json.dumps(test_keys).encode()).hexdigest(),
                       "test_count": len(test_keys)}
    result["data"][name] = {"dropped_as_current_run_py": dropped, "n_segments": len(ids), **modes}

# Resume trusts the run_key despite altered configuration.
old = plan.copy()
old.loc[0, ["status", "run_dir"]] = ["done", "synthetic_previous_lr_0.001"]
with patch.object(rm.pd, "read_csv", return_value=old), patch.dict(rm.CONST, learning_rate=.002):
    resumed = rm.build_manifest()
    result["resume_changed_learning_rate"] = {"status": resumed.iloc[0].status,
        "run_dir": resumed.iloc[0].run_dir, "new_command_lr": rm.build_cmd(resumed.iloc[0])[rm.build_cmd(resumed.iloc[0]).index("--learning_rate") + 1]}

# No subprocess can execute: all calls are mocked, as are log/manifest writes.
single = actual_manifest.iloc[:1].copy()
saved = []
with patch.object(rm.pd, "read_csv", return_value=single), \
     patch.object(rm, "save_manifest", side_effect=lambda d: saved.append(d.copy())), \
     patch.object(Path, "mkdir"), patch("builtins.open", mock_open()), \
     patch.object(Path, "read_text", return_value="synthetic success without run_dir"), \
     patch.object(rm.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as child:
    try:
        _, output = capture(lambda: rm.execute(False, None))
        result["native_manifest_execution"] = {"error": None, "stdout": output}
    except Exception as e:
        result["native_manifest_execution"] = {"error": type(e).__name__ + ": " + str(e),
            "last_saved_status": saved[-1].iloc[0].status, "mock_child_calls": child.call_count}
# Isolate other logic defects after reproducing the native dtype error above.
# This changes ONLY an in-memory fixture, never the production manifest/code.
single = actual_manifest.iloc[:1].copy()
single["duration_s"] = single.duration_s.astype(object)
saved = []
with patch.object(rm.pd, "read_csv", return_value=single), \
     patch.object(rm, "save_manifest", side_effect=lambda d: saved.append(d.copy())), \
     patch.object(Path, "mkdir"), patch("builtins.open", mock_open()), \
     patch.object(Path, "read_text", return_value="synthetic success without run_dir"), \
     patch.object(rm.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as child:
    _, output = capture(lambda: rm.execute(False, None))
    result["success_without_artifacts_after_fixture_dtype_conversion"] = {"mock_child_calls": child.call_count,
        "status": saved[-1].iloc[0].status, "run_dir": saved[-1].iloc[0].run_dir, "stdout": output}
with patch.object(rm.subprocess, "run", side_effect=AssertionError("dry run spawned process")) as child:
    _, output = capture(lambda: rm.execute(True, None))
    assert child.call_count == 0
    result["dry_run"] = {"child_calls": 0, "printed_commands": output.count("\n# [")}

# All collector inputs below are synthetic, in memory. Never read real outputs.
y = np.array([[1, 2], [2, 4], [3, 6]], dtype=float)[..., None]
p = y.copy()
p[:, 0, 0] = 0
persist = np.broadcast_to(np.arange(3, dtype=float)[:, None, None], y.shape).copy()
result["constant_horizon_corr"] = {"collector": cm.metrics(p, y)["corr"],
    "core_test_and_existing_anchored": float(np.nanmean(CORR(p, y))),
    "collector_horizons": cm.corr_per_horizon(p, y).tolist()}
anchored = p - (p[:, :1, :] - persist[:, :1, :])
assert np.allclose(anchored[:, :1, :], persist[:, :1, :])
assert np.allclose(anchored, p - p[:, :1, :] + persist[:, :1, :])
result["anchored_formula"] = {"first_horizon_equals_last_observation": True,
                              "same_translation_as_existing_script": True}

def fake_collect(man, missing=None, args_override=None, bad_shape=False):
    def fake_args(path, **kwargs):
        key = path.parent.name
        row = man[man.run_key == key].iloc[0]
        args = dict(rm.CONST, criterion=row.loss, fold=int(row.fold),
                    data_path=rm.DATASETS[row.dataset]["data_path"],
                    split_file=rm.DATASETS[row.dataset]["split_file"],
                    input_cols=rm.CONST["input_col"].split(","),
                    exog_cols=["Past10Min"] if row.exog == "full" else [],
                    exog_in=1 if row.exog == "full" else 0)
        args.update(args_override or {})
        return json.dumps(args)
    def fake_load(path):
        if missing and path.parent.parent.name == missing[0] and path.parent.name == missing[1]:
            raise FileNotFoundError(2, "synthetic missing checkpoint outputs", str(path))
        return {"pred.npy": p, "true.npy": y[:1] if bad_shape else y, "persist.npy": persist}[path.name].copy()
    with patch.object(cm.pd, "read_csv", return_value=man), \
         patch.object(Path, "exists", return_value=True), patch.object(Path, "read_text", fake_args), \
         patch.object(cm.np, "load", side_effect=fake_load), patch.object(cm, "load_seg_ids", return_value=np.array([1, 1, 2])):
        return capture(lambda: cm.collect(Path("synthetic_manifest.csv")))

done = plan.copy()
done["status"] = "done"
done["run_dir"] = done.run_key.map(lambda k: "synthetic/" + k)
collected, _ = fake_collect(done)
assert len(collected) == 144
result["complete_synthetic_collection"] = {"rows": len(collected), "all_MSE_finite": bool(np.isfinite(collected.mse).all()),
    "has_segment_corr": any("corr" in c and "seg" in c for c in collected.columns),
    "has_val_metrics": any("val" in c for c in collected.columns)}
partial, warning = fake_collect(done, missing=(done.iloc[0].run_key, "outputs_alt"))
_, sanity_output = capture(lambda: cm.sanity(partial))
result["missing_one_checkpoint"] = {"rows": len(partial), "warning": warning, "sanity": sanity_output,
    "affected_group_folds": partial[(partial.dataset == "drycut") & (partial.exog == "none") &
        (partial.loss == "mse") & (partial.checkpoint == "best_mse") & (partial.anchoring == "raw")].fold.tolist()}
missing_config = collected[~((collected.dataset == "old") & (collected.exog == "full") & (collected.loss == "huber"))]
_, result["missing_entire_config_sanity"] = capture(lambda: cm.sanity(missing_config))
def synthetic_main(frame):
    tables = {}
    def save_table(df, path, **kwargs):
        tables[Path(path).name] = df.copy()
    with patch.object(cm, "collect", return_value=frame), patch.object(sys, "argv", ["collect_matrix.py"]), \
         patch.object(Path, "mkdir"), patch.object(pd.DataFrame, "to_csv", save_table):
        _, log = capture(cm.main)
    return tables, log
tables, log = synthetic_main(collected)
result["synthetic_main"] = {"table_shapes": {k: list(v.shape) for k, v in tables.items()},
    "headline": log.split("--- anchored / best_corr 的 headline ---")[-1],
    "summary_has_mse": "mse_mean" in tables["summary_by_config.csv"].columns,
    "wide_columns": list(tables["results_wide.csv"].columns)}
nan_frame = collected.copy()
bad = (nan_frame.run_key == done.iloc[0].run_key) & (nan_frame.checkpoint == "best_corr") & (nan_frame.anchoring == "anchored")
nan_frame.loc[bad, ["corr", "mse"]] = np.nan
nan_tables, _ = synthetic_main(nan_frame)
ns = nan_tables["summary_by_config.csv"]
result["nan_fold_summary"] = ns[(ns.dataset == "drycut") & (ns.exog == "none") & (ns.loss == "mse") &
    (ns.checkpoint == "best_corr") & (ns.anchoring == "anchored")][["n_folds", "corr_mean", "mse_mean"]].to_dict("records")
mislabeled, _ = fake_collect(done.iloc[:1], args_override={"criterion": "huber", "fold": 3,
    "early_stop_metric": "mse", "data_path": "wrong_dataset.csv", "exog_col": "HL01"})
result["wrong_args_accepted"] = {"rows": len(mislabeled), "manifest_label": mislabeled.iloc[0][["loss", "fold", "checkpoint"]].to_dict(),
                                "actual_args_override": {"criterion": "huber", "fold": 3, "early_stop_metric": "mse", "data_path": "wrong_dataset.csv", "exog_col": "HL01"}}
try:
    mismatch, _ = fake_collect(done.iloc[:1], bad_shape=True)
    result["shape_mismatch"] = {"accepted": True, "rows": len(mismatch), "raw_mse": float(mismatch.iloc[0].mse)}
except Exception as e:
    result["shape_mismatch"] = {"accepted": False, "error": type(e).__name__ + ": " + str(e)}
base_zero = pd.DataFrame([{"run_key": "z", "checkpoint": "best_corr", "anchoring": m, "mse": v}
                         for m, v in [("persistence", 0.), ("raw", 1.), ("anchored", 0.)]])
result["zero_baseline"] = cm.add_improve(base_zero)[["anchoring", "improve_mse_pct"]].to_dict("records")
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")
    nan_metrics = cm.metrics(np.full_like(y, np.nan), y)
result["nan_prediction_accepted"] = {"metrics": nan_metrics, "warning_count": len(ws)}
bad_points = pd.DataFrame({"window_idx": [100, 101, 102], "segment": [7, 8, 9]})
with patch.object(Path, "exists", return_value=True), patch.object(cm.pd, "read_csv", return_value=bad_points):
    result["invalid_window_ids_accepted"] = cm.load_seg_ids(Path("synthetic"), 3).tolist()

# Serialize nonfinite synthetic results explicitly, keeping strict JSON valid.
def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
assert all(digest(p) == h for p, h in result["sha256"].items()), "Inputs changed during checks"
print(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False))
