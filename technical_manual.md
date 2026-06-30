# Technical Manual

## 1. Project Overview

### 1.1 Purpose
This repository implements time-series forecasting workflows for water-level prediction, centered on linear-family models:
- `Linear`
- `NLinear`
- `DLinear`
- `DLinearMix`
- `DLinearMix2`

Primary training entrypoint:
- `run.py`

Primary analysis entrypoints:
- `analyze_best_models_overview.py`
- `analyze_full_inference.py`

### 1.2 Current Goal (from code and workspace artifacts)
Based on available scripts, run outputs, and analysis outputs, the current practical goal is:
1. Train many model configurations for input-to-target mappings (mostly targeting `HL01`).
2. Select best runs by test MSE.
3. Analyze performance by horizon and by segment.
4. Re-run full inference (train/val/test coverage) for best runs and produce segment-horizon diagnostics.

Business objective beyond this technical workflow: **Unknown from current code inspection.**

## 2. Current Project Status

### 2.1 Implemented and Available Now
Implemented and runnable in current code:
1. End-to-end training + test output generation (`run.py` + `exp/exp_Main2.py`).
2. Segment-aware dataset splitting and windowing (`data_provider/Data_Loader.py`).
3. Model variants for single-input and mixed-input/exogenous setups (`models/`).
4. Aggregated and per-horizon metrics output generation.
5. Segment-level ranking and point-level export during testing.
6. Post-run overview analysis (`analyze_best_models_overview.py`).
7. Full-inference analysis on best checkpoints (`analyze_full_inference.py`).
8. Visualization utility for run outputs (`visualize.py`).

### 2.2 Evidence in Workspace
Observed evidence of active experimentation:
- Many run folders in `runs/` including `DLinear`, `DLinearMix`, `DLinearMix2` settings.
- Existing summary CSV: `runs/summary/DLinearMix2_summary.csv`.
- Existing analysis artifacts in `analysis/` and `analysis/full_inference/`.

### 2.3 Notable Direction Notes
`meeting_recap.txt` records incremental goals and future ideas (lag distribution + boxplot integration, horizon-level MSE/Corr tables, model interaction ideas, graph extraction ideas). These are notes, not guaranteed implemented features.

## 3. Folder Structure

Top-level structure relevant to current workflows:

- `run.py`
- `analyze_best_models_overview.py`
- `analyze_full_inference.py`
- `analyze_full_inference_lag.py` (backward-compatible alias to `analyze_full_inference.py`)
- `visualize.py`
- `data_provider/`
- `exp/`
- `models/`
- `utils/`
- `dataset/`
- `runs/`
- `analysis/`
- `checkpoints/` (legacy/older checkpoint folders)
- `test_results/` (legacy/older test plot folders)
- `logs/`
- `Source_Code/` (older reference snapshot)

### 3.1 Dataset Files Observed
In `dataset/`:
- `water_level_all.csv`
- `water_level_all2.csv`
- `water_level_all3.csv`
- `water_level_rain_all4.csv`
- `wra_cogate_obs_long.csv`
- `wra_cogate_obs_wide.csv`
- `wra_cogate_obs_wide_gate_opening.csv`
- `wra_cogate_pqid_fullname_map.csv`

### 3.2 No Standard Project Manifest Found
The following were not found:
- `README*`
- `requirements*.txt`
- `pyproject.toml`

## 4. Main Tasks and Workflows

### 4.1 Training + Testing Workflow
1. User runs `run.py` with model/data args.
2. Script builds unique setting name and run directory under `runs/`.
3. Script saves runtime config (`run_args.json`, `run_cmd.txt`).
4. `Exp_Main` (from `exp/exp_Main2.py`) trains model and early-stops.
5. Best checkpoint loaded.
6. Test metrics and artifacts written under `run_dir/outputs/`.

### 4.2 Best-Run Overview Workflow
1. `analyze_best_models_overview.py` scans run folders with required output files.
2. Loads run metadata from `run_args.json` and metrics from `outputs/metrics.npy`.
3. Chooses best setting per `input_col->target` pair by test MSE.
4. Produces overview CSV, horizon overlays, relative horizon overlays, and segment fanplots.

### 4.3 Full Inference Workflow
1. `analyze_full_inference.py` scans runs and selects best setting per pair by test MSE.
2. Loads checkpoint and generates/collects point-level predictions for test/train/val.
3. Aggregates segment-horizon metrics and lag-distribution style outputs.
4. Writes full-inference outputs under timestamped subfolder in `analysis/full_inference/`.

## 5. Core Files and Modules

## 5.1 `data_provider/`

### 5.1.1 `data_provider/Data_Factory.py`
**What it does**
- Factory function to build dataset + dataloader for `train`, `val`, `test`, or `pred` mode.

**Key function**
- `data_provider(args, flag)`

**Main inputs**
- `args` fields including `data`, `root_path`, `data_path`, `seq_len`, `label_len`, `pred_len`, `features`, `target`, `input_col`, `exog_col`, `segment_col`, `stride_train`, `stride_eval`, `batch_size`, `num_workers`, `embed`, `freq`, `train_only`, `model`.
- `flag`: `train` / `val` / `test` / `pred`.

**Main outputs**
- Dataset object (`Dataset_Custom` or `Dataset_Pred`)
- `torch.utils.data.DataLoader`

**Connections**
- Used by `exp/exp_Main2.py` and `exp/exp_Main.py`.

### 5.1.2 `data_provider/Data_Loader.py`
**What it does**
- Defines dataset classes for model training/testing and prediction.

**Key classes/functions**
- `Dataset_Custom(Dataset)`
- `Dataset_Pred(Dataset)`
- `Dataset_Custom._parse_col_spec(...)`
- `Dataset_Custom.__read_data__(...)`
- `Dataset_Custom.__getitem__(...)`
- `Dataset_Custom.__len__(...)`
- `Dataset_Custom.inverse_transform(...)`

**Main inputs**
- CSV path from `root_path` + `data_path`.
- Mandatory `date` column.
- Feature mode (`S`, `M`, `MS`), optional segment split via `segment_col`, optional multi-input/exogenous columns.

**Main outputs**
- Windowed tensors: `seq_x`, `seq_y`, `seq_x_mark`, `seq_y_mark`.
- Optional metadata arrays used later for segment/horizon analysis:
  - `valid_starts`
  - `window_segment_ids`
  - `dates`

**Connections**
- Core data source for experiments in `exp/`.
- Segment metadata is consumed downstream by test/analysis scripts.

## 5.2 `exp/`

### 5.2.1 `exp/exp_Basic.py`
**What it does**
- Base experiment class with device acquisition and abstract training/testing hooks.

**Key class/methods**
- `Exp_Basic`
- `_acquire_device()`
- abstract placeholders: `_build_model`, `_get_data`, `vali`, `train`, `test`

**Connections**
- Parent class for `Exp_Main` implementations.

### 5.2.2 `exp/exp_Main2.py`
**What it does**
- Main experiment implementation currently used by `run.py`.

**Key class/methods**
- `Exp_Main`
- `_build_model()`
- `_get_data(flag)`
- `_select_optimizer()`
- `_select_criterion()`
- `_forward(batch_x)`
- `_slice_output(outputs, batch_y)`
- `vali(...)`
- `train(setting)`
- `_run_train_epoch(...)`
- `test(setting, test=0)`
- `_run_inference(...)`
- `_save_horizon_mse(...)`
- `_save_segment_metrics(...)`
- `_append_summary_csv(...)`
- `_write_run_overview(...)`
- `predict(setting, load=False)`

**Main inputs**
- Runtime args from `run.py` (namespace containing data/model/training/output config).

**Main outputs**
- Checkpoint: `checkpoints/checkpoint.pth`
- Test artifacts under `outputs/` including metrics arrays, horizon CSV/PNG, segment-level CSV/GZ, meeting table.
- Summary CSV append (path from `args.summary_csv`, usually `runs/summary/<model>_summary.csv`).

**Connections**
- Training/testing engine used by `run.py`.
- Outputs consumed by both analysis scripts and `visualize.py`.

### 5.2.3 `exp/exp_Main.py`
**What it does**
- Another experiment implementation with similar interface and older logic.

**Current usage**
- `analyze_full_inference.py` imports `Exp_Main` from this file.
- `run.py` does **not** use this file; it imports from `exp_Main2.py`.

**Risk note**
- This split introduces potential behavior mismatch between training/testing pipeline and full-inference re-run pipeline.

## 5.3 `models/`

### 5.3.1 `models/Linear.py`
- Class: `Model`
- Behavior: direct linear projection from input window (`seq_len`) to horizon (`pred_len`), per-channel or shared.
- Input tensor: `[B, L, C]`
- Output tensor: `[B, P, C]`

### 5.3.2 `models/NLinear.py`
- Class: `Model`
- Behavior: subtract last timestep, linear projection, add last timestep back.
- Input tensor: `[B, L, C]`
- Output tensor: `[B, P, C]`

### 5.3.3 `models/DLinear.py`
- Classes: `moving_avg`, `series_decomp`, `Model`
- Behavior: decomposition into trend + residual, then linear forecast on both components.
- Requires odd `dlinear_kernel_size`.

### 5.3.4 `models/DLinearMix.py`
- Classes: `moving_avg`, `series_decomp`, `Model`
- Behavior: early channel mixing (`Linear(in_channels -> 1)`), then DLinear decomposition/forecast on mixed signal.
- Uses `mix_in`/`enc_in`, and validates `branch_in + exog_in` consistency.

### 5.3.5 `models/DLinearMix2.py`
- Functions/classes: `_parse_col_spec`, `moving_avg`, `series_decomp`, `DLinearBranch`, `ExogenousEncoder`, `HorizonWiseFusion`, `FlattenFusion`, `Model`.
- Behavior: branch-wise DLinear per main input channel + optional exogenous encoder + fusion head.
- Supports two fusion styles:
  - horizon-wise shared fusion (`flatten_fusion=False`)
  - flattened fusion (`flatten_fusion=True`)
- Validates channel layout consistency.

## 5.4 `utils/`

### 5.4.1 `utils/metrics.py`
Functions:
- `RSE`, `CORR`, `MAE`, `MSE`, `RMSE`, `MAPE`, `MSPE`, `metric`

Used by experiment testing for global metrics.

### 5.4.2 `utils/tools.py`
Functions/classes:
- `adjust_learning_rate(...)`
- `EarlyStopping`
- `dotdict`
- `StandardScaler`
- `visual(...)`
- `test_params_flop(...)`

Used by experiment training/testing loops.

### 5.4.3 `utils/timefeatures.py`
Classes/functions:
- time feature classes (`MinuteOfHour`, `HourOfDay`, `DayOfWeek`, etc.)
- `time_features_from_frequency_str(...)`
- `time_features(...)`

Used by dataset preprocessing for timestamp encodings.

## 5.5 `run.py`

**What it does**
- Parses CLI args, configures channel counts for mix models, creates run folder, saves run config, then calls train and test.

**Key functions**
- `_safe_name`
- `_parse_csv_cols`
- `_configure_mix_model_args`
- `_accelerator_available`
- `_cleanup_run_dir`
- `_cleanup_run_outputs`
- `main`

**Main outputs**
- `runs/<setting>/run_args.json`
- `runs/<setting>/run_cmd.txt`
- `runs/<setting>/checkpoints/checkpoint.pth`
- `runs/<setting>/outputs/*`
- `runs/summary/<model>_summary.csv` (appended by experiment class)

## 5.6 `analyze_full_inference.py`

**What it does**
- Picks best run per pair and produces full split coverage segment-horizon metrics.

**Key classes/functions**
- `RunInfo` dataclass
- `_safe_int`, `_parse_csv_list`, `ensure_dir`, `_safe_name`, `build_output_dir`
- `discover_runs`, `load_run_info`, `pass_filters`, `select_best_runs`
- `topk_per_pair`, `pair_name`, `_safe_corr`
- `_build_points_for_split`, `_metrics_from_points`, `run_full_inference`
- `plot_best_horizon_mse_boxplot_per_pair`, `plot_horizon_metric_boxplot_per_pair`
- `main`

**Main outputs**
Written to timestamped folder in `analysis/full_inference/`:
- `full_segment_horizon_metrics.csv`
- `best_horizon_segment_metrics.csv`
- `lag_horizon_distribution.csv`
- `top30_mse_full_inference.csv`
- `top30_corr_full_inference.csv`
- `top30_overlap_segments_full_inference.csv`
- corresponding PNG charts
- additionally writes `segment_horizon_points_full.csv.gz` into each selected run's `outputs/`.

## 5.7 `analyze_best_models_overview.py`

**What it does**
- Generates best-run overview and visualization artifacts from run outputs.

**Key classes/functions**
- `RunInfo` dataclass
- `_safe_int`, `_parse_csv_list`, `ensure_dir`
- `discover_runs`, `load_run_info`, `pass_filters`, `pair_name`
- `plot_horizon_overlay`, `build_horizon_relative_table`, `plot_horizon_overlay_relative`
- `plot_segment_fan`
- `main`

**Main outputs**
In `analysis/`:
- `run_overview.csv`
- `horizon_mse_overlay.csv`
- `horizon_mse_overlay.png`
- `horizon_mse_overlay_relative.csv`
- `horizon_mse_overlay_relative.png`
- `segment_fanplots/*.png`
- `segment_fanplots/segment_fanplot_draw_records.csv`
- `segment_fanplots/segment_fanplot_counts.csv`

## 6. Data Pipeline

### 6.1 Supported Input Format
Confirmed by code and sample headers:

1. Base water-level format example (`dataset/water_level_all.csv`):
   - `date,HL01,HL02,HL03,HL04,HL05,HL06`

2. Segment/rain format example (`dataset/water_level_rain_all4.csv`):
   - includes `date`, segment metadata columns (`SegmentStart`, `SegmentEnd`, `segment_id`, `WinStart`, `WinEnd`), `isRain`, water-level columns, and rainfall accumulations.

### 6.2 Required Columns for Core Training
- Always required: `date`
- Required target: value passed by `--target`
- `features=S`: requires effective input column (`--input_col` or fallback target) and target.
- `features=M/MS`: uses non-date columns; MS forces target as last dimension.
- Segment workflow requires valid `--segment_col` existing in CSV.

### 6.3 Split and Window Rules
- If `segment_col` set:
  - segment-level split: approx 70% train, 10% val, 20% test by ordered segments.
  - windows do not cross segment boundaries.
- Else:
  - row-index split fallback.
- Window length uses `seq_len` and `pred_len`, with configurable `stride_train` and `stride_eval`.

## 7. Model Pipeline

### 7.1 Training/Test Pipeline
1. Build model from `_MODEL_DICT` in `exp/exp_Main2.py`.
2. Build dataloaders via `data_provider`.
3. Train with Adam + MSE + early stopping.
4. Load best checkpoint.
5. Run test and write global/horizon/segment outputs.

### 7.2 Full Inference Pipeline
1. Analysis script selects best checkpoints.
2. Loads experiment/model and checkpoint.
3. Reconstructs points for train/val/test where available.
4. Aggregates metrics per segment and horizon.

## 8. Training / Testing / Inference Usage

### 8.1 Training + Testing (single run)
Example command:

```bash
python run.py \
  --model DLinear \
  --data custom \
  --root_path ./dataset \
  --data_path water_level_rain_all4.csv \
  --features S \
  --input_col HL02 \
  --target HL01 \
  --segment_col segment_id \
  --seq_len 60 \
  --pred_len 15 \
  --stride_train 1 \
  --stride_eval 1
```

### 8.2 Training + Testing (mix model with exogenous)

```bash
python run.py \
  --model DLinearMix2 \
  --data custom \
  --root_path ./dataset \
  --data_path water_level_rain_all4.csv \
  --features S \
  --input_col HL02,HL03 \
  --exog_col isRain \
  --target HL01 \
  --segment_col segment_id \
  --seq_len 60 \
  --pred_len 15
```

### 8.3 Best-run overview analysis

```bash
python analyze_best_models_overview.py --runs_root ./runs --out_dir ./analysis --models DLinear --targets HL01
```

### 8.4 Full-inference analysis

```bash
python analyze_full_inference.py --runs_root ./runs --out_dir ./analysis --models DLinear --targets HL01
```

### 8.5 Visual inspection utility

```bash
python visualize.py --mode topk --output_root ./runs --k 10
python visualize.py --mode segment --run_dir ./runs/<setting> --segment 220 --horizon 7 --points_source full
```

### 8.6 Important CLI Arguments (high impact)
- Data selection: `--root_path`, `--data_path`, `--features`, `--input_col`, `--exog_col`, `--target`, `--segment_col`
- Sequence settings: `--seq_len`, `--pred_len`, `--label_len`
- Window sampling: `--stride_train`, `--stride_eval`
- Model config: `--model`, `--individual`, `--dlinear_kernel_size`, `--flatten_fusion`, `--branch_in`, `--exog_in`, `--mix_in`
- Optimization: `--train_epochs`, `--batch_size`, `--learning_rate`, `--patience`, `--lradj`
- Hardware: `--use_gpu`, `--use_amp`, `--use_multi_gpu`, `--devices`
- Output hygiene: `--output_root`, `--clean_run_dir`, `--clean_outputs_only`

## 9. Analysis Scripts

### 9.1 `analyze_best_models_overview.py`
- Requires per-run files:
  - `outputs/metrics.npy`
  - `outputs/mse_horizon.csv`
  - `outputs/mse_segment_combined.csv`
  - `outputs/segment_horizon_points.csv.gz`
- Runs lacking required files are skipped.

### 9.2 `analyze_full_inference.py`
- Requires at least `outputs/metrics.npy` for run selection.
- Requires checkpoint in `checkpoints/checkpoint.pth` for selected runs.
- Attempts to use test points file if already present; otherwise regenerates.

### 9.3 `visualize.py`
- Reads run `outputs/` files and supports modes:
  - `topk`
  - `meeting`
  - `horizon`
  - `segment`
- For segment mode and full points, depends on `segment_horizon_points_full.csv.gz` (usually created by full-inference analysis).

## 10. Outputs and Results

### 10.1 Per-run Outputs (`runs/<setting>/`)
- `run_args.json`
- `run_cmd.txt`
- `run_overview.txt`
- `checkpoints/checkpoint.pth`
- `outputs/metrics.npy`
- `outputs/pred.npy`, `outputs/true.npy`
- `outputs/mse_horizon.csv`, `outputs/mse_horizon.png`
- `outputs/mse_segment_combined.csv`
- `outputs/segment_horizon_points.csv.gz`
- `outputs/segment_horizon_rank.csv`
- `outputs/meeting.csv`

### 10.2 Aggregate Analysis Outputs
In `analysis/`:
- overview and overlay CSV/PNG artifacts
- `segment_fanplots/` artifacts

In `analysis/full_inference/<timestamp>__.../`:
- full segment-horizon metrics and top-k overlap diagnostics

## 11. Known Issues and Limitations

1. Pipeline split risk:
   - `run.py` trains/tests with `exp/exp_Main2.py`.
   - `analyze_full_inference.py` re-infers using `exp/exp_Main.py`.
   - Potential mismatch if implementations diverge.

2. Project reproducibility metadata is incomplete:
   - No dependency manifest (`requirements.txt` / `pyproject.toml`) found.

3. Some outputs are prerequisites:
   - Analysis scripts skip runs missing expected files.

4. `MAPE` / `MSPE` in `utils/metrics.py` divide by `true` directly:
   - Can become unstable/infinite when true values are zero.

5. `CORR` implementation scales by `0.01`:
   - This is non-standard and can confuse interpretation if not documented externally.

6. Legacy/parallel code paths exist (`Source_Code/`, `exp_Main.py`, `exp_Main2.py`, `checkpoints/`, `test_results/`):
   - Increases maintenance ambiguity.

7. Some runtime assumptions are implicit:
   - Data columns and semantic meaning (especially segment/rain fields) depend on external data conventions.

8. Environment setup is not codified in repository files.

## 12. Recommended Next Steps

1. Unify experiment implementation:
   - Decide whether `exp_Main.py` or `exp_Main2.py` is canonical.
   - Make `analyze_full_inference.py` use the canonical one.

2. Add explicit dependency and environment setup files.

3. Add a formal project README describing:
   - dataset schema
   - command cookbook
   - output artifacts
   - expected workflow sequence

4. Add validation checks before run:
   - assert required columns for selected mode
   - warn on impossible split/window configurations

5. Add lightweight tests for:
   - dataset split/window logic
   - metric calculations
   - analysis file contracts

6. Decide archival strategy for legacy folders (`checkpoints/`, `test_results/`, `Source_Code/`) to reduce confusion.

## Questions for the Project Owner

1. Which experiment implementation should be the long-term single source of truth: `exp/exp_Main.py` or `exp/exp_Main2.py`?
2. What exact Python/package versions were used for the latest trusted results?
3. Which dataset file is the production baseline for future runs: `water_level_all.csv`, `water_level_rain_all4.csv`, or another file?
4. Should `CORR` remain scaled by `0.01`, or should it be changed to standard Pearson correlation scale?
5. Is `Source_Code/` still active or only historical backup?
6. Are there required data privacy or governance constraints for sharing run outputs?

## Most Important Files for Future Work

1. `run.py`
2. `exp/exp_Main2.py`
3. `data_provider/Data_Loader.py`
4. `models/DLinearMix2.py`
5. `analyze_full_inference.py`
6. `analyze_best_models_overview.py`
7. `runs/summary/DLinearMix2_summary.csv`
8. `analysis/run_overview.csv`
9. `analysis/full_segment_horizon_metrics.csv`
