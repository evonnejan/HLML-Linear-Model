import argparse
import atexit
import fnmatch
import random
import torch
import numpy as np
import os, json, sys, time
import shutil
import pandas as pd
from exp.exp_Main2 import Exp_Main # 注意這裡的 Import 路徑要符合你的檔案大小寫


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU/CUDA/MPS) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
    # Reduce non-determinism from cuDNN; harmless on MPS/CPU.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _safe_name(text):
    return str(text).replace('/', '-').replace(' ', '_')


def _parse_csv_cols(text):
    if text is None:
        return []
    if isinstance(text, str):
        return [item.strip() for item in text.split(',') if item.strip()]
    return [str(item).strip() for item in text if str(item).strip()]


def _expand_col_patterns(patterns, available_cols):
    """Expand glob patterns (e.g. 'HL*', '*gate_opening*') against the actual
    CSV column list. Plain names are passed through unchanged. Order is
    preserved and duplicates are removed."""
    seen = set()
    out = []
    for p in patterns:
        if any(ch in p for ch in '*?[]'):
            matches = [c for c in available_cols if fnmatch.fnmatchcase(c, p)]
            if not matches:
                raise ValueError(
                    f"Pattern '{p}' matched no columns. Available: {available_cols}"
                )
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    out.append(m)
        else:
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _validate_split_args(parser, args):
    """強制 split 方式必須明講，避免靜默回退到舊的內建切法。

    --split_mode 刻意沒有預設值：忘了帶就直接報錯，而不是安靜地用另一種切法
    跑完 —— 後者會讓整批實驗結果不可比且難以察覺。
    """
    if not getattr(args, "segment_col", None):
        for name in ("split_mode", "split_file", "fold"):
            if getattr(args, name, None) is not None:
                parser.error(f"--{name} 需要同時設定 --segment_col。")
        return

    if args.split_mode is None:
        parser.error(
            "已設定 --segment_col，必須明確指定 --split_mode：\n"
            "  --split_mode file --split_file dataset/splits_<dataset>.csv"
            "   （依 window 數切分、含重疊防護、可搭配 --fold 做 rolling-origin CV）\n"
            "  --split_mode builtin"
            "                                        （舊的依段數 70/10/rest，無重疊防護）"
        )
    if args.split_mode == "file":
        if not args.split_file:
            parser.error("--split_mode file 需要 --split_file（由 build_splits.py 產生）。")
        if not os.path.exists(args.split_file):
            parser.error(f"--split_file 找不到檔案：{args.split_file}")
    else:  # builtin
        if args.split_file:
            parser.error("--split_mode builtin 不可與 --split_file 併用；要用 split 檔請改 --split_mode file。")
        if args.fold is not None:
            parser.error("--fold 只能搭配 --split_mode file 使用。")


def _expand_col_args(args):
    """Read CSV header, expand wildcards in --input_col and --exog_col."""
    csv_path = os.path.join(args.root_path, args.data_path)
    if not os.path.isfile(csv_path):
        return
    # Read only the header row for cheapness.
    available = list(pd.read_csv(csv_path, nrows=0).columns)

    for attr in ("input_col", "exog_col"):
        raw = getattr(args, attr, None)
        cols = _parse_csv_cols(raw)
        if not cols:
            continue
        expanded = _expand_col_patterns(cols, available)
        if expanded != cols:
            print(f"[col expand] --{attr}: {raw}  =>  {','.join(expanded)}")
        setattr(args, attr, ",".join(expanded) if expanded else None)


def _drop_constant_columns(args):
    """Pre-flight check: drop columns with std=0 on the train split from
    --input_col / --exog_col. Such columns become a constant 0 stream after
    StandardScaler and waste model capacity. Booleans are tested as 0/1."""
    input_cols = _parse_csv_cols(args.input_col)
    exog_cols = _parse_csv_cols(getattr(args, 'exog_col', None))
    candidates = input_cols + exog_cols
    if not candidates:
        return

    csv_path = os.path.join(args.root_path, args.data_path)
    if not os.path.isfile(csv_path):
        return  # let Data_Loader raise the proper error
    df = pd.read_csv(csv_path)

    # Identify the train slice using the same rule as Data_Loader.
    seg_col = getattr(args, 'segment_col', None)
    if seg_col and seg_col in df.columns:
        if "SegmentStart" in df.columns:
            seg_info = df[[seg_col, "SegmentStart"]].drop_duplicates().copy()
            seg_info["SegmentStart"] = pd.to_datetime(seg_info["SegmentStart"])
            seg_ids_sorted = seg_info.sort_values("SegmentStart")[seg_col].tolist()
        else:
            seg_info = df[[seg_col, "date"]].drop_duplicates().copy()
            seg_info["date"] = pd.to_datetime(seg_info["date"])
            seg_ids_sorted = (
                seg_info.groupby(seg_col)["date"].min().sort_values().index.tolist()
            )
        nseg = len(seg_ids_sorted)
        train_n = max(1, int(nseg * 0.7))
        train_ids = set(seg_ids_sorted[:train_n])
        df_train = df[df[seg_col].isin(train_ids)]
    else:
        num_train = int(len(df) * 0.7)
        df_train = df.iloc[:num_train]

    to_drop = []
    for c in candidates:
        if c not in df_train.columns:
            continue  # let Data_Loader produce the proper missing-column error
        col = df_train[c]
        if col.dtype == bool:
            col = col.astype(int)
        non_nan = col.dropna()
        # A column is "constant" if all non-NaN values are identical.
        # nunique() handles floating-point exact-equality cleanly here because
        # the dataset stores already-quantised sensor readings; for true
        # near-constant floats this would still flag legitimate constants.
        if non_nan.nunique() <= 1:
            value = float(non_nan.iloc[0]) if len(non_nan) > 0 else float("nan")
            to_drop.append((c, value))

    if not to_drop:
        return

    drop_set = {c for c, _ in to_drop}
    print("[constant-column auto-drop]")
    for c, value in to_drop:
        bucket = "input_col" if c in input_cols else "exog_col"
        print(f"  '{c}' is constant (value={value}) on train split — dropping from --{bucket}")
    input_cols = [c for c in input_cols if c not in drop_set]
    exog_cols = [c for c in exog_cols if c not in drop_set]
    args.input_col = ",".join(input_cols) if input_cols else None
    args.exog_col = ",".join(exog_cols) if exog_cols else None


def _configure_mix_model_args(args):
    """
    Normalize channel-related args for mix-style models.

    Canonical rule:
    - input_col list controls branch channels
    - exog_col list controls exogenous channels
    - enc_in / mix_in only carry total channel count
    """
    input_cols = _parse_csv_cols(args.input_col)
    exog_cols = _parse_csv_cols(args.exog_col)

    if len(input_cols) == 0:
        raise ValueError(f"{args.model} requires --input_col, e.g. --input_col HL02,HL03")

    overlap = sorted(set(input_cols) & set(exog_cols))
    if overlap:
        raise ValueError(
            f"--input_col and --exog_col cannot share columns; duplicates: {overlap}. "
            f"Each column must appear in exactly one of the two lists."
        )

    inferred_branch_in = len(input_cols)
    inferred_exog_in = len(exog_cols)

    if args.branch_in is None:
        args.branch_in = inferred_branch_in
    elif int(args.branch_in) != inferred_branch_in:
        raise ValueError(
            f"branch_in ({args.branch_in}) must equal number of input_col ({inferred_branch_in})"
        )

    if args.exog_in is None:
        args.exog_in = inferred_exog_in
    elif int(args.exog_in) != inferred_exog_in:
        raise ValueError(
            f"exog_in ({args.exog_in}) must equal number of exog_col ({inferred_exog_in})"
        )

    inferred_mix_in = int(args.branch_in) + int(args.exog_in)
    if inferred_mix_in <= 0:
        raise ValueError("branch_in + exog_in must be positive")

    if args.mix_in is None:
        args.mix_in = inferred_mix_in
    elif int(args.mix_in) != inferred_mix_in:
        raise ValueError(
            f"mix_in ({args.mix_in}) must equal branch_in + exog_in ({inferred_mix_in})"
        )

    args.enc_in = int(args.mix_in)
    args.input_cols = input_cols
    args.exog_cols = exog_cols


def _accelerator_available():
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    return has_cuda or has_mps


def _cleanup_run_dir(run_dir: str):
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)


def _cleanup_run_outputs(run_dir: str):
    outputs_dir = os.path.join(run_dir, 'outputs')
    if os.path.isdir(outputs_dir):
        shutil.rmtree(outputs_dir)

def main():
    # 使用 argparse 來建立參數設定
    parser = argparse.ArgumentParser(description='Time Series Forecasting with Linear Models')

    # --- 基本設定 ---
    parser.add_argument('--model', type=str, default='Linear', help='model name, options: [NLinear, DLinear, Linear, DLinearMix, DLinearMix2]')
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='water_level_all.csv', help='data file')
    parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--input_col', type=str, default=None, help='input column(s), e.g., HL02 or HL02,HL03')
    parser.add_argument('--exog_col', type=str, default=None, help='optional exogenous column(s), e.g., isRain or isRain,HL06')
    parser.add_argument("--segment_col", type=str, default=None, help="e.g., segment_id; if set, windows will not cross segments")
    parser.add_argument("--split_mode", type=str, default=None, choices=["file", "builtin"],
                        help="REQUIRED when --segment_col is set. 'file': use --split_file from "
                             "build_splits.py (window-count boundaries, overlap guard, rolling-origin CV). "
                             "'builtin': legacy 70/10/rest by segment count, no overlap guard. "
                             "No default on purpose — the split must be an explicit choice.")
    parser.add_argument("--split_file", type=str, default=None,
                        help="segment-wise split assignment CSV from build_splits.py; "
                             "required by (and only valid with) --split_mode file")
    parser.add_argument("--fold", type=int, default=None,
                        help="use fold_<k> column of --split_file for train/val (rolling-origin CV); "
                             "test hold-out stays fixed across folds")
    parser.add_argument('--target', type=str, default='HL01', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='min', help='freq for time features encoding, options:[s, min, h, D, B, W, ME]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--output_root', type=str, default='./runs', help='root folder for all run outputs')
    parser.add_argument(
        '--clean_run_dir',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='remove existing run_dir before writing outputs (safe with fixed run_id/manual reruns)'
    )
    parser.add_argument(
        '--clean_outputs_only',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='remove existing run_dir/outputs before test() writes new outputs'
    )
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')

    # --- 預測長度設定 ---
    # 假設我們看過去 60 分鐘 (60步)，預測未來 15 分鐘 (15步)
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=30, help='start token length (for Transformer, Linear models mostly ignore this)')
    parser.add_argument('--pred_len', type=int, default=15, help='prediction sequence length')

    # --- 模型架構設定 ---
    parser.add_argument('--stride_train', type=int, default=1)
    parser.add_argument('--stride_eval', type=int, default=1)
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size (你的 HL01~HL06 共 6 個變數)')
    parser.add_argument(
        "--individual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use individual linear layer per channel"
    )
    parser.add_argument('--dlinear_kernel_size', type=int, default=25,
                    help='DLinear decomposition moving average kernel size (odd number)')
    parser.add_argument('--mix_in', type=int, default=None,
                    help='total channels used by mix models (normally auto-inferred from input_col/exog_col)')
    parser.add_argument('--branch_in', type=int, default=None,
                    help='branch channel count for branch-wise mix models (auto-inferred from input_col)')
    parser.add_argument('--exog_in', type=int, default=None,
                    help='exogenous channel count for branch-wise mix models (auto-inferred from exog_col)')
    parser.add_argument(
        "--flatten_fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="for DLinearMix2: use flatten fusion instead of horizon-wise fusion"
    )
    parser.add_argument('--fusion_hidden_dim', type=int, default=32,
                    help='for DLinearMix2: hidden dim of fusion MLP')
    parser.add_argument('--exog_emb_dim', type=int, default=16,
                    help='for DLinearMix2: GRU hidden size = embedding size of exogenous encoder')
    parser.add_argument('--dropout', type=float, default=0.1,
                    help='for DLinearMix2: dropout for exogenous/fusion MLP blocks')

    # --- 訓練設定 ---
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers (Windows 建議設為 0 以免卡死)')
    parser.add_argument('--train_only', type=bool, default=False, help='train only or not')
    parser.add_argument('--test_flop', type=bool, default=False, help='See flop or not')
    parser.add_argument('--output_attention', type=bool, default=False, help='whether to output attention in encoder')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for Python/NumPy/PyTorch (CPU/CUDA/MPS) reproducibility')
    parser.add_argument('--criterion', type=str, default='mse', choices=['mse', 'huber'],
                        help='training loss: mse (default) or huber (SmoothL1Loss); huber down-weights outlier-event gradients')
    parser.add_argument('--huber_beta', type=float, default=1.0,
                        help='for --criterion huber: transition point (in scaled units) between quadratic and linear regions')
    parser.add_argument('--warmup_epochs', type=int, default=8,
                        help='for lradj=warmup_exp: linear warm-up length (0.1*lr → lr)')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.92,
                        help='for lradj=warmup_exp: per-epoch exp decay factor after warm-up')
    parser.add_argument('--lr_floor_ratio', type=float, default=0.01,
                        help='for lradj=warmup_exp: floor LR as ratio of base learning_rate')
    parser.add_argument(
        '--early_stop_metric',
        type=str,
        default='mse',
        choices=['mse', 'mae', 'corr'],
        help='metric used to pick the best checkpoint and trigger early stopping (default: mse)'
    )
    parser.add_argument(
        "--save_test_plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-batch test plots (default: disabled)"
    )
    
    # --- 硬體與混合精度設定 ---
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True if _accelerator_available() else False, help='use gpu (CUDA or MPS)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids for multi-gpu, e.g., 0,1')

    args = parser.parse_args()

    _validate_split_args(parser, args)

    run_id = time.strftime("%Y%m%d-%H%M%S")

    if args.model in ('DLinearMix', 'DLinearMix2'):
        _expand_col_args(args)
        _drop_constant_columns(args)
        _configure_mix_model_args(args)

    input_col_name = args.input_col if args.input_col else args.target
    if getattr(args, 'exog_col', None):
        input_col_name = f"{input_col_name}+exog({args.exog_col})"
    setting = (
        f"{_safe_name(args.model)}_"
        f"{_safe_name(input_col_name)}-to-{_safe_name(args.target)}_"
        f"sl{args.seq_len}_pl{args.pred_len}_"
        f"st{args.stride_train}-{args.stride_eval}_"
        f"{run_id}"
    )

    args.setting = setting
    args.run_id = run_id
    args.run_dir = os.path.join(args.output_root, args.model, setting)
    args.checkpoints = os.path.join(args.run_dir, 'checkpoints')
    args.summary_csv = os.path.join(args.output_root, args.model, f'{args.model}_summary.csv')

    if args.clean_run_dir:
        _cleanup_run_dir(args.run_dir)

    os.makedirs(args.run_dir, exist_ok=True)

    class _Tee:
        """Write to console + log file. Collapse tqdm-style \\r updates so log keeps only the final state per line."""

        def __init__(self, console, log_file):
            self.console = console
            self.log_file = log_file
            self._log_partial = ""

        def write(self, data):
            self.console.write(data)
            if not data:
                return
            buf = self._log_partial + data
            if '\n' in buf:
                *lines, tail = buf.split('\n')
                for line in lines:
                    if '\r' in line:
                        line = line.split('\r')[-1]
                    self.log_file.write(line + '\n')
                self._log_partial = tail
            else:
                self._log_partial = buf

        def flush(self):
            self.console.flush()
            self.log_file.flush()

        def flush_partial(self):
            if self._log_partial:
                line = self._log_partial
                if '\r' in line:
                    line = line.split('\r')[-1]
                self.log_file.write(line)
                self._log_partial = ""

        def isatty(self):
            return False

    _train_log = open(os.path.join(args.run_dir, 'train.log'), 'w', encoding='utf-8', buffering=1)
    _stdout_tee = _Tee(sys.__stdout__, _train_log)
    _stderr_tee = _Tee(sys.__stderr__, _train_log)
    sys.stdout = _stdout_tee
    sys.stderr = _stderr_tee

    def _close_train_log():
        try:
            _stdout_tee.flush_partial()
            _stderr_tee.flush_partial()
            _train_log.flush()
        except Exception:
            pass
        # Restore original streams before closing log, so any later flush by
        # Python shutdown won't try to touch the closed log file.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        try:
            _train_log.close()
        except Exception:
            pass

    atexit.register(_close_train_log)

    def save_run_config(setting, args):
        folder = args.run_dir
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, "run_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)

        with open(os.path.join(folder, "run_cmd.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(sys.argv) + "\n")

    save_run_config(setting, args)

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    # Seed before instantiating Exp (model init / DataLoader sampling depend on RNG).
    _set_seed(int(args.seed))
    print(f'>>>>>>> seeded RNGs with seed={args.seed} <<<<<<<')

    # 開始訓練
    print('>>>>>>> start training : {} >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp = Exp(args)  # 實例化實驗
    exp.train(setting)
    
    # 開始測試
    print('>>>>>>> testing : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    if args.clean_outputs_only:
        _cleanup_run_outputs(args.run_dir)
    exp.test(setting)

if __name__ == '__main__':
    main()