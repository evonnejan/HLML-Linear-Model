import argparse
import torch
import os, json, sys, time
from exp.exp_Main import Exp_Main # 注意這裡的 Import 路徑要符合你的檔案大小寫


def _safe_name(text):
    return str(text).replace('/', '-').replace(' ', '_')


def _parse_csv_cols(text):
    if text is None:
        return []
    if isinstance(text, str):
        return [item.strip() for item in text.split(',') if item.strip()]
    return [str(item).strip() for item in text if str(item).strip()]


def _accelerator_available():
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    return has_cuda or has_mps

def main():
    # 使用 argparse 來建立參數設定
    parser = argparse.ArgumentParser(description='Time Series Forecasting with Linear Models')

    # --- 基本設定 ---
    parser.add_argument('--model', type=str, default='Linear', help='model name, options: [NLinear, DLinear, Linear, DLinearMix]')
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='water_level_all.csv', help='data file')
    parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--input_col', type=str, default=None, help='input column(s), e.g., HL02 or HL02,HL03')
    parser.add_argument('--exog_col', type=str, default=None, help='optional exogenous column(s), e.g., isRain or isRain,HL06')
    parser.add_argument("--segment_col", type=str, default=None, help="e.g., segment_id; if set, windows will not cross segments")
    parser.add_argument('--target', type=str, default='HL01', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='min', help='freq for time features encoding, options:[s, min, h, D, B, W, ME]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--output_root', type=str, default='./runs', help='root folder for all run outputs')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')

    # --- 預測長度設定 ---
    # 假設我們看過去 60 分鐘 (60步)，預測未來 15 分鐘 (15步)
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=30, help='start token length (for Transformer, Linear models mostly ignore this)')
    parser.add_argument('--pred_len', type=int, default=15, help='prediction sequence length')

    # --- 模型架構設定 ---
    parser.add_argument('--stride_train', type=int, default=3)
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
    
    run_id = time.strftime("%Y%m%d-%H%M%S")

    if args.model == 'DLinearMix':
        input_cols = _parse_csv_cols(args.input_col)
        exog_cols = _parse_csv_cols(args.exog_col)
        if len(input_cols) == 0:
            raise ValueError("DLinearMix requires --input_col, e.g. --input_col HL02,HL03")
        args.mix_in = len(input_cols) + len(exog_cols)
        args.enc_in = args.mix_in

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
    args.run_dir = os.path.join(args.output_root, setting)
    args.checkpoints = os.path.join(args.run_dir, 'checkpoints')
    args.summary_csv = os.path.join(args.output_root, 'summary.csv')
    os.makedirs(args.run_dir, exist_ok=True)
    
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

    # 開始訓練
    print('>>>>>>> start training : {} >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp = Exp(args)  # 實例化實驗
    exp.train(setting)
    
    # 開始測試
    print('>>>>>>> testing : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

if __name__ == '__main__':
    main()