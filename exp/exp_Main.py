from data_provider.Data_Factory import data_provider
from exp.exp_Basic import Exp_Basic
# from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from models import DLinear, DLinearMix, Linear, NLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os, time, csv
from datetime import datetime
import json
import importlib
from pathlib import Path

import warnings
import matplotlib.pyplot as plt

try:
    tqdm = importlib.import_module('tqdm').tqdm
except Exception:
    tqdm = None

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.use_amp = bool(getattr(self.args, 'use_amp', False) and self.device.type == 'cuda')
        if getattr(self.args, 'use_amp', False) and not self.use_amp:
            print('AMP is enabled only on CUDA; running without AMP on current device')

    def _checkpoint_dir(self):
        if hasattr(self.args, 'run_dir'):
            path = os.path.join(self.args.run_dir, 'checkpoints')
        else:
            path = os.path.join(self.args.checkpoints, self.args.setting if hasattr(self.args, 'setting') else '')
        os.makedirs(path, exist_ok=True)
        return path

    def _outputs_dir(self):
        if hasattr(self.args, 'run_dir'):
            path = os.path.join(self.args.run_dir, 'outputs')
        else:
            path = os.path.join('./results', self.args.setting if hasattr(self.args, 'setting') else '')
        os.makedirs(path, exist_ok=True)
        return path

    def _plots_dir(self):
        if hasattr(self.args, 'run_dir'):
            path = os.path.join(self.args.run_dir, 'plots')
        else:
            path = os.path.join('./test_results', self.args.setting if hasattr(self.args, 'setting') else '')
        os.makedirs(path, exist_ok=True)
        return path

    def _plot_segment_horizon_case(self, points_df, segment, horizon, out_path, title):
        part = points_df[(points_df['segment'] == segment) & (points_df['horizon'] == horizon)].copy()
        if len(part) == 0:
            return

        part = part.sort_values('target_time')
        x = pd.to_datetime(part['target_time'])

        plt.figure(figsize=(12, 4.5))
        plt.plot(x, part['true'].to_numpy(dtype=float), label='GroundTruth', linewidth=2)
        plt.plot(x, part['pred'].to_numpy(dtype=float), label='Prediction', linewidth=2)
        plt.title(title)
        plt.xlabel('Target Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()

    def _plot_ranked_cases(self, points_df, rank_df, metric_name, ascending, top_k, out_dir, prefix):
        if len(rank_df) == 0:
            return

        used = rank_df.dropna(subset=[metric_name]).sort_values(metric_name, ascending=ascending).head(top_k)
        for idx, row in enumerate(used.itertuples(index=False), start=1):
            metric_val = getattr(row, metric_name)
            segment_val = getattr(row, 'segment')
            horizon_val = int(getattr(row, 'horizon'))
            title = f"{prefix} #{idx}: segment={segment_val}, horizon=t+{horizon_val}, {metric_name}={metric_val:.4f}"
            out_path = os.path.join(out_dir, f"{prefix.lower()}_{idx:02d}_seg-{segment_val}_h-{horizon_val}.png")
            self._plot_segment_horizon_case(points_df, segment_val, horizon_val, out_path, title)

    def _build_model(self):
        model_dict = {
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Informer': Informer,
            'DLinear': DLinear,
            'DLinearMix': DLinearMix,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.device.type == 'cuda':
            devices_text = str(getattr(self.args, 'devices', str(getattr(self.args, 'gpu', 0))))
            device_ids = [int(item.strip()) for item in devices_text.split(',') if item.strip() != '']
            model = nn.DataParallel(model, device_ids=device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = self._checkpoint_dir()

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epochs_trained = 0
        for epoch in range(self.args.train_epochs):
            epochs_trained = epoch + 1
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            train_iter = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Train {epoch + 1}/{self.args.train_epochs}",
                leave=False
            ) if tqdm is not None else train_loader

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_iter):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if tqdm is not None and hasattr(train_iter, 'set_postfix'):
                    train_iter.set_postfix(loss=f"{loss.item():.5f}")

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        self.epochs_trained = epochs_trained
        
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self._checkpoint_dir(), 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        segment_ids_all = []
        sample_ptr = 0
        plots_dir = None
        save_test_plots = bool(getattr(self.args, 'save_test_plots', False))

        self.model.eval()
        with torch.no_grad():
            test_iter = tqdm(test_loader, total=len(test_loader), desc='Test', leave=False) if tqdm is not None else test_loader
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_iter):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                batch_size_now = pred.shape[0]
                if hasattr(test_data, 'window_segment_ids') and test_data.window_segment_ids is not None:
                    segment_ids_all.append(test_data.window_segment_ids[sample_ptr:sample_ptr + batch_size_now])
                sample_ptr += batch_size_now

                if save_test_plots and i % 20 == 0:
                    if plots_dir is None:
                        plots_dir = self._plots_dir()
                    # 取第一筆樣本做示意圖
                    x0 = batch_x.detach().cpu().numpy()[0]
                    y_true0 = true[0]
                    y_pred0 = pred[0]

                    # inverse 回 HL01 真實尺度（你的 test_data.inverse_transform 應該對應 scaler_y）
                    y_true0_inv = test_data.inverse_transform(y_true0.reshape(-1, y_true0.shape[-1])).reshape(-1) if test_data.scale else y_true0.reshape(-1)
                    y_pred0_inv = test_data.inverse_transform(y_pred0.reshape(-1, y_pred0.shape[-1])).reshape(-1) if test_data.scale else y_pred0.reshape(-1)

                    # === 對齊這一筆 sample 在 dataset 裡的位置 ===
                    # test_loader: shuffle=False, drop_last=False，因此第 i 個 batch 的第 0 筆 = dataset 的 index i*batch_size
                    ds_idx = i * test_loader.batch_size
                    s_begin = int(test_data.valid_starts[ds_idx]) if test_data.valid_starts is not None else ds_idx

                    # future 對應到的 row index 範圍
                    t0 = s_begin + self.args.seq_len
                    t1 = t0 + self.args.pred_len

                    # 取出 future 的真實時間軸（長度 pred_len）
                    x_time = test_data.dates[t0:t1]

                    # ✅ 檢查：這個 future 的 raw HL01，應該要等於 inverse 後的 y_true0（允許極小誤差）
                    y_true_raw = test_data.y_raw[t0:t1]
                    max_abs_diff = float(np.max(np.abs(y_true_raw - y_true0_inv)))
                    if max_abs_diff > 1e-3:
                        print(f"[WARN] y_true mismatch: max_abs_diff={max_abs_diff:.6f} (check scaling / indexing)")

                    x_col = getattr(self.args, "input_col", None) or self.args.target
                    y_col = self.args.target

                    # HL02 -> HL01：只畫 future 的 HL01 True vs Pred
                    if x_col != y_col:
                        title = f"{self.args.model} | {y_col} | {str(x_time[0])} ~ {str(x_time[-1])}"
                        visual(y_true0_inv, y_pred0_inv,
                            os.path.join(plots_dir, f"{i}_future.png"),
                            x=x_time, title=title)
                    else:
                        # HL01 -> HL01（同欄位）才畫 history+future（可選）
                        title = f"{self.args.model} | {y_col} | {str(test_data.dates[s_begin])}"
                        gt = np.concatenate((test_data.y_raw[s_begin:s_begin+self.args.seq_len], y_true_raw), axis=0)
                        pred_line = np.concatenate((test_data.y_raw[s_begin:s_begin+self.args.seq_len], y_pred0_inv), axis=0)
                        x_all = test_data.dates[s_begin:s_begin+self.args.seq_len+self.args.pred_len]
                        visual(gt, pred_line, os.path.join(plots_dir, f"{i}.png"), x=x_all, title=title)
                        
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        
        # 將數值還原為真實水位
        if test_data.scale:
            # 因為 StandardScaler 只吃 2D 陣列，所以我們先把它壓平，轉完再塑型回來
            shape_preds = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, shape_preds[-1])).reshape(shape_preds)
            trues = test_data.inverse_transform(trues.reshape(-1, shape_preds[-1])).reshape(shape_preds)

        # result save
        outputs_dir = self._outputs_dir()

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        sq_err = np.square(preds - trues)
        if sq_err.ndim == 3:
            horizon_mse = sq_err.mean(axis=(0, 2))
        else:
            horizon_mse = sq_err.mean(axis=0)

        pd.DataFrame({
            'horizon': np.arange(1, len(horizon_mse) + 1, dtype=np.int64),
            'mse': horizon_mse.astype(float)
        }).to_csv(os.path.join(outputs_dir, 'mse_horizon.csv'), index=False, encoding='utf-8-sig')

        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, len(horizon_mse) + 1), horizon_mse, marker='o')
        plt.xlabel('Horizon (t+k)')
        plt.ylabel('MSE')
        plt.title('Horizon-wise MSE')
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'mse_horizon.png'), dpi=180)
        plt.close()

        has_segment_metrics = False
        segment_combined_rows = []
        points_df = None
        if len(segment_ids_all) > 0:
            segment_ids = np.concatenate(segment_ids_all, axis=0)
            if len(segment_ids) == preds.shape[0]:
                has_segment_metrics = True
                segment_rows = []
                segment_horizon_rows = []

                if preds.ndim == 3 and preds.shape[2] == 1:
                    pred_viz = preds[:, :, 0]
                    true_viz = trues[:, :, 0]
                elif preds.ndim == 3:
                    pred_viz = preds.mean(axis=2)
                    true_viz = trues.mean(axis=2)
                else:
                    pred_viz = preds
                    true_viz = trues

                start_indices = test_data.valid_starts if test_data.valid_starts is not None else np.arange(pred_viz.shape[0], dtype=np.int64)
                points_rows = []

                for seg_id in pd.unique(segment_ids):
                    mask = (segment_ids == seg_id)
                    seg_sq_err = sq_err[mask]
                    seg_overall = {
                        'segment': seg_id,
                        'num_windows': int(mask.sum()),
                        'horizon': 'all',
                        'mse': float(seg_sq_err.mean())
                    }
                    segment_rows.append(seg_overall)
                    segment_combined_rows.append(seg_overall)

                    if seg_sq_err.ndim == 3:
                        seg_horizon_mse = seg_sq_err.mean(axis=(0, 2))
                    else:
                        seg_horizon_mse = seg_sq_err.mean(axis=0)

                    for horizon_idx, horizon_val in enumerate(seg_horizon_mse, start=1):
                        row_item = {
                            'segment': seg_id,
                            'horizon': int(horizon_idx),
                            'mse': float(horizon_val)
                        }
                        segment_horizon_rows.append(row_item)
                        segment_combined_rows.append(row_item)

                    seg_window_indices = np.where(mask)[0]
                    for local_idx in seg_window_indices:
                        s_begin = int(start_indices[local_idx])
                        for horizon_idx in range(self.args.pred_len):
                            t_idx = s_begin + self.args.seq_len + horizon_idx
                            if t_idx >= len(test_data.dates):
                                continue
                            y_t = float(true_viz[local_idx, horizon_idx])
                            y_p = float(pred_viz[local_idx, horizon_idx])
                            points_rows.append({
                                'segment': seg_id,
                                'window_idx': int(local_idx),
                                'horizon': int(horizon_idx + 1),
                                'target_time': pd.to_datetime(test_data.dates[t_idx]).strftime('%Y-%m-%d %H:%M:%S'),
                                'true': y_t,
                                'pred': y_p,
                                'abs_err': abs(y_t - y_p),
                                'sq_err': (y_t - y_p) ** 2
                            })

                pd.DataFrame(segment_combined_rows).to_csv(
                    os.path.join(outputs_dir, 'mse_segment_combined.csv'),
                    index=False,
                    encoding='utf-8-sig'
                )

                points_df = pd.DataFrame(points_rows)

                rank_rows = []
                grouped = points_df.groupby(['segment', 'horizon'])
                for (seg_id, horizon_id), grp in grouped:
                    true_arr = grp['true'].to_numpy(dtype=float)
                    pred_arr = grp['pred'].to_numpy(dtype=float)
                    if len(true_arr) >= 2:
                        corr_val = float(np.corrcoef(true_arr, pred_arr)[0, 1])
                    else:
                        corr_val = np.nan
                    mse_val = float(np.mean((pred_arr - true_arr) ** 2))
                    rank_rows.append({
                        'segment': seg_id,
                        'horizon': int(horizon_id),
                        'num_points': int(len(grp)),
                        'mse': mse_val,
                        'corr': corr_val
                    })

                rank_df = pd.DataFrame(rank_rows).sort_values(['mse', 'corr'], ascending=[True, False])
                rank_df.to_csv(
                    os.path.join(outputs_dir, 'segment_horizon_rank.csv'),
                    index=False,
                    encoding='utf-8-sig'
                )

                points_df.to_csv(
                    os.path.join(outputs_dir, 'segment_horizon_points.csv.gz'),
                    index=False,
                    encoding='utf-8-sig',
                    compression='gzip'
                )

                meeting_rows = []
                meeting_rows.extend(rank_df.dropna(subset=['mse']).sort_values('mse', ascending=True).head(50).assign(category='best_mse').to_dict(orient='records'))
                meeting_rows.extend(rank_df.dropna(subset=['mse']).sort_values('mse', ascending=False).head(50).assign(category='worst_mse').to_dict(orient='records'))
                meeting_rows.extend(rank_df.dropna(subset=['corr']).sort_values('corr', ascending=False).head(10).assign(category='best_corr').to_dict(orient='records'))
                meeting_rows.extend(rank_df.dropna(subset=['corr']).sort_values('corr', ascending=True).head(10).assign(category='worst_corr').to_dict(orient='records'))
                pd.DataFrame(meeting_rows).to_csv(
                    os.path.join(outputs_dir, 'meeting.csv'),
                    index=False,
                    encoding='utf-8-sig'
                )

        if not has_segment_metrics:
            print('segment-wise metrics skipped (segment_col not set or ids unavailable)')

        csv_path = getattr(self.args, 'summary_csv', os.path.join(os.getcwd(), 'results_summary.csv'))
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fieldnames = [
            "time", "run_id", "setting", "run_dir", "device", "model",
            "input_col", "output_col",
            "seq_len", "label_len", "pred_len",
            "stride", "kernel_size",
            "epochs", "batch_size", "lr", "lradj",
            "MSE", "MAE", "corr"
        ]

        # corr 目前是 (pred_len,) 的 array（每個 horizon 一個），存成可解析字串
        if isinstance(corr, np.ndarray):
            corr_str = ";".join([f"{c:.6f}" for c in corr.reshape(-1)])
        else:
            corr_str = str(corr)
    
        kernel_size = ""
        if getattr(self.args, "model", "") == "DLinear":
            kernel_size = str(getattr(self.args, "dlinear_kernel_size", ""))

        epochs_run = int(getattr(self, "epochs_trained", self.args.train_epochs))
        run_dir_abs = os.path.abspath(getattr(self.args, 'run_dir', ''))
        outputs_abs = os.path.abspath(outputs_dir)
        overview_path = os.path.join(run_dir_abs, 'run_overview.txt')

        overview_lines = [
            f"run_dir: {run_dir_abs}",
            f"device: {self.device}",
            f"model: {self.args.model}",
            f"input_col: {getattr(self.args, 'input_col', None) or self.args.target}",
            f"target: {self.args.target}",
            "",
            "outputs:",
            f"- metrics.npy",
            f"- pred.npy",
            f"- true.npy",
            f"- mse_horizon.csv",
            f"- mse_horizon.png",
            f"- mse_segment_combined.csv",
            f"- segment_horizon_rank.csv",
            f"- segment_horizon_points.csv.gz",
            f"- meeting.csv",
            "",
            f"outputs_abs: {outputs_abs}"
        ]
        with open(overview_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(overview_lines) + '\n')

        overview_url = Path(overview_path).resolve().as_uri()

        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": getattr(self.args, 'run_id', ''),
            "setting": setting,
            "run_dir": f"=HYPERLINK(\"{overview_url}\",\"open_overview\")",
            "device": str(self.device),
            "model": self.args.model,
            "input_col": getattr(self.args, "input_col", None) or self.args.target,
            "output_col": self.args.target,
            "seq_len": int(self.args.seq_len),
            "label_len": int(self.args.label_len),
            "pred_len": int(self.args.pred_len),
            "stride": int(self.args.stride_train),
            "kernel_size": kernel_size,
            "epochs": epochs_run,
            "batch_size": int(self.args.batch_size),
            "lr": float(self.args.learning_rate),
            "lradj": str(getattr(self.args, "lradj", "")),
            "MSE": float(mse),
            "MAE": float(mae),
            "corr": corr_str
        }

        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        np.save(os.path.join(outputs_dir, 'metrics.npy'), np.array([float(mae), float(mse), float(rmse), float(mape), float(mspe), float(rse)]))
        np.save(os.path.join(outputs_dir, 'pred.npy'), preds)
        np.save(os.path.join(outputs_dir, 'true.npy'), trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            best_model_path = os.path.join(self._checkpoint_dir(), 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        
        # result save
        outputs_dir = self._outputs_dir()

        np.save(os.path.join(outputs_dir, 'real_prediction.npy'), preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(os.path.join(outputs_dir, 'real_prediction.csv'), index=False)

        return