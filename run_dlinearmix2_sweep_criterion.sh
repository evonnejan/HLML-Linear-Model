#!/usr/bin/env bash
# 3 seq_len x 2 fusion x 3 criterion sweep for DLinearMix2.
# No-HL01 cross-station prediction; seed=42; --early_stop_metric mse.
# 18 runs total, ~5 min/run, total ~90 min.

START_TS=$(date +%s)
LOG=/tmp/sweep_criterion_progress.log
echo "=== Sweep[criterion] start: $(date) ===" > "$LOG"

run_one() {
  sl=$1
  ll=$2
  fusion_flag=$3
  criterion=$4
  huber_beta=$5
  tag=$6
  extra=""
  if [ "$fusion_flag" = "true" ]; then
    extra="--flatten_fusion"
  fi
  echo "--- [$tag] sl=$sl ll=$ll fusion=$fusion_flag criterion=$criterion beta=$huber_beta start $(date +%H:%M:%S) ---" | tee -a "$LOG"
  python run.py \
    --model DLinearMix2 \
    --data custom \
    --data_path water_level_rain_gate_all.csv \
    --segment_col segment_id \
    --features S --target HL01 \
    --input_col 'HL02,HL03,HL04,HL05,HL06' \
    --exog_col 'isRain,Past10Min,Past1Hr,Now,*gate_opening*' \
    --seq_len "$sl" --pred_len 15 --label_len "$ll" \
    --batch_size 64 --train_epochs 80 --patience 15 \
    --learning_rate 1e-3 --dropout 0.1 \
    --early_stop_metric mse \
    --criterion "$criterion" --huber_beta "$huber_beta" \
    --lradj warmup_exp \
    --warmup_epochs 8 --lr_decay_gamma 0.92 --lr_floor_ratio 0.01 \
    --seed 42 \
    $extra \
    >> "/tmp/sweep_criterion_${tag}.log" 2>&1
  rc=$?
  echo "--- [$tag] end $(date +%H:%M:%S) rc=$rc ---" | tee -a "$LOG"
}

# 3 sl x 2 fusion x 3 criterion = 18 runs
# fusion=false (horz)
run_one 30 15 false mse   1.0  M_sl30_horz_mse
run_one 30 15 false huber 0.5  M_sl30_horz_hub05
run_one 30 15 false huber 1.0  M_sl30_horz_hub10
run_one 45 22 false mse   1.0  M_sl45_horz_mse
run_one 45 22 false huber 0.5  M_sl45_horz_hub05
run_one 45 22 false huber 1.0  M_sl45_horz_hub10
run_one 60 30 false mse   1.0  M_sl60_horz_mse
run_one 60 30 false huber 0.5  M_sl60_horz_hub05
run_one 60 30 false huber 1.0  M_sl60_horz_hub10
# fusion=true (flat)
run_one 30 15 true  mse   1.0  M_sl30_flat_mse
run_one 30 15 true  huber 0.5  M_sl30_flat_hub05
run_one 30 15 true  huber 1.0  M_sl30_flat_hub10
run_one 45 22 true  mse   1.0  M_sl45_flat_mse
run_one 45 22 true  huber 0.5  M_sl45_flat_hub05
run_one 45 22 true  huber 1.0  M_sl45_flat_hub10
run_one 60 30 true  mse   1.0  M_sl60_flat_mse
run_one 60 30 true  huber 0.5  M_sl60_flat_hub05
run_one 60 30 true  huber 1.0  M_sl60_flat_hub10

END_TS=$(date +%s)
echo "=== Sweep[criterion] done: $(date)  elapsed=$((END_TS - START_TS))s ===" | tee -a "$LOG"
touch /tmp/sweep_criterion_done
