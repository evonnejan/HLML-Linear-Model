#!/usr/bin/env bash
# 3 seq_len x 2 fusion sweep WITHOUT HL01 in input (cross-station prediction).
# Fixed seed=42. ~7 min/run, total ~42 min.

START_TS=$(date +%s)
LOG=/tmp/sweep_noHL01_progress.log
echo "=== Sweep[noHL01] start: $(date) ===" > "$LOG"

run_one() {
  sl=$1
  ll=$2
  fusion_flag=$3
  tag=$4
  extra=""
  if [ "$fusion_flag" = "true" ]; then
    extra="--flatten_fusion"
  fi
  echo "--- [$tag] seq_len=$sl label_len=$ll fusion=$fusion_flag start $(date +%H:%M:%S) ---" | tee -a "$LOG"
  python run.py \
    --model DLinearMix2 \
    --data custom \
    --data_path water_level_rain_gate_all.csv \
    --segment_col segment_id --split_mode builtin \
    --features S --target HL01 \
    --input_col 'HL02,HL03,HL04,HL05,HL06' \
    --exog_col 'isRain,Past10Min,Past1Hr,Now,*gate_opening*' \
    --seq_len "$sl" --pred_len 15 --label_len "$ll" \
    --batch_size 64 --train_epochs 80 --patience 15 \
    --learning_rate 1e-3 --dropout 0.1 \
    --early_stop_metric corr \
    --lradj warmup_exp \
    --warmup_epochs 8 --lr_decay_gamma 0.92 --lr_floor_ratio 0.01 \
    --seed 42 \
    $extra \
    >> "/tmp/sweep_noHL01_${tag}.log" 2>&1
  rc=$?
  echo "--- [$tag] end $(date +%H:%M:%S) rc=$rc ---" | tee -a "$LOG"
}

run_one 30 15 false N1_sl30_fusionH
run_one 45 22 false N2_sl45_fusionH
run_one 60 30 false N3_sl60_fusionH
run_one 30 15 true  N4_sl30_fusionF
run_one 45 22 true  N5_sl45_fusionF
run_one 60 30 true  N6_sl60_fusionF

END_TS=$(date +%s)
echo "=== Sweep[noHL01] done: $(date)  elapsed=$((END_TS - START_TS))s ===" | tee -a "$LOG"
touch /tmp/sweep_noHL01_done
