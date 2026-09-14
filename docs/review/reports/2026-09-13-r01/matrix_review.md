# run_matrix.py／collect_matrix.py 專項審查 — 2026-09-14

本次沿用使用者指定的報告資料夾，僅審兩支矩陣腳本及其必要上下游接線；不是重做全專案review。程式與dataset內容為事實來源，PROGRESS.md與腳本說明為受審宣稱。本輪不修改程式、不訓練、不執行真正的run_matrix run、不載入模型或checkpoint、不計算既有test的新性能。

**後續更新（2026-09-14）：** 使用者另行授權修復validation取樣，Data_Factory已改val為shuffle=False／drop_last=False，8個無訓練測試通過。因此原C-04的隨機丟樣本缺口已修；下方兩支矩陣腳本的10項finding尚未修復。上述「不修改程式」及佐證JSON的hash描述原審查當時狀態；最新修正與六項問題的詳細解法見 [matrix_remediation.md](matrix_remediation.md)。

## 0. 摘要

**結論：基本矩陣與anchored計算方向正確，但目前不能視為可放心啟動24-run的版本。** 本機pandas 3.0.1會在首個run結束後寫duration_s時拋TypeError；續跑缺實驗設定識別，合表另有來源校驗、缺漏及非有限值處理問題。尚未落實開發／CV與最終test分期。

- **審查者／日期：** Codex，2026-09-14。
- **版本：** `3be09d28aae1357775acc8d0cdbe8a115cba6daf`；兩支腳本與上下游／資料hash存於佐證JSON。
- **接手狀態：** report.md與CHANGES.md已有上一輪未提交修改；本輪保留其內容。沒有commit／push。
- **證據：** 現行程式、selected-column CSV統計，以及mock子程序／合成arrays驗證。沒有新的模型性能結果。
- **方向判定：** 前進，但應先修執行中斷與資料／結果契約，再啟動批次。
- **增量：** 本補充10則專項finding（6 Major、4 Minor、0 Blocker）；MX-C03是原C-05在新合表器的具體延伸，不另宣稱發現全新根因。原報告24則原始統計不改。原C-03／C-08仍影響此矩陣；C-04審查時存在，已於後續授權修復，另外列出而不重複新增ID。
- **使用者已裁決：** 主呈現採best_corr＋anchored，且同時明列MSE；不要求本輪先解決雙指標唯一排名。跨old／drycut共用固定test維持暫緩。

## 1. Block D — 資料與矩陣輸入

### 本block查核宣稱

1. 矩陣恰有2 dataset × 2 exog × 2 loss × 3 fold，24個唯一run。
2. 每筆命令的資料與split成對，input不含target，exog不含isRain／min_since_rain。
3. Sidecar確實驗證資料與split的配對及必要參數。
4. 實際CSV與split的segment／fold涵蓋吻合。
5. 同一dataset的test在fold間固定；none／full是否也使用相同窗口需查實際NaN。

### 通過與限制

現有manifest為24列pending；命令有24個唯一key，所有CLI旗標均能在run.py parser找到。兩套資料分別對應自己的split；input固定HL02～HL06，none不傳exog_col，full為三種雨量＋gate欄，均不含isRain／min_since_rain。固定segment_col及file模式符合目前矩陣用途；此矩陣不是先前討論之通用Segment／Row介面的實作，不把它缺row選項當bug。

依現行run.py的常數欄規則排除north_gate_opening_4，再按loader的75列window與NaN規則只讀重算：

| Dataset | Fold | Train none／full | Val none／full | Test none／full |
|---|---|---|---|---|
| drycut | 1 | 16983／16721 | 5661／5656 | 6240／6240 |
| drycut | 2 | 22644／22377 | 6487／5417 | 6240／6240 |
| drycut | 3 | 29131／27794 | 5638／5635 | 6240／6240 |
| old | 1 | 16286／15710 | 5325／5325 | 5523／5523 |
| old | 2 | 21611／21035 | 6010／5238 | 5523／5523 |
| old | 3 | 27621／26273 | 5311／5311 | 5523／5523 |

同一dataset的none／full test不只數量相等，(segment_id, origin)清單hash也相同。fold test由base split取用，所以同dataset各fold亦固定。這次沒有重算跨dataset共同test清單；先前共用test方案暫緩。

Train／val的none與full樣本有實際差異，尤其fold 2；不能把exog對照解讀成完全只加減欄位、訓練樣本不變。它仍可比較「各自按可用欄位篩選資料的整套流程」在同dataset固定test上的結果。使用者已允許各run val不同；此處說明結論邊界，不要求現在強制共用val或跨dataset test。

### MX-D01 · Preflight沒有驗證宣稱的data／split配對

- **區塊／子類：** D，輸入一致性。
- **嚴重度：** Major。
- **位置：** `run_matrix.py:104-117`；`build_splits.py:256-271,345-359`。
- **現象：** preflight只確認檔案存在及6項sidecar參數；沒有比對sidecar.data_path、實際CSV／split內容hash、fold欄位或stride。將記憶體中的old split_file改指drycut split，實際preflight仍印「皆一致」。目前DATASETS硬編碼配對本身正確。
- **影響：** 配錯或同路徑重產資料後仍可能通過前置檢查；之後可能才被loader擋下，若ID重疊則可能誤套分派。不能把此preflight當成配對正確的證明；本輪沒有執行錯配訓練，未聲稱這組必然完成或已發生leakage。
- **建議：** 比對正規化來源路徑、保存與核對資料／split hash，查fold集合及segment涵蓋；none／full共用segment分派可保留，但分清split計數的欄位集與每run實際窗口數。
- **信心度／證據：** 高；JSON `preflight_actual`、`preflight_wrong_dataset_pair`、`data`。

## 2. Block C — 驅動、續跑及合表

### 本block查核宣稱

1. 計畫保存足以固定實驗設定，續跑不混入不同設定的done列。
2. 命令組裝可對接run.py，dry-run不執行子程序。
3. 成功狀態與完整可讀的run產物一致。
4. 合表校驗manifest標籤、run_args及checkpoint來源。
5. Anchored公式、corr與MSE可與上游指標對照。
6. 缺檔、非有限值、形狀錯誤、零baseline與缺fold不產生誤導性的完整結果。

### MX-C01 · 首個run結束後duration_s寫入會中斷矩陣

- **區塊／子類：** C2，執行bug。
- **嚴重度：** Major；實務上是啟動批次前必修項，不以論文結果無效的Blocker定義混稱。
- **位置：** `run_matrix.py:181,201-223`。
- **現象：** 本機pandas=3.0.1；實際manifest以keep_default_na=False載入後duration_s是str dtype。第222行寫入float拋 `TypeError: Invalid value '0.0' for dtype 'str'`。以實際manifest首列作fixture、子程序成功回傳的mock已重現。
- **影響：** 第一組不論成功或失敗，寫duration時都可能在同一行中斷；磁碟最後保存的是running，沒有保存此次done／failed與run_dir。重啟會再跑該組，浪費已完成訓練，也違反「失敗不中斷、逐組續跑」宣稱。不是dry-run能測出的問題。
- **建議：** 讀manifest後以明確schema將duration_s轉為數值／nullable浮點，空值不用空字串；完成狀態、耗時與路徑一起驗證後原子保存。加入mock成功／失敗／恢復狀態測試。
- **信心度／證據：** 高；JSON `environment`、`native_manifest_execution`，最後一次mock save仍為running。無真實子程序執行。

### MX-C02 · 續跑可混合不同設定或資料版本

- **區塊／子類：** C2，重現性／續跑。
- **嚴重度：** Major。
- **位置：** `run_matrix.py:134-149,158-171,180-184`。
- **現象：** run_key只有dataset／exog／loss／fold，manifest未凍結CONST、flags、實際command或資料hash。修改learning_rate由0.001為0.002後，同key舊done仍保留；pending則用當前CONST重建新命令。
- **影響：** 同一張矩陣可混入不同learning_rate、seed、模型設定或同路徑新資料，後續照樣按相同配置合表。重新plan也不會解決done復用。
- **建議：** plan保存完整設定、命令、程式／data／split fingerprint；run使用已保存設定，resume須完全吻合，否則要求新matrix_id或明確新計畫。不能只增加一兩項key而漏其餘參數。
- **信心度／證據：** 高；JSON `resume_changed_learning_rate`：done舊路徑保留，而新command learning_rate=0.002。

### MX-C03 · 合表corr與主流程／既有anchored工具的常數規則不一致

- **區塊／子類：** C2，原C-05延伸。
- **嚴重度：** Major。
- **位置：** `collect_matrix.py:43-65`；`utils/metrics.py:8-15`；`compute_anchored_mse.py:47-54`。
- **現象：** collector對常數horizon記NaN後nanmean；主test及既有anchored總corr將它算0。合成兩步預測，第一步為常數、第二步完美相關：collector=1.0，上游≈0.5。
- **影響：** 相同pred／true在矩陣表與run summary得到不同corr；部分horizon退化時，主表可能因跳過退化步而看起來更好。這不代表NaN規則本身錯，問題是無共同定義且不報有效步數。
- **建議：** 統一主流程與離線工具的明確corr契約；若採NaN，附valid_horizons／total_horizons及退化原因，不能只輸出nanmean。Anchored平移公式本身一致，不能由此推論指標也完全一致。
- **信心度／證據：** 高；JSON `constant_horizon_corr`、`anchored_formula`。

### MX-C04 · 合表未驗證run來源，checkpoint標籤寫死

- **區塊／子類：** C2，來源／標籤契約。
- **嚴重度：** Major。
- **位置：** `collect_matrix.py:40,101-119,138-152`；`exp/exp_Main2.py:406-417`。
- **現象：** loss／fold／dataset直接取manifest，不核對run_args；outputs固定標best_corr、outputs_alt固定標best_mse。實際上上游outputs是primary metric，當early_stop_metric=mse時兩者相反。合成args刻意改成huber／fold3／另一data_path／primary=mse，仍產6列，第一列標mse／fold1／best_corr。target guard也只用assert查input_col，沒有查exog_col。
- **影響：** 錯指run_dir、舊done或人工manifest的結果可被歸入錯誤實驗因子或checkpoint。目前CONST=corr時目錄對應正確，並非聲稱當前24條指令已反標。
- **建議：** 以保存的run metadata核對dataset、loss、fold、seed、欄位、split與config fingerprint；checkpoint標籤由實際primary／alt metadata決定或明確拒絕不符者。target查input與exog，使用顯式例外而非可被-O移除的assert。現行run.py／loader已有target防護，此處是合表驗收缺口。
- **信心度／證據：** 高；JSON `wrong_args_accepted`。

### MX-C05 · 缺漏、NaN與錯位輸入仍能產生看似正常的summary

- **區塊／子類：** C2，結果完整性。
- **嚴重度：** Major。
- **位置：** `collect_matrix.py:54-65,85-95,120-127,170-179,189-211`。
- **現象：** 缺一個outputs_alt時只警告並跳過，剩141列；sanity仍稱每組3 folds，因為沒有按checkpoint／anchoring查。整個配置缺失也不會進入groupby而被發現。NaN預測不被拒絕；summary mean跳過無效值，但n_folds按列數count，合成案例有一fold的corr／MSE=NaN，仍報n_folds=3且平均只用其餘2fold。
- **其他同類契約缺口：** pred有3個window、true只有1個window會經NumPy broadcasting產出MSE；segment對應只查去重後長度，window_idx=[100,101,102]配n=3也被接受，沒有核對0..n−1或一window對應唯一segment。
- **影響：** 部分失敗或錯位的結果可參與排名，表面折數與實際有效折數不符；錯配真值或segment時指標失去語意。現有matrix全pending，沒有聲稱已產生這種正式結果。
- **建議：** 依預期matrix與checkpoint建立完整網格，正式模式缺漏即失敗；探索部分結果需顯式partial模式。檢查shape完全一致、N>0、P符合pred_len、單target、有限值與window_id一一對應；每項指標另報有效fold數及缺失原因，再輸出summary。
- **信心度／證據：** 高；JSON `missing_one_checkpoint`、`missing_entire_config_sanity`、`nan_fold_summary`、`shape_mismatch`、`invalid_window_ids_accepted`。合成main確實執行，CSV寫入被攔截保存在記憶體。

### MX-C06 · Persistence MSE=0時改善百分比除零

- **區塊／子類：** C2，指標邊界。
- **嚴重度：** Minor。
- **位置：** `collect_matrix.py:157-167`。
- **現象：** baseline MSE=0時，raw MSE=1得到−inf，baseline／anchored MSE=0得到NaN。
- **影響：** 完美persistence的樣本集會輸出非有限改善率；其百分比在數學上未定義，不能納入一般mean／std或把0/0當0%改善。
- **建議：** 標為未定義及baseline_zero，保留絕對MSE／誤差差值；摘要排除百分比時需報有效數。
- **信心度／證據：** 高；JSON `zero_baseline`。未查到目前真實test恰為0，故不升級為當前必現故障。

### MX-C07 · 子程序exit 0就標done，未驗收run_dir及產物

- **區塊／子類：** C2，狀態恢復。
- **嚴重度：** Minor。
- **位置：** `run_matrix.py:174-176,211-223`；`collect_matrix.py:109-113`。
- **現象：** run_dir靠stdout中的單引號repr解析，找不到也回空字串；子程序exit0仍設done。隔離MX-C01的dtype障礙後，mock成功但無run_dir的輸出重現done＋空路徑。collector將空字串變Path('.')，exists通過後讀根目錄run_args.json而失敗。
- **影響：** done不保證可收集，重跑又跳過done；解析失敗或產物遺失需人工修復。當前正常Namespace輸出的regex可匹配，不聲稱必然解析失敗。
- **建議：** 用結構化完成記錄或預先指定run_id，不依賴人類log；標done前核對run_args及必要產物，並提供reconcile／重驗done能力。manifest寫入宜採atomic replace，避免中斷時半份CSV；本輪未做斷電或並行壓力試驗。
- **信心度／證據：** 高；JSON `success_without_artifacts_after_fixture_dtype_conversion`，fixture的dtype轉換僅在記憶體進行，未修改受審程式。

## 3. Block M — 使用者需求

### 本block查核宣稱

1. 開發／CV只用train／val，方法固定後才讀test。
2. 同時支持高corr與低MSE的比較，而非只突出單一數字。
3. fold統計忠實描述rolling訓練歷史與固定test，沒有宣稱獨立seed重複或不同test期間。
4. 能查看segment層級的弱點；與主流程已存在的segment corr分清楚。
5. 不把暫緩的跨dataset共用test視為本輪必須新增的功能。

### 需求對照

| 使用者方向 | 實際狀態 | 判斷 |
|---|---|---|
| 2 dataset × none/full × mse/huber × 3 folds | 24條命令，無漏項或重複 | 符合目前腳本／矩陣宣稱 |
| 不放isRain／min_since_rain；target不得餵模型input／exog | 命令符合；run.py及loader已有防護 | 符合；collector自身guard仍不足見MX-C04 |
| 外部分派且segment明確 | 每組皆傳segment_col、split_mode=file、split_file、fold | 符合 |
| 高corr且低MSE；best_corr＋anchored為主 | long／summary都有兩指標，但headline缺絕對MSE、wide只有corr | 部分符合，見MX-M01 |
| Anchored事後算、不重訓 | 平移公式正確，讀persist取最後觀測值 | 符合；指標契約差異見MX-C03 |
| 開發／CV只用train／val | driver直接呼叫現有run.py，仍每epoch test並在結尾test；collector只讀test npy，沒有val指標欄 | 未完成，原C-08仍有效 |
| Val每run內固定完整 | 後續已改val shuffle=False且drop_last=False，8個測試通過 | C-04隨機丟樣本缺口已修；空集合／NaN防護另待處理 |
| 能看低corr segment | 主流程另有metrics_segment.csv；新collector只算segment MSE median／p90，未彙整segment corr | 現有主流程可看，新合表缺此診斷；未把尚未要求實作的window-wise當新bug |
| 跨old／drycut共用test暫緩 | 分別6240／5523個test windows | 依最新裁決接受暫緩；不將分數差直接歸因於模型優劣 |

### 仍會影響這批矩陣的既有finding

- **C-08（Major）：** `exp/exp_Main2.py:240-244,288-290`建立並逐epoch評估test，`run.py:510-516`訓練後直接test。兩個checkpoint均由val選取，沒有發現程式直接用test loss反向傳播／選checkpoint；問題是test仍暴露於開發決策，且沒有凍結後獨立final階段。收集「已預先固定的多個方法」之最終test是允許的，本問題不因多方法共用test就自動成立。
- **C-04（原Major，後續已修）：** 審查時 `data_provider/Data_Factory.py:16-30`讓val走shuffle=True／drop_last=True，batch_size=64會丟棄不滿尾批。使用者後續要求先修此項，現已明確走shuffle=False／drop_last=False並驗證完整取樣；8個測試通過。不把歷史finding刪掉，也不聲稱舊checkpoint已重選。
- **C-03（Major）：** `run.py:122-183`選常數欄依builtin前70%段，沒有遵循各fold的train範圍。矩陣使用fold1～3仍會走此路徑；本次兩資料都刪north_gate_opening_4，不代表一般fold選欄防護已修。

Row scaler的C-02不在此矩陣的實際file＋segment路徑上；不把它誤報為這24條命令必然洩漏。

### MX-M01 · 已選定的headline沒有明列MSE

- **區塊／子類：** M，呈現需求。
- **嚴重度：** Minor。
- **位置：** `collect_matrix.py:196-219`。
- **現象：** summary_by_config.csv已有mse_mean／mse_std；console headline只顯corr與相對baseline改善率，results_wide.csv則僅有anchored／best_corr的corr。
- **影響：** 不足以直接從主表同時判斷使用者要求的高corr與低MSE。改善百分比不等於絕對MSE，尤其跨dataset baseline本身不同。
- **建議：** 保留best_corr＋anchored主視角，headline與wide明列corr_mean／std、mse_mean／std、baseline MSE、改善率與有效fold數；best_mse及raw保持附表。無須自行新增複合權重或選唯一贏家。
- **信心度／證據：** 高；使用者本輪Q-MX1回覆；JSON `synthetic_main`。主表選best_corr本身已獲接受，不列為bug。

## 4. Block R — 宣稱核對

### 本block查核宣稱

1. 「sidecar驗配對」有實作支持。
2. 「合表與既有anchored定義不漂移」涵蓋公式與指標規則。
3. 「各fold test難度不同」符合loader的實際分派。
4. 「144列」及一致性檢查可識別缺少checkpoint／整組配置。
5. PROGRESS所稱「內建自我檢驗」能找到對應assert或檢查分支。

### MX-R01 · Fold統計說明與實際固定test不符

- **區塊／子類：** R，統計解讀。
- **嚴重度：** Minor。
- **位置：** `collect_matrix.py:16-21,157-162`；`data_provider/Data_Loader.py:64-93`。
- **現象：** add_improve註解宣稱「各fold test難度不同」，但loader在同dataset固定base test，本次窗口清單也一致。fold的train逐漸擴張且歷史重疊、val期間不同；固定seed42，並非3個獨立seed或3份獨立test。
- **影響：** mean±std可作描述，但它反映不同訓練歷史／checkpoint選擇流程在同test上的變動，不能把std解讀成獨立重複實驗的穩定性證明，也不能說改善百分比就自動讓不同dataset test可比。對同一baseline，改善率只是MSE的線性重表示。
- **建議：** 更正fold與test說明，保留各fold結果及有效數，不從3fold std推導獨立樣本信賴區間；跨dataset維持各自評估母體的解讀限制。
- **信心度／證據：** 高；JSON `data`、loader分派程式。這是解讀／文件bug，不是groupby mean或sample std算術錯誤。

### 其他宣稱的澄清

- Anchored平移公式相同已驗證；「與舊工具不漂移」不能延伸到corr的NaN規則，見MX-C03。
- 完整合成24-run在collector主入口產144列long、48列summary、8列wide，符合組合數宣稱；但這是資料完整的條件式結果，缺失保護見MX-C05。
- PROGRESS.md:107稱「內建自我檢驗」，受審collector沒有檢查anchored／persistence h1相等的assert或分支。這個數學不變量是真的，本輪合成測試也驗過，但應區分「過去人工驗證」與「程式每次自動驗證」。
- `per_segment_mse`的p90是segment MSE第90百分位，並非最差10%事件的平均；函式說明用worst-decile可能誤導。主summary僅保留segment median的fold平均，p90只在long表。
- long中的epochs取run_args.train_epochs，代表最大訓練epoch數，不是early stopping後實跑數；若用於成本解讀，應改名max_epochs或補actual_epochs。duration_s是子程序總耗時，包含test／輸出，不是純訓練耗時。

## 5. 給使用者的問題

Q-MX1已在審查途中提出並獲答：使用者選擇「以best_corr＋anchored為主，同時明列MSE」。據此MX-M01只要求補主表MSE，不否定主視角、不擅自改checkpoint策略。本專項無其他未答問題；原報告未答問題仍保留。

## 6. 證據與限制

可重現程式：[matrix_checks_2026-09-14.py](evidence/matrix_checks_2026-09-14.py)；輸出：[matrix_checks_2026-09-14.json](evidence/matrix_checks_2026-09-14.json)。從repo根目錄執行：

```sh
.HLML_Linear_venv/bin/python -B docs/review/reports/2026-09-13-r01/evidence/matrix_checks_2026-09-14.py
```

腳本只向stdout輸出。實際CSV只讀需要欄位；hash採分塊讀取。兩支矩陣module載入不執行main；唯一execute(False)呼叫完全mock subprocess.run、log與manifest寫入，不會啟動run.py。collector的arrays、run_args、segment IDs與CSV輸出均在記憶體mock，未讀取真實模型結果。AST核對run.py參數但沒有執行其入口。實際dry-run亦以禁止任何子程序的mock保護，確認0次子程序、24條命令。

首次佐證執行被新發現MX-C01中斷；之後增加捕捉該例外、記錄實際manifest dtype的probe。為繼續查其他邏輯，僅第二個mock fixture把duration_s暫轉object，受審程式與manifest完全未改。一次追加證據腳本的patch因上下文不符未套用，隨即更正；最終重跑全部probe成功。合成輸出包含刻意NaN／inf，以字串序列化，不能誤當真實性能。

未驗證GPU／MPS、實際訓練成功率、24-run性能、並行執行／斷電恢復；不因mock通過就聲稱正式實驗已跑通。hash核對涵蓋兩支腳本、必要上下游、現有manifest、兩份train CSV、兩份split CSV及sidecar，全數在probe前後不變。
