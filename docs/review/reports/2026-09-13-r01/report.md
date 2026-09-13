# Research Review Report — 2026-09-13 r01

本次新增檔案與後續修訂詳見 [變更紀錄](CHANGES.md)。

**最新修正狀態（2026-09-14，第二輪討論）：** 已依使用者明確要求修復C-01的target輸入防護，24個不含訓練的回歸測試通過；其他修正尚未套用。暫不使用isRain／MSR的決策已取代前輪旗標提案。原始24則finding與分級是09-13審查快照；第5d節記錄修正後狀態、剩餘問題與最新設計。程式已改動，原始file:line與「核心hash未變」核對僅適用於各自當時版本。

**2026-09-14 討論更新：** 使用者已確認validation完整固定、target輸入防護及開發／最終test隔離方向，並提出旗標配置與近期研究順序。具體決策、尚未裁決細節與修正驗收見第5c節；不代表程式已修復。原始findings及09-13佐證保留。

- **審查者：** Codex，外部審查
- **審查時間：** 2026-09-13T20:03:07+08:00（首次完成核對見 `evidence/verification.txt`；後續完整性複核見 `evidence/completeness.txt`，時間與變更歷程見 `CHANGES.md`）
- **repo commit：** fd29037
- **context 時效：** 部分過期。文件自稱 09-10 查證，09-13 有局部修正；其後 f0affcb 改動 build_splits 錯誤訊息。以現行程式為準。
- **證據狀態：** 有歷史與煙霧結果；無符合正式現行原則的完整比較結果。本次新增的性能數字僅為資料可直接計算的 persistence，未訓練任何模型。

## 0. 摘要

### 完成狀態與適用範圍

**D → C → M → R 四個 block 已完成本次規範範圍內的審查；尚未完成的是作者裁決與問題修復，以及明列受限的驗證。** 這份報告不是「全部通過」或整條系統已證明安全的認證。2026-09-13 的補充複核將完成狀態、覆蓋與限制集中列明，24則finding的內容與定級未改。

| 工作 | 狀態 | 交付／限制 |
|---|---|---|
| 資料 D、程式 C、方法 M、文件 R | 已完成規定範圍的閱讀、抽查與判定 | 第1–4節；詳細覆蓋見附錄 |
| T-01～T-33 | 33條均已給判定與依據 | 含部分相符、無法判定及宣稱有問題，不等於33條全部通過 |
| Findings | 原始24則；C-01程式防護已修復，其餘23則尚未結案 | 原始分級1 Blocker／17 Major／6 Minor；尚未結案1 Blocker／16 Major／6 Minor。C-01歷史實驗分類仍待Q-C1 |
| 使用者裁決 | 原始11題；本階段Q-M1／M3已以排除兩旗標裁決，其餘9題仍有待答內容 | 第5節保留原問題；第5d節為最新回覆，先前第5c節保留歷史討論 |
| SQL、正式性能、硬體與其他受限驗證 | 未驗證，原因已列 | 附錄集中列出補齊條件；禁止訓練與範圍限制仍有效 |
| 報告／變更紀錄完整性 | 已補充核對 | `evidence/completeness.txt`；保留首次佐證，不覆寫歷史輸出 |

閱讀順序建議：先看本節與唯一Blocker C-02，再看第5節待裁決事項，最後按附錄覆蓋表查證。**「完整 review」指指定範圍已逐項處理，不代表所有可能輸入、實際模型效益或範圍外依賴都已驗證。**

| 嚴重度 | 數量 |
|---|---|
| Blocker | 1 |
| Major | 17 |
| Minor | 6 |
| Nit | 0 |

24則finding，不含11個待裁決問題及假設。**唯一已定Blocker是C-02的row模式scaler leakage；已驗過的segment+split-file路徑不受此項影響。** M-02若被當成可部署方法，其非因果輸入將構成另一個Blocker；目前依「診斷對照」的未確認假設列Major。方法設計的Major表示論文結論前提未完備，不等於已證明正式實驗無效。

### 與前一份報告的 delta

- **前一份：** 本次為第一份；開始時 reports/ 只有 TEMPLATE.md。
- **新增：** D-01～D-02、C-01～C-09、M-01～M-05、R-01～R-08。
- **已解決：** 無前次report ID可比較。context的OI-03撤回、OI-06修正已獨立驗證，不冒充本次修復。
- **仍存在：** 無前次report可計delta；OI-01、OI-04、OI-05分別對應C-01、M-01、C-03。

### 方向判定

**前進。** 可重現的資料／split與正確的file模式scaler是實質進展；但應先修評估契約、釐清可因果特徵與共同評估母體，再把36-run探索結果升格為論文證據。優先處理C-02（修或阻擋row路徑）、C-03～05、C-08與M-02～04；roadmap的cross-attention應等簡單ablation確認必要性。

### 一句話結論

資料切分已可重現，但「沒有時間重疊」仍不足以保證可部署、可比較、可投稿的研究結論。

### 給使用者的問題（原始摘要；最新回覆見第5c節）

完整功課與影響見第5節；以下每題已在所屬block完成時提出。09-14已收到部分相關決策，尚不能將任何一題所有細節視為已結案；最新狀態以第5c節為準。

| 問題 | 待裁決事項 |
|---|---|
| Q-D1 | 正式drycut維持Past10Min，或原意是Past1Hr？ |
| Q-D2 | 5分鐘限制應套用最終模型輸入，或允許段內延用？ |
| Q-D3 | 是否有設備依據證明所有負開度都代表全關？ |
| Q-C1 | 早期base包含HL01是刻意對照或疏漏？ |
| Q-C2 | 是否曾用test數字影響選模、調參或roadmap？ |
| Q-M1 | isRain是否只作不可部署的診斷對照？ |
| Q-M2 | 主要比較同時刻預測準確度，或整套選樣策略及coverage？ |
| Q-M3 | exog「無」是無衍生欄，或移除全部雨量／閘門？ |
| Q-M4 | 當期論文先驗證水位預測，或必須驗證預警效用？ |
| Q-R1 | 水位單位公分與HL02–06全在上游有何現場依據／例外？ |
| Q-R2 | median／p90正式要用MSE還是RMSE？ |

## 1. Block D — 資料

### 本 block 我打算驗證的宣稱

1. dataset 檔案總表、CSV 欄數／列數與原始寬表分鐘連續性。
2. 閘門重建前後 NaN 率與共有值一致性，逐欄合併兩條路徑一致。
3. drycut 179 段、old 159 段、段長及 isRain 分布與舊核心列數一致。
4. drycut 實際雨量欄位、0.5 mm 量化、L 與 buffer 條件及 meta 重現。
5. min_since_rain 全域因果計算、NaN／最大值來源與影響 windows。
6. 段內 gate ffill 不跨段、殘餘 NaN 排除及實際觀測新鮮度。
7. split／fold 時間重疊、資訊是否可於推論當下取得。
8. 現行讀寫路徑與譜系、歷史檔誤用、資料交付是否完整。

### Findings

#### D-01 · drycut 的實際雨量欄與研究宣稱相反
- **區塊：** D
- **子類：** 欄位語意／譜系
- **嚴重度：** Major
- **位置：** `build_drycut_segments_meta.py:32,80,101`；`docs/review/context/01-data.md:94`（§4.2、§5.2）
- **現象：** 實際預設 `Past10Min`，以此重算 179 段起訖全部相符；`Past1Hr` 重算只有 164 段且不符。文件卻宣稱現行採 Past1Hr，並用它解釋資料膨脹。
- **為什麼是問題：** 方法定義、事件數、buffer 理由及重現條件會被錯誤描述；實驗結果不得按 Past1Hr 方法解釋。
- **建議：** 明訂 intended rain_col；若維持現況，修正文件並將 rain_col 納入產物 manifest／檔名。若改用 Past1Hr，須視為新資料版本。
- **信心度：** 高（意圖待 Q-D1 裁決）
- **證據：** `evidence/data.txt` 的 drycut_reconstruction；命令 `.HLML_Linear_venv/bin/python -B docs/review/reports/2026-09-13-r01/evidence/audit_data.py`。

#### D-02 · 段內無上限 ffill 使五分鐘新鮮度限制失效
- **區塊：** D
- **子類：** 缺值／欄位語意
- **嚴重度：** Major（暫定；待 Q-D2）
- **位置：** `build_training_csv_from_meta.py:99-106`；`Data_From_SQL_all.py:143-148`
- **現象：** 寬表先淘汰逾 5 分鐘觀測，下游又無上限 ffill。drycut 有 4,463 個填補儲存格距段內上次有效「分鐘值」超過 5 分鐘，最大至少 574 分鐘；old 為 1,132 格、105 分鐘。補驗有效windows：確實有1,521／1,096個window的**輸入部分**含age>5分鐘值，輸入中最大age下界559／90分鐘，不只是被NaN filter丟掉的列。
- **為什麼是問題：** 模型收到的值不再符合 T-06「過舊的觀測不應視為當前狀態」；長時間未回報被當成狀態不變，且缺少 freshness 指示。
- **建議：** 明訂五分鐘是量測有效上限或只適用合併；保留原始觀測時刻／age、缺測標記，依確認後的物理語意限制 carry-forward。
- **信心度：** 高（程式及數值）；嚴重度背景待確認。
- **證據：** `evidence/data.txt` training_stats；`evidence/supplement.txt` stale_values_reach_valid_model_inputs；全檔逐段 ffill 重算與 train 值 mismatch=0。`meeting_recap.txt:26-27`的確同時寫5min合併與段內ffill，支持兩階段可能是刻意設計；最終freshness意圖仍須裁決。

### 通過的查核與限制

`evidence/data.txt` 驗得寬表 511,679 列、23 欄、無缺分鐘；drycut 54,590×30／179 段，old 50,489×30／159 段；段長 130/200/2040 及 181/261/1521；isRain 分別 33,110/21,480 與 31,409/19,080。isRain 的核心判斷全數吻合。閘門重建共有值 949,041 格 `isclose` 全相同，任一 NaN 79.83736%→7.05501%。

MSR 全域重算全相符；60／70 NaN 全在第一段，換成 isRain 會多 60／70 windows。24,240 分鐘來自 2025-06-05 17:09 的段首 buffer，上次 Past10Min>0 是 05-19 21:09；不是重置或計數 bug。資料是否漏報需 SQL／外部觀測確認。

「179 段可用」只在按長度估算成立；加入現行 NaN filter 後 drycut 有 8 段、old 有 6 段零 windows。訓練任一 gate NaN 實為 3.49148%／3.41857%，不是文件的 3.48%／3.41%；數字錯誤統一列 Block R。

drycut 無重疊、最小間隔 61 分鐘；old 24 群涵蓋 56 段，重複分鐘 1,642，各 split 與三個 fold 跨 partition 群數均 0。段內時間網格全連續。現行輸入明列水位／exog 時不包含時間邊界欄；isRain 本身仍是離線 segment 標記，推論可取得性留待 Block M。

原始 gate 負值不是零星現象：south_gate_opening_3 有 227,756 個負值，south_gate_opening_1 最小 -7.1336。不能由數值自行證明它們代表全關，列 Q-D3，尚不直接判定標籤錯誤。

### 未驗證項目

| 項目 | 原因 |
|---|---|
| SQL 原始取數重現 | 無 SQL 連線；本次只查核本機資料及現行程式 |

## 2. Block C — 程式碼

### 本 block 我打算查證的項目

1. T-01～T-33 逐條比對，涵蓋 12 支 MUST-REVIEW 核心。
2. CLI 欄位展開、target 防護、常數欄選擇與 split/fold 一致性。
3. scaler 是否只 fit 真正 train；NaN filter、stride、seq/label 邊界。
4. split／fold 產物重現、重疊防護 raise 與空／極短資料反例。
5. Data Factory 的 train/val/test 參數、評估樣本完整性。
6. 主模型兩種 fusion、無 exog、通道順序及形狀契約。
7. vali/test 的尺度、corr、baseline、segment 彙總與 checkpoint 選用。
8. 種子／run manifest／summary 能否識別現行 pipeline 實驗；SQL 依賴與裝置選擇。

### C1 意圖一致性：T-01 ~ T-33 判定表

| ID | 判定 | 依據（file:line） | 備註 |
|---|---|---|---|
| T-01 | **不符** | `run.py:45-66,186-238`；`Data_Loader.py:247-250` | HL* 接受 HL01；C-01。歷史 base 意圖待 Q-C1 |
| T-02 | **宣稱有問題** | `exp_Main2.py:212-219`；`run.py:359-363` | 提供 pooled corr，但不是形狀相似的保證；預設早停仍 MSE，示例顯式 corr；M-01 |
| T-03 | 無法判定（效果） | `exp_Main2.py:547-548`；`DLinearMix2.py:295-347` | 當下 target 可合法取錨；delta training 未實作，不能判新穎性或效果 |
| T-04 | 相符（密碼部分） | `Data_From_SQL_all.py:18-38`；`Data_From_SQL_4.py:38-39` | AST 脫敏查核 DB_USER/PASSWORD 來自環境；SQL 依賴在範圍外，未做全 git 歷史秘密掃描 |
| T-05 | 相符 | `Data_From_SQL_all.py:135-148`；`rebuild_gate_columns.py:39-53` | 部分回報／負值／5min 邊界合成資料結果一致；來源 SQL 完整性未驗 |
| T-06 | **不符／意圖待裁決** | `Data_From_SQL_all.py:143-148`；`build_training_csv_from_meta.py:105` | 合併有效、最終 freshness 被無限 ffill 改變；D-02 |
| T-07 | 相符（程式）；無法判定（物理） | `rebuild_gate_columns.py:52`；`build_training_csv_from_meta.py:106` | 三處 clip 一致；負值語意待 Q-D3 |
| T-08 | 相符（受審核心） | `Data_From_SQL_all.py:104-107`；`build_training_csv_from_meta.py:220-221` | 唯一受審組裝路徑有段內 ffill；其他輔助工具不在 C 範圍 |
| T-09 | 相符 | `build_drycut_segments_meta.py:39-55` | 保留未移除 runs；現存 179 段可重現，rain_col 描述錯見 D-01 |
| T-10 | 相符 | `build_drycut_segments_meta.py:40-43` | eq(0) 對 NaN=False；現有雨量無 NaN |
| T-11 | 相符（現行參數） | `build_drycut_segments_meta.py:89-99,107-108`；`build_training_csv_from_meta.py:118-125` | 第一層限制參數、第二層驗 meta，不是等價檢查；allow-overlap 只豁免第二層；現有間隔 61min |
| T-12 | **宣稱有問題** | `build_training_csv_from_meta.py:135-144`；`Data_Loader.py:436-489` | 179 是按長度的可用數，實際 171 段有 windows；混合動態應分層評估，未證明有害 |
| T-13 | 相符 | `Data_Loader.py:456`；`build_splits.py:272` | need=96+15；111 列恰產 1 window |
| T-14 | 相符 | `build_training_csv_from_meta.py:214-221` | 先算後切、實際全列重算相同；16.8 天來源已解釋 |
| T-15 | 部分相符 | `build_splits.py:248`；`run.py:268` | split 預設 MSR，CLI exog 預設 None；新矩陣未寫。已棄用 sweep 不作現行 bug 證據 |
| T-16 | 相符（保留欄位）；宣稱有問題（可部署性） | `build_training_csv_from_meta.py:91,215` | isRain/MSR 均在 CSV；isRain 用未來定義事件邊界，M-02 |
| T-17 | 相符（CLI file 模式） | `run.py:69-98`；`Data_Loader.py:155-166` | CLI 必須明講；直接實例化仍可 builtin 並警告，不能聲稱所有路徑禁止回退 |
| T-18 | 相符（同欄位／排序／stride） | `build_splits.py:58-77`；`Data_Loader.py:179-185,457-489` | 兩套 CSV 各四種 fold × 三 split 共 24 次 loader 均一致；builder 不排序段內 date，亂序自訂檔不保證一致 |
| T-19 | 相符（現存 split） | `build_splits.py:171-175`；`Data_Loader.py:69-93` | 同段單一分派；split_file 未驗 duplicate ID／非法標籤／時間倒序，任意外部 CSV 需另驗 |
| T-20 | 相符（現存 meta） | `build_splits.py:95-106,140-150` | 24 群零跨 partition；全 blocked 與單點 blocked 都 raise；兩份 split 位元組重現 |
| T-21 | 相符（固定性）；**數字不符** | `Data_Loader.py:83-86` | drycut test 各 fold=5,700，不是 5,592；old=4,839；R-01 |
| T-22 | 相符 | `build_splits.py:209-214` | train prefix 逐折包含前折 val；資料分派驗得無倒序 |
| T-23 | 相符（3 折）；無法判定（充分性） | `build_splits.py:253`；現存 `splits_*.csv` | drycut val 段數38/32/28，old20/28/23；事件不等於統計獨立，M-03 |
| T-24 | 相符（CLI） | `run.py:75-98,381` | 所有 CLI main 經驗證；Dataset API 本來沒有 split_mode，需限縮宣稱 |
| T-25 | 相符（未實作）；宣稱有問題（歸因） | `Data_Loader.py:307-318` | train scaler 並不憑空造成 raw 單位 bias；distribution shift 未解決不等於該用隨機 split |
| T-26 | 相符（現行窗口） | `Data_Loader.py:441-468` | x/exog/target 全覆蓋且 fillna 前算 mask；多檢查未來 exog 會改變樣本母體，M-03 |
| T-27 | 相符 | `Data_Loader.py:179-185,474-489` | loader 先按段／date 排序，並非只信檔案原始順序；實際段內連續 |
| T-28 | **部分不符** | `Data_Loader.py:207-221,281-315`；`run.py:135-151` | segment scaler 正確；row 模式全檔 fit=C-02；constant selector=C-03 |
| T-29 | **不符** | `run.py:120-183,410,450` | 未使用 fold train；drop 訊息在 Tee 建立前，run_args 只留刪後欄位；C-03 |
| T-30 | 相符 | `exp_Main2.py:445-450,547-548,617-694` | persistence 取 label_len-1 的 target，同一 scaler inverse；不是 HL02 錨 |
| T-31 | 相符（mse/corr 模式） | `exp_Main2.py:255-267,406-428` | 兩個 checkpoint 都評估；MAE 模式為 MAE+Corr，不是 MSE+Corr；主早停限制兩者訓練長度 |
| T-32 | 相符（實作）；無法判定（效益） | `exp_Main2.py:132-139,198-200`；`run.py:350-351` | beta=1 在標準化單位；需以各 fold 的 target std 換算物理門檻，不能推定改善洪峰 |
| T-33 | 相符（正常 CLI 路徑） | `run.py:201-238`；`Data_Loader.py:247-250`；`DLinearMix2.py:312-321` | input→exog 順序正確；兩 fusion／有無 exog forward 皆 [2,15,1]；任意外部 tensor 欄序不受保護 |

### Findings

#### C-01 · HL* 展開仍可違反禁止 target 歷史的核心原則
- **區塊：** C
- **子類：** C1 意圖一致性
- **嚴重度：** Major
- **位置：** `run.py:45-66,186-238`；`data_provider/Data_Loader.py:247-250`
- **現象：** AST 提取純函式後，實際 CSV header 的 HL* 展成 HL01–06 並通過 configure；程式只禁止 input/exog 相互重疊。PROGRESS §5 仍引導此配置。
- **為什麼是問題：** 會讓聲稱「不使用 HL01 歷史」的實驗不符合設計；歷史／煙霧 metadata 確有使用此設定，但不能直接視為未來資料 leakage。現行 code-map 示範明列 HL02–06，未證明正式現行結果已受污染，故不泛稱 Blocker。
- **建議：** 在正式模型入口拒絕 target 出現在 input 或 exog，輸出展開後欄位；依 context 已記錄的決定，不增加放行參數，sanity 可獨立入口。歷史 base run 意圖待 Q-C1。
- **信心度：** 高
- **證據：** `evidence/code.txt` target_guard；`evidence/run_inventory.json`。

#### C-02 · row split 的 scaler 使用 val/test 資料
- **區塊：** C
- **子類：** C2 bug／leakage
- **嚴重度：** Blocker（限定未指定 segment_col 的路徑）
- **位置：** `data_provider/Data_Loader.py:207-221,281-318,328-340,372-385`
- **現象：** row 分支設 df_train=df_raw，後面 fit 前沒有套 num_train。20 列測例前14列全0、後6列全100；真正 train mean=0，實際 fitted mean=30、n_samples_seen=20。
- **為什麼是問題：** 直接用 hold-out 決定 train 的標準化參數；`run.py:265,269` 預設無 segment，這仍是可達路徑。
- **建議：** row 模式 fit 限制於真正 train 前綴，或正式研究入口拒絕 row 模式；不要因此否定已驗通過的 file split 分支。
- **信心度：** 高
- **證據：** `evidence/code.txt` row_scaler_leak；`evidence/row_scaler_fixture.csv`。

#### C-03 · 常數欄選擇看到了 fold 以後的資料，且未完整留下刪欄紀錄
- **區塊：** C
- **子類：** C1 意圖一致性／C2 bug
- **嚴重度：** Major
- **位置：** `run.py:135-183,385-388,410,450,478-479`
- **現象：** 用前70%段決定常數，與 split_file/fold 無關。drycut fold1/2 分別多看59/21段；old多看42/22段。現存候選欄的刪除集合恰好一致（north_gate_opening_4），故尚無實際結果改變證據。stdout drop 發生在 train.log Tee 建立前，json 只存刪後設定。
- **為什麼是問題：** 特徵選擇的因果邊界不成立；未來 fold／特徵變更可造成資訊使用或通道錯刪，原始輸入契約也難追溯。
- **建議：** 共用正式 split 選出的 train IDs；保存 original/expanded/dropped/retained 欄位、理由與 train 範圍。
- **信心度：** 高
- **證據：** `evidence/code.txt` constant_selector；`evidence/cli_effects.txt` 確認本次刪欄不改 window 計數。

#### C-04 · validation 隨機丟樣本，早停每輪比較的資料不固定
- **區塊：** C
- **子類：** C2 bug
- **嚴重度：** Major
- **位置：** `data_provider/Data_Factory.py:26-28,62-67`；`exp/exp_Main2.py:194-219,289-297`
- **現象：** val 落入與 train 相同的 shuffle=True、drop_last=True。batch64 時 drycut fold1 4,294只評4,288；old fold1 4,569只評4,544。兩次 sampler 的被丟集合不同；小於 batch 的 val 會全空。
- **為什麼是問題：** validation 不是完整固定評估集；靠近早停門檻時 checkpoint 選擇受隨機漏樣影響。
- **建議：** val 使用 shuffle=False、drop_last=False，檢查零批次／非有限分數；test 已採正確設定。
- **信心度：** 高
- **證據：** `evidence/code.txt` val_sampling。

#### C-05 · corr 的零變異規則在 validation 與 test 不一致
- **區塊：** C
- **子類：** C2 bug／指標契約
- **嚴重度：** Major
- **位置：** `exp/exp_Main2.py:215-219,455-457,648-652`；`utils/metrics.py:8-15`
- **現象：** vali 與 persistence per-horizon 跳過零變異；全域 test CORR 加 epsilon 後回0並納入均值。同一個兩 horizon 測例（第一個完美、第二個常數）得到 vali corr=1、test mean=0.5。
- **為什麼是問題：** 同名主指標在選模與報告時意義不同；乾段、常數預測尤其容易觸發。
- **建議：** 統一零變異／少樣本規則，報有效 horizon／事件數；明確決定常數預測的懲罰與 undefined 指標處理。
- **信心度：** 高
- **證據：** `evidence/code.txt` corr_zero_variance（實際 vali 函式 AST 提取執行，無訓練）。

#### C-06 · 「每 partition 足夠 window」並未被強制
- **區塊：** C
- **子類：** C2 bug／邊界條件
- **嚴重度：** Minor（現存 split 未觸發）
- **位置：** `build_splits.py:163-169,188-206`
- **現象：** 只限制至少一段，不限制 n_windows>0。三段 windows=[10,0,0] 仍返回 train/val/test，後兩者零樣本。
- **為什麼是問題：** 違反檔頭「Sufficient eval data」；換 seq_len／欄位後可能產生不能評估的 split，晚到訓練／評估才失敗。
- **建議：** 同時驗參數正數、段數、每 partition/fold 有效 window 下限；不成立直接拒絕。
- **信心度：** 高
- **證據：** `evidence/code.txt` empty_eval_partition、boundary probes。

補驗另發現greedy邊界會誤稱全域無解：windows=[1,1,7,1,1,1]、最後三段同一重疊群時，先選b1=3使b2候選全blocked而raise；其實b1=2、b2=3可得到安全的[2,7,3] windows。防洩漏的raise應保留，但訊息只能稱「目前候選區間無解」；若要求優先安全/充分樣本、比例其次，應回看前一邊界。此為可行性缺口，不是現行split有leakage。證據：`evidence/supplement.txt` greedy_split_false_infeasible。

#### C-07 · 保留的 predict 入口不遵循 DLinearMix2 的訓練資料契約
- **區塊：** C
- **子類：** C2 bug
- **嚴重度：** Major（推論接線缺口；不影響現行 test 路徑）
- **位置：** `data_provider/Data_Factory.py:20-25,46-55`；`data_provider/Data_Loader.py:576-591`；`exp/exp_Main2.py:1047-1064`
- **現象：** pred 改用 Dataset_Pred，不傳 input/exog 或既有 scaler；features=S 只產 HL01 一欄。實測 x=[1,96,1]，重新 fit 整份資料54,588個非NaN target；與五個鄰站＋exog模型通道及尺度不符。另 inverse_transform 接3D preds 亦不符 sklearn 2D 介面。
- **為什麼是問題：** 若把此入口當即時預警推論會報通道錯或使用不同尺度；目前 CLI main 只 train→test，故不聲稱既有 test 結果因此無效。
- **建議：** 提供共用欄位與 train scaler 的 inference dataset；明確禁止未支援的模型進入 Dataset_Pred。
- **信心度：** 高
- **證據：** `evidence/cli_effects.txt` pred factory。

#### C-08 · final hold-out 每個 epoch 與每個 CV run 都被評估並顯示
- **區塊：** C
- **子類：** C1 意圖一致性／評估流程
- **嚴重度：** Major（未證明人為按 test 選模）
- **位置：** `exp/exp_Main2.py:241-243,289-297,381-385`；`run.py:498-504`
- **現象：** 每輪 test 指標寫進 console/log；每個 run 結束又 test 兩個 checkpoint。實際 early-stopping 傳入的是 vali_score，沒有自動以 test 更新權重的證據。
- **為什麼是問題：** 宣稱 CV 是 model selection、test 是 final hold-out，但流程讓 test 持續暴露，增加人工選模與反覆調參污染；單純固定 test IDs 不足以保證盲測。
- **建議：** CV 階段只評 val；凍結方法與 checkpoint policy 後才一次解封 test；若已用 test 影響設計，應如實標示並新增時間外 hold-out。
- **信心度：** 高（暴露路徑）；歷史決策影響待 Q-C2。
- **證據：** 上述 file:line；`evidence/run_inventory.json` 有09月煙霧輸出。

#### C-09 · 實驗產物缺少足以鎖定資料／split／程式版本的 manifest
- **區塊：** C
- **子類：** C3 可重現性
- **嚴重度：** Minor
- **位置：** `run.py:474-484`；`exp/exp_Main2.py:899-914`；`build_training_csv_from_meta.py:165-171`
- **現象：** run_args 有路徑與 fold，但不保存資料與 split 内容 hash、程式 commit、scaler、前處理版本；summary 缺 dataset/split/fold。source 指紋只含 meta bytes、寬表 path/size/整秒mtime與選項，不含程式版本或寬表內容。
- **為什麼是問題：** 同名 dataset 曾就地重建，僅看 run_args 不能證明產生結果時使用哪版資料；36-run 表格也不能單靠 summary 分辨因子。
- **建議：** 一份 manifest 保存 commit、dataset/meta/split hash、選項、完整欄位、scaler；summary 納入 dataset、fold、split identity。
- **信心度：** 高
- **證據：** 實際 run metadata 與上述程式寫檔欄位；本次 evidence 已另存 dataset hash。

### 最該補測試的三個地方

1. loader／scaler／factory：segment 與 row train-only fit、欄位契約、seq+pred邊界、NaN／stride及完整固定 val。
2. split 與 overlap：連通群、全blocked／單點blocked、零window、不同fold expanding、split檔與loader樣本對齊。
3. 評估契約：target最後觀測錨、同尺度persistence、corr零變異、雙checkpoint獨立輸出與不碰hold-out的選模流程。

這三組測試直接防止洩漏或指標錯誤。未執行現有 `tests/test_merge_gate_data.py`，因其測試已棄用腳本，不在受審核心。

### 未驗證項目

| 項目 | 原因 |
|---|---|
| 訓練收斂、梯度、實際 checkpoint 重載結果、MPS/CUDA／AMP | 禁止訓練；本次只做 CPU 無梯度形狀探針，不存權重 |
| SQL 取數、上游 water/rain 對齊是否含未來資訊 | 無 SQL 原始水雨資料；`_4/_5` 是被地圖排除的現行依賴，範圍矛盾列 R-03 |
| 任意外部 meta/split 完整安全性 | 現行產物已驗；未宣稱所有錯誤輸入均被拒絕。label_len>seq_len、非法標籤、缺分鐘應另做契約測試 |

## 3. Block M — 方法與方向

### 本 block 我打算驗證的宣稱

1. 現行條件是否真的「沒有任何結果」，煙霧測試與正式研究證據如何區分。
2. pooled／segment corr、anchored 與 persistence 是否支持形狀學習與預警宣稱。
3. isRain 推論時是否可得，以及選事件／選NaN windows是否引入未來資訊。
4. drycut/old 及 exog 因子是否在同一評估母體上公平比較。
5. 三折單seed、事件單位、季節分布及 hold-out policy 是否足以支撐結論。
6. baseline／ablation／新穎性證據與36-run計畫是否對齊論文問題。
7. roadmap 的依據是否可由現行證據支持，下一步優先順序是否需調整。

### 證據狀態判定

**有舊結果與煙霧結果；無符合正式現行建模原則、可支撐方法優劣的完整結果。** `run_inventory.json` 找到125份 run_args（runs123、sanity2）；其中3份使用 drycut，都是1 epoch且含HL01。09-10 file/fold2煙霧確有有限的 [5700,15,1] pred/true/persist；09-02煙霧是[5592,15,1]，與目前樣本數不同。不能照文件宣稱「全部09-02以前、現行完全沒有任何模型實驗」，也不能把這些煙霧當正式原則下的研究驗證。

原始 anchored metadata 的9412 windows、raw20883.8867／adj1537.3794／persist2430.1643 與 context 舊數字吻合；該 run input 排除HL01。這只能證實檔案保存了該組數字；不證明現行條件的效果，也不證明其「形狀學得極好」解釋。此 block 以設計與計畫為主，另審已存在的歷史推論是否過度。

### Findings

#### M-01 · pooled corr 無法單獨支持形狀學習的核心結論
- **區塊：** M
- **子類：** 評估協議／過度詮釋
- **嚴重度：** Major
- **位置：** `exp/exp_Main2.py:212-219,753-756`；`docs/review/context/03-evidence.md:50`（§4）；`docs/model_roadmap.md:24`
- **現象：** 跨window／事件水位差可主導相關。合成反例每事件斜率全反向（各corr=-1），pooled仍0.99996675。現行資料不需模型的persistence，在drycut test pooled corr=0.980869、old=0.979425。
- **為什麼是問題：** corr高可能只是當下level與未來level相近，不能由adj corr=0.988推定模型學好15分鐘變化，更不能由一次後處理改善確定HL01缺席是唯一根因。per-segment corr已存在，但其 all列把window×horizon攤平，也不是每個15分鐘forecast的形狀分數。
- **建議：** 保留pooled作次要比較，事前固定event宏觀彙總及相對persistence的配對skill；加delta誤差／方向、峰值與時間偏差診斷。所有corr報有效事件及零變異數；不可把負corr用abs排名當「最差」。anchored若用於實際輸出，須完整定義為方法的一部分並與raw/persistence同報。
- **信心度：** 高（指標反例）；模型實際形狀品質未驗證。
- **證據：** `evidence/method.txt` pooled_counterexample、population；歷史anchored metadata。

`adj[k]=pred[k]-pred[0]+HL01[t]` 使用當下觀測，無未來target洩漏；但把 **t+1** 強制設為 t 的值，與delta-target的 `pred_abs[k]=delta[k]+HL01[t]` 不等價。若主張保留全部未來動態，必須承認第一步被鎖成persistence；建議將二者列不同方法。

#### M-02 · isRain 對照包含推論當下不可知的未來事件資訊
- **區塊：** M
- **子類：** leakage 論證／問題設定
- **嚴重度：** Major（若當成可部署方法則為 Blocker；待 Q-M1）
- **位置：** `build_drycut_segments_meta.py:40-55`；`build_training_csv_from_meta.py:91`；`docs/review/context/02-method-eval.md:149`（§5）
- **現象：** drycut 判斷是否為長乾run需看後續雨量，isRain再由整段核心邊界產生。合成兩份資料在01:55之前完全相同，只改02:10之後有無降雨，01:55的isRain由False變True；兩者均在保留buffer內。
- **為什麼是問題：** 不是只有「讓模型知道split結構」，而是輸入會隨未來改變。三層不重疊論證保證partition分鐘不共用，並未保證特徵可因果計算。以此宣稱即時預警效果會無效。
- **建議：** isRain僅作明確標示的非因果診斷對照，不参与正式可部署模型優劣結論；或重新定義只依過去雨量的旗標。選事件可用於回溯事件評估，但須說明母體條件，不能等同連續上線預警。
- **信心度：** 高
- **證據：** `evidence/method.txt` isRain_future_counterexample。

#### M-03 · 36-run 比較混合了資料選樣、前處理與模型效益
- **區塊：** M
- **子類：** 實驗設計／評估母體
- **嚴重度：** Major
- **位置：** `build_training_csv_from_meta.py:75-106`；`data_provider/Data_Loader.py:441-468`；`docs/review/context/02-method-eval.md` §5
- **現象：** drycut/old test有5700／4839個origin，只4521個共同；drycut另1179、old另318。persistence MSE在兩者已不同（2821.8924／3278.9951），故模型分數不能直接歸因於切法。共有44,429個日期的gate有2,590格NaN狀態不同（皆有值時沒有數值差），來自段界重置ffill。去掉exog會令全資料可用windows從33381→34518、31440→32678；本次test恰不變，但dev／fold與fit母體會改變。
- **為什麼是問題：** 相同前處理函式不代表相同可用樣本；單看不同test上的MSE／corr不能回答哪個建模或exog策略更好。未來exog缺值也参与整段need的篩選，評估母體包含部署當下未知的存活條件。
- **建議：** 預先固定共同原始時間hold-out與forecast origins，以共同有效樣本做配對比較，再另報各pipeline完整coverage與分層結果；ablation主分析固定NaN mask及切分。若研究目標是整套選樣policy，將coverage與錯失事件當成輸出，不硬解釋成模型優勢。兩套fold的val日期不同，也不可把相同fold號當同一事件集合。
- **信心度：** 高
- **證據：** `evidence/method.txt` common_test_origins／shared_date_gate_comparison；`evidence/data.txt` windows及fold期間。

#### M-04 · 即時預警的成功判準與連續時間評估尚未定義
- **區塊：** M
- **子類：** 問題設定／評估協議
- **嚴重度：** Major
- **位置：** `docs/review/context/00-overview.md:26-49`；`exp/exp_Main2.py:454-489`；`dataset/rain_segments_meta_drycut_L3h_buf60.csv`
- **現象：** 任務有1分鐘/15步/HL01，但未定義要預警哪個水位／變化事件、最小提前量、容許誤報或漏報；主流程只有數值誤差與corr，且只取54,590/511,679分鐘（約10.7%）事件窗。
- **為什麼是問題：** 能在挑過的事件窗預測水位，不等於可在全年連續運轉的預警系統；沒有乾期誤報母體就無法評估干擾成本。單純補一個任意corr數字不是合理的成功定義。
- **建議：** 若當期論文定位為水位forecasting benchmark，先把預警列用途、主判準固定為共同事件上對persistence的配對skill與不確定區間；若要主張warning，需使用者依運作需求訂事件定義、提前量／誤報容忍，加入連續dry/wet hold-out。門檻在看正式test前凍結，不依本次test baseline調門檻。
- **信心度：** 高；實務效用門檻須作者決定。
- **證據：** 程式輸出的實際指標欄位及Block D保留分鐘數。

#### M-05 · 矩陣能篩選配置，尚不能識別架構貢獻或穩定優勢
- **區塊：** M
- **子類：** 新穎性／實驗設計
- **嚴重度：** Major
- **位置：** `models/DLinearMix2.py:52-75,104-117,130-164,260-293`；`docs/review/context/02-method-eval.md` §5；`docs/model_roadmap.md:61-113`
- **現象：** 36 runs只比較資料2×exog3×loss2×fold3×seed1，沒有可識別branch DLinear、GRU或fusion收益的對照，delta參數化也不在矩陣。單seed不能估計初始化變異，3個expanding folds也不是3個獨立重複試驗。
- **為什麼是問題：** branch線性時序投影+GRU+MLP的組合本身不足以證明新穎性與必要性；「不放HL01」是可明確定義的限制，但在即時預警已有該站感測器時，需說明為何此限制有用。
- **建議：** 36-run定位為探索／篩選；先在固定可因果評估集比較persistence、鄰站直接線性映射/正則化回歸、同輸入的單純GRU或等容量MLP，再做無exog、簡單pooling替代GRU、fusion等關鍵ablation。選定少數配置後補多seed、事件／重疊群的配對區間及逐fold結果。可考慮以「有限目標資訊條件下的有效預測」為貢獻，需依新數據證明而非從delta形式判定價值。
- **信心度：** 中（設計缺口高；完整文獻新穎性未作系統性搜尋）。
- **證據：** 受審模型與矩陣因子；[LTSF-Linear原論文](https://arxiv.org/abs/2205.13504)提供簡單線性基線的直接先例；[水文benchmark研究](https://hess.copernicus.org/articles/25/5517/2021/)採固定評估期間比較神經與概念模型。兩者僅作外部研究背景，不作本專案實作或15分鐘閘門水位性能的權威。

### 方向判定的完整論述

**前進，但尚不宜直接把36 runs當作論文驗證矩陣。** 前進的具體證據是原始分鐘網格可查、179段meta可重現、gate逐欄合併一致、split位元組可重現且跨partition重疊為0、file模式scaler與24個loader計數一致。這些是可信研究的基礎，不只是文件整理。

優先順序建議為：先修可達leakage路徑或明確阻擋，修val完整性/corr/特徵選擇；確認D區塊物理語意；凍結可因果欄位、共同評估時間與主指標、禁止CV持續解封test；再用少量現行原則下的baseline＋raw/anchored/delta診斷驗證roadmap根據。exog有效性與簡單pooling ablation應早於cross-attention；只有確認瓶頸才增加horizon-aware結構。delta是合理候選，不是已證實的根因療法。

三層防leakage在**現有meta+split的時間重疊層面**成立；它不涵蓋isRain未來依賴、row scaler、fold常數選擇、hold-out反覆查看、跨欄位採樣母體。不可再用「0 leakage」概括整條pipeline。

3-fold可以作目前單年資料的探索設計，無須為湊折數壓縮事件；有效單位更接近事件／相互關聯的降雨過程，而非上萬個重疊windows。drycut val38/32/28是段數，不自動保證獨立性；old20/28/23且有union群。單一7/18–8/03 test只支持該時間外切片的結果，不能泛化全年或所有颱風。現有資料test濕分鐘比例確較train高（drycut33.74% vs26.88%；old35.08% vs28.15%），但沒有外部事件標註可把每段稱為颱風。

未執行訓練，因此不判斷模型是否優於baseline、delta是否有效、exog是否必要。沒有現行正式模型結果不是另列的bug；缺的是能回答論文問題的驗證設計與可辨識證據。

## 4. Block R — 審查文件本身

### 本 block 我打算驗證的宣稱

1. context 的數字是否對上本次資料與loader，區分過期歷史與宣稱現況。
2. 抽查至少10個file:line，判斷是否只是小漂移或指向不同函式。
3. 「無模型結果」、run總數、anchored日期及污染判定是否有實物支援。
4. MUST-REVIEW與實際import圖、資料譜系是否矛盾或漏掉執行依賴。
5. 架構能力、buffer可用性、0 leakage、test季節性是否過度宣稱。
6. README／PROTOCOL／block／TEMPLATE／spec的範圍、提問及ID/delta schema是否一致。
7. 明載meeting要求是否被漏記或換了單位／指標，授權缺口是否有足夠來源證據。

### Findings

#### R-01 · 證據台帳混用舊版本數字與現行統計
- **區塊：** R
- **子類：** R1 事實錯誤
- **嚴重度：** Major（主要影響現行window核對；細小計數誤差不另灌水）
- **位置：** `docs/review/context/01-data.md:11,62-67,110-111,150-151`；`03-evidence.md:24,40-43`；`05-traceability.md` T-21；`06-open-issues.md:129-131`
- **現象：** drycut目前split是23551/4130/5700，台帳仍寫23151/4043/5592；舊數字確對應09-02早期煙霧的5592輸出。test目前從07-18起，不是泛稱6–8月。gate NaN實為3.49148%/3.41857%，正確四捨五入3.49%/3.42%；資料夾20 CSV+2 sidecar+1歷史report，不是22 CSV；雨量9欄不是8。數個CSV缺最後換行，wc-l扣header會少一筆：gate opening實際283873筆（另見data.txt inventory）。
- **為什麼是問題：** loader正確也會被錯誤期望判成失配；現況證據混版本使研究不可追溯。±4%也不精確：4294/4641/4627相對平均4520.67最大偏差約5.01%。
- **建議：** 生成帶hash／精確欄位／seq/pred/stride／方法版本的統計表；歷史數值註明當時snapshot，不充當現行驗收值。資料列數用CSV parser分塊驗證，不只計換行。
- **信心度：** 高
- **證據：** `evidence/data.txt`、`code.txt`、`method.txt`、`docs.txt`。Rain-col根本錯誤另列D-01、不重複計數。

#### R-02 · 「所有結果09-02以前、完全無現行實驗」被實際產物否定
- **區塊：** R
- **子類：** R1 事實錯誤／R2 內部一致性
- **嚴重度：** Major
- **位置：** `docs/review/context/03-evidence.md:10-13,47,61-64,74`；`00-overview.md` §7；`06-open-issues.md` OI-01
- **現象：** 125份run_args（runs123、兩個sanity）中有3份drycut一epoch run；09-10 file/fold2有完整5700筆預測。台帳同頁先說全部09-02前，後面又承認09-02/09-10煙霧；「全repo116」也不符可讀metadata總數。anchored產生日期在同頁寫06-30及06-03，應區分計算／討論／訓練時間；本機json本身不記生成時間，不能靠文件確定計算日期。
- **為什麼是問題：** 引導Block M選錯證據狀態，隱藏實際test已暴露的事實。煙霧含HL01且無資料hash，既不能稱完全沒結果，也不能當符合正式原則的結果。
- **建議：** 明列三種狀態：歷史研究結果／流程煙霧／正式可比較結果；每項附run path、資料hash、split identity及用途。若125/116用不同去重或篩選，將分母規則寫清楚。
- **信心度：** 高（產物存在與形狀）；歷史生成時刻未驗證。
- **證據：** `evidence/run_inventory.json`；`evidence/method.txt` smoke_artifact及historical_anchored_metadata。

#### R-03 · 審查範圍與資料譜系漏掉現行執行依賴
- **區塊：** R
- **子類：** R2 內部一致性／R4 遺漏
- **嚴重度：** Major
- **位置：** `docs/review/context/04-code-map.md` §1、§4；`01-data.md:23-35,70`；`Data_From_SQL_all.py:18-38,199-237`；`exp/exp_Main2.py:41`
- **現象：** `_all`不只借SQLServerClient，還實際import並呼叫`_4`的water/rain/segment函式與`_5`的gate loaders；地圖把兩者列歷史，且只揭露_4 client。`utils.tools`的EarlyStopping/LR排程與`utils.timefeatures`亦未列核心。譜系圖暗示_all讀gate-opening CSV，實際_all是SQL→記憶體pivot；讀本機CSV的是rebuild。rain_segments_meta雖被標歷史，卻是現行old對照的必要上游；run.py預設仍讀歷史water_level_all.csv。
- **為什麼是問題：** 照「只審MUST-REVIEW、禁止審歷史」會漏掉取數時間對齊、早停等實際計算邏輯；不能据此宣称整條SQL→結論已完成安全審查。
- **建議：** 用可達函式分級：_4/_5現行被呼叫部分納入，真正舊main排除；把training utility與推論/anchored工具的覆蓋要求講清楚。重畫SQL路徑與本機重建路徑兩條譜系，標old meta為「歷史定義、現行對照上游」。
- **信心度：** 高
- **證據：** `evidence/docs.txt` core imports；本次遵守既定範圍，未擅自審被排除的上游實作。

#### R-04 · 「單一context無法表達horizon延遲」是過度的架構斷言
- **區塊：** R
- **子類：** R3 過度宣稱
- **嚴重度：** Major
- **位置：** `docs/review/context/06-open-issues.md:123-127`；`docs/model_roadmap.md:95-113`；`models/DLinearMix2.py:130-164,334-345`
- **現象：** 文件引用151–158稱broadcast，實際該處是FlattenFusion，最後Linear輸出15個不同horizon，各輸出可有不同context權重；broadcast只在horizon-wise的341行。即使共享MLP也同時讀不同branch forecast，不能由共享context推出所有horizon反應相同。
- **為什麼是問題：** 將尚未實測的壓縮瓶頸寫成表達能力不可能，會錯誤支持高成本cross-attention的優先順序。
- **建議：** 改為「固定維context可能有資訊瓶頸，需要ablation」，分清兩種fusion；用延遲控制實驗驗證再做結構改動。
- **信心度：** 高
- **證據：** 已讀模型全檔；`evidence/docs.txt` reference probe、`code.txt`兩種fusion形狀探針。此為結構推理，不聲稱現有模型已學會延遲。

#### R-05 · 「buffer=0不可用／179段100%可訓練」混淆長度與有效性
- **區塊：** R
- **子類：** R3 過度宣稱
- **嚴重度：** Minor
- **位置：** `docs/review/context/01-data.md:69,125,150-151`；`05-traceability.md` T-12
- **現象：** buf0有72段符合111列，並非完全不可用；isRain全True只代表該特徵無資訊，可刪欄，不代表其他欄無法訓練。buf60雖179段按長度合格，NaN後仍8段無樣本。
- **為什麼是問題：** 把資料保留偏好包裝成必要條件，會遮蔽可用子集／coverage的真正取捨。
- **建議：** 同時報length-eligible与NaN-valid段/windows，將「不可用」改為「未達本研究保留所有短段的coverage目標」。
- **信心度：** 高
- **證據：** `evidence/data.txt` drycut_reconstruction／training_stats；正常loader已驗。

#### R-06 · protocol／template／spec存在會影響後續審查的一致性缺口
- **區塊：** R
- **子類：** R2 內部一致性／R5 可用性
- **嚴重度：** Minor
- **位置：** `docs/review/protocol/PROTOCOL.md` §0.6、§4；`protocol/block-C-code.md` C1；`reports/TEMPLATE.md` §§0、3、4、5；`2026-09-10-research-review-system-design.md` 狀態、§§6–7
- **現象：** §0.6要求四block未答問題，§4摘要仍只說Block R；template只在§5放問題、M/R又漏「事前5–10宣稱」欄；Block C仍稱OI-01～06皆未處理，實際OI-03撤回、OI-06已修；spec仍「尚未實作／三個block」，無標示已被現行protocol取代。ID雖要求沿用，未定義finding合併／拆分／撤回時怎麼記。
- **為什麼是問題：** 下次照template執行可能漏提問或漏列宣稱；沿用已撤回疑點會放大誤報，也不利delta。
- **建議：** 以現行protocol為作業版本、spec標歷史；對齊template與問題索引，增狀態、首次ID與supersedes欄位，無需重編既有ID。
- **信心度：** 高
- **證據：** 本次逐檔讀過所有15份既有review Markdown；`evidence/docs.txt`文件清冊。

#### R-07 · 部分file:line指向不同邏輯，不能當精確追溯
- **區塊：** R
- **子類：** R1 事實錯誤／R5 可用性
- **嚴重度：** Minor（整份程式仍可讀；關鍵架構誤判另列R-04）
- **位置：** `docs/review/context/04-code-map.md` §1；`05-traceability.md` T-01、T-28、T-29；`02-method-eval.md` §1.2
- **現象：** run.py169–174現在是常數刪欄流程，不是input/exog overlap；正確在201–206。constant函式完整範圍120–183，文件119–156漏後半。buffer外擴現在107–108，106只是segment_id。build_splits行數313，不是304。
- **為什麼是問題：** 本專案已因跳讀六行出過誤報，錯定位會再引導片段式漏審；但小漂移本身不等於實验無效。
- **建議：** 引用commit＋函式名＋關鍵行，補自動抽查；下表提供本次實際位置。
- **信心度：** 高
- **證據：** `evidence/docs.txt` 15組reference probes。

#### R-08 · 物理與外部權威宣稱缺少可查來源，不能升格為既定事實
- **區塊：** R
- **子類：** R3 過度宣稱／R4 遺漏
- **嚴重度：** Minor（待來源補齊；不直接判定物理或法律結論錯誤）
- **位置：** `docs/review/context/00-overview.md:10`；`01-data.md:86-101`；`06-open-issues.md:152-153`；`docs/rain_event_definition_comparison.md:10-18`
- **現象：** 資料CSV只見HL欄名，無站點拓樸／單位metadata；文件稱HL02–06全上游、水位公分，以及「官方Past10Min／有效降雨>0.5」沒有精確來源。授權處只寫Apache-2.0與缺NOTICE，未附所複製上游revision與license。
- **為什麼是問題：** 欄名與0.5量化不等於物理單位或官方事件定義的證明；引用論證不可只沿用文件自身。缺NOTICE本身也不足以認定Apache條件違反。
- **建議：** 補站點／感測器資料字典、拓樸及外部來源；授權先固定實際上游版本與條款。Apache 2.0 §4(d) 的NOTICE保留義務有「上游包含NOTICE」前提，不能把所有衍生專案都必須新建NOTICE當普遍規則；本次不作授權合規結論。[Apache官方條款](https://www.apache.org/licenses/LICENSE-2.0)
- **信心度：** 高（來源缺口）；物理語意待Q-R1。
- **證據：** `evidence/docs.txt` dataset schema／license filename掃描；外部條款僅用來查核文件措辭，不作本專案Tier0。

### `file:line` 抽查結果

| 文件宣稱的位置 | 實際內容是否相符 | 備註 |
|---|---|---|
| `run.py:44-66` glob | 相符（差1行） | 函式45–66 |
| `run.py:169-174` overlap guard | **不符** | 實際201–206；引用處是constant處理 |
| `run.py:119-156` constant | 部分 | 函式120–183，引用截斷 |
| `DLinearMix2.py:144-165`兩fusion | 部分 | 只涵蓋FlattenFusion；HorizonWise在120–141 |
| `DLinearMix2.py:151-158`broadcast | **不符** | 實際FlattenFusion層；broadcast341 |
| `Data_Loader.py:247-250`欄序 | 相符 | input→exog去重 |
| `Data_Loader.py:277`target | 相符 | df_y_cur_raw=target |
| `Data_Loader.py:307-311`scaler | 相符（mix/segment） | row上游df_train錯仍會走此處 |
| `Data_Loader.py:457-472`NaN filter | 相符 | 包含前綴和邊界 |
| `Data_Loader.py:474-489`不跨段 | 相符 | 段內列舉 |
| `exp_Main2.py:547`取錨 | 相符 | batch_y label_len-1 |
| `build_drycut_segments_meta.py:91-92`L檢查 | 相符 | OI-03確為已撤回誤報 |
| `build_drycut_segments_meta.py:106-107`buffer | 部分 | 正確107–108 |
| `build_splits.py:143-149`blocked raise | 相符 | OI-06修正生效 |
| `exp_Main2.py:184-221`vali | 相符 | 函式至220 |
| `exp_Main2.py:696-880`segment metrics | 相符 | all/per-horizon corr確存在 |

### 其餘已知疑點的處理

OI-03不再列finding，L約束存在；OI-06的raise與split不變已獨立重驗。OI-02是已棄用腳本的過期設定，本次不按現行bug修復優先；真正有效樣本差異列M-03。OI-17的BOM已由實際pandas/loader讀取通過。OI-19的x_raw在排除外來碼與本次evidence後只有三處賦值、沒有讀取；屬已知無效狀態，不另新增Nit。資料表已涵蓋全部20個CSV，僅漏歷史report.txt，沒有缺交任何本機核心CSV。

meeting5/20同時寫「四種MSE」與median/p90 RMSE，程式實算全為MSE；p90確有明文依據，故不誤報成應算最差10%平均。單位選擇待Q-R2，未將未確認意圖硬列bug。訓練日誌的train_mse在Huber時其實是loss值，應改名train_loss；屬低影響命名，不另列finding。

## 5. 給使用者的問題

下列11題已依D→C→M→R分四批主動提出（3／2／4／2題），截至09-13首次完成尚未獲答。09-14相關決策與剩餘疑點見第5c節；本節保留原始問題與功課，不把未回答的歷史背景視為已確認。不計嚴重度。

### Q-D1 · 正式drycut維持Past10Min，還是原意為Past1Hr？
- **我查到的：** `build_drycut_segments_meta.py:32,80,101`預設Past10Min；`audit_data.py`驗得179段起訖全部匹配，Past1Hr則164段且不匹配。
- **我的暫定判斷：** 文件寫錯現行方法，信心高；不替作者決定研究意圖。
- **需要你決定的：** 維持Past10Min並修文件，或另產Past1Hr版本？
- **影響：** D-01的修正方向及資料版本。

### Q-D2 · 五分鐘限制要套用最終輸入，還是只套在第一次合併？
- **我查到的：** `build_training_csv_from_meta.py:105`無上限ffill；資料重算得到4,463個drycut填補格age>5min，補驗1,521個有效window實際輸入含此值，最久至少559min。
- **我的暫定判斷：** 與T-06文字衝突，暫列Major；meeting兩階段處理記載顯示可能刻意，信心中。
- **需要你決定的：** 最終輸入也限制5min，或允許延用最後狀態並標freshness？
- **影響：** D-02嚴重度與ffill政策；若沿用，仍需把實際量測與估計狀態區分。

### Q-D3 · 負開度可一律視為全關，有設備或現場確認嗎？
- **我查到的：** `rebuild_gate_columns.py:52`直接clip；raw gate南3有227,756個負值、南1最小-7.1336，見`data.txt`。
- **我的暫定判斷：** 實作一致但物理解讀不明，信心低。
- **需要你決定的：** 是否有可引用依據支持一律當0，或須調查／分型？
- **影響：** 是否新增資料品質Major；目前只列疑問與R-08來源缺口。

### Q-C1 · 早期base sweep包含HL01是刻意對照或疏漏？
- **我查到的：** `run.py:45-66,186-238`實測接受HL01–06；125份metadata中13份明列input含HL01，包括三個09月煙霧。
- **我的暫定判斷：** 正式入口缺防護確定；歷史動機未記載，信心低。
- **需要你決定的：** 早期base是有意的對照實驗，還是當時誤展開？
- **影響：** C-01對歷史實驗的分類／污染範圍，不改變正式入口防護建議。

### Q-C2 · 是否曾用test數字影響模型或研究決策？
- **我查到的：** `exp_Main2.py:289-297`每epoch計算test，但自動early stopping只吃val；`method.txt`確認09月煙霧有實際test輸出。
- **我的暫定判斷：** 暴露路徑已證實、人工污染未證實，信心中。
- **需要你決定的：** 實際只用val決策，或曾用test挑模型／調參／決定roadmap？
- **影響：** C-08應只隔離未來test流程，或現有test須降級為開發證據並新增hold-out。

### Q-M1 · isRain是否只作非因果、不可部署的診斷對照？
- **我查到的：** 現行drycut函式反例中，僅改未來雨量便改變既往isRain；`method.txt`保留相同過去、不同未來的具體時刻。
- **我的暫定判斷：** 非因果性確定，信心高；對照用途未知。
- **需要你決定的：** 是否同意限定為診斷對照、不拿它支持正式預警能力？
- **影響：** M-02目前Major；若作可部署性能證據則升Blocker。

### Q-M2 · 比較目標是同時刻準確度，還是整套選樣policy？
- **我查到的：** `audit_method.py`驗得test origins=5700/4839，共同4521；同一persistence的MSE亦2821.89/3279.00。
- **我的暫定判斷：** 直接比較各自分數不能隔離準確度效益，信心高。
- **需要你決定的：** 主分析固定同一批forecast origins，或把coverage／漏掉事件也當研究輸出？
- **影響：** M-03的評估母體及論文可主張的效果。

### Q-M3 · 矩陣exog「無」是無衍生旗標，還是全部外生資訊移除？
- **我查到的：** 文件§5只寫isRain/無/MSR；roadmap說整條exog拿掉。`data.txt`兩種「無」產生不同window總數：33441/31510對34518/32678。
- **我的暫定判斷：** 因子定義有兩種合理解讀，信心高。
- **需要你決定的：** 保留雨量／閘門只移除isRain/MSR，或全部exog移除？
- **影響：** M-03採樣控制與M-05是否真的完成整體exog ablation。

### Q-M4 · 當期論文先驗證水位預測，或必須驗證實際預警效用？
- **我查到的：** 程式只報MSE/MAE/corr；事件窗只覆蓋全年約10.7%，persistence已有約0.98 pooled corr。
- **我的暫定判斷：** 任意corr門檻不能代表預警成功，信心高。
- **需要你決定的：** 先定位為有限HL01資訊的forecasting，或當期即要warning？若後者，預警事件、提前量與誤報容忍的業務需求為何？
- **影響：** M-01/M-04的主指標、乾期／連續hold-out及研究優先順序。

### Q-R1 · 「水位公分、HL02–06皆上游」可由哪些現場資料確認？
- **我查到的：** dataset schema與檔名檢索未有站點拓樸／單位字典，只有文件自述；gate PqId map不包含HL站位關係。
- **我的暫定判斷：** 無法從欄名驗證物理拓樸或單位，信心低。
- **需要你決定的：** 兩項宣稱是否完全正確，若有例外請指出及說明來源？
- **影響：** R-08，以及M-05能否把模型解釋為上游傳遞／退水機制。

### Q-R2 · median／p90正式使用MSE還是RMSE？
- **我查到的：** `meeting_recap.txt:43-47`混寫MSE與RMSE；`exp_Main2.py:822-826`實際全用MSE，p90定義本身與meeting一致。
- **我的暫定判斷：** 是意圖單位歧義，非p90演算法bug，信心高。
- **需要你決定的：** 統一MSE並更正文案，或median/p90用RMSE以保留水位單位？
- **影響：** T-02／T-31附近評估意圖的說明，可能新增功能要求；目前不因歧義計bug。

## 5b. 審查期間所做的假設

| 09-13原始假設（當時均未確認；09-14更新見第5c節） | 依據 | 若假設錯誤的後果 |
|---|---|---|
| 現有Past10Min資料先作受審基準，未替作者改研究雨量意圖 | 程式與179段meta相符 | D-01可能需產新版本而非只修文件 |
| 5min是值得保留的最終freshness需求，但刻意持續估計狀態亦有可能 | T-06文字與meeting兩階段語意相衝 | D-02需重定級／改成來源與狀態標記建議 |
| 負值物理語意與測站單位／拓樸尚未確認 | 缺設備／站點metadata | 取得依據後可關閉疑問，或升為資料品質問題 |
| 不認定任何歷史run已被人為test選模污染；也不認定早期HL01是無意 | 程式只能證明配置與暴露，不能證明人的決策 | C-01/C-08影響範圍需按回答更新 |
| isRain只允許作非因果診斷對照，不作可部署headline | 避免把已證非因果特徵誤當線上方法 | M-02若用於正式預警，升Blocker |
| 同時列出兩種「無exog」與兩種比較目標，不替作者選定 | 矩陣文字不足；兩者樣本數不同 | M-03/M-05主分析設計需收斂 |
| 不自行發明預警效用數字門檻；水位forecasting作可繼續的最低研究範圍 | 缺實務閾值及誤報／提前量需求 | 若當期需要warning，M-04需求與資料工作需提前 |
| 目前MSE欄名忠實描述實算值，尚不把RMSE需求當既定bug | 最新程式與meeting文字存在歧義 | Q-R2確認後調整報表意圖判定 |

## 5c. 2026-09-14 使用者裁決與討論紀錄

**狀態原則：** 以下區分「使用者已確認的方向」、「審查者建議」及「仍待確認」，不將討論誤記為程式已修復。原有禁止訓練／commit／push持續有效；本輪先在報告資料夾保存決策，是否立即擴大至受審程式修改另行釐清。

### 已確認方向與具體修正規格

| 對應 | 使用者回覆／決策 | 具體處理與驗收 | 尚未確認或完成 |
|---|---|---|---|
| C-04 | early stopping必須比較同一份validation，確實要修改 | `Data_Factory.py`把val明確設成shuffle=False、drop_last=False；完整保留尾批，拒絕空評估；驗收同一資料各ID恰評一次、同一模型重評分數一致 | 尚未修改程式；train loader是否保留drop_last是另一件事，不能一併推定 |
| C-01 | target出現在input或exog、違反正式原則時應直接報錯 | `run.py`在glob展開後、constant刪欄前檢查，避免target先被刪掉而掩蓋錯誤；正式loader也防直接API繞過。保留最後一個HL01觀測作persistence／anchored的既有用途，禁止的是模型輸入歷史 | 尚未修改；早期含HL01實驗是刻意或疏漏仍待Q-C1 |
| C-08 | 開發／CV只用train+val；方法及checkpoint政策固定後才作預先規劃的最終test比較 | `exp_Main2.py`移除訓練期間test loader／評估；`run.py`預設不自動test，提供明確的最終評估階段，載入已凍結設定與既有checkpoint，不能為了test再訓練；驗收開發路徑不觸及test評估，最終階段不呼叫train | 已確定未來政策，程式尚未修改；Q-C2的歷史test是否已影響設計仍未答 |
| M-02／03 | 提議old不放isRain，drycut放min_since_rain | 保留這個提案；CSV欄名實際是`isRain`。old與drycut的完整input／exog清單需列明，不能把「不放isRain」解讀為移除雨量與gate | drycut是否完全排除isRain、old能否也用MSR、是否採配對因子矩陣仍待確認；尚未結案Q-M1／M3 |
| M-01 | 現階段使用pooled corr搭配anchored，之後嘗試delta；承認部分segment仍差 | 記為近期研究方向。raw／anchored／persistence須同集合並列；保留segment分布與低corr案例，不必等模型改進才揭露 | raw或anchored的corr用於選checkpoint、零變異處理尚待定；不能預先保證anchored／delta會改善所有段 |
| M-03 | 承認old與drycut test範圍不同，仍希望先探索 | 可先在dev／val探索；同時規劃共同origins的配對表與各自coverage表，避免用final test反覆摸索 | 是否正式採同母體比較／完整policy評估，Q-M2仍待定；不能把承認問題當成接受不可比較的最終結論 |

### C-02 適用範圍澄清

本次已證實的「df_train指向整檔」scaler bug位於未指定segment_col的row路徑，且需有hold-out並啟用scale才涉及該標準化洩漏；指定有效segment_col會走另一分支。24個現行segment+split-file/fold loader查核通過train-only fit。這不是「只要指定segment_col就排除所有leakage」：builtin segment split仍可能跨重疊事件，C-03的constant selector也仍未依fold選train，C-07的Dataset_Pred另有重fit全檔問題。

### shuffle／drop_last及corr計算的建議

shuffle是在本專案的window dataset中打亂window取樣順序，不會打亂每個window內的96分鐘。drop_last決定是否捨棄不足batch_size的最後一批；10筆、batch4會是4+4+2，設True時捨棄最後2筆。兩者合用就會每輪遺漏不同window；關閉shuffle但保留drop_last則會固定漏資料，仍非完整validation。[PyTorch DataLoader官方定義](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。train可因隨機梯度訓練需要打亂獨立window；本次C-04先只修val。

**審查者建議，尚未視為作者已採納：** 保留Pearson、每個horizon跨固定windows計算，再等權平均；用同一函式處理val/test/baseline，各horizon值與有效數均輸出。零變異Pearson在數學上未定義，應報NaN／原因，不能把epsilon得到的0冒充標準Pearson。[SciPy官方定義](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)。近零變異／有限值規則也需固定並記錄，不能看test後才調。

若繼續以corr選模，建議在固定val上先依真值定義可評horizon集合；所有候選使用同一集合。該集合中任何pred變成常數，該候選的完整corr分數無效，不得只平均剩餘好看的horizon來更新best-by-corr。可另留既定best-by-MSE checkpoint，但不可靜默把corr選模切換成MSE；全程無有效corr候選則明確報錯／標示。這是審查者提出的保守選模政策，不是Pearson定義強制要求。理由是數學定義一致、候選比較分母固定、避免預測退化反而得利；不是依哪種算法分數較高決定。

### pooled反例與anchored／delta的精確區分

原始反例是5個事件、每事件20點：事件水位L依序為0、1000、2000、3000、4000，事件內j=0…19，真值=L+j，預測=L−j。每事件內一升一降，corr=-1；跨事件仍是高水位對高預測、低水位對低預測。Var(L)=2,000,000、Var(j)=33.25，兩者在這個合成設計中共變異為0，故pooled corr=(2,000,000−33.25)/(2,000,000+33.25)=0.999966750552772。這是事件間水位差主導結果；不是程式把-1平均成正數，也不證明真實模型每事件都反向。

`anchored[h]=pred[h]-pred[1]+HL01[t]`（此處h從1起算）在同一forecast加同一常數，不能改變其內部升降、峰值位置或非零變異的15點corr；跨window因各自位移不同，pooled／segment跨window corr卻可能改變。第一步被固定為HL01[t]，因此不是「任意horizon的delta重建」。

delta訓練目標應明訂為`delta[h]=HL01[t+h]-HL01[t]`，最後`pred_abs[h]=pred_delta[h]+HL01[t]`，第一步可不為0。使用正確錨且同物理單位時，絕對誤差與delta誤差代數相同；效果若改善，來自參數化／可學性等實際差異，沒有「改成delta必然修好低corr段」的保證。不同單位標準化或Huber beta也會改變實際優化問題，需要控制。合成數值與平移不變性見 `evidence/discussion_math_2026-09-14.txt`，不含模型訓練。

### 因子矩陣與共同評估的建議

old無旗標、drycut有MSR是兩個整套配置，可以探索，但同時改切法與特徵，不能歸因於drycut。建議固定其他雨量／gate後，比較old/drycut × 無衍生旗標/MSR的2×2設計；「無衍生旗標」不等於「無全部exog」。本機old與drycut都有MSR，技術上可做；這是建議，尚未替作者重定正式矩陣。

現有test共同起點為4521；未來凍結後可以按絕對日期／horizon join結果，先檢查真值一致、兩側皆可用及前處理條件，再作配對比較。各自完整集合另報coverage與事件分布。開發時先在共同val日期上採同樣原則；相同fold號不是相同val日期，必要時先定共同dev時間範圍。無須等delta或新架構才把比較條件定好，也不能把「先嘗試」當成繼續反覆查看final test。

### 原始問題的最新狀態

- Q-C1／Q-C2：未來防護與test政策已確認；歷史動機／是否用test決策尚未回答，保留原問題。
- Q-M1／Q-M3：已有old／drycut旗標提案，但完整欄位與是否配對尚未定案。
- Q-M2：已承認母體不同；研究比較目標與正式評估表仍待確認。
- Q-D1／D2／D3、Q-M4、Q-R1／R2：本輪未取得對應明確答覆。維持未驗證假設，不因已回覆其他問題而自動結案。

## 5d. 2026-09-14 第二輪討論與C-01修正

### 授權、完成範圍與驗收

使用者本輪明確要求「Target防護的方向已確認的話就先幫我修改一下程式碼」，因此解除本項修正與必要測試的報告資料夾外寫入限制；不擴大成所有findings的修復授權。禁止訓練、commit、push仍有效。C-02關閉row、C-04完整val與C-08評估分期目前仍是待實作方向；本輪沒有替換split或重產dataset。

**C-01已修復的程式缺口：** `run.py`新增 `_validate_target_inputs`，glob展開後、constant刪欄前拒絕target；直接configure也驗證。`Dataset_Custom.__init__`對DLinearMix／DLinearMix2直接呼叫API的input／exog加同樣檢查，且在讀檔前失敗。使用當前target參數，不硬編碼HL01。這項修正不限制target作label或當下錨點，也不宣稱更改了其他model家族的輸入政策。

`tests/test_target_input_guard.py`共24個測試通過，涵蓋直接target字串／列表／空白、exog、HL*、改用其他target、CLI在刪常數前拒絕、兩種mix model及train/val/test API拒絕、合法無exog／有exog配置，以及真實loader仍提供正確target labels和最後觀測錨。所有測試只讀／合成資料；run.py只取AST參數函式與preflight語句，未import或執行main，未呼叫train、model forward或checkpoint。

執行：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .HLML_Linear_venv/bin/python -B -m pytest -p no:cacheprovider tests/test_target_input_guard.py -q`。輸出見 `evidence/target_guard_tests_2026-09-14.txt`。run.py保留既有CRLF，Data_Loader保留LF與原本無結尾換行；以 `git -c core.whitespace=cr-at-eol diff --check`檢查，未修改Git設定。

**版本界限：** §§1–4與33條T表保留原始審查證據，因此T-01及C-01原文描述的是修正前；不得再用原始target_guard accepted=True代表現況。原始audit_code.py包含故意違反新政策的反例，直接整支重跑會被新防護拒絕；原始audit_completeness.py也會因已授權的核心hash變化失敗。兩者保留為歷史快照，新的驗收用本節測試與 `evidence/target_guard_fix_verification_2026-09-14.txt`；後續審查應更新程式位置。

### 本輪已確定與尚待實作的方向

| 項目 | 最新決定／解釋 | 現況 |
|---|---|---|
| segment_col／row | 使用者表示一定有segment_col，希望關閉row；建議正式CLI要求segment_col、split_mode=file及split_file，loader直接API也拒絕缺segment_col。欄名錯／不存在也應直接報錯，不能回退 | 本輪僅提出做法，尚未改C-02；builtin是否一併禁用是建議，不把兩者混為同一問題 |
| validation | shuffle=False、drop_last=False已確認；另要拒絕空評估與非有限選模分數 | C-04尚未實作；NaN Pearson與空資料／數值錯誤須分型，不應把NaN改成0或默默改用其他指標 |
| isRain／MSR | 使用者決定本階段兩者都不放；取代第5c節old無旗標、drycut有MSR及2×2提案 | 排除的是兩個模型輸入欄，不是刪CSV欄或移除其他雨量／gate。Q-M1／M3本階段已裁決；矩陣腳本尚未建立，split window統計日後須按新欄位重算 |
| 因子矩陣test | 使用者希望矩陣能用test比較；可以對預先凍結的各配置作同一最終test比較 | 仍不在每epoch／CV探索時用test決定設定；看test後挑贏家屬描述性比較，再修改設計即進入新開發輪，不能冒稱未參與設計的驗證 |
| pooled corr | 使用者仍思考，不預先替換主指標或採納第5c節的選模規則 | 維持C-05待修，corr算法與選模政策尚待定案 |

### 空集合、NaN與「常數」的軸向

空集合可由NaN window filter丟光、segment太短或drop_last導致零batch；現行vali在沒有pred時回傳三個NaN。若選模分數不是有限數值，就無法合法比較checkpoint。NaN既不大於也不小於正常數值；不能把它當作「分數很差」或「沒有進步」而照常走成功流程。MSE／MAE NaN通常需查資料或數值；Pearson NaN還可能是常數序列的正常未定義狀態，應按事先固定的corr政策處理。

Pearson分母是兩組偏離各自平均值的平方和乘積之平方根。一組全為常數時，該組每項偏差皆為0，分母為0；沒有可供判斷的共同變化，因此不是corr=0或corr=1。即使真值與預測都同樣全為100，MSE=0但Pearson仍未定義。[SciPy pearsonr定義](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)

本專案vali的pooled corr沿pred陣列的**window軸**計算：固定h，取所有window的pred[:,h,0]和true[:,h,0]。所以「預測常數」是這個h在所有window都給相同值，不必是單次15分鐘曲線全平。例如A窗口[100,100,100]、B[110,110,110]、C[120,120,120]雖各自平坦，各h跨窗口仍有變異；反過來每個窗口都輸出[100,101,102]，各窗口曲線上升，但每個h跨窗口都是常數。第5c節anchored不改單次曲線corr，與此處pooled軸向不同，兩者不能混用。

### 共用test與rolling的安排

使用者提出先讓old test與drycut相同，再分train/val。這個順序可行，且與rolling相容：先固定不參與開發的test原始時段／forecast origins，再於較早dev部分做rolling-origin expanding folds。現行Data_Loader._load_split_file已將各fold的test固定取自base split；build_splits.assign_folds只處理train+val。兩份split各自的fold穩定test，並不代表old和drycut之間已共用test。

對兩個方法，val不在數學上強制完全相同；若要以val分數直接比較或在相近難度下選設定，建議共用val時間界限／起點。train可保留old與drycut各自選出的不同事件，因為那是研究要比較的訓練選樣。old同樣可以用rolling；例：fold1 train=A、val=B；fold2 train=A+B、val=C；fold3 train=A+B+C、val=D；三折全程保留同一T不評估，最後固定方案才評T。最終採哪個fold checkpoint、或按val選出的訓練長度在dev重訓，要在test前預定，不能用T挑最好的fold。

若直接共用drycut test，應固定它的**日期＋96分鐘輸入＋15分鐘目標**及共同前處理；不要把drycut segment_id複製到old split，兩者ID語意不同。old訓練必須排除共用test保留區域及跨界／重疊事件。若old CSV不含drycut需要的窗口，應從共同原始寬表建立獨立eval資料入口，或明確使用共同可用起點；不能靠重標split就創造缺失資料。前者比較「不同訓練選樣、同一評估方法」，各自原生coverage需另報。

本輪按最新無旗標配置（保留Past10Min／Past1Hr／Now及六個非恆定gate）重新只讀查核：現有test仍5700／4839個起點，共同4521。兩套base test原始列都從2025-07-18 10:10開始，但結束不同，內部起點也不同；可見同一開始日期還不足以視為同一test。drycut fold1 val列範圍2024-12-24～2025-03-05；old為2025-01-15～2025-03-15，確實不是同一期間。詳見 `evidence/split_discussion_2026-09-14.txt`。本輪未生成或覆寫split。

## 6. 附錄

### 覆蓋與執行限制

12支MUST-REVIEW均逐檔閱讀：Data_From_SQL_all、rebuild_gate_columns、build_drycut_segments_meta、build_training_csv_from_meta、build_splits、Data_Loader、Data_Factory、run、DLinearMix2、exp_Main2、exp_Basic、utils/metrics。T-01～T-33全數有判定；15份既有review Markdown全數閱讀；未審已棄用／外來程式。沒有前次review，未做虛構delta。

未執行`run.py`或任何sweep，未呼叫任何train函式、反向傳播或optimizer step，未讀寫checkpoint權重，未寫runs/test_results/checkpoints。`run.py`只作靜態文字解析與AST純函式探針。模型測試只CPU、eval、no_grad且使用合成零輸入。SQL重現、正式模型性能、硬體測試與完整文獻新穎性均明示未驗證。

所有新增腳本、fixture、日誌、報告都在本次資料夾；Python以`-B`停用bytecode寫入。未執行git commit/push。全域python與.venv缺pandas，因此使用現有`.HLML_Linear_venv/bin/python`（pandas3.0.1/numpy2.4.3/torch2.10.0），未安裝套件。大檔用chunksize或明確usecols讀取，未整檔載入大CSV。

### evidence/ 內容清單與重跑方式

工作目錄為repo根目錄；除docs audit用標準Python，其餘用`.HLML_Linear_venv/bin/python -B`。

| 檔案 | 內容／指令 |
|---|---|
| `audit_data.py` → `data.txt` | `.../python -B docs/review/reports/2026-09-13-r01/evidence/audit_data.py`；20 CSV streaming hash/正確列數、網格、NaN、meta重現、MSR、重疊及gate |
| `audit_code.py` → `code.txt` | 同前換audit_code.py；24個loader/fold、split位元組重現、scaler、val sampler、純函式反例及無梯度shape |
| `row_scaler_fixture.csv` | C-02最小反例；只在報告資料夾產生20列CSV |
| `audit_method.py` → `method.txt` | 評估origin母體、persistence、季節統計、pooled/isRain反例、只讀歷史metadata/array shape |
| `audit_docs.py` → `docs.txt` | core hashes/imports、15份review文件last commit、file:line probes、資料／license清冊 |
| `audit_supplement.py` → `supplement.txt` | valid window中實際stale輸入、target錨抽查、greedy split的安全替代解 |
| `run_inventory.json` | 125份run_args的可比較欄位、路徑、是否有輸出；不存帳密 |
| `cli_effects.txt` | CLI刪常數gate後計數不變；pred factory通道與scaler反例 |
| `commands.md` | run inventory／CLI effects的實際抽查命令及外部背景來源 |
| `verification.txt` | 首次完成時的schema、count、檔案範圍與git狀態核對；保留歷史版本 |
| `audit_completeness.py` → `completeness.txt` | 後續完整性複核；標準Python、串流hash比對12支核心與20個CSV、報告schema／連結、原始佐證完整性及git狀態 |
| `discussion_math_2026-09-14.py` → `discussion_math_2026-09-14.txt` | 5事件pooled反例精確拆解與forecast平移相關不變性的純算術驗證 |
| `discussion_verification_2026-09-14.txt`／`discussion_verification_2026-09-14_final.txt` | 09-14首次核對因待產生的輸出連結不存在而失敗，保留該紀錄；後者為最終重驗。沿用audit_completeness.py，不覆寫09-13紀錄 |
| `target_guard_tests_2026-09-14.txt` | C-01修正後24個回歸測試輸出；測試程式在repo的tests/test_target_input_guard.py |
| `split_discussion_2026-09-14.txt` | 無isRain/MSR配置下的test起點交集，以及各dataset／fold實際日期界限；只讀loader查核 |
| `verify_target_guard_fix_2026-09-14.py` → `target_guard_fix_verification_2026-09-14.txt` | C-01授權修正後的範圍、dataset／未修改核心hash、原始findings及新測試清單核對 |

`.source.sha256`是受審程式的來源指紋，不是本次計算的CSV內容hash；本次真正CSV hash位於data.txt。read-only probes產生的stdout已保存；statistical facts與待確認意圖分列，外部背景不取代程式／dataset的權威。

### 規範覆蓋索引（後續完整性複核補充）

下表表示要求已有對應處理與證據；「已覆蓋」含發現問題或明示未驗證，不表示驗收通過。D/C/M/R開始前的宣稱分別為8／8／7／7條。

| 規範要求 | 已完成的處理／證據 | 結論或界限 |
|---|---|---|
| D1 譜系、現行讀寫、孤兒／幽靈依賴 | 核心讀寫路徑與 `docs.txt` import清單；R-03 | SQL與CSV重建是兩條路徑；old meta仍是現行對照上游，不能依文件直接排除 |
| D2 至少8項統計 | `data.txt`：20 CSV清冊、分鐘網格、列欄數、NaN、段數／長度、旗標及MSR | 超過8項；差異見D-01、R-01、R-05 |
| D3 欄位語意 | `data.txt`重算meta、核心標記、MSR；`code.txt`合併等價探針 | 負開度物理意義待Q-D3；isRain非因果性見M-02 |
| D4 缺值與ffill | `data.txt`與 `supplement.txt` | 驗過段內限制、段首NaN、有效window及陳舊值實際輸入；D-02 |
| D5 隱性leakage | T-16、T-26、D-02、M-02／03 | 明列特徵不含邊界欄；事件標記與NaN選樣另有因果限制 |
| D6 歷史檔誤用 | 核心預設路徑檢索及R-03 | `run.py:265`仍預設歷史 `water_level_all.csv`；棄用sweep不視為現行流程 |
| D7 交付完整性 | `data.txt` inventory、`docs.txt`資料夾清冊 | 本機核心CSV齊全；SQL原始取數與物理metadata不在可驗證資料內 |
| C1 意圖一致性與OI-01～06 | 33條T表、第4節已知疑點處理 | OI-03撤回、OI-06修正經獨立核對；其餘依適用路徑判定 |
| C2 正確性與邊界 | `code.txt`、`cli_effects.txt`、`supplement.txt` | 24組loader/fold/split、兩份split重現、合成邊界及指標探針；不是任意錯誤輸入的窮盡測試 |
| C3 優化與可維護性 | C-09、三組最該補測試項目；重複window計數與gate合併交叉核對 | 記錄可重現性與契約缺口；未作效能benchmark，不因iterrows存在就臆測瓶頸 |
| M0 證據狀態 | `run_inventory.json`、`method.txt` | 有歷史／煙霧產物，沒有正式現行條件的完整比較 |
| M1 問題設定 | M-04、Q-M4 | forecast與warning成功判準須分清；業務門檻待作者 |
| M2 新穎性與可辯護性 | M-05、R-04及已列第一手背景來源 | 架構必要性缺對照；未作完整文獻新穎性認證 |
| M3 評估協議 | M-01、C-05／08、`method.txt` persistence與corr反例 | 不能只靠pooled corr或事後anchored支持形狀／預警效果 |
| M4 實驗設計 | M-03／05、共同origins與季節統計 | 36-run屬探索；同母體、多seed、事件群區間仍需正式研究驗證 |
| M5 三層防leakage | T-11／19／20、D/C/M交叉判定 | 現存partition時間不重疊成立；不涵蓋特徵、scaler或人工test選模 |
| M6 方向及roadmap | 第3節完整論述 | 前進；先評估契約與簡單對照，再判斷是否增加模型複雜度 |
| R1 數字與至少10處file:line | 第4節16列抽查表、`docs.txt`15組探針 | R-01／02／07；表格列數與probe分組數不同，不是漏失 |
| R2 內部一致性 | R-03／06及OI處理 | 文件範圍與實際依賴、protocol與template均有缺口 |
| R3 過度宣稱 | R-04／05／08及M-01 | 區分實作事實、模型能力推理與未查證物理說法 |
| R4 遺漏 | R-03／08、Q-R2及meeting核對 | 未把範圍外依賴默認通過；MSE／RMSE意圖待答 |
| R5 可用性與跨次比對 | R-06／07、首次delta、問題與假設表 | 首次無前報可比；本次追加清楚的完成狀態與變更歷程 |

### 12支MUST-REVIEW逐檔覆蓋

| 檔案 | 審查焦點與可追溯結果 |
|---|---|
| `Data_From_SQL_all.py` | 逐欄gate合併、5min邊界及讀寫依賴；T-04～08、D-02、R-03、merge_equivalence探針；未連SQL |
| `rebuild_gate_columns.py` | 本機重建與SQL組裝的合併結果一致性、clip、備份／dry-run程式閱讀；T-05～08；未執行就地重建 |
| `build_drycut_segments_meta.py` | rain_col、run-length、L/buffer與meta重現；D-01、T-09～11、M-02 |
| `build_training_csv_from_meta.py` | 先MSR後切段、isRain、ffill、overlap及指紋；D-02、T-12～16、C-09 |
| `build_splits.py` | window計數、連通群、blocked邊界、fold與位元組重現；T-18～23、C-06 |
| `data_provider/Data_Loader.py` | 欄序、train scaler、NaN mask、stride與predict；C-02／07、T-26～30 |
| `data_provider/Data_Factory.py` | split參數透傳、val sampler、pred dataset；C-04／07 |
| `run.py` | 靜態與AST純函式查核欄位展開、防護、constant、split、seed／manifest；C-01／03／09；未執行入口 |
| `models/DLinearMix2.py` | branch／exog契約、兩fusion、有無exog的CPU無梯度forward；T-33、R-04、M-05 |
| `exp/exp_Main2.py` | val/test、corr、persistence錨、segment metrics、checkpoint政策；C-05／08、M-01；未訓練或重載權重 |
| `exp/exp_Basic.py` | `_acquire_device`靜態閱讀：CUDA→MPS→CPU選擇與fallback訊息；本項未另發finding，未驗硬體執行或多GPU編號映射 |
| `utils/metrics.py` | CORR與validation零變異規則交叉反例；C-05 |

### 尚未完成的驗證與補齊條件

| 未驗證事項 | 原因／目前能說到哪裡 | 補齊條件 |
|---|---|---|
| SQL原始取數、水雨上游對齊與觀測真實性 | 本機CSV可核對；沒有SQL原始水雨資料；_4/_5現行依賴被既定範圍排除 | 提供可讀原始資料，並另行明確納入現行依賴；本次未連線 |
| 裝置、多GPU、AMP、梯度、收斂與checkpoint重載 | 只做CPU合成無梯度探針；未驗證訓練品質或硬體相容性 | 在另外授權的驗證工作中安排；本次禁止訓練仍有效 |
| 正式模型效益、delta／exog貢獻、統計顯著性 | 舊結果與煙霧不能替代正式比較；persistence數字不是模型優越性證據 | 裁決評估契約並修復相關問題後，另行授權正式實驗 |
| 任意外部meta／split及非法參數的完整拒絕行為 | 已驗現行產物與部分反例；未窮盡缺分鐘、duplicate ID、非法label、label_len>seq_len等組合 | 建立資料契約與輸入驗收測試；不能把本報告當成所有自訂資料都安全的證明 |
| 完整runtime依賴與輔助工具 | 依現行規範只審12支核心；R-03指出utils.tools/timefeatures、上游及推論／anchored輔助工具的覆蓋缺口 | 先明訂擴充清單，再做後續審查；本次未把它們默認通過 |
| 設備負值、單位、站位拓樸與預警業務需求 | 資料欄名不足以驗物理意圖；Q-D3、Q-R1、Q-M4待答 | 使用者確認並提供可追溯依據 |
| 完整文獻新穎性、實際上游授權版本及全git歷史秘密掃描 | 第一手背景查核與目前檔案檢索不足以支持這些完整結論 | 分別進行明確範圍的文獻、來源與歷史稽核；本次未作此認證 |

### 裁決、修復與下一次驗收的順序

1. **先處理研究有效性：** C-02的row scaler修復或阻擋；C-03／04／05的特徵選擇與評估契約；C-08的test隔離。使用者先答Q-C2、Q-M1可決定既有證據用途與M-02定級。
2. **固定資料與研究意圖：** Q-D1／D2／D3、Q-M2／M3／M4及Q-R1／R2，決定資料版本、可因果特徵、共同母體、正式指標與物理解讀；Q-C1用於歷史結果分類。
3. **修訂宣稱與可重現性：** D-01、C-09、R區塊文件錯誤與範圍缺口，讓實作與文件能用同一版本驗收。
4. **才進入正式方法驗證：** 先baseline與關鍵ablation，再多seed／事件群不確定性，最後判斷複雜架構是否必要。這是建議順序，尚未執行任何修復或訓練。

任何finding只有在後續取得修正與驗證證據後才可標「已解決」；此次補齊報告不等於解決24則finding。後續完整性核對可從repo根目錄執行：`python3 -B docs/review/reports/2026-09-13-r01/evidence/audit_completeness.py`；輸出另存於本次報告資料夾，避免覆寫首次佐證。
