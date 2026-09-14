# 本次審查的變更紀錄

本紀錄適用於 `docs/review/reports/2026-09-13-r01/`。依使用者要求，後續每次變更都追加記錄：時間、檔案、改了什麼、原因、驗證與影響，並在對話中告知。不得用後來的修訂覆蓋既有紀錄。

原有範圍限制持續有效：只能寫入本次報告資料夾；不得執行訓練、git commit 或 git push。要求記錄變更，不視為授權修正受審程式或修改 `PROGRESS.md`。

**授權更新（005）：** 使用者於09-14第二輪討論明確要求先修改target防護；本項程式與必要回歸測試已獲授權寫入報告資料夾外。其他修改仍依原範圍，不執行訓練／commit／push。

## 001 · 首次完整審查（事後補記）

- **原始完成核對時間：** 2026-09-13T20:08:30+08:00，出處為 `evidence/verification.txt`。逐次編輯未保存獨立時間，不推測或補造。
- **受審 commit：** `fd2903764d6e6f4f6334d2ad71048d1293d40a24`。
- **變更範圍：** 新增以下16個檔案，全部位於本次報告資料夾。
- **原因：** 依 D → C → M → R 完成外部審查，保存可核對的報告與佐證。

| 新增檔案 | 用途 |
|---|---|
| `report.md` | 四個block、33條意圖對照、24則finding、11個未答問題及假設 |
| `evidence/audit_data.py` | 資料清冊、hash、分鐘網格、缺值、meta、MSR與重疊查核 |
| `evidence/data.txt` | 資料查核輸出 |
| `evidence/audit_code.py` | loader/scaler/split/validation與模型形狀的非訓練探針 |
| `evidence/code.txt` | 程式探針輸出 |
| `evidence/row_scaler_fixture.csv` | 20列合成資料，用於重現row模式scaler洩漏 |
| `evidence/audit_method.py` | 評估母體、persistence、非因果特徵與相關係數反例 |
| `evidence/method.txt` | 方法查核輸出與既有實驗產物的只讀核對 |
| `evidence/audit_docs.py` | 核心程式hash/import、文件清冊及file:line抽查 |
| `evidence/docs.txt` | 文件與範圍查核輸出 |
| `evidence/audit_supplement.py` | 有效輸入中的陳舊gate值、target錨與split可行性補驗 |
| `evidence/supplement.txt` | 補驗輸出 |
| `evidence/run_inventory.json` | 125份既有run_args的選定欄位與產物存在性清冊 |
| `evidence/cli_effects.txt` | 常數欄移除後的樣本數及pred資料入口查核 |
| `evidence/commands.md` | 實際抽查命令、可重跑片段與外部背景來源 |
| `evidence/verification.txt` | 原始完成時的schema、finding數量、核心hash與檔案範圍核對 |

### 審查過程中對上述新增檔案的修訂

1. **逐block更新 `report.md`：** 先寫待驗宣稱，完成後立即寫入findings與證據，再彙整摘要、意圖判定、問題及假設。
2. **更正CSV列數計法：** `audit_data.py`初版計換行數後扣header；發現兩份CSV最後無換行後，改為分別記錄換行數與實際列數，重跑 `data.txt`。正確值為gate wide 646,922列、gate opening 283,873列，並同步反映於報告R-01及 `commands.md`。
3. **更正檔名檢索輸出：** `audit_docs.py`曾以空切片輸出測站metadata候選清單；改成實際按檔名關鍵字篩選，再重跑 `docs.txt`。該清單只代表檔名檢索，不能證明所有檔案內容均無相關資訊。
4. **補強D-02證據：** 在 `audit_supplement.py` / `supplement.txt` 確認陳舊gate值確實進入有效window的輸入，再更新 `report.md`；drycut/old分別1,521/1,096個window，最大age下界559/90分鐘。
5. **補充C-06可行性反例：** 現行greedy切分會在存在安全替代邊界時報無解；補驗結果與影響寫入 `supplement.txt`、`report.md`，未誤認為現存split有時間重疊。
6. **完成報告文字與格式核對：** 補上schema固定欄位、證據索引、嚴重度統計、11個未答裁決及未確認假設，並修整部分繁簡用字。沒有把使用者未回答的問題當成同意。

### 原始驗證與結果

- 24則finding：1 Blocker、17 Major、6 Minor、0 Nit；Blocker限定於未指定 `segment_col` 的row模式scaler路徑。
- 33條T對照均有判定，11題裁決尚未獲答。
- 12支受審核心程式hash未變；git tracked diff為空，git status只有本次報告資料夾未追蹤。
- 原始mtime核對未發現本次資料夾外的workspace檔案改動；該核對排除Git內部與既有virtualenv，範圍詳見 `evidence/verification.txt`。
- 沒有修正程式或dataset，沒有修改 `PROGRESS.md` 或其他既有文件，沒有訓練／commit／push。

## 002 · 補上變更紀錄與入口

- **記錄時間：** 2026-09-13T20:11:17+08:00（本批變更開始前記錄）。
- **觸發要求：** 使用者「有任何的改動都要記錄下來，讓我知道」，接著要求「請繼續」。
- **新增：** `CHANGES.md`（本檔），補記首次審查新增檔案與過程修訂，建立後續追加紀錄方式。
- **修改：** `report.md`，僅在開頭新增本紀錄的連結。
- **原因：** 讓使用者集中查到改動內容、原因及驗證，並保留原有報告資料夾寫入限制。
- **影響：** 不變更finding內容、定級或問題答覆狀態；不重新執行資料查核或訓練。`evidence/verification.txt`保留為原始完成時的紀錄，不覆寫歷史驗證。
- **驗證：** 在記憶體移除新增連結後，`report.md`的SHA-256與本批變更前完全一致（`5c6b25aa890c4303e25de32805248198ad2cb683a555ff3c2a1e7137c844025a`），確認審查內容未變。資料夾現有17個檔案；HEAD未變，git tracked diff為空，git status仍只有本次報告資料夾未追蹤。

## 003 · 補齊完成狀態、覆蓋表與驗證界限

- **記錄時間：** 2026-09-13T20:30:29+08:00（本批變更開始前快照）；實際完成查核時間見 [completeness.txt](evidence/completeness.txt)。
- **觸發要求：** 使用者詢問是否全部review完成，要求繼續未完部分，並把 `report.md`／`CHANGES.md` 補充清楚完整。
- **原有缺口：** 四個block內容已完成，但完成範圍、受限驗證與待裁決狀態散落各節，容易把「完成review」誤解成全部驗證通過或問題已修復。

| 動作 | 檔案 | 改了什麼／原因 |
|---|---|---|
| 修改 | [report.md](report.md) | 摘要新增完成狀態表，區分已審、待裁決、未驗證與未修復；附錄補規範覆蓋索引、12支核心逐檔覆蓋、未驗證項目及補齊條件、後續處理順序；更新驗證與佐證索引 |
| 修改 | `CHANGES.md` | 追加本筆，記錄補充動機、具體檔案與查核方式，保留001／002歷史內容 |
| 新增 | [audit_completeness.py](evidence/audit_completeness.py) | 只讀完整性查核；保存本批變更前的15份佐證hash及24則finding正文hash，核對原始佐證、12支核心、20個CSV、報告schema與本機Markdown連結；不import專案或執行訓練 |
| 新增 | [completeness.txt](evidence/completeness.txt) | 保存上述查核的時間、結果與產物hash；另存新輸出，沒有覆寫首次verification |

### 本批驗證與影響

- **實際查核命令：** `python3 -B docs/review/reports/2026-09-13-r01/evidence/audit_completeness.py > docs/review/reports/2026-09-13-r01/evidence/completeness.txt`。
- **檢查範圍：** 20份CSV以1 MiB分塊hash比對首次資料佐證，沒有將大檔整檔載入記憶體；12支核心比對首次 `docs.txt` hash。報告逐則核對必要欄位、24則finding正文、33條T、11題Q及D/C/M/R事前宣稱數量。
- **裁決狀態：** 本次再次提出Q-D1、Q-C2、Q-M1三題，供使用者優先回覆；沒有將未回答改成同意，其餘問題仍列第5節。
- **結論影響：** 本批補的是可讀性、覆蓋與可核對性；24則finding正文與定級未變，未標記任何finding已解決。1 Blocker／17 Major／6 Minor／0 Nit的適用條件維持原文。
- **寫入範圍：** 僅上述4個報告資料夾內檔案；本批後共19個檔案。沒有修復受審程式、修改dataset或 `PROGRESS.md`，沒有訓練／commit／push。
- **驗證結果：** 查核PASS：原有15份佐證、12支核心與20份CSV的hash均未變；24則finding正文未變，33條T、11題Q與各block宣稱數均通過核對，git tracked diff為空。完成後另修整本批新增文字的三處繁體用字並更新核對輸出；`completeness.txt`保存本批最終結果。001／002中16與17個檔案的數字仍代表各自當時快照。

## 004 · 記錄09-14裁決、corr解釋與後續修正規格

- **記錄時間：** 2026-09-14T01:48:30+08:00（本批寫入前快照）；驗證執行時間另見本批輸出。
- **觸發要求：** 使用者逐點回覆八項嚴重問題，確認validation、target防護與test隔離方向，提出old／drycut旗標及pooled+anchored→delta的研究順序，並要求解釋相關計算。
- **本批前快照：** report SHA-256=`67c838999f1ad3060e52a6fba5ee639f27ddac04965d83beb00914304b620191`；CHANGES SHA-256=`e588fdc762dbdfab519f8d06c4f11c7c3f9f38f200a95902827739a7b21b2d4a`；資料夾19檔。

| 動作 | 檔案 | 改了什麼／原因 |
|---|---|---|
| 修改 | [report.md](report.md) | 新增09-14更新提示與第5c節：記錄已確認方向、尚未定案細節、三項修正位置及驗收條件；解釋scaler範圍、shuffle/drop_last、建議corr規則、pooled反例與anchored/delta差別；提出2×2及共同評估建議。把「均未獲答」改成原始狀態加最新回覆索引，保留原始11題 |
| 修改 | `CHANGES.md` | 追加004，保存本輪決策更新、檔案清單、驗證與寫入限制；001～003不改寫 |
| 新增 | [discussion_math_2026-09-14.py](evidence/discussion_math_2026-09-14.py) | 只用標準函式庫重現原始5事件反例、計算variance分解、驗證單一forecast加常數不改corr；沒有import專案 |
| 新增 | [discussion_math_2026-09-14.txt](evidence/discussion_math_2026-09-14.txt) | 純算術stdout；明示合成反例不是實測模型性能 |
| 新增 | [discussion_verification_2026-09-14.txt](evidence/discussion_verification_2026-09-14.txt) | 初次驗證紀錄；核心、dataset、finding/schema通過，但Markdown連結指向尚待產生的本檔，因此檔案存在檢查失敗；保留失敗輸出 |
| 新增 | `evidence/discussion_verification_2026-09-14_final.txt` | 連結目標已存在後的最終重驗輸出；使用既有audit_completeness.py，保留首次失敗與09-13佐證 |

### 裁決與影響

- C-04、C-01、C-08已取得未來修正方向，但本批尚未修改受審程式，沒有將finding標為已解決。
- Q-C1／C2的歷史動機／test使用情況仍未答；Q-M1／M3有旗標提案、Q-M2有母體差異共識，但完整矩陣與主比較目標尚未定案。原始11題都還有待答內容，不再描述為完全沒有收到任何回覆。
- pooled+anchored作近期方向、delta作後續候選已記錄；未宣稱它們保證高corr／低MSE。統一Pearson、固定可評horizon及退化候選處理屬審查者建議，尚未冒充使用者已採納。
- 因原始使用者明訂只能改報告資料夾，已提出兩個釐清：是否現在授權三項程式修正，以及矩陣是否採old/drycut×無旗標/MSR；等待回答，不因經過時間而放行依賴工作。
- 本輪新增第一手參考為PyTorch DataLoader與SciPy pearsonr官方說明，連結存於報告第5c節；僅支援API／數學語意，專案事實仍以程式與dataset為準。

### 驗證方式

1. `python3 -B docs/review/reports/2026-09-13-r01/evidence/discussion_math_2026-09-14.py`，stdout保存於同名txt。
2. 以標準Python subprocess執行既有 `audit_completeness.py`、capture_output=True，完成後才寫stdout。初次驗證因新增log的Markdown連結目標尚未建立而失敗；保存失敗輸出後，修正本紀錄與佐證索引，再重驗至 `_final.txt`。未變更驗證判準；最終輸出artifact清單包含初次失敗紀錄，不含最終輸出本身。
3. 24則finding正文保持原樣；定級仍1 Blocker／17 Major／6 Minor／0 Nit。33條T與11個原始Q保留，更新僅在狀態與討論節。最終PASS結果以 `discussion_verification_2026-09-14_final.txt` 為準。

**寫入範圍：** 僅本表6個報告資料夾內檔案；完成後共23檔。受審程式、dataset、PROGRESS.md均未改；沒有訓練、checkpoint存取、commit或push。

## 005 · 實作target防護，更新無旗標與共同評估討論

- **記錄時間：** 2026-09-14T02:18:07+08:00（修改前快照）。
- **明確授權：** 使用者「Target防護的方向已確認的話就先幫我修改一下程式碼」。本筆將C-01及必要測試視為已獲准的範圍外寫入；不再就同一項目要求確認。
- **修改前hash：** run.py=`5e77124257d673fa7211900d1d0be902388b18cfd7e4df91394ba73988ca98fb`；Data_Loader.py=`9b6c37a4782a9b6c9647e192f652901f6ada18b3c506ede8bab2bdc514fde9d2`；report.md=`f0a0e7bf7873c0162429eda8f9ef2c50ce2eae8269d3d0591e2b95d5052364fe`；CHANGES.md=`7ac916969cd31d6635577baf09e06086948feb7086a5d9ae685fe0bad4e54cdd`。

| 動作 | 檔案 | 內容及原因 |
|---|---|---|
| 修改 | [run.py](../../../../run.py) | 新增target防護；glob展開後、常數欄移除前檢查，configure直接呼叫亦拒絕target混入input/exog；保留CRLF |
| 修改 | [Data_Loader.py](../../../../data_provider/Data_Loader.py) | Dataset_Custom的mix模型在讀檔前驗input/exog，防直接API繞過CLI；target仍可作label／最後觀測錨；保留原本無結尾換行 |
| 新增 | [test_target_input_guard.py](../../../../tests/test_target_input_guard.py) | 24個不含訓練的回歸案例，驗target直寫／glob／exog／API與合法資料路徑 |
| 修改 | [report.md](report.md) | 加最新修正狀態及第5d節；C-01程式缺口標為已修，保留歷史finding；記兩旗標暫不用、row關閉建議、NaN與corr軸向、共同test及rolling設計。原始11題中本階段Q-M1／M3已裁決，另外9題仍有待答內容 |
| 修改 | `CHANGES.md` | 頂端追加本項授權更新，並追加005，列明所有改動與驗證 |
| 新增 | `evidence/target_guard_tests_2026-09-14.txt` | pytest輸出：24 passed in 1.28s |
| 新增 | `evidence/split_discussion_2026-09-14.txt` | 在不放isRain/MSR、保留原始雨量與非恆定gate時，以現行Dataset只讀核對test交集與各fold列時間範圍；未寫dataset |
| 新增 | [verify_target_guard_fix_2026-09-14.py](evidence/verify_target_guard_fix_2026-09-14.py) | 驗證只改兩支核心及新增必要測試、其他10核心與20CSV／原始15佐證不變；檢查原始24finding正文、33T、11原始Q、文件連結、diff與換行格式 |
| 新增 | `evidence/target_guard_fix_verification_2026-09-14.txt` | 上述驗證結果與實際tracked diff；新的程式版本hash，保留原始佐證 |

### 驗證與過程修整

- 執行 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .HLML_Linear_venv/bin/python -B -m pytest -p no:cacheprovider tests/test_target_input_guard.py -q`，已通過24案例；停用bytecode、pytest cache與外部plugin自動載入，不產生訓練輸出。
- 首次一般 `git diff --check`將run.py既有CRLF樣式的新行視為尾端空白；保留全檔CRLF，以命令區域的 `-c core.whitespace=cr-at-eol`重驗通過，未改Git設定。apply_patch曾加上Data_Loader結尾換行，已移除以保持原格式，最終diff只含功能變更。
- 只讀新配置查核仍得到drycut／old test起點5700／4839、共同4521；兩者rolling val日期不同。現有split未改，因此其n_windows仍是原配置統計，不冒充新矩陣已重建完成。
- 原始audit_code.py／audit_completeness.py保存舊版本判準，不能用其失敗推論新target防護有錯；不覆寫它們，另用本筆回歸測試與 `python3 -B docs/review/reports/2026-09-13-r01/evidence/verify_target_guard_fix_2026-09-14.py`核對新版本，stdout另存本筆verification。
- 本筆完成1項程式防護修復；原始統計24則（1 Blocker／17 Major／6 Minor）保留，尚未結案23則（1／16／6）。歷史HL01實驗分類仍待Q-C1；未把舊run自動改成合法或重跑。

**未執行：** row停用、val設定／NaN選模政策、test分期入口、矩陣腳本、dataset／split重產及PROGRESS.md修改。這些方向與本次明確要求的target修正分開記錄。沒有訓練、模型forward、checkpoint存取、commit或push。

## 006 · 評估split介面、window-wise corr與共用固定test

- **記錄時間：** 2026-09-14T02:49:26+08:00（第一筆新佐證生成時）。
- **觸發要求：** 使用者接受segment流程要求segment_col、詢問old是否需split_file，並要求保留未來row模式需求；詢問現有window-wise corr、評估分期改動量，主張val僅用於各run選checkpoint並請評估共同固定test。
- **本輪接手狀態：** HEAD已為 `1c87927b5989d5c747f4250897a087c95358d0d2`，git status為空；前輪target防護與報告已在這個既有commit內，summary也已增加split_mode／split_file／fold。這些是本輪開始前的repo狀態，本輪沒有commit；保留已存在的其他變更。

| 動作 | 檔案 | 內容／原因 |
|---|---|---|
| 修改 | [report.md](report.md) | 新增第5e節與頂端更新提示：切分單位與分派來源分開、split_file可選、明確保留未來row模式、old/builtin實際查核、目前沒有window-wise corr、分期與共同test的相對改動量、允許各run獨立val的共同test方案 |
| 修改 | `CHANGES.md` | 追加006，記錄接手版本、需求調整、查核與寫入範圍；001～005保留 |
| 新增 | `evidence/assessment_2026-09-14.json` | 記本輪初始HEAD、乾淨status、六支相關程式hash；按實際段界及現行70/10/rest規則重算builtin分派與跨partition重疊群 |
| 新增 | `evidence/assessment_refresh_2026-09-14.json` | 同輪外部HEAD／split更新後，按60分鐘輸入、15步預測、無兩旗標只讀重驗test數量、交集與新split hash |
| 新增 | `evidence/assessment_verification_2026-09-14.txt` | 核對外部版本變動、相關程式hash未變、原始finding正文保留、dataset只有兩份split不同及當前diff範圍；不聲稱同輪全部檔案均未變 |

### 已記錄的需求與判斷

- segment模式要求segment_col；未來仍需可明確選用的row模式，不再建議永久刪除row能力。分派來源可以是file或程式規則；不自動把漏填segment_col當成選擇row。C-02在未修train-only scaler前仍未結案。
- old／drycut與file／builtin是不同軸：兩套CSV均可走現有兩種分派路徑。old builtin本次111／15／33段，24個重疊群但恰好0群跨partition；沒有誤報成目前已跨界洩漏。builtin仍缺一般性防護，固定test或更改比例後需重驗。
- 現行主流程只有跨window的逐horizon與segment彙總corr；window-wise可由既有pred/true或points檔離線補算，不需訓練。本輪沒有新增指標實作，也沒有查看test新的性能分數。
- 使用者定位val為各run選checkpoint用途，因此不強制old/drycut共用val；要求各run內固定完整。共同test方案可行，主要新增獨立eval資料／origin定義與train scaler注入，不需改模型架構。
- 分期入口屬小至中型流程調整；完整共同test屬中型資料接線工作。這是靜態評估，沒有聲稱已有完整patch或保證工時。

### 查核與限制

- 先以 `rg`查corr呼叫、分組／axis，再讀exp_Main2、metrics、compute_anchored_mse及相關CLI／loader接線；關鍵位置存於第5e節。只為識別現有指標查看少數舊工具呼叫，沒有展開完整舊工具審查。
- pandas只讀train CSV的segment_id、SegmentStart、WinStart、WinEnd並去重段落；按現行builtin段數比例指派，對WinStart／WinEnd排序求重疊連通群，保存群數與跨partition數。未呼叫loader、模型或訓練，未產生split。
- 首次完整性核對在「HEAD與開始時相同」的assert中斷；查得期間出現既有工作之外的commit `1ce6302`，其訊息記seq_len改60並重產split。該commit亦納入本輪已寫的assessment.json與部分report內容；本agent沒有執行git commit，也沒有回退或覆寫該commit。
- 重新只讀查核：六支相關程式hash仍與本輪開始相同；20個CSV中只有兩份splits_train內容與原始審查不同。使用目前split、seq_len=60／pred_len=15與無兩旗標，loader test計數6240／5523均與split計數吻合，共同5198；沒有計算模型或baseline分數。已把第5e節的評估母體數字更新並清楚標出96分鐘舊快照，避免延用過期數字。
- 最終核對保留上述版本事件，以refresh記錄固定最新HEAD與split hash，確認程式及原始findings不變、文件連結與diff範圍；stdout存於本筆verification。未以舊HEAD的固定hash判準否定正常版本前進。

**本agent本輪共修改2檔、新增3檔，均在報告資料夾。** 中途外部commit已納入其中部分內容，故最終git status不代表本輪所有寫入的完整清單；以本表為準。本agent未改核心程式、測試、dataset、PROGRESS.md或Git設定；未執行訓練、模型推論、checkpoint存取、commit、push。原始finding定級與C-01修正狀態不變。

## 007 · 確認切分介面、釐清corr定義並暫緩共同test

- **記錄日期：** 2026-09-14。
- **觸發要求：** 使用者確認Segment／Row／外部分派方向，詢問現行pooled與segment corr算法，確認高corr且低MSE目標，並暫緩共用固定test。
- **接手版本：** HEAD=`3be09d28aae1357775acc8d0cdbe8a115cba6daf`，git status為空；既有commit非本輪操作。
- **修改前SHA-256：** report.md=`c346c928f88714479bf42e4bfce9443c173c890ecb4b74ba9019652ac41c2665`；CHANGES.md=`6cbf1277acf66646110a70c780b674d4733ff0f156c965ec18f2c9f947a55278`。

| 動作 | 檔案 | 內容／原因 |
|---|---|---|
| 修改 | [report.md](report.md) | 新增頂端第四輪裁決與第5f節：確認切分介面方向、精確記錄三種既有corr的軸向、對照尚未實作的window-wise、保留零變異差異及MSE互補解讀、將共用test安排改為暫緩；更新摘要的回覆索引與已裁決題數，避免與後文矛盾 |
| 修改 | `CHANGES.md` | 追加007，記錄本輪兩檔變更、查核範圍與未實作事項；保留001～006 |

### 查核與限制

- 直接閱讀 `utils/metrics.py:8-15`、`exp/exp_Main2.py:184-220,445-457,696-789`，確認全域逐horizon跨windows後平均、segment all攤平、segment逐horizon及零變異規則。佐證為上述版本程式，沒有新增性能數字或證據檔。
- 相關程式修改前SHA-256：exp/exp_Main2.py=`d518323065a5e018031c46191ccd1fde33252a45065c30a3e7afae20f43c5c3f`；utils/metrics.py=`592c57491903a19b3bc57105972fb11e1bd6cce27f361684f50b7461d8374c1b`。
- 本批驗證通過：git diff --check無格式錯誤、原始第1～4節及既有CHANGES 001～006前綴完整保留、兩支corr相關程式hash不變、git diff僅上述兩份文件且無未追蹤新檔。核對時HEAD仍為上述接手版本；不需要訓練或新增回歸測試。

**本輪僅修改上述2份報告文件，沒有新增檔案。** 新介面、corr規則統一、window-wise、開發／最終test分期均未在本輪實作；共同test明確暫緩。未修改核心程式、dataset、PROGRESS.md；未執行訓練、模型推論、checkpoint存取、commit或push。

## 008 · 審查run_matrix／collect_matrix並記錄主表裁決

- **記錄日期：** 2026-09-14。
- **觸發要求：** 使用者要求查兩支腳本的bug與需求符合度；審查中確認「以best_corr＋anchored為主，同時明列MSE」。
- **接手版本：** HEAD=`3be09d28aae1357775acc8d0cdbe8a115cba6daf`；report.md／CHANGES.md有第007筆未提交修改，本輪保留。未修改原始第1～4節finding或覆寫先前證據。

| 動作 | 檔案 | 內容／原因 |
|---|---|---|
| 修改 | [report.md](report.md) | 新增矩陣專項入口與第5g節，記通過項目、需修問題、與原C-03／C-04／C-05／C-08關聯及新主表裁決 |
| 修改 | `CHANGES.md` | 追加008，記本輪所有檔案、驗證過程與範圍；001～007保留 |
| 新增 | [matrix_review.md](matrix_review.md) | 先建立D→C→M→R查核清單，補實際資料統計、10則專項finding、需求符合表、已答Q-MX1及未驗證範圍。6 Major／4 Minor；不改原24則統計 |
| 新增 | [evidence/matrix_checks_2026-09-14.py](evidence/matrix_checks_2026-09-14.py) | 可重現只讀查核：實際manifest／CLI／sidecar／CSV窗口；全mock的執行狀態與合成合表、缺漏／NaN／標籤／shape反例。無模型或訓練呼叫 |
| 新增 | [evidence/matrix_checks_2026-09-14.json](evidence/matrix_checks_2026-09-14.json) | 保存Python／pandas／NumPy版本、15個輸入檔hash、24命令核對、各fold窗口數與所有合成probe結果；非模型性能 |

### 查核結果與過程修訂

- 本機Python3.14.3／pandas3.0.1；實際manifest duration_s為str。第一次mock execute在第222行寫float時拋TypeError，佐證JSON重導向當時只有空檔；沒有真實子程序或實驗產物被建立。這是新發現MX-C01，非訓練失敗。
- 更新probe捕捉該錯誤及最後保存狀態running，再以僅記憶體dtype轉object的fixture驗證其他成功路徑缺口；沒有修改受審manifest或程式。另一次證據腳本的patch因上下文不符未套用，隨即更正，沒有額外檔案改動。
- 補上完整collector main的記憶體驗證，攔截CSV寫入：完整情境144列long／48列summary／8列wide；缺checkpoint剩141列卻仍通過廣義3fold檢查；一fold的corr／MSE為NaN時summary仍報n_folds=3。
- 其他重現：改learning_rate後done仍復用；old錯配drycut split仍過preflight；常數horizon令collector corr=1而core≈0.5；錯誤run_args仍按manifest歸類；shape broadcasting、錯位window_idx及零baseline未被適當處理。
- 實際dataset用selected columns按75列窗口重算；同dataset的none／full test清單hash相同，fold test沿base固定。train／val可用窗口有差異，完整數量見專項報告；未取得任何新模型分數。
- 證據腳本最終完整重跑成功；內建斷言確認15個輸入檔probe前後hash不變。文件最後再核對diff格式、連結、finding數量、歷史正文與寫入範圍；沒有新增專案tests或訓練測試。

**本輪修改2檔、新增3檔，全部在本報告資料夾。** 未修改run_matrix.py、collect_matrix.py、run.py、其他核心程式、dataset、experiments/manifest.csv、PROGRESS.md；未執行訓練、推論、checkpoint讀寫、commit或push。發現的bug仍待修復，本輪沒有把「審查完成」寫成「程式已修好」。

## 009 · 修正validation完整取樣，詳解六項矩陣問題

- **記錄日期／接手版本：** 2026-09-14，HEAD=`3be09d28aae1357775acc8d0cdbe8a115cba6daf`。接手已有第007／008筆的未提交報告與佐證，全部保留。
- **授權與範圍：** 使用者明確要求先修改validation shuffle／drop_last；本項核心程式與必要回歸測試獲授權，其餘六項問題、主表MSE及dev／final分期本輪提供詳細解法，不擅自擴大實作。

| 動作 | 檔案 | 內容／原因 |
|---|---|---|
| 修改 | `data_provider/Data_Factory.py` | 一行條件由只判test改為val或test，讓val使用shuffle=False、drop_last=False；保留CRLF |
| 新增 | `tests/test_validation_loader.py` | 真實DataLoader配記憶體索引Dataset；5種val長度各重複3次，另驗train／test／pred，共8個測試；不讀dataset、不載模型、不訓練 |
| 修改 | [report.md](report.md) | 新增最新修正提示與第5h節、更新現況為C-01／C-04已修及其餘22則未結案；保留原始24則finding正文與歷史分級 |
| 修改 | [matrix_review.md](matrix_review.md) | 加後續修正提示，更新需求表及C-04現況；其餘10項專項finding保持未修，歷史證據保留 |
| 新增 | [matrix_remediation.md](matrix_remediation.md) | 詳述六項問題的例子、影響、修復步驟與驗收；澄清主表加MSE與dev／final分期，標示已實作與待實作範圍 |
| 新增 | [evidence/validation_loader_tests_2026-09-14.txt](evidence/validation_loader_tests_2026-09-14.txt) | 保存本次8個無訓練測試結果，8 passed in 1.94s |
| 修改 | `CHANGES.md` | 追加009，記錄本輪全部改動、程式hash及驗證；001～008保留 |

### 驗證與限制

- 命令：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .HLML_Linear_venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_validation_loader.py`；8 passed。使用-B並關閉pytest cache／plugin autoload，未新增cache或其他測試產物。
- Data_Factory修改前SHA-256=`639256123532f78d5eced63c636474d855379e945e788e69e347d414d056bac8`，修改後=`ee6bd5950e36a16733969014d8cbdc719e22140f3df2ce10141ec3f6114b12fb`。原matrix_checks JSON保留舊hash作歷史證據；沒有覆寫舊測試結果。
- 測試確認非空val小於batch仍產生一批、尾批不丟棄、每次樣本順序固定且完整；train保留隨機取樣與丟尾批，test保留完整評估，pred保留batch_size=1。
- 原C-04隨機丟樣本程式缺口已修；未驗證／未實作空dataset拒絕、NaN／inf與未定義corr的checkpoint防護、歷史checkpoint重選。既有vali先串接batch再計指標，不需要為尾批增加新的loss加權程式。
- 後續靜態核對包括一行程式diff、相關資料與矩陣程式hash、歷史finding正文及CHANGES前綴、文件連結與本輪寫入範圍；不執行完整matrix probe重寫歷史佐證。
- 預設git diff --check將原檔保留的CRLF之CR報為行尾空白；未為此改寫整檔換行。使用單次 `git -c core.whitespace=cr-at-eol diff --check` 通過，沒有修改Git設定。其餘hash、歷史正文、文件連結與檔案範圍核對通過。

**本輪修改4檔、新增3檔。** 報告資料夾外僅修改Data_Factory及新增其回歸測試；沒有修改兩支matrix腳本、run.py、exp_Main2、dataset、manifest或PROGRESS.md。未執行訓練、模型推論、checkpoint存取、commit或push。

## 010 · 複核使用者貼入修法，支持corr分階段處理

- **日期／版本：** 2026-09-14，HEAD=`3be09d28aae1357775acc8d0cdbe8a115cba6daf`；接手已有前輪Data_Factory、測試、報告及佐證的未提交修改，全部保留。
- **要求：** 審查附件六項修法是否可用，並評估先改collector、保留utils CORR。這次僅評估方案，沒有執行修復。
- **附件：** `/Users/zkc/.codex/attachments/90e81b5c-ef69-4376-b75c-36fc6edd955c/pasted-text.txt`；hash存於本輪JSON，附件未修改。

| 動作 | 檔案 | 內容／原因 |
|---|---|---|
| 修改 | [matrix_remediation.md](matrix_remediation.md) | 新增後續方案提示及逐項複核：schema文字空值、hash／plan缺項、split路徑與值契約、corr分期條件、metadata一致性、set與assert缺口；更正early stopping呼叫關係，記未答Q-P1 |
| 修改 | [report.md](report.md) | 新增頂端提示與第5i節，摘要方案可接受條件及#4分期範圍；原findings與修正狀態不變 |
| 修改 | `CHANGES.md` | 追加010，記本輪檔案、查核與限制，001～009保留 |
| 新增 | [evidence/proposal_checks_2026-09-14.py](evidence/proposal_checks_2026-09-14.py) | 只讀實際manifest／sidecar，加記憶體schema／StringIO round-trip、hash盲點、微尺度corr、set去重／assert反例及AST選模接線；不訓練、不寫正式產物 |
| 新增 | [evidence/proposal_checks_2026-09-14.json](evidence/proposal_checks_2026-09-14.json) | 保存上述結果、pandas版本、附件及14個受查檔hash；全為方案查核，無新模型性能 |

### 結果與限制

- 貼文Float64讀法能寫135.7並成功round-trip；全域空值會產生pd.NA文字。加上僅數值欄na_values的替代probe後重跑成功，選填文字保留空字串。
- 確認config hash不隨FOLDS／LOSSES變動，且CONST未含huber_beta等實際預設；sidecar正確配對的路徑原始字串不同，解析後才相同。Set掩蓋重複、optimize=1會移除assert也已重現。
- AST及直接讀碼證實現行early stopping以vali的np.corrcoef選模，非utils CORR；支持將collector規格先明確化，utils的legacy test／anchored輸出遷移另行處理。未將「分期同意」標成C-05已解決。
- 貼文std min=171.8／267.6沒有來源，已提Q-P1；目前未答，僅作未獨立驗證的單次觀察，不推論整批均不觸發。沒有為尋找對應結果而計算舊test的新性能。
- 兩次probe均完成，受查14檔在probe前後hash不變；本輪僅文件與佐證。最終另核對文件連結／diff、歷史正文及檔案範圍，保留前輪已授權的一行Data_Factory修改。

**本輪修改3檔、新增2檔，皆在報告資料夾。** 未修改任何核心程式／測試／dataset／manifest，未訓練、推論、讀寫checkpoint、commit或push。第009筆的validation修正仍是最近一次程式修正；六項方案與主表／dev-final工作仍待實作。
