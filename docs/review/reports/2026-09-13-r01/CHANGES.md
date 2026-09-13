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
