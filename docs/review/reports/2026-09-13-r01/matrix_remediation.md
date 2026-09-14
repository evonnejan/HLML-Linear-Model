# 矩陣六項問題的修復設計與validation修正 — 2026-09-14

本文件補充matrix_review.md，區分「已實作」與「建議設計」。使用者本輪明確要求立即修改validation shuffle／drop_last，其餘要求為解釋六項問題與解法；本輪僅實作validation取樣修正，沒有擴大修改矩陣、corr定義、主表或test分期。

**後續方案複核（2026-09-14）：** 使用者貼入六項修法，並傾向#4先只改collector、保留utils.metrics.CORR。逐項核對後支持分階段處理，但貼文有early stopping呼叫路徑錯誤，以及schema／hash／split／重複檢查缺口；詳細判定見文末「貼入修法的逐項複核」。這次只審方案與補紀錄，沒有實作貼文中的修法。

## 已實作：固定且完整的validation

`data_provider/Data_Factory.py:15`從 `if flag == 'test'` 改為 `if flag in ('val', 'test')`，讓validation使用shuffle=False、drop_last=False與args.batch_size。原檔CRLF保留，程式diff只有一行。Train、test、pred其餘行為不變。

以10個window、batch_size=4為例：以前val先洗牌，只評兩批共8個，遺漏的2個可隨epoch改變；現在每次按固定順序評4＋4＋2共10個。打亂的是window的取樣順序，不是window內每分鐘的順序。`Exp_Main.vali`原本將全部batch串接再計算MSE／corr，因此尾批不會被當成與完整batch等權的一個平均值。

新增 `tests/test_validation_loader.py`，使用真實PyTorch DataLoader與只在記憶體回傳索引的Dataset。5種val長度（1、3、4、5、11），每種連跑3次，核對無漏樣、無重複、固定順序與尾批；另外驗train隨機且drop_last=True、test保留尾批、pred batch_size=1。共8個測試通過，無模型、資料檔或訓練。輸出見 `evidence/validation_loader_tests_2026-09-14.txt`。

原C-04的隨機丟樣本程式缺口已修復；過去選出的checkpoint不會因此自動更新。這一行不處理空dataset、NaN／inf或未定義corr的checkpoint選擇，亦不停止每epoch test。那些仍需獨立驗收。

## 1. 耗時欄位型別使第一組結束後中斷（MX-C01）

**觸發流程：** plan在duration_s放空字串 → execute以keep_default_na=False讀CSV → 本機pandas3.0.1推論為str欄 → 子程序結束 → 嘗試寫浮點秒數 → TypeError。先前JSON的0.0是mock瞬間結束的值，真正訓練後寫例如135.7也有同樣型別衝突。

**影響：** 第一組已耗時完成，驅動器卻還沒把done、run_dir、finished持久保存就中斷。磁碟最後狀態仍running；重新執行會再跑這組，失敗組同樣走到duration賦值，所以「失敗繼續下一組」也無法保證。

**建議實作順序：**

1. 定義manifest欄位schema：fold為整數，duration_s為可空浮點數，狀態／路徑為字串，未有耗時用缺值而非文字。
2. 新plan與每次讀CSV都套用schema。對既有檔將duration_s的空字串轉缺值，再用 `pd.to_numeric(..., errors='raise').astype('Float64')`；只有new plan設dtype還不夠，CSV重新讀入會再次推論型別。不應把非空的非法耗時悄悄coerce掉。
3. 組內完成時先核對狀態及產物，再一起保存finished、duration_s、run_dir、status。寫同目錄暫存檔後replace，可降低中斷留下半份CSV的風險；恢復流程則核對已有產物再決定重跑。
4. 用mock成功、非零returncode與中途例外驗證兩組以上：首組完成後可前進下一組、CSV重新讀入仍正確、重啟會跳過有效done。

僅降低pandas版本可能暫時迴避現象，沒有修好manifest未定義schema的根因。本輪未實作此修正。

## 2. 改設定後續跑混用新舊結果（MX-C02）

**例子：** 已完成 `drycut__none__mse__f1`，使用learning_rate=0.001；後來把CONST改為0.002並續跑。該done仍保留，fold2／3用0.002，合表卻把它們當同一設定的三fold。換seed、seq_len、模型flags或覆寫同路徑CSV也有同類問題。

**建議：** plan建立不可靜默變動的matrix_id及設定快照，保存模型、輸入欄位、seq_len／pred_len／stride、loss、learning_rate、seed、early-stop規則、flags、資料／split hash與相關程式版本。每列保存完整resolved command及其run設定fingerprint；固定部分可放共同manifest metadata，不必重複24份。

run使用計畫中保存的設定；resume逐列核對該設定與run_args。若不符，明確拒絕沿用done，並建立新的matrix或新run身份，不把舊結果改標成新設定。只看git commit不夠，未提交程式與同名CSV也可能變；只在run_key加learning_rate亦不夠，會漏其他影響因素。

訓練身份與評估版本可分開：修改表格欄名不需要重訓；調整anchored或corr計算規則則應建立新的eval版本、保留舊輸出與同一checkpoint的對應，不能覆寫到看不出兩版差異。評估規則仍應在最終test比較前固定。

**驗收：** 同設定可resume；改任一影響因素或原地換CSV必被辨識；collector不能把不同fingerprint聚成同組。設定凍結後才有可靠的mean±std。

## 3. Sidecar只驗參數，未驗資料與split配對（MX-D01）

Sidecar目前是split旁的JSON，記錄seq_len、欄位等產生參數。兩份split本來就可能以完全相同參數生成，所以「參數相同」不代表「這份split是由這份資料生成」。實測將old指向drycut split仍過preflight；loader稍後可能因segment缺失擋下，但preflight的配對保證已不成立。

**建議三層核對：**

1. 來源：核對正規化data_path及CSV內容hash，避免同路徑被重產或複製錯檔；也記split CSV自身hash，使舊JSON配新CSV能被辨識。
2. 契約：核對segment欄名、唯一segment ID、所需fold欄、合法分派值及資料中的ID涵蓋；時間重疊群不可跨partition的規則另需保留，不可只看ID有出現。
3. 計數：明記split的n_windows基於哪組欄位、need及stride，必要時重算每run實際窗口數。none與full共用segment指派是可行的，無須因exog不同而強制生成不同split；但兩者NaN篩選後的train／val數量可能不同，不宜宣稱完全相同訓練樣本的純加欄對照。

目前兩份資料硬編碼配對本身正確；這項是缺少防錯保證，沒有證據表示當前24條命令已配錯。本輪不重產split。

## 4. 同份預測在合表與主流程算出不同corr（MX-C03）

假設預測2步，每步都有多個window可比較。第1步所有預測都一樣，因此其Pearson未定義；第2步預測與真值完全線性一致，corr=1。

- 主流程：分母加epsilon，第1步結果0，第2步約1；平均約0.5。
- 新合表：第1步記NaN，nanmean忽略它；只剩第2步，平均1。

兩份數字都可能出現在同一實驗的報表，讀者會以為合表改善了模型，實際是計算集合／未定義處理不同。把全部NaN直接補0並非統計定義上的通用修正。

**建議統一契約：** 一個共用函式計算逐horizon Pearson，明確指定axis、零變異及有限值規則；validation、test、anchored與collector都呼叫同一規則。真值或預測常數時建議回NaN，並記valid_horizons／total_horizons、constant_pred_count、constant_true_count。若只平均有效步，應顯示例如「corr=1，有效1/2步」，而非裸露的1。

「有限但常數」與「NaN／inf預測」要分開：前者MSE仍有意義、corr未定義；後者是輸入／數值失敗，應中止有效結果驗收。所有horizon皆未定義時不能宣稱有可用corr；以corr選checkpoint時應明確報錯或使用預先定義的處理，不能臨時悄悄換MSE。若要指定最低有效比例，需事先定義，而非看完test才決定。

**驗收：** 同arrays在各入口的corr、有效數及未定義原因一致；覆蓋全有效、部分常數、全常數及非有限案例。這是建議，當前corr規則本輪未改。

## 5. 合表使用錯誤因子或checkpoint標籤（MX-C04）

Manifest像實驗清單，run_args像產物實際履歷。現在collector直接信清單上的dataset／loss／fold，沒有核對履歷；若run_dir指錯，huber fold3的結果可被標成mse fold1，讀表時無法察覺。

Checkpoint還有獨立問題：上游outputs代表「主要選模指標的checkpoint」，不永遠是best_corr；outputs_alt代表另一個。當early_stop_metric=corr時目前對應正確；若是mse，兩個目錄的語意交換，新collector卻照樣寫死標籤。

**建議：** 收集每run前核對manifest與run_args的模型、dataset hash／split、fold、criterion、seed、欄位、seq_len及設定fingerprint。Checkpoint額外保存實際選模指標、檔案／hash及對應outputs子目錄，collector由此取名；若矩陣限定primary=corr，也可以遇到不符直接拒收，不能默默換標籤。

Target防護同時查展開後input_cols與exog_cols，使用顯式ValueError，避免assert被Python -O移除。這是collector對舊／錯指產物的驗收；現行run.py與loader的target防護已存在。

**驗收：** 故意交換兩個run_dir、更動fold／loss、primary改mse，必須拒收或按正確metadata顯示；不產生看似正常但標籤錯誤的列。

## 6. 缺漏、NaN或錯位結果被當成完整比較（MX-C05）

目前完整矩陣預期24個run × 2 checkpoint × 3 variant＝144列。若某run缺outputs_alt，collector警告後跳過，只剩141列；sanity僅按dataset／exog／loss看是否出現fold1～3。缺失fold的outputs仍在，所以這個粗檢查依然通過。整個配置消失時，它甚至不會出現在groupby中。

另有「列存在、指標無效」：一個fold的MSE／corr為NaN，groupby mean只平均其餘2fold，但n_folds=count仍寫3。甚至pred有3個window、true只有1個時，NumPy broadcasting可能把那1個真值重複套到3個預測，照樣算出數字。

**建議：**

1. 從凍結matrix列舉所有預期run×checkpoint×variant，與實際輸出作缺漏、重複及多餘項目比對。正式final模式缺必要產物即失敗；若要看部分進度，以明確partial模式輸出缺漏清單，不沿用完整報告的成功狀態。
2. 載入時要求pred／true／persist形狀完全相同、N>0、P與設定相符、target數正確且數值有限，禁止隱式broadcast修補。window_idx須恰覆蓋0..N−1，每window有唯一segment；若有origin／target_time應一起核對對齊。
3. Summary分開記expected_folds、present_folds、valid_corr_folds與valid_mse_folds。合法常數資料的corr未定義不能被改標成MSE失效；但不能只報有利子集平均而不揭露有效數與原因。各候選若有效horizon集合不同，也必須明列，不能當同等完整指標排名。
4. 先驗收後輸出正式表；保留manifest、缺漏原因、指標版本與evaluation母體身份，讓結果可追溯。

**驗收：** 缺一checkpoint、整組配置、重複fold、NaN／inf、空arrays、shape不符及window錯位都應被識別；完整情境仍產144列，且每組預期／存在／有效數可核對。

## 主表明列MSE到底是什麼意思

目前collector已算MSE；問題是不同輸出「秀了哪些欄位」，並非缺少計算。

| 輸出 | 現況 | 建議 |
|---|---|---|
| results_long.csv | 每run／checkpoint／variant都有corr、MSE等 | 保留完整明細 |
| summary_by_config.csv | 已有corr_mean／std、mse_mean／std | 保留，補有效數與缺漏狀態 |
| Console headline | best_corr＋anchored，只秀corr及相對persistence改善率，沒秀絕對MSE | 同列加mse_mean／std |
| results_wide.csv | best_corr＋anchored，各fold只有corr | 各fold同時列corr與MSE，或另提供MSE樞紐表 |

例如同一方法主表應能直接看到「corr=…，MSE=…，相對persistence改善=…」，不用另外打開完整CSV找MSE。改善率80%不代表絕對MSE小：baseline若為10000，改善80%後仍為2000；baseline100、改善50%後MSE是50。跨dataset的baseline可不同，更應同時呈現絕對數字。

best_corr指以val corr選的checkpoint；anchored指對該checkpoint的預測做既定平移校正。主表corr與MSE都來自這同一份anchored預測，不把best_corr的corr和best_mse的MSE拼成虛構方法。使用者已選這個主視角，best_mse可以留附表；主表加欄不需要重訓、改模型或新增loss。前提是有可用產物且指標契約／來源驗收先統一。本輪尚未修改主表。

## 「開發／CV只看train／val，最後才看test」的具體流程

現在driver的plan只是建立命令清單；run階段呼叫run.py後，每epoch算val與test，結束又評估兩個checkpoint。它的plan→run不是研究意義的dev→final隔離；collector只讀test npy，亦不是CV val報表。

建議分成：

1. **Dev／CV：** train更新權重，固定完整val決定checkpoint及早停。保存每fold的val指標；若要在開發階段比較anchored，應在固定val也保存必要pred／true／persist或直接算相同版本指標。原始val corr選best_corr是目前規則，不因要看anchored就自動改選模方式。
2. **凍結：** 固定候選矩陣、特徵、資料分派、early-stop／checkpoint選擇、anchored公式、corr規則、報表欄位與fold彙整方法。將它們寫入可核對的計畫。
3. **Final test：** 獨立入口載入指定checkpoint、該run的欄位及train scaler，在預先定義test評估，再交collector合表。單純run.py新增train_only=True不等同dev模式，因為現行train_only也改變資料分派，可能把全部資料當train，不能拿它當停用test的快捷鍵。

技術修改主要在exp_Main2.train停止建立／評估test、run.py將訓練和final入口解耦、driver提供明確phase與完成狀態、collector辨識val或test來源。獨立final入口需要復用訓練時的scaler與欄位，而不是在test重新fit。這比只刪一個print多，但不需重寫模型；本輪只解釋設計，未實作分期。

最終test可一次比較事先固定的多個方法、raw／anchored及預定checkpoint；不能看完test再據此修改方法而仍把同一test當完全未用過的驗證。共用固定test暫緩是跨old／drycut評估母體的另一項決定，不撤回dev／final分工。各dataset本身的fold test目前已固定；並非每fold都有一份不同test。

## 貼入修法的逐項複核 — 2026-09-14

來源：使用者附件pasted-text.txt，SHA-256=`2643b5b70c9904da4d5f90f9b0ce5f0d1e5ac93cc50d869a21e92c03183c8971`。本節是方案審查，不表示程式修復已套用；先前八項測試通過的validation修正保持有效。

### #1 型別與原子寫入：方向正確，必須定義文字空值契約

貼文的Float64 schema可解決duration_s賦值，且記憶體CSV round-trip保留135.7已驗證。但全域 `na_values=['']` 也把run_dir／error／finished變成pd.NA；目前下游有Path(run_dir)、error[:160]等操作。空路徑或空錯誤進入這些分支會TypeError，缺status的比較也可能產生「NA不能轉bool」；不是每筆合法pending都立即觸發，但介面已改，不能不處理。

建議保持選填文字欄為空字串，只對數值欄指定空值（以貼文MANIFEST_DTYPES為前提）：

```python
df = pd.read_csv(
    path,
    dtype=MANIFEST_DTYPES,
    keep_default_na=False,
    na_values={"fold": [""], "duration_s": [""]},
)
```

這個版本在本機驗過：duration_s可寫135.7、error／run_dir保留空字串。之後再驗證必填run_key／dataset／exog／loss／fold／status非空、fold合法且key唯一；done必須有非空路徑。使用nullable Int64不表示允許正式run缺fold。Status統計改用duration_s.notna()而非與空字串比較；plan、execute、status及collector都應使用同一reader。

同目錄tmp＋os.replace可用；保留建立父目錄，並明確只允許單一writer。同一個固定.csv.tmp不能支持兩個驅動器同時寫入；如要支援並行需鎖定manifest並使用獨有暫存檔。原子替換可避免讀到半截CSV，並不等於任意斷電的durability保證。別把這段擴張成尚未驗證的並行／斷電支援。

貼文稱「Blocker」可理解為阻擋啟動批次；沿原review嚴重度定義仍保留MX-C01為Major，避免將執行中斷誤稱已造成研究結果無效。

### #2 Hash與凍結：兩欄hash有幫助，但貼文payload仍不完整

目前payload包含CONST／FLAGS／EXOGS／DATASETS，沒有LOSSES／FOLDS；記憶體改folds由[1,2,3]為[1,2]，或移除huber，hash完全不變。Grid改變不一定要求重訓相同設定的既有run，但必須改變計畫身份／預期完整性，不能被當成同一份凍結plan。

此外huber_beta、dlinear_kernel_size、fusion_hidden_dim、lradj等實際訓練參數來自run.py預設，未在CONST中；改預設或模型程式也不改此hash。應保存resolved training設定及相關程式／資料身份，另保存完整grid快照。每run的身份包含該列dataset／exog／loss／fold，計畫身份包含預期列集合；不要只hash大字典而不保存可讀快照。

只檢查done還不夠：從plan到run之間pending也可能受現在的全域設定改變。Run須執行凍結參數或先明確拒絕配置漂移；collector亦從同一快照核對。貼文的(a)新manifest可接受；(b)改回pending須建立正確的新run身份與設定，重新核對同配置其他done，不能只重跑眼前一列而保留同組舊參數結果。

建議完整SHA-256留在稽核metadata，12字元僅作顯示短碼。舊manifest沒有hash時不能直接把當前hash貼上當成歷史證明，應明確遷移／標記未驗證。Hash資料檔需串流讀取，資料與split分開記錄較易定位。

訓練config與eval_version分開正確；eval_version不只進long，summary／wide及其metadata也應能追溯。不同評估版本拒絕混合或納入分組鍵，不可只留一欄卻被groupby聚掉。

### #3 Sidecar：三層架構可用，路徑與分派規則需要修正

現有sidecar.data_path是 `dataset/train_old.csv`，矩陣data_path是 `train_old.csv`；直接字串相等會誤擋正確配對。需按約定root解析成同一檔案後比較，再核對hash。完整資料hash與split自身hash在檔案寫完後生成；讀回驗證才接受。新增hash應綁定已查核的資料／split配對，不能對未知的舊產物直接補hash便宣稱生成譜系已證明。

貼文「split值合法(train/val/test/空)」要分欄處理：base `split`只允許train／val／test，不允許空值；`fold_k`允許train／val／空，空表示該fold未用的dev或base test列。Loader的test由base split決定，fold欄不該另寫test以改寫它。

除了set相等，split一個segment只能一列，需檢查重複ID與矛盾分派。重疊防護要檢查每個fold的有效train／val／base test分派，不能只驗base；時間順序與非空有效評估窗口仍應驗收。

貼文14369／14179屬舊PROGRESS快照。本次現行seq_len60的drycut fold1 train為16983／16721（none／full），old為16286／15710，見matrix_checks JSON。數字過期不推翻「none／full有效窗口可能不同」這個原則；目前同dataset的test none／full反而相同，不能泛稱所有partition都不同。

### #4 同意先只改collector，但修正理由與過渡規格

**貼文的重要事實錯誤：** 現行DLinearMix2主流程的early stopping沒有呼叫utils.metrics.CORR。`exp_Main2.py:184-220`的vali自行用np.corrcoef，跳過常數horizon；`:288-297`以vali_metrics選分數與checkpoint。utils CORR經metric在最終test／baseline報表使用，compute_anchored_mse.py也會呼叫它。只改utils CORR不會直接改變這條路徑的選模分數；若未來讓vali也改接共同函式，才要評估選模行為變化。其他歷史Exp類別不在本輪全面核對範圍。

**分階段建議仍合理：** 先改善collector的定義、有效性與可追溯性，另一次集中處理上游及歷史報表相容性。這次不是因utils CORR正被early stopping使用才必須保留，而是避免把不同入口的遷移併成未完整驗收的大修改。

第一階段最小契約：

- 保留collector已有的常數→NaN語意；新增per-horizon有效mask與原因，並將totals／counts寫入CSV。統一使用float64計算統計，先拒絕shape不符、空window與NaN／inf輸入；計算後也確認係數有限，不能只查std。
- 每run×checkpoint×variant分別記錄corr_definition／eval_version、n_valid_horizons／n_total_horizons、invalid_horizons及原因；raw、anchored、persistence分別查，不能只查raw。
- 常數horizon用warning或logging，不能把「assert」當成警告。AssertionError會中止，且assert在-O下可消失；真正需要拒絕的輸入用顯式ValueError。
- 部分有效可以保留MSE與有效horizon的corr平均，但必須標為partial_horizons；全部無效時corr=NaN、有效0/P，不做nanmean空集合。這與缺產物的partial report是兩種不同狀態。
- Fold summary不能只報有限平均值數；至少同時能看n_valid_corr_folds（有有限corr）與n_full_horizon_corr_folds（全部horizon有效）。某fold僅1/15有效時，不應被當成完整15步的corr與其他配置無條件並列排名。
- 上游Corr與collector新定義不混在同一欄／同一均值。保留歷史輸出，記清何者是legacy epsilon規則；本階段只能說差異已顯式管理，不能說C-05跨入口一致性已全部修復。

「std不為0」也不是兩算法必相等的充分條件。記憶體反例p=t=[0,1e-8,2e-8]，std≈8.16e-9>0；真正Pearson為1，但epsilon版本≈0.00019996，因分母加數主導結果。實際水位的正常尺度可能使此差異微小，仍應按明確數值容差理解相等，而非宣稱數學定義一致。

貼文std min=171.8／267.6未附run_dir／checkpoint／variant，現有報告無來源可核對，experiments/亦無預測產物。已於審查途中提問Q-P1。未獲答前僅視為使用者提供的單次觀察，不能升格為全部24 runs、兩checkpoint、三variant皆不觸發；這不阻擋同意分階段方案。

這項延期也不表示整批矩陣已可啟動：duration/schema等已確認執行缺口，以及另行確認的dev／final test分期仍需依各自範圍處理。上一輪validation shuffle／drop_last修正保持已完成。

### #5 來源與checkpoint標籤：接受方向，改用凍結metadata

以early_stop_metric推導primary／alt名稱符合目前Exp_Main.test接線，或對不符合矩陣primary=corr的run明確拒收，兩者皆可。仍應核對輸出真屬該checkpoint；只有run_args名稱不能證明手動搬動／舊outputs已被覆寫的產物身份，可保存checkpoint對應與outputs metadata。

貼文seed比對CONST[seed]應改為凍結plan.seed；dataset不能只靠檔名映射，使用已驗資料身份。exog_level不能只看exog_in是否大於0，因為不同欄位集合可有相同維度；需記預期展開欄位、已記錄的常數刪欄與實際欄位及順序。Args hash與plan hash要採相同canonical payload，不能拿完整run_args（含run_dir／時間戳）去比只含CONST的大字典。Train identity與run_args核對後才能接受manifest標籤。

### #6 完整性：預期網格正確，但set與assert不足

`actual=set(rows)`會把重複消掉。實測同一key兩列，missing=0、extra=0，仍有重複。必須先以duplicated或Counter檢查唯一性，再用set找缺／多項。

Expected需由凍結plan的實際run列及預定checkpoint／variant展開，不能重新用目前DATASETS／EXOGS／LOSSES／FOLDS拼格線；也不能只從done列推expected，否則未完成的run會被隱藏。Checkpoint期望需跟#5的primary／alt契約一致。

Shape檢查應用顯式例外；以optimize=1編譯貼文assert，故意不等shape仍通過已重現。另驗N>0、P、target數與有限性、window／segment一一對應。四種fold數按唯一fold計，不按列數；常數horizon所致有效性不足另依#4顯示。

正式模式缺必要產物報錯、--partial明確標示的方向可接受。正式與partial輸出應分開命名或帶狀態metadata，不能讓部分結果覆寫既有正式報告。相對baseline改善率仍需保留baseline=0的未定義處理；主表增加同一best_corr＋anchored的corr／MSE mean±std與有效數可直接採用。

### 本輪驗證與未答問題

佐證：[proposal_checks_2026-09-14.py](evidence/proposal_checks_2026-09-14.py)、[proposal_checks_2026-09-14.json](evidence/proposal_checks_2026-09-14.json)。讀實際manifest／sidecar、以StringIO驗schema與CSV round-trip、記憶體驗hash盲點／set去重／epsilon微尺度反例、AST確認vali選模接線。未呼叫run.py入口、模型、訓練或真實test評分，沒有把方案套到核心程式。

Q-P1（審查途中提出，待回覆）：std min=171.8／267.6是哪個run_dir、checkpoint及raw／anchored／persistence？答案只影響「目前未觸發」的證據範圍，不影響本輪對分階段修法的支持。
