# Block D — 資料審查

> 前置：已讀 `PROTOCOL.md`（特別是 §0.5 執行紀律、§1 材料分級）與 `../context/01-data.md`。
> 產出：`report.md` 的 Block D 段落，findings 編號 `D-NN`。

---

## D0 先列出你要驗的東西

在報告裡先寫下你打算驗證的 5–10 個具體宣稱，再逐一執行。

## D1 資料譜系是否成立

`../context/01-data.md` §2 畫了一條 lineage。逐段確認：

- 每個「現行」檔案是否真的由它宣稱的程式產生？（讀該程式的輸出路徑）
- 中間產物的欄位是否真的流進下游？（比對欄名、列數量級）
- 有沒有**孤兒檔案**：被列為現行、但沒有任何現行程式讀它？
- 有沒有**幽靈依賴**：現行程式讀取的檔案沒有出現在總表裡？

## D2 宣稱的統計是否驗得出來

`../context/01-data.md` §3 與 `../context/03-evidence.md` §2 列了一批數字
（列數、欄數、時間範圍、NaN 率、段數、`isRain` 分布、`min_since_rain` 範圍）。
**抽驗至少 8 項**，用 `wc -l` / `awk` / pandas 分塊。不符即為 finding。

特別驗算：
- `all_minute_wide.csv` 是否真的是**連續 1 分鐘網格、無缺分鐘**
- 閘門任一欄 NaN 7.06%、訓練檔 ffill 後 3.48%
- `train_old.csv` 的 `isRain=True` 列數是否等於 `water_level_all2.csv` 的列數（交叉驗證）
- 段數 179 / 159，段長分布

## D3 欄位語意是否與程式一致

`../context/01-data.md` §4 宣稱了幾個「語意陷阱」。對照程式碼確認：

- `isRain` 是否真的是「核心/buffer 標記」而非降雨旗標？（`build_training_csv_from_meta.py:75-97`）
- `min_since_rain` 是否真的在**切段前**全域計算？（`build_training_csv_from_meta.py:53-73` 的呼叫順序）
- 降雨判定用的是 `Past1Hr` 還是 `Past10Min`？量化粒度 0.5mm 的說法成立嗎？
- 閘門負值 clip 到 0 的行為，在幾支程式裡是否一致？

## D4 缺值處理是否合理

- 段內 gate ffill（`build_training_csv_from_meta.py:99-108`）：ffill 跨不跨段？段首無值時怎麼辦？
- 殘餘 3.48% NaN 的 window 是否確實被丟棄，而非被 `fillna(0)` 混進訓練？
- `min_since_rain` 的 60–70 列 NaN 落在哪裡？影響多少 window？

## D5 隱性 leakage

- 有沒有任何欄位**編碼了切分結構**？（`isRain` 是已知的一個——確認還有沒有別的）
- `WinStart` / `WinEnd` / `SegmentStart` / `SegmentEnd` 這些欄位有沒有可能被誤當成特徵餵進模型？
- 有沒有使用了未來資訊的衍生欄（例如以整段統計量回填）？

## D6 歷史遺留檔是否被誤用

`../context/01-data.md` §3 把一批檔案標為「歷史遺留」。
grep 全 repo，確認**現行程式**沒有任何一支還在讀它們
（特別注意 `water_level_rain_gate_all.csv`——它與 `train_old.csv` 只差一個欄位，極易混用）。

## D7 交付完整性

`dataset/` 已 gitignore。確認：審查者手上的資料夾是否包含審查所需的全部檔案？
若有缺，在報告中列出缺哪些、影響哪些驗證。
