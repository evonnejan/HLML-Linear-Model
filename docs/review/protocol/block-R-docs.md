# Block R — 審查文件本身

> 前置：已讀 `PROTOCOL.md`。
> 產出：`report.md` 的 Block R 段落，findings 編號 `R-NN`。
> **對象：`docs/review/` 底下的所有 `.md`（context / protocol / README / TEMPLATE / spec）。**

---

## 為什麼要有這個 block

`docs/review/context/` 是使用者委託撰寫的「受審宣稱」，**本身就可能有錯**。
已經發生過一次：先前的 `OI-03` 宣稱「`L >= 2*buffer` 沒有被程式強制」，
但該檢查其實存在於 `build_drycut_segments_meta.py:91-92`——撰寫時漏看了那六行。

所以這個 block 的任務是：**把審查文件本身也當成受審對象**，找出不合理、不一致、
過度宣稱或誤導之處，並**提出問題讓使用者裁決**。

## R1 事實錯誤

- `file:line` 是否真的指向它宣稱的程式碼？（抽查至少 10 處）
- 引用的數字是否與你在 Block D 驗算的結果一致？
- 有沒有像 OI-03 那樣「宣稱某個檢查不存在，但其實存在」的情況？反之亦然？

## R2 內部一致性

- `context/` 各檔之間有無矛盾？（例如 `01-data.md` 說某檔是歷史遺留，
  `04-code-map.md` 卻把讀它的程式列為現行核心）
- `05-traceability.md` 的 T-NN 與 `06-open-issues.md` 的 OI-NN 是否互相對得上？
- `PROTOCOL.md` 要求的東西，`TEMPLATE.md` 是否都有對應欄位？
- 分級（現行核心／一次性／已棄用）是否與程式碼的實際 import 關係相符？

## R3 過度宣稱與措辭

- 有沒有把「推斷」寫得像「事實」？`05-traceability.md` 要求標 `[有出處]` / `[推斷]`，
  抽查標註是否誠實。
- 嚴重度是否被誇大或淡化？（例如把潛在風險寫成現行 bug，或反過來）
- 有沒有為了讓自己好看而模糊處理的地方？

## R4 遺漏

- MUST-REVIEW 清單有沒有漏掉現行 pipeline 會用到的程式？
- `01-data.md` 的檔案總表有沒有漏檔，或把現行檔標成歷史遺留？
- `05-traceability.md` 有沒有漏掉重要的建模意圖？
  （對照 `meeting_recap.txt` 與 `docs/model_roadmap.md`，看有哪些明確表達過的要求沒被列進 T-NN）
- `06-open-issues.md` 有沒有迴避掉某些顯而易見的問題？

## R5 可用性

- 一個對本專案毫無認識的人，照 `README.md` 的閱讀順序走，
  能不能建立起足夠的判斷基礎？
- `PROTOCOL.md` 的指示有沒有模稜兩可、會導致不同審查者做出不同範圍的地方？
- `TEMPLATE.md` 的 schema 是否足以支撐跨次比對（ID 沿用、delta、方向判定）？

## 輸出方式

Block R 的 findings 用 `R-NN` 編號，**嚴重度照 `PROTOCOL.md` §4 的定義**
（例如「`file:line` 指錯 → 讓審查者查錯地方」通常是 Major）。

另外，**凡是你覺得「這個宣稱我沒辦法判定對錯，但它看起來可疑」的，
一律列進報告末尾的「給使用者的問題」清單**，用問句寫，讓使用者自己裁決。
這類項目不計入嚴重度統計。
