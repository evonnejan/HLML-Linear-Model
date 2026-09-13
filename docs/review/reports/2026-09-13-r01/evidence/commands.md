# 本次抽查命令

全部從 `/Users/zkc/Desktop/HLML/Linear` 執行。所有 `>` 指向本資料夾；不執行訓練。

```sh
.HLML_Linear_venv/bin/python -B docs/review/reports/2026-09-13-r01/evidence/audit_data.py > docs/review/reports/2026-09-13-r01/evidence/data.txt
.HLML_Linear_venv/bin/python -B docs/review/reports/2026-09-13-r01/evidence/audit_code.py > docs/review/reports/2026-09-13-r01/evidence/code.txt
.HLML_Linear_venv/bin/python -B docs/review/reports/2026-09-13-r01/evidence/audit_method.py > docs/review/reports/2026-09-13-r01/evidence/method.txt
python3 -B docs/review/reports/2026-09-13-r01/evidence/audit_docs.py > docs/review/reports/2026-09-13-r01/evidence/docs.txt
.HLML_Linear_venv/bin/python -B docs/review/reports/2026-09-13-r01/evidence/audit_supplement.py > docs/review/reports/2026-09-13-r01/evidence/supplement.txt
```

資料inventory初版直接計換行；交叉檢查pandas筆數後，發現兩份CSV最後無換行，已改為記錄newline_count與實際records並重跑。最終data.txt為更正後結果，沒有把自己的少算一列當資料錯誤。

## Run inventory（實際執行的Python）

```python
from pathlib import Path
import json, collections
out = Path('docs/review/reports/2026-09-13-r01/evidence/run_inventory.json')
rows = []
for root in ['runs', 'runs_sanity', 'runs_sanity_seg', 'checkpoints', 'test_results']:
    for p in Path(root).rglob('run_args.json'):
        d = json.loads(p.read_text())
        rows.append({'path': str(p), **{k: d.get(k) for k in [
            'model', 'data_path', 'split_mode', 'split_file', 'fold',
            'input_col', 'exog_col', 'run_id', 'seed', 'train_epochs'
        ]}, 'has_outputs': (p.parent/'outputs/pred.npy').exists()})
result = {'count': len(rows),
          'by_data': dict(collections.Counter(r['data_path'] for r in rows)),
          'rows': rows}
out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
```

## CLI effects（實際執行的Python，stdout → cli_effects.txt）

```python
import sys, contextlib, io
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import pandas as pd
from build_splits import count_windows_per_segment
from data_provider.Data_Factory import data_provider
from types import SimpleNamespace
for tag in ['drycut_L3h_buf60', 'old']:
    d = pd.read_csv('dataset/train_'+tag+'.csv')
    cols = ['HL01','HL02','HL03','HL04','HL05','HL06','min_since_rain',
            'Past10Min','Past1Hr','Now'] + [c for c in d
            if 'gate_opening' in c and c != 'north_gate_opening_4']
    n = count_windows_per_segment(d, seg_col='segment_id', check_cols=cols,
                                  need=111, stride=1)
    sp = pd.read_csv('dataset/splits_train_'+tag+'.csv')
    print(tag, 'after CLI constant-drop', int(n.n_windows.sum()),
          'split count mismatch segments', int(n.n_windows.ne(sp.n_windows).sum()))
a = SimpleNamespace(data='custom',model='DLinearMix2',root_path='dataset',
    data_path='train_drycut_L3h_buf60.csv',seq_len=96,label_len=30,pred_len=15,
    features='S',target='HL01',embed='timeF',freq='min',batch_size=64,
    stride_train=1,stride_eval=1,num_workers=0)
with contextlib.redirect_stdout(io.StringIO()):
    ds, dl = data_provider(a, 'pred')
x, *_ = next(iter(dl))
print('pred factory x_shape', list(x.shape),
      'scaler fitted rows', int(ds.scaler.n_samples_seen_))
```

## 靜態閱讀與範圍

逐檔 `cat` / `nl -ba` / `sed -n` 閱讀README、PROTOCOL、00–06 context、template、四block規範、spec與12支MUST-REVIEW；PROGRESS/roadmap/meeting僅作待驗證意圖背景。

```sh
git status --short
git log -5 --format='%h %ad %s' --date=iso
git log -1 --format='%H %aI'
git ls-files dataset
rg --files --hidden -g AGENTS.md
rg --files --hidden -g '*test*.py' -g '!Source_Code/**' -g '!*venv*/**' -g '!docs/review/reports/**'
rg -n --glob '*.py' --glob '!Source_Code/**' --glob '!docs/review/reports/**' 'x_raw'
rg -n '緯度|經度|座標|標高|站序|上游|下游|單位|公分|厘米' docs/review/context/00-overview.md docs/review/context/01-data.md Data_From_SQL_all.py data_provider/Data_Loader.py
rg --files dataset docs | rg -i 'station|location|site|coordinate|map|字典|測站'
```

未找到AGENTS.md；未讀外來參考碼實作。DB檢索僅輸出匹配數與脫敏AST欄位類型，未保存連線秘密。git ls-files dataset為空，確認dataset不在現行索引；不代表清查過全部git歷史。

## 外部背景查核（2026-09-13）

使用web工具查詢並開啟以下第一手來源；不取代專案的Tier 0，只用於Block M研究背景與Block R外部宣稱校正。

- [Zeng et al., LTSF-Linear原論文](https://arxiv.org/abs/2205.13504)：線性基線是已有工作；未由此推定本專案的15分鐘水位效果或新穎性結論。
- [HESS rainfall–runoff benchmark](https://hess.copernicus.org/articles/25/5517/2021/)：研究設計比較使用固定期間，對照LSTM與概念水文模型；不是本專案同任務的直接baseline規範。
- [Apache 2.0官方條款](https://www.apache.org/licenses/LICENSE-2.0)：核對§4(d) NOTICE義務有上游NOTICE前提；未確定本機Source_Code的實際上游revision，故不作法律合規判定。

搜尋結果中的二手blog、討論區及未開啟的候選文章未作論證依據。
