# minispconv

## 語言慣例

**版控內的程式碼一律英文** —— identifier、docstring、行內註解、錯誤訊息、
commit message。錯誤訊息等同公開介面（會出現在使用者的 traceback）。

**設計論述用中文寫**，放在 `docs/design/`。這是專案的產品之一（`docs/ROADMAP.md`
通篇在論證「對上游可辯護的差異」），值得用母語寫到該有的密度。

## 設計論述放哪裡

判準是**會不會隨實作變動**：

| 文字 | 位置 | 語言 |
| --- | --- | --- |
| 為什麼這樣選、與 spconv 的差異、被否決的方案 | `docs/design/*.md` | 中文 |
| docstring：介面契約、參數語意、關鍵陷阱 | code | 英文，精簡 |
| 行內註解：這行為什麼長這樣 | code | 英文，一兩句 |

只要一段論述會因為改了迴圈而失效，它就該留在 code 裡，不進 design note。
反過來，會被單獨閱讀、且不隨實作改變的，搬進 `docs/design/`，code 裡留一行指標
（`See docs/design/xxx.md for the full rationale.`）。

搬移時**不要稀釋論述** —— 被否決的方案與其代價要一併記錄，那才是 design note
的價值所在。

## 改完程式要跑什麼

```bash
pixi run check
```

依序跑 `ruff format` → `ruff check --fix` → `pyright` → `pytest`。**順序有意義**：
format 會改動程式碼，先跑才不會讓 lint 去抓格式化之後才出現的問題。

寫成單一入口而不是三行指令，是為了讓「該跑什麼」只有一份定義 ——
之後加工具改 `pyproject.toml` 的 `[tool.pixi.tasks]`，這裡不用動。

## 其他

- `refs/` 是本地的參考資料夾（上游程式碼、開發日誌、探針），不進版控，需獨立備份
- `docs/` 只放已精煉的產品：`ROADMAP.md` 與 `design/*.md`。**版控內的任何檔案都不得引用 `refs/` 的路徑** —— clone 的人沒有那個資料夾
- 實作寫死 3D，oracle 保持 NDim 泛型（見 `docs/design/oracle.md`）
