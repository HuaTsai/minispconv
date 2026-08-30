# Design note：rulebook oracle

> 對應 `src/minispconv/oracle.py`。這份文件只寫**不隨實作變動**的論述；
> 任何會因為改了迴圈而失效的說明，留在 code 裡當行內註解。

---

## 為什麼 oracle 在 package 內，不在 `tests/`

L1、L2、L3 的測試都要 import 它，而且它的身分是**規格**而非測試工具 ——
「`(o, k, i)` 該有哪些 pair」這個問題的答案就是這個檔案。放進 `tests/` 會暗示
它是可拋棄的輔助程式；它不是，它是專案的唯一真理來源。

---

## 三條刻意的設計原則

### 1. 維持 NDim 泛型

純 Python、慢無所謂；實作端寫死 3D。這個不對稱是刻意的：泛型讓**手算的 2D 小例**
能驗證 3D 實作的邏輯，而維持泛型在一個不追求效能的暴力實作裡幾乎不花力氣。

對應 `docs/ROADMAP.md` 的「範圍決定：只做 3D」。

### 2. 獨立的列舉路徑 —— oracle 的全部價值所在

| | 列舉空間 | 需要的推導 |
| --- | --- | --- |
| 實作端 | kernel 空間：`for k in range(K)` 反解 `o` | 整除檢查、ceil、邊界夾擠 |
| oracle | (output, kernel) 空間正向窮舉 | `i = o*s - p + k*d`，然後查 `i` 在不在 active 集合 |

正向公式**就是 conv 的定義本身**，不需要任何反解技巧。

**兩條路徑不共用任何一行推導。** 任何「讓 oracle 快一點」的重構，只要讓它靠近實作的
形狀，就是在銷毀這個價值 —— 兩個共用了推導的實作會一起錯，而且錯得一模一樣，測試全綠。

這條原則的具體後果：oracle 不准 import 實作端的任何東西，包含看起來無害的
offset table 產生器（見下方「已知的接縫」第 2 點）。

### 3. 只回傳數學上的 pair 集合，不回傳 rulebook 的實體佈局

佈局（`[K, 2, N]`？扁平表 + 每組起點索引？）是 L1 的效能決策、會改好幾次；
oracle 不該跟著改。實作端各自寫 `to_canonical()` 往上轉來比對。

---

## 分層：canonical form vs rulebook 佈局

這兩件事常被混為一談，但它們是不同的層：

| | oracle canonical | L1/L2 rulebook 佈局 |
| --- | --- | --- |
| 容器 | `frozenset[Pair]` | 按 kernel offset 分組的 tensor |
| 識別 | 絕對座標 `in_pos` / `out_pos` | active 表的序號 `in_idx` / `out_idx` |
| tap | tuple `(kz, ky, kx)` | 扁平 `k ∈ 0..K-1` |
| 順序 | 無 | 有 |
| 會不會改 | 不會 | 會，好幾次 |

轉換是**單向**的：`rulebook + 座標表 → canonical`。反過來做不到，因為 canonical
沒有序號資訊。這是對的方向 —— 比對只需要單向。

### `Pair` 的四個取捨

- **三個座標欄位缺一不可。** 少了 `tap`，gather → GEMM → scatter 就不知道這一筆
  該乘上權重張量的哪一片。
- **座標存絕對值，不存 active 表的序號。** 序號依賴 active 表的排序，而排序是實作
  細節（L1 和 L2 可能不同）。存座標 → 兩個實作只要算出同一件事就一定比得過。
- **`frozen=True` + 全 tuple 欄位 → 可 hash。** canonical form 的重點就是能丟進 `set`
  做集合相等比對；`list` 欄位或非 frozen 都會讓 `__hash__` 變成 `None`。
- **`order=True`** 讓它可排序，之後 `L3.d` 的 determinism mode 要驗「順序」時直接用。

### 為什麼是 `frozenset` 而不是 sorted list

曾考慮把「排序後的 pair 序列」當成正規形式，讓順序不對就算失敗。**否決。**

determinism 是 `L3.d` 才要買的**性質**，不是 rulebook 的**定義**。寫進正規形式等於
強迫 `L2.3` 在還沒開始優化時就付 sort 的代價，而那時連 rulebook 生成要花多久都還不
知道。`order=True` 這條縫留著就夠 —— 要驗順序時 sort 一次再比。

代價要誠實記下：**L2 的 atomicAdd 不定序不會被預設測試抓到**，得靠 `L3.d` 專屬的
測試補。這是知情的取捨，不是疏漏。

### 附帶好處：output 座標表被順便驗了

regular conv（`s > 1`）的 output 座標表是實作端自己建的，那張表**本身**也需要驗證。
因為 canonical 存的是絕對座標，實作只要少建一個 output site，pair 集合就會少一整批，
比對直接抓到 —— 不需要另外寫一個「output 座標表對不對」的測試。

---

## 已知的接縫（`to_canonical()` 尚未定案的細節）

1. **需要兩張座標表，不是一張。** `in_idx → in_pos` 查 input 表，`out_idx → out_pos`
   查 output 表。SubM 時兩者同一張，regular conv 時不同。傾向讓 rulebook 物件持有這
   兩個參照 —— `indice_key` 共用機制本來就要求 rulebook 記得自己是為哪組座標建的。

2. **tap 扁平化順序必須是單一真理來源。** oracle 的 `tap` 是 tuple，實作端的 `k` 是
   扁平整數，轉換靠 offset table。那張表 ROADMAP 規劃要當 CUDA constant bank 的查表用，
   **它必須只有一份**。若 oracle 用 `itertools.product` 隱式產生、實作端另外寫一份
   z-y-x row-major，兩邊某天會不一致，而症狀是「pair 集合對、tap 全錯」—— 很難看出來。

   注意這跟原則 2（不共用推導）不衝突：offset table 是**資料**，不是推導。共用一張表
   跟共用一套反解邏輯是兩回事。

3. **`batch` 從序號撈回來。** `Pair.batch` 在實作側是 `in_indices[in_idx][0]`。這確認了
   canonical 的資訊量比 rulebook 多，所以測試只能寫成「實作轉上來 == oracle」，
   不能反過來從 oracle 生 rulebook。
