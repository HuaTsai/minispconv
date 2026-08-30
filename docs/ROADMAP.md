# minispconv 開發路線圖

> 不含時間；以**版本演進**為主軸。程式碼全部自己寫 —— 這份文件是**規格 + 設計決策 + 驗收 + 讀物**。
> 本檔在 `docs/`，進版控。內文的 `geometry.h` / `indice.cu.h` / `reordering.cu.h` 指
> [traveller59/spconv v1.2.1](https://github.com/traveller59/spconv/tree/fad3000249d27ca918f2655ff73c41f39b0f3127/include/spconv)
> 的 `include/spconv/`（Apache-2.0，v1.2.1 是 branch 不是 tag，故連結 pin 在 commit）。

---

## 範圍決定：只做 3D

minispconv **只支援 3D sparse convolution**。這是刻意的範圍決定，不是未完成的功能：

- **spatial sparsity 是點雲與 voxel grid 的性質** —— 2D 影像本來就稠密，沒有 sparse conv 可發揮的餘地
- **4D 的使用者基數接近零**
- 泛型維度的代價是**測試矩陣、可讀性、API 複雜度**，而那個代價換來的覆蓋率沒有對應需求

明確的範圍宣告本身是差異化的一部分（對照上游只有 license header 的空 `docs/API.md`）。

**實作寫死 3D、oracle 保持泛型** —— 這個不對稱是刻意的：`S0.1` 的暴力 oracle 是純 Python、
慢無所謂，維持任意維度不花力氣，卻能用手算的 2D 小例驗證 3D 實作的邏輯。

實作端 3D-specific 的只有三處，很薄：**座標表示（`[N,4]` 的 b,z,y,x）、offset 表的產生、
hash key 的線性化**。未來要擴充是機械性工作，不是重寫。
用一個 `constexpr int kNDim = 3;` 集中，**不要讓 `3` 散落成魔術數字**。

---

## 主設計線：在 kernel 空間列舉，不在 output 空間

`geometry.h` 難讀且有 bug 的根因，是它**在 output 空間列舉候選**：

- 先算每維的 `[lower, upper]` 區間 → 三行擠了 ceil 慣用法與幾何量（`- 1 + stride` 不是幾何量，是整數 ceil 的配料）
- 在區間內以 `d` 為步長跳 → 需要 `counterSize`、`counter[]` 里程表進位、`numPoints`
- `s > 1 且 d > 1` 時步長假設失效 → **靜默算錯**（已用暴力法確認：只有 `s == 1` 或 `d == 1` 才正確）

**改在 kernel 空間列舉，上述全部消失**：

```
for k in 0 .. kernelVolume-1:          ← 純量迴圈，與維度無關
    (dz, dy, dx) = offset_table[k]     ← 表是資料；換維度只是換表
    對每軸算 num = pos + p - d_axis·dilation
    整除檢查（num % s == 0）+ 邊界檢查
    o_axis = num / s
```

- **不需要 `counter[]` 里程表**，也不需要 template 遞迴或 `if constexpr` 展開 ——
  外層是一維範圍，沒有巢狀要攤平
- **offset 表查表 = 零除法**。host 端算好 `kernelVolume × 3` 的表當 kernel 參數傳 →
  進 constant bank（`c[0x0][...]`），warp 內廣播讀取幾乎免費。
  那張表本來就需要（rulebook 要記 kernel offset），順手當查表用
- 整除檢查明確寫出 → `s>1 且 d>1` 不會靜默算錯
- **零 warp divergence**：每個 thread 都跑剛好 `kernelVolume` 圈。
  output 空間列舉的圈數依賴 `input_pos`，`s>1` 時 warp 會分岔
- 迴圈次數 `kernelVolume`（vs 上游的 `kernelVolume / s^3`）—— 但 **SubM 層 `s=1`，兩者相同**，
  而 SubM 是 spconv 網路裡最多的層

**這是 minispconv 對上游的第一個可辯護差異。** L1 就要落地，並寫進 design note。

---

## 硬性邊界（先定死，避免範圍蔓延）

| 打                              | 不打                                    |
| ------------------------------- | --------------------------------------- |
| **3D**、forward、FP32、Linux、單 GPU | backward（stretch，卡滿一週即棄）；**2D / 4D**（見「範圍決定」） |
| SubM + regular sparse conv      | FP16 Tensor Core 效能                   |
| 可讀性、測試、文件、Nsight 證據 | 多 arch wheel 覆蓋、autotuner、int8/FP8 |

**應變規則**：hash table 卡 > 2 週 → 改用 cuCollections 當依賴，火力集中 forward 優化。

**驗收數字**：

| 里程碑            | 標準                            |
| ----------------- | ------------------------------- |
| L1 forward        | 與 spconv 單層數值比對，誤差 ~0 |
| CUDA 優化 round 1 | benchmark vs spconv ≥ 1/10 速度 |
| CUDA 優化 round 2 | ≥ 1/5 速度                      |
| backward          | stretch，卡滿一週即棄           |

---

## Stage 0 — Oracle 與地基（純 Python，不碰 GPU）

目的：先有**唯一真理來源**，之後每一版都對著它驗。

| 版本   | 內容                                                                                                                    | 驗收                        |
| ------ | ----------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `S0.1` | 暴力 rulebook oracle：**維持 NDim 泛型**（純 Python，慢無所謂），對任意 `(ks,s,p,d)` 窮舉所有 `(o,k)` pair。實作端寫死 3D，oracle 不跟著寫死 | 手算的 2D 例子逐項吻合 |
| `S0.2` | 測試 fixtures：2D 小例（5×5 grid、4 個 active point、3×3 kernel，**手算可驗，只餵 oracle**）+ 3D 隨機點雲生成器（可控密度／shape／seed，餵實作） | fixture 可重現（固定 seed） |
| `S0.3` | **環境 gate**：獨立 pixi env 裝 spconv 並跑通一層 SubMConv3d                                                            | 能拿到 spconv 的輸出張量    |

**`S0.3` 的已查證細節**（避免踩雷）：

- `spconv-cu126` 2.3.8 有 `cp312` + `manylinux_2_28_x86_64` wheel，本機 python 3.12 / linux-64 相符
- 它拉 `cumm-cu126`，與 dev env 的 `cuda-toolkit 13.x` 衝突 → **必須開獨立 pixi feature/environment**
  （例如 `oracle = ["cu12", "oracle"]`），別污染 default env
- prebuilt arch 最高 sm_90；RTX 4060 Laptop 是 **sm_89**、AGX Orin 是 **sm_87**，兩張卡都在覆蓋範圍內
- **若卡住不阻塞後續**：`S0.1` 的 oracle 已足以驗證正確性，spconv 對拍延到 `L1.3` 再解；
  真的裝不起來就退回 CPU 版 spconv 或 docker

**讀**：

- `geometry.h` —— 讀完能回答「為什麼從 input 推 output，而不是相反」
- spconv `docs/spconv2_algo.pdf`（732 KB 投影片，repo 內唯一的演算法說明）

---

## Stage 1 — L1：PyTorch reference 實作

目的：**正確性優先，效能無所謂**。這一層之後永遠是 GPU kernel 的 oracle。

| 版本   | 內容                                                                            |
| ------ | ------------------------------------------------------------------------------- |
| `L1.0` | 座標 hash + **SubM rulebook**（`s=1, d=1` 起手，最單純）                        |
| `L1.1` | gather → GEMM → scatter-add forward（SubM 完整可跑）                            |
| `L1.2` | **regular sparse conv**（`s>1`，output active site 擴張、需要建 output 座標表） |
| `L1.3` | 與 spconv 單層數值對拍                                                          |
| `L1.4` | （選）inverse / transposed —— 只有走 CenterPoint 才需要                         |

**要做的設計決策**（沒有標準答案，值得先想清楚再寫）：

1. **rulebook 的資料佈局** —— `[kernelVolume, 2, N]`？還是按 offset 分組的扁平表 + 每組起點索引？
   決定因素在消費端：同一個 kernel offset 的所有 pair 必須連續，才能湊成一次 GEMM
2. **SubM 的 output-site 判定** —— dense grid 查表還是 dict？記憶體 vs 速度的取捨在 L2 會原樣重演
3. **hash key 的線性化方式** —— batch index 要不要併進 key？int32 上限 2^31 何時會溢位？
   （上游 #604 / #706 / #656 三個 open issue 全是這個。解法不是改 int64，是 **int32 + 明確 assert**）
4. **integer dtype** —— torch 2.12 的 `index_select` / `index_add_` / `gather` / `scatter_add_`
   **全部接受 int32**（已實測），所以 L1 可全程 int32，不需要像上游那樣中轉 int64
5. **API 形狀** —— `indice_key` 機制要不要留？rulebook 只依賴座標與幾何參數、不依賴 features，
   所以幾何相同的多層（例如一串 stride=1 的 SubM）可共用同一份 rulebook。**這條縫第一天就要留**

**驗收**：

- 對 `S0.1` oracle：所有 `(o,k)` pair 完全一致（含 `s>1`、`d>1` 的參數組合）
- 對 spconv：單層 forward 輸出誤差 ~0
- **不變量測試**：內部點（離邊界 ≥ `(ks-1)·d`）產生的 pair 數恆等於 kernel volume
- 把上游 issue #679 / #450 的 repro 納入測試集

**讀**：

- Graham & van der Maaten, _Submanifold Sparse Convolutional Networks_, arXiv:**1706.01307** §2–3
  （較完整版：arXiv:**1711.10275**，CVPR'18）—— 只讀 submanifold 的動機，別讀 segmentation 實驗
- `reordering.cu.h` 的 `gatherGenericKernel` / `scatterAddGenericKernel`
  （**先讀消費端**，才知道 rulebook 該長什麼樣；`Vec` / `VecBlock` / `batch*` 變體全部跳過，那是 L3 教材）
- `indice.cu.h` 的 `prepareIndicePairsKernel`（regular）與 `getSubMIndicePairsKernel`（SubM）
  —— 名字含 `Hash` 的先跳過
- SECOND (Yan et al. 2018, _Sensors_) 的 sparse conv 那節 —— 選讀，只為理解 spconv 的原始場景

---

## Stage 2 — L2：CUDA v0（正確性優先，先別優化）

| 版本   | 內容                                                                                                                   | 為什麼這個順序                         |
| ------ | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `L2.0` | **build 管線打通**：一個 trivial kernel 走完 `cpp_extension` → `torch.library.custom_op` → `register_fake` → `opcheck` | 先把工具鏈的坑一次踩完，之後只剩演算法 |
| `L2.1` | gather / scatter-add kernel（rulebook 仍由 L1 在 CPU 產生）                                                            | 最直白的 kernel，先建立信心            |
| `L2.2` | 端到端 GPU forward（中間 GEMM 直接呼叫 `torch.matmul` / cuBLAS）                                                       | 有可量測的 baseline 了                 |
| `L2.3` | rulebook 生成上 GPU（hash table）                                                                                      | 最難，留到最後                         |

**`L2.0` 必須一次做對的四件事**（做錯很難查）：

1. **stream** —— kernel 一定要 launch 在 `at::cuda::getCurrentCUDAStream()`。
   用預設 stream 會跟 PyTorch 的 op race；**小 workload 下測試會過**，等 benchmark 或 profiling 才炸
2. **dtype 分派** —— `AT_DISPATCH_FLOATING_TYPES_AND_HALF`
3. **tensor 進 kernel** —— `torch::Tensor` 是 host 端物件（refcount + virtual dispatch），進不了 kernel。
   用 `data_ptr<T>()` + 明確傳 shape（建議），或 `packed_accessor32`。
   本專案 rulebook shape 已知且簡單，裸指標更快也更好讀，Nsight 行號直接對回自己的 `.cu`
4. **fake tensor** —— `register_fake` 寫 meta 實作。**sparse conv 的 output size 是 data-dependent**，
   這裡必然要面對（上游把 `numActOut` 從 host 傳進來，代價是每層一次 D2H sync，
   也就是 `ops.py:939` 那個打斷 async pipeline 的已知效能問題 —— 同一個決策同時造成效能損失與 compile 不相容）

**要做的設計決策**：

1. **`kernelSize` 是編譯期常數還是 runtime 值** —— **這個決定會滲透整個 API，L1 動筆前就要定**。
   常數（`template <int KS>`）→ offset 表可進暫存器、迴圈可完全展開、任何除法被折成乘法位移；
   runtime → 一份 binary 通吃所有 kernel size。
   上游選 runtime，然後用 codegen 把組合爆炸炸回來（正是 §2 批評的那件事）。
   **建議的中間點**：3×3×3 走 template 特化，其餘走通用路徑，並在 design note 寫清楚為什麼。
   （NDim 已不是問題 —— 寫死 3D，見「範圍決定」）
2. **gather kernel 的 thread mapping** —— 一 thread 一個元素？一 thread 一個 feature vector？
   一個 warp 一個 feature vector？直接決定 coalescing 品質
3. **scatter-add 的衝突處理** —— `atomicAdd`？還是先按 output index 排序再 segment reduce？
   （上游 `batchScatterAddGenericKernel` 的註解自己在猜：`this may due to atomicAdd?`
   —— **把這個問號變成有數據的答案，就是 minispconv 的素材**）
4. **hash table 設計** —— open addressing + linear probing？cuckoo？table 大小與 load factor？
   注意應變規則：卡 > 2 週就換 cuCollections

**驗收**：

- 每個 kernel 對 L1 逐元素比對（`torch.allclose` 不夠，rulebook 要 exact match）
- `torch.library.opcheck` 全綠
- **property test**：隨機 sparse pattern × shape × channel 掃過，GPU 與 L1 結果一致
- CI 每 commit 跑測試（**上游的 CI 一個測試都不跑** —— 這是可驗證的對比）

**讀**：

- PyTorch custom op 官方教學（`torch.library.custom_op` / `register_fake` / `register_autograd`）
  —— 本機 torch 2.12.1 這些 API 全都在（已實測）。**這是 2019 年 spconv 沒趕上的班車**，
  當年要 pybind + `autograd.Function` + 自己處理 TorchScript，才留下 `// torch.jit only support int64` 那道疤
- **PMPP Ch 11 Prefix Sum (Scan)** —— rulebook 需要 exclusive scan 定位每個 offset 的寫入起點
- PMPP 的 atomics / histogram 章 —— scatter-add 的衝突處理直接相關
- ⚠️ **PMPP Ch 14 Sparse Matrix 跟這裡不是同一件事**：Ch 14 是 CSR/ELL 的 SpMV（矩陣元素稀疏），
  sparse conv 是 spatially sparse（空間位置稀疏、每個 active 位置的 feature 是稠密的）。優先度低
- `indice.cu.h` 的 hash 版 kernel（`*HashKernel`）—— 這時候才讀

---

## Stage 3 — L3：優化與量測

**做法沿用 cuda-warmup 的 ReduceSum V0→V5 模式**：每版一個假設 → 一份 `.ncu-rep` → 一段 design note。
**先寫下猜哪裡是瓶頸，再用 Nsight 驗** —— 猜錯的紀錄跟猜對的一樣有價值。

| 版本    | 內容                                                                                    |
| ------- | --------------------------------------------------------------------------------------- |
| `L3.0`  | benchmark 框架 + Nsight 流程（沿用 `scripts/ncu-row` 的做法）+ baseline 數字            |
| `L3.1+` | 逐版優化（下面是候選清單，不是固定順序 —— 哪個贏由量測決定）                            |
| `L3.d`  | determinism mode：rulebook 生成後 sort `(filter_offset, input_idx)`，並量化 sort 的代價 |

**優化候選清單**（上游 `reordering.cu.h` 的 `Generic` / `Vec` / `VecBlock` 三個變體，
就是同一個 gather 的三種向量化程度，等於把 L3 路徑寫在同一個檔裡供對照）：

- memory coalescing：thread ↔ channel 維度的對應方式
- ILP / grid-stride loop + unroll（上游的 `NumILP`）
- 向量化載入（`float4`）—— 需要 channel 對齊，注意小 channel（4/16）場景會失效
- shared memory staging / block-level tiling
- batched GEMM 減少 kernel launch 次數
- **Map 階段優化** —— 領域競爭焦點已從 GEMM 移到 rulebook 生成
  （TorchSparse++ 量到 mapping 佔總 runtime 最高 50%）

**Benchmark 公平性紀律**（不然一句被戳破）：

- 按官方指南選 algo：FP32 明確設 Native、FP16 用預設 implicit GEMM
- 明確控制 TF32 開關**並在報告揭露**
- 三組對照：spconv default / spconv Native / minispconv
- 呈現 channel sweep（4→256）× kernel size × 密度**矩陣**，不是單一數字
- **FP32 是公平戰場**（無 Tensor Core），FP16 誠實承認打不過

**驗收**：round 1 ≥ spconv 1/10 速度；round 2 ≥ 1/5。

**讀**：

- spconv `docs/PERFORMANCE_GUIDE.md`（官方自認的 footguns，逐字讀）
- TorchSparse++ arXiv:**2311.12862** §6.3（mapping 佔比 50%）
- Minuet arXiv:**2401.06145**（hash 的 L2 hit ratio 僅 19–36%，sort + binary search 做到 93%+）
  —— ⚠️ 它的端到端 baseline 是 ME / TorchSparse，**沒直接跑 spconv 2.x**，引用時必註明
- NVIDIA Lidar_AI_Solution 的 `3DSparseConvolution`（NVIDIA 自己重寫的版本）
- spconv-triton（2026-07 出現的社群重寫，production drop-in 定位）—— 確認與 minispconv 的教學定位不撞

---

## Stage 4 — Artifact（對外產出）

這是 portfolio 的實際交付物，**不是收尾雜務**：

- **每個 kernel 一頁 design note**：block 切法、memory access pattern 圖解、Nsight 佐證
  （上游 `docs/API.md` 是只有 license header 的空檔案 —— 這裡客觀超越很容易）
- **透明行為表格**：精度 × 演算法 × 硬體組合，明確標出不支援的參數組合。
  例如 `s>1 且 d>1`：minispconv 要嘛正確處理、要嘛 assert 擋掉，**不能靜默算錯**
- **README 效能表**：三組對照 + 完整 sweep 矩陣
- **開 issue**：`conv.py:339` 參數錯位、`select_by_index` 一呼叫必炸。
  repo 凍結大概率不理，但 issue 本身 = 「我真的讀了」的公開證據
- **blog** hook：「每月 12 萬+ 下載的庫，上游已沉默 19 個月」

---

## Stage 5 —（可棄）nuScenes + CenterPoint 端到端

只在 L3 走完後做。目標：swap 掉 spconv 層後 **mAP ≥ 原版 95%**。

前置成本要先睜眼看清楚：nuScenes 資料集下載、OpenPCDet 環境、模型權重，
以及 minispconv 必須補齊 CenterPoint 實際用到的所有層型（很可能包含 `L1.4` 的 inverse conv）。

**卡住就砍掉** —— Stage 0–4 的 artifact 完整性不依賴這一段。
