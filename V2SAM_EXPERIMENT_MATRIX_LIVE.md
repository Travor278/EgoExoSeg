# V2-SAM 实验矩阵与复现进度（持续更新）

> 最后更新：2026-08-14 06:45（Asia/Shanghai）
> 状态：corrected Ego-only epoch 1、official Exo-init epoch 0 的完整 Ego2Exo FULL-V2-TEST，以及 DROID/BehaviorSim 上 B/Visual 与 C/Fusion 四项 Exo2Ego zero-shot full 均已完成并独立复算。Ego-only epoch 2 尚未进入正式训练：三次唯一 smoke 分别在 machine-output、validation-loop、DINO binding 问题修复后推进到真实 checkpoint resume，当前最后门禁已定位为审计器对内容完全相同的 MMEngine `HistoryBuffer` 做对象相等比较所产生的假阳性，等待 TDD 修复与独立复审。Seeta 容器重启使 Joint-natural epoch 1 在 iter 29,330 中断；最新可精确恢复的 accumulation 边界仍为 iter 28,000，而 recovery smoke 的执行证据仍缺独立信任根，尚未恢复。
> 本文是便于讨论的“活文档”；机器可验证的 JSONL ledger、run manifest、checkpoint SHA 和 raw metrics 才是最终证据。

## 1. 当前最重要的结论

1. 论文里的 **Single Expert 特指 Fusion Expert（专家 C）**，不是“任意一个单专家”。
2. **Multi-Experts 是 A（Anchor）+ B（Visual）+ C（Fusion）三套独立专家输出，再用 PCCS / Cycle-Points 逐对象选择**；公开仓库没有一个可直接下载的“ABC 总 checkpoint”。
3. 目前最可靠的官方单专家起点是方向专用 Fusion checkpoint：
   - Ego checkpoint 在 Ego2Exo PROBE200 上 per-pair IoU `50.18`；
   - Exo checkpoint 在 Exo2Ego PROBE200 上 per-pair IoU `51.36`；
   - Exo checkpoint 跨方向到 Ego2Exo 仍有 `36.47`，因此它是“从 Exo 迁移到 Ego”的合理初始化；
   - 反向跨域很差：Ego checkpoint 到 Exo2Ego 只有 `3.49`。
4. 已跑的旧 Ego-only LR `1e-6` 和旧 Joint LR `1e-6` **不能证明小学习率无效**。它们在构造函数内过早加载 checkpoint，随后 MMEngine 初始化覆盖了冻结的 DINO；这些结果已标为 `EARLYLOAD_INVALID`，排除学习率因果判断和模型选择。
5. 修正版使用 top-level `load_from`，并在第一个 forward 前逐张量验证冻结 DINO、专家权重和别名。代码已通过独立审查；新的 Ego-only 与 Joint 实验会重新回答“小 LR 是否有效”。

## 2. 阅读表格前必须统一的口径

### 2.1 方向

| 简写 | 含义 |
| --- | --- |
| Ego2Exo | 以第一视角图像/提示，在第三视角目标帧上分割对应对象 |
| Exo2Ego | 以第三视角图像/提示，在第一视角目标帧上分割对应对象 |

方向不是一个额外分类 head。数据集通过交换 query / target 图像和对应 mask 表达任务方向。

### 2.2 指标不能混排

| 指标 | 用途 | 能否与论文主表直接比较 |
| --- | --- | --- |
| paper mIoU | 论文协议下的主表指标 | 只能与同论文协议比较 |
| stock IoU | 旧公开 evaluator 的聚合 | 否 |
| per-object IoU | 每个对象等权后平均 | 否 |
| per-pair IoU | 每个图像 pair 等权后平均；当前 clean gate 的主指标 | 否 |
| PCCS / Cycle-Points | 无 GT 的多专家选择结果 | 只能与同 selector 比较 |
| GT-IoU / SaveMask oracle | 用 GT 选最佳专家的上界分析 | 永远不能进入正式排名 |

本文将 paper mIoU 与自建 stock / per-object / per-pair 分开列出。`TEST-PROBE200-V1` 只用于兼容性诊断，不能选择最终模型。

### 2.3 Exo2Ego `61.56` 指标专项复核

结论：`61.56` 的算术复算正确，但它是 **internal GATE per-pair IoU**，不是论文 mIoU，也不是独立 v2 test 成绩。禁止把它与论文 Fusion Exo2Ego `47.3` 放在同一排名中。

| 同一个 official Fusion Exo checkpoint | Scope / 聚合 | Exo2Ego IoU |
| --- | --- | ---: |
| 论文报告 | FULL-V2-TEST / paper mIoU | 47.30 |
| 本次官方 full 实测 | FULL-V2-TEST / self per-object | 46.38 |
| 本次官方 full 实测 | FULL-V2-TEST / self per-pair | 51.42 |
| corrected 初始化 gate | GATE128-V1 / stock legacy truncation | 51.28 |
| corrected 初始化 gate | GATE128-V1 / self per-object | 55.55 |
| corrected 初始化 gate | GATE128-V1 / self per-pair | 61.56 |

专项复核对 `204` pairs / `497` objects 的 raw metrics 重新计算得到：stock `51.2844`、per-object `55.5532`、per-pair `61.5636`，与 summary 完全一致；mismatch / missing / extra 均为零。per-pair 先在每个 pair 内平均对象，再让每个 pair 等权，因此该 gate 上比 per-object 高 `6.01` pp。单对象 pair 共 `55` 个且平均 IoU `80.92`，是这一聚合抬升的主要来源之一。

范围方面，历史名称 `GATE128-V1` 是按 group 选取后的标签，并不表示恰好 128 pairs；实际 Ego2Exo / Exo2Ego 是 `217 / 204` pairs。Exo gate 只含 `2` 个 take。它与本次 clean fine-tune train 在 pair ID 和 take group 上都无重叠，但这 `204/204` 条记录与官方 full-train Exo2Ego JSON 中的记录一致；公开方向配置使用 Exo2Ego FullTrain。由于官方 checkpoint 没有随附可核验的训练 manifest，只能作高置信推断而非密码学实证：official Exo checkpoint 很可能已经训练过这些样本。故该 gate 只能作为相同 scope 下的**保持能力 / 灾难遗忘探针**，不能作为未见数据泛化成绩。

推理代码审计未发现 target GT mask 内容进入模型预测：对象数来自 query prompt mask，target GT 只提供最终插值分辨率。因而当前没有“把 GT mask 像素喂给模型”的证据；高值的已确认来源是 scope 和聚合口径，而不是 GT 内容泄漏。

后续规则：GATE/VAL 的相对 delta 仍可用于安全早停和内部选点，但必须标为 `internal_retention_probe`、`paper_comparable=false`；最终候选只在预注册的 FULL-V2-TEST 上报告，并将 paper mIoU 与 self per-object/per-pair 分列。

### 2.4 状态标签

| 标签 | 含义 |
| --- | --- |
| `PAPER` | 论文报告值，未由本次运行重算 |
| `OFFICIAL_MEASURED` | 官方 checkpoint 在已锁定数据/代码上的实测值 |
| `REPRODUCED_VALID` | 本次复现且初始化、数据、评测、哈希合同均有效 |
| `EARLYLOAD_INVALID` | 实验确实运行过，但 checkpoint 初始化时序无效，不能用于选模或 LR 因果判断 |
| `IN_PROGRESS` | 已预注册，正在 smoke / gate / train |
| `BLOCKED` | 缺少满足正式合同的实现或证据，未生成伪结果 |

## 3. 数据集与本次数据构造

### 3.1 论文 Ego-Exo4D v2 数据

论文给的是约数：

| 方向 | Train pairs | Train masks | Train classes | Test pairs | Test masks | Test classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ego2Exo | 约 110K | 约 523K | 约 28 | 约 41K | 约 200K | 约 35 |
| Exo2Ego | 约 123K | 约 567K | 约 29 | 约 47K | 约 219K | 约 35 |

本地原始训练 JSON 的精确记录数为 Ego2Exo `110,118`、Exo2Ego `123,381`。

### 3.2 本次 Mini / clean grouped split

原 Mini 过滤后曾使用 Ego `87,135`、Exo `97,255`。为了避免训练与选模泄漏，当前正式实验改为按 take / video 整组留出：

| 方向 | Clean train | GATE128-V1 | VAL512-V1 | 说明 |
| --- | ---: | ---: | ---: | --- |
| Ego2Exo | 86,319 | 217 | 599 | 整组分割，因此会略超过目标 128 / 512 |
| Exo2Ego | 96,476 | 204 | 575 | pair 与 take/video group 均无交叉 |

其他固定评测 scope：

| Scope | Ego2Exo | Exo2Ego | 用途 |
| --- | ---: | ---: | --- |
| TEST-PROBE200-V1 | 200 pairs / 737 objects | 200 pairs / 728 objects | 内部回归诊断，不选模 |
| 官方 full test 实际 shard 计数 | 40,517 pairs / 100,223 objects | 46,515 pairs / 109,253 objects | 官方 Fusion 同方向 full 实测 |

## 4. Single Expert、Multi-Experts 与代码对应

### 4.1 专家定义

| 专家 | 输入/作用 | 是否训练 | 论文 Ego2Exo / Exo2Ego mIoU |
| --- | --- | --- | ---: |
| A Anchor | DINO 稀疏对应 + 原 SAM2 decoder | Anchor expert 不训练，复用原 SAM2 decoder | `38.7 / 41.6` |
| B Visual | visual prompt，训练 VPMatcher + decoder | 是 | `36.2 / 46.6` |
| C Fusion | 同时融合 anchor prompt 与 visual prompt，但只有一套独立 decoder | 是；这就是论文 Single Expert | `44.5 / 47.3` |
| A+B | 两专家候选 + PCCS | selector 无参数、无需训练 | `42.7 / 48.2` |
| A+B+C | 三专家候选 + PCCS | 三套专家分别存在；selector 无参数、无需训练 | `46.3 / 49.6` |

### 4.2 仓库与分支定位

| 代码来源 | 主要作用 | 不能误解成什么 |
| --- | --- | --- |
| 公开 V2-SAM main | `v2sam_visual`、`v2sam_fusion` 单专家训练/测试 | 不含完整 TripleDecoder / PCCS 正式实现 |
| 内部 V2-SAM-O main | 多个单专家与 DINO 实验目录 | 不是论文最终三专家推理包 |
| `feat/dinov3` | 增加 DINOv3 sparse correspondence | 不是多专家训练 |
| `feat/dinov3-done` | 无共同 merge-base 的项目快照 | 不能当作 main 上普通增量分支 |
| `feat/MOE` | dual-decoder / MOE 原型快照 | 不是论文最终 PCCS 结果 |
| `wrz` | Expert1/2/3、TripleDecoder、cycle / SaveMask 等最终项目包快照 | TripleDecoder 主要是推理组合器，不是“把三专家一起重新训练”的总模型 |

`wrz` 中 SaveMask 按 GT IoU 选最佳专家，只能作为 oracle 上界；论文正式结果对应不读 GT 的 PCCS / Cycle-Points。

## 5. 论文报告值（paper mIoU，独立参考）

| 组合 | Ego2Exo | Exo2Ego | Total / 备注 |
| --- | ---: | ---: | --- |
| A Anchor | 38.7 | 41.6 | 单专家 A |
| B Visual | 36.2 | 46.6 | 单专家 B |
| C Fusion / Single Expert | 44.5 | 47.3 | total 45.9 |
| A+B PCCS | 42.7 | 48.2 | 两专家 |
| A+B+C PCCS / Multi-Experts | 46.3 | 49.6 | total 48.0 |

论文训练补充：24 epoch、AdamW、LR `4e-5`、BF16、warmup 5% + cosine、报告 per-device batch 16、gradient accumulation 4。论文同时写 8 张 GPU 与 effective batch 64，数学表述存在不一致，复现时必须单独记录 `physical_bs × GPU × accumulation`，不能照抄一个模糊的“batch 64”。

## 6. 官方 checkpoint 实测

### 6.1 Fusion Expert（C）官方 full，同方向

| Checkpoint → 评测方向 | Scope | stock IoU | per-object IoU | per-pair IoU | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| official Fusion Ego → Ego2Exo | FULL-TEST-V2 | 44.65 | 40.24 | 44.93 | `OFFICIAL_MEASURED` |
| official Fusion Exo → Exo2Ego | FULL-TEST-V2 | 49.04 | 46.38 | 51.42 | `OFFICIAL_MEASURED` |

注意：这些是自建三种聚合，不应写成论文 mIoU。

### 6.2 Fusion Expert（C）2×2 PROBE200

| 初始化 checkpoint → 评测方向 | stock IoU | per-object IoU | per-pair IoU | 解读 |
| --- | ---: | ---: | ---: | --- |
| Exo ckpt → Ego2Exo | 35.86 | 29.29 | 36.47 | 当前 Exo→Ego 迁移起点 |
| Exo ckpt → Exo2Ego | 54.26 | 46.00 | 51.36 | Exo 同方向强基线 |
| Ego ckpt → Ego2Exo | 51.96 | 45.68 | 50.18 | Ego 同方向强基线 |
| Ego ckpt → Exo2Ego | 4.14 | 4.17 | 3.49 | 强方向专化，反向几乎失效 |

### 6.3 Fusion Expert（C）2×2 VAL512（正式内部基线）

Run `OFFICIAL-FUSION-C-VAL512-V1-8f56d32-20260812T220500Z-RETRY6` 已完成。四个 job 均通过 exact pair/object accounting、external-DINO closure、data-only overlay semantic gate、strict receipt、artifact SHA 与 terminal manifest 审计；这些仍是 `internal_retention_probe / paper_comparable=false`。

| Checkpoint → VAL | pairs / objects | stock IoU | per-object IoU | per-pair IoU |
| --- | ---: | ---: | ---: | ---: |
| Ego → Ego2Exo | 599 / 995 | 86.4518 | 82.8429 | 84.8519 |
| Ego → Exo2Ego | 575 / 993 | 28.2629 | 23.3114 | 23.7923 |
| Exo → Ego2Exo | 599 / 995 | 55.2130 | 49.6047 | 53.7748 |
| Exo → Exo2Ego | 575 / 993 | 78.7061 | 72.5813 | 75.4179 |

全部 mismatch/missing/extra=`0/0/0`。run manifest SHA256 `b6b773ef6292beede5392c91854a704e0cb7546257b47459c90fb4f54dadd8e4`，post-run audit SHA256 `02f3bbb6bb3f6f502590ba97f34dcb0fbd76951abc2821a44620fdbbb8e77411`。外部 monitor 在 job 4 manifest 已落盘且 evaluator 刚退出的瞬间记录了一条生命周期 race `FATAL`；随后用同一已核哈希审计代码只读复核 job 4 与 terminal manifest 均通过，未修改 run。该 monitor 告警保留为运维证据，不否定四个已原子完成的 job。

### 6.4 Visual Expert（B）2×2 PROBE200

| 初始化 checkpoint → 评测方向 | stock IoU | per-object IoU | per-pair IoU |
| --- | ---: | ---: | ---: |
| Visual Exo → Ego2Exo | 16.96 | 17.21 | 21.30 |
| Visual Exo → Exo2Ego | 38.05 | 29.50 | 36.00 |
| Visual Ego → Ego2Exo | 31.84 | 24.08 | 28.09 |
| Visual Ego → Exo2Ego | 4.91 | 3.44 | 4.38 |

结论：当前公开 Fusion 明显强于 Visual；官方权重也显示强烈方向专化。

## 7. 已跑训练实验

### 7.1 历史 Joint LR `1e-5` epoch 1

| 训练 | 方向 | stock IoU | per-object IoU | per-pair IoU | 当前用途 |
| --- | --- | ---: | ---: | ---: | --- |
| Exo-init，Ego+Exo natural concat，LR `1e-5`，epoch 1 | Ego2Exo PROBE200 | 26.45 | 19.05 | 24.18 | 历史诊断，不作为当前最佳候选 |
| 同上 | Exo2Ego PROBE200 | 27.21 | 22.80 | 30.52 | 历史诊断，不作为当前最佳候选 |

该运行使用旧 checkpoint 加载流程；在完成同等初始化时序复算前，不把下降归因于 LR 或 joint training 本身。

### 7.2 Ego-only LR `1e-6` epoch 1（已判无效）

训练条件：official Fusion Exo 初始化；只训练 Ego2Exo Mini；physical batch 4、accumulation 16、effective batch 64；LR `1e-6`；contrastive boost 关闭；epoch 1 / iter 21,792。

| 评测方向 | stock IoU | per-object IoU | per-pair IoU | 相对相同 official Exo-init PROBE 基线 |
| --- | ---: | ---: | ---: | --- |
| Ego2Exo | 21.74 | 12.21 | 18.07 | per-pair `-18.40` pp |
| Exo2Ego | 13.09 | 6.46 | 14.83 | per-pair `-36.53` pp |

状态：`EARLYLOAD_INVALID`。冻结 DINO 相对官方权重发生了 `303,128,589 / 303,227,920` 个元素变化，relative L2 `1.01748`；DINO 不在 optimizer 中，因此这不是正常训练更新，而是初始化时序错误。禁止续 epoch 2/3，也禁止据此断言 LR `1e-6` 太大或太小。

### 7.3 Joint-natural LR `1e-6` clean gate（已判无效）

训练条件：official Fusion Exo 初始化；clean grouped Ego `86,319` + Exo `96,476`，natural ratio `47.22% : 52.78%`；batch 可混方向；共享模型、共享五项 loss；physical batch 4、accumulation 16；LR `1e-6`；boost 关闭。

| Checkpoint | Ego2Exo per-pair IoU | Exo2Ego per-pair IoU | 决策 |
| --- | ---: | ---: | --- |
| direct init / iter 0 | 24.71 | 61.56 | 有效的直接 checkpoint internal-retention GATE 基线；非 paper mIoU |
| iter 500 | 8.78 | 52.27 | 旧 runner 仅诊断并继续 |
| iter 2000 | 10.12 | 50.76 | both below initialization，自动停止 |

状态：iter 500/2000 均为 `EARLYLOAD_INVALID`，排除选模和 LR 因果；iter 0 是未训练的 direct checkpoint 基线，仍可保留作同 scope 对照。

### 7.4 同事曾报告的 Ego stock epoch 序列

| Epoch | Ego stock IoU | 证据级别 |
| ---: | ---: | --- |
| 1 | 25.74 | `colleague_reported / NEEDS_EVIDENCE` |
| 2 | 29.45 | `colleague_reported / NEEDS_EVIDENCE` |
| 3 | 30.83 | `colleague_reported / NEEDS_EVIDENCE` |
| 4 | 30.87 | `colleague_reported / NEEDS_EVIDENCE` |

目前只有 guide 中的文字值，缺原始 code commit、config、data manifest、checkpoint SHA 和 summary，因此不与本次结果做正式排名。

截至 `2026-08-12 22:09 CST`，corrected Ego-only 尚在 epoch 1（`15,400 / 21,584`），Joint-natural 尚在 epoch 1（`10,920 / 45,712`），均未生成 epoch-1 checkpoint 或 epoch-1 VAL。中途 internal GATE 的 Ego stock 为：Ego-only iter 2000 `29.78`、Joint iter 2000 `22.16`。Ego-only 的 `29.78` 数值上高于同事 epoch 1 的 `25.74` 且接近其 epoch 2 的 `29.45`，但前者是 `217`-pair training-domain GATE，后者据 guide 指向 `ego2exo_test.json`；在同事原始证据和本次 FULL-V2-TEST stock 均补齐前，不得据此宣称超过。

## 8. 正在进行的 corrected POSTINIT 实验

### 8.0 `POSTINIT` 的含义

`POSTINIT` 不是推理后处理，也不是一种新模型，而是 checkpoint 的**加载时序合同**：旧 `EARLYLOAD` 在模型构造函数内先加载官方 checkpoint，随后 MMEngine 的 `init_weights()` 又覆盖了部分已加载参数；这些参数中的 DINO 随后被冻结，训练也无法把它修回。`POSTINIT` 则先完成全部框架初始化，再用 top-level `load_from` 加载官方 checkpoint，并在第一个 forward 前逐张量验真。

```text
EARLYLOAD（无效）                         POSTINIT（有效）
构建模型                                  构建模型
  → 构造函数内加载官方 checkpoint            → MMEngine 完成 init_weights
  → MMEngine 再执行 init_weights              → top-level load_from 加载官方 checkpoint
  → 冻结 DINO 等参数被覆盖                     → step-0 参数与别名验真
  → 从错误起点开始训练                         → 第一个 forward / 正式训练
```

因此 `FT-EXOINIT-POSTINIT-EGOONLY-LR1E6` 表示：官方 Exo2Ego Fusion checkpoint 初始化、在框架初始化后加载、只用 Ego2Exo 微调、学习率 `1e-6`。有效 POSTINIT 必须证明 DINO `368` tensors / `303,227,920` elements 按运行时 dtype 语义逐值一致、direct/sparse DINO 为同一对象、专家参数覆盖完整且 missing/unexpected/shape/value mismatch 为零。旧 `EARLYLOAD_INVALID` 中冻结 DINO 有 `303,128,589 / 303,227,920` 个元素偏离官方权重，因此只能作为故障证据，不能判断小 LR 或 joint training 是否有效。

两条实验都使用 official `fusion_exo2ego_full.pth`，SHA256：

`0e6e2589e2055cb64b92c2967c7c5796bb9530a720531c6511cce82cfa012de3`

修正版的硬门禁：

- `model.pretrained_pth = None`，由 MMEngine top-level `load_from` 在模型初始化后加载；
- 第一个 forward 前验证 standalone DINO `368` tensors / `303,227,920` elements；
- direct DINO 与 sparse-correspondence DINO 必须是同一对象且逐值一致；
- 验证 decoder / matcher / constr prompt FCS 的官方 checkpoint 覆盖；
- step-0 attestation 必须原子落盘并由 runner 再次验真；
- missing、unexpected、shape/value mismatch、篡改、超时或 trainer 提前退出均 fail-closed。

| Experiment ID | 机器 | Train data | LR / batch | 当前状态 | 下一结果点 |
| --- | --- | --- | --- | --- | --- |
| `FT-EXOINIT-POSTINIT-EGOONLY-LR1E6-NB-E1` | `.96` | clean Ego2Exo 86,319 | `1e-6`; 4×acc16=64 | `EPOCH1_FULL_EGO_DONE`：checkpoint SHA `88d9c14f…06fd`；FULL Ego stock `40.72`，accounting全0 | epoch2 formal=`0`；第三次 smoke 已真实恢复到 epoch1/iter21584，当前先修 HistoryBuffer 内容比较假阳性并复审，再做唯一 fresh smoke |
| `FT-EXOINIT-POSTINIT-JOINTNAT-LR1E6-NB-E1` | Seeta 4090 | clean Ego 86,319 + Exo 96,476 natural concat | `1e-6`; 4×acc16=64 | `INFRA_INTERRUPTED`：容器重启；日志到 iter `29,330` 全finite，`iter_29000.pth` 含optimizer但非安全边界 | reviewed恢复只允许从`iter_28000.pth`重放1000 batches；先补10-check smoke launcher |

iter 500 只做诊断；iter 2000 才执行预注册停止规则。只有 corrected run 通过 step-0 attestation 后的结果，才能回答“小 LR 和单向/混训哪个更好”。

### 8.2 Corrected Ego-only epoch 0/1（训练、内部 VAL 与 FULL 已完成）

Checkpoint：epoch `1`、iter `21,584`、size `3,404,365,822 B`、SHA256 `88d9c14f12882d232e04694bd4b41505b2a1dc13de27052024bf18e2cecd06fd`；optimizer state 完整，训练五项 loss 与 grad 全 finite。run manifest 终态为 `epoch_1_evaluated`，训练与评测 PGID 均已清场。

| 内部 VAL512 方向 | pairs | objects | stock IoU | per-object IoU | per-pair IoU | accounting |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ego2Exo | 599 | 995 | 65.61 | 64.02 | 67.50 | mismatch/missing/extra=`0/0/0` |
| Exo2Ego | 575 | 993 | 69.14 | 66.22 | 69.19 | mismatch/missing/extra=`0/0/0` |

Ego2Exo raw 的独立复算确认：`599` pairs / `995` objects；stock 是展平后只取前 `599` 个对象，得到 `65.6068`，而全 `995` 对象均值是 `64.0195`。被 stock 丢弃的后 `396` masks 平均 IoU 为 `61.62`，因此旧截断使该 scope 的 stock 抬高约 `1.59` pp。另有 `312/599` 个单对象 pair，其 pair 均值 `76.57`，使 per-pair 再高于 per-object `3.48` pp。

该 VAL 只有 `3` 个 take；其中一个 take 为 `72` 个单对象 pair，平均 `80.28`。此外 `508/599`（`84.8%`）记录可在 official Exo2Ego FullTrain 中找到同 key、query/target 路径互换、对象 key 相同的反向样本。Corrected Ego fine-tune 自身与 VAL 在 ID/take 上无重叠，但 official Exo-init 很可能已见过这些反向图像/对象；由于官方 checkpoint 无训练 manifest，此处标为高置信 exposure inference，不写成严格泄漏实证。故三个内部数值都偏高，不能说明 full-test 达到 60+。

这些是 training-domain internal VAL，只能证明 epoch 1 在固定内部 scope 上未崩溃；在同 scope official-init VAL 基线补齐前，也不能量化其中多少提升来自训练。为比较同事 Ego stock epoch 1 `25.74`，已按 `CORRECTED-EPOCH-MATRIX-FULL-V2-V1` 在同一 `ego2exo_test.json` 上完成 report-only full evaluation。`9/9` shards 按原序合并后共 `40,517/40,517` pairs、`100,223` objects，mismatch/missing/extra=`0/0/0`；checkpoint strict load `1703/1703`，所有 raw/summary 数字有限，独立复算与正式 summary 逐键一致。

| FULL-V2-TEST / Ego2Exo | Epoch 0 official Exo-init IoU / Dice | Corrected epoch 1 IoU / Dice | Epoch1−Epoch0 IoU |
| --- | ---: | ---: | ---: |
| stock legacy | 31.3251 / 36.1655 | **40.7199 / 46.3351** | **+9.3948 pp** |
| per-object | 28.7670 / 33.1451 | 37.6067 / 42.8869 | +8.8397 pp |
| per-pair | 32.6455 / 37.3758 | 41.7770 / 47.3459 | +9.1315 pp |

FULL 的 stock `40.72` 说明内部 VAL 的 `65.61` 确实主要来自 scope/聚合偏置，不能外推；它仍比同事报告的 epoch-1 stock `25.74` 高 `14.98` pp，比同事报告序列最高值 `30.87` 高 `9.85` pp。两边都指向 `ego2exo_test.json` 与 legacy stock 定义，但同事原始 checkpoint、raw metrics 与 manifest 尚未取得，因此此处只能写成“同 JSON/同报告口径下高于同事报告值”，不能写成已完成可审计复现。该 FULL 结果严格标为 `report_only / excluded_from_training_decision`，不据此启动 epoch 2。

Epoch 1 FULL 证据：combined raw SHA256 `a83165a50dbd601e801903b01498a62a55055dd22a425b714065874ba3ad8705`；summary SHA256 `bff4028c408e51196d68138273cc99d0175cf3fcb62f3df692f3c63681c00c52`；独立 summary SHA256 `7bdd234d313f33bd041f7c9855cb2e3313225c37761363fa02c1e8082d22855c`；run manifest SHA256 `832d733c5e2855663486ef78d0edf002888e874f197043ce6919ca864e50f4d8`。Epoch 0 FULL 同样为 `40,517/40,517` pairs、`100,223` objects、accounting全0；combined raw SHA256 `b155e3fff9706f3730df18074e6af77bb9d7a2ccedf644d7f7107b08489cdec5`，terminal manifest SHA256 `1fb381fda5fe211fcb7c5a4ed840d6b476ae4bc299ba1937ea80e9ec3a7eed8a`。两次均为 `report_only / excluded_from_training_decision`。

现在 official Exo-init 的同 scope VAL512 基线也已补齐。corrected epoch 1 相对初始化：Ego2Exo stock/per-object/per-pair 分别 `+10.39 / +14.41 / +13.72` pp；Exo2Ego 分别 `-9.56 / -6.36 / -6.22` pp。双向简单平均则 stock `66.96→67.37`（`+0.42` pp）、per-object `61.09→65.12`（`+4.02` pp）、per-pair `64.60→68.34`（`+3.75` pp）。因此结论是“Ego 专项适配明显成功，同时有显著 Exo 遗忘”；不是全面双向提升。按预注册的 Ego-only 条件“epoch 1 主方向有增益才续”，它具备进入 epoch 2 的资格；实际启动前仍需通过 epoch-boundary resume 连续性与 provenance 门禁。FULL 结果不参与该决定。

### 8.1 Corrected gate 实时结果

| 轨道 | Checkpoint | Ego2Exo per-pair IoU | Exo2Ego per-pair IoU | 相对 init | 决策 |
| --- | --- | ---: | ---: | --- | --- |
| Ego-only POSTINIT | init | 24.71 | 61.56 | 基线 | internal-retention gate；非 paper mIoU |
| Ego-only POSTINIT | iter 500 | 25.35 | 61.50 | Ego `+0.64` pp；Exo `-0.07` pp | diagnostic continue |
| Ego-only POSTINIT | iter 2000 | 32.76 | 53.30 | Ego `+8.05` pp；Exo `-8.27` pp | stop gate passed；continue epoch 1 |
| Joint-natural POSTINIT | init | 24.71 | 61.56 | 基线 | internal-retention gate；非 paper mIoU |
| Joint-natural POSTINIT | iter 500 | 24.37 | 61.12 | Ego `-0.34` pp；Exo `-0.44` pp | diagnostic continue |
| Joint-natural POSTINIT | iter 2000 | 27.39 | 59.78 | Ego `+2.68` pp；Exo `-1.79` pp | stop gate passed；continue epoch 1 |

旧 EARLYLOAD Joint 在 iter 500 的同 scope per-pair 是 Ego `8.78` / Exo `52.27`。两条 corrected 轨道都没有出现这种冻结特征被覆盖后的灾难性下降，因此当前结果支持“初始化时序修复有效”。到 iter 2000，Ego-only 的 Ego 增益更大（`+8.05` pp），但 Exo 遗忘也明显（`-8.27` pp）；Joint-natural 的 Ego 增益较小（`+2.68` pp），却把 Exo 回落控制在 `-1.79` pp。按同一 internal-retention scope 的双向简单平均，Ego-only 为 `43.03`，Joint-natural 为 `43.58`：当前 Joint 略优于双向保持，Ego-only 更适合作为 Ego 专用候选；这些数值均不是论文可比成绩。两者都继续到 epoch 1，固定 VAL512 只用于内部选点，最终候选再做预注册 FULL-V2-TEST。

## 9. 尚未补齐的正式对照

| 优先级 | 实验 | 目的 | 触发/停止条件 |
| --- | --- | --- | --- |
| P0 | Corrected Ego-only LR `1e-6` | 测 Exo checkpoint 是否能专门适配 Ego | 先过 step0/500/2000；epoch 1 有增益才续 |
| P0 | Corrected Joint-natural LR `1e-6` | 测自然比例混训能否保留 Exo 并提高 Ego | 先过双向 gate；两向都下降则停 |
| P1 | 胜者续 epoch 2 / 3 | 找 2–3 epoch 内最佳 checkpoint | 连续评测增益 <0.5 pp 或关键方向回落 >2 pp 时停 |
| P1 | 官方 Fusion 2×2 VAL512 | 建立固定 internal-retention 对照 | 固定 599/575 pairs，per-pair 为内部主指标；非 paper mIoU |
| P2 | Ego-only LR `3e-6` | 当 `1e-6` 稳定但明显欠拟合时提高 LR | iter 2000 明显下降则停 |
| P2 | Ego-only / Joint LR `3e-7` | 当 `1e-6` 仍出现有效初始化下遗忘时降低 LR | 只作为诊断分支，不盲跑 3 epoch |
| P2 | Joint balanced homogeneous batch | 区分 natural concat / 混方向 batch 是否有影响 | 只有 corrected natural joint 出现方向偏置才跑 |
| P0（推理） | Official A+B+C PCCS VAL | 复现论文 Multi-Experts | 行为级证明 PCCS 不读 GT、保存 final masks、统一 metric 重算后才解锁 |
| P3 oracle | ABC GT-IoU / SaveMask | 测 selector 上界 | 永远标 oracle，不参与正式排名 |

当前 TripleDecoder / PCCS 路径为 `BLOCKED`：现有 `wrz` 包尚未满足行为级 selector 合同。评测脚本会 fail-closed，不会用关键字扫描或 GT oracle 伪装成正式 PCCS 数字。

### 9.1 并行资源与任务隔离规则

- Seeta 资源审计（2026-08-12 17:05）：GPU `49,140 MiB`，Joint 训练占 `10,889 MiB`，空闲 `37,620 MiB`；RAM 可用约 `926 GiB`，CPU `208` 核，系统盘可用 `43 GiB`，持久盘可用 `666 GiB`。
- 单个 Fusion 推理约占 `5 GiB`，因此允许与 Joint 训练并行，但每个 job 启动前必须重新检查空闲显存不少于 `20 GiB`；四个方向/ckpt 组合只串行执行，不互相并发。
- 暂不在 Seeta 叠加第二条正式训练：两条训练会竞争同一 GPU 算力，并且 gate 时还需要同时保留 paused-train 与 evaluator 显存。待 Fusion VAL 完成或测得并行吞吐后再决定是否开 LR `3e-7` 训练。
- Official Fusion 2×2 VAL512 的前五次正式尝试均 fail-closed，未产出指标且未影响 Joint：依次暴露 SAM2 路径、DINO basename、config dump inode碰撞、external-DINO checkpoint closure，以及 `Config.dump→reload` 破坏 MMEngine `LazyObject` callable。修复后的 data-only overlay 方案使用 tracked base config 单次加载、attested JSON overlay 同对象应用、真实1-record semantic smoke，并在独立复审后由 retry6 完成四个正式 job。前五份失败 run 仍保留并排除指标；retry6 为唯一有效 VAL512 2×2 run。
- Seeta 在 `2026-08-13 11:46:45` 发生容器重启。Joint 原 runner/train PGID 消失，日志最后到 iter `29,330`，五项 loss/grad 均finite，fatal/OOM/missing为0；最后完整 checkpoint `iter_29000.pth` 已验证含optimizer。由于 batch accumulation=`16` 且中途 checkpoint 可能不保存未step的累积梯度，加之普通 EpochBased resume 可能从 dataloader epoch开头重放，当前禁止直接resume。恢复方案必须先确定安全optimizer边界 checkpoint，并重建同 seed/sampler/worldsize 的完整顺序、精确跳过已完成batch，再由独立review和短smoke证明不重复样本。
- Joint恢复审计已确定 `iter_29000 = 1812×16+8`、`iter_28500` 余4，二者checkpoint都不含参数梯度或 OptimWrapper inner count，不能精确恢复；最新安全边界为 `iter_28000`（Adam step1750、scheduler28000，SHA256 `b7ec075bb70ab1c3430193d0390947c81edda3f4d3b240222a4218f1b05dcdcd`）。reviewed方案从同一 LengthGroupedSampler 全序列精确跳过前28000 batches，只执行 suffix17712 batches到45712；full/prefix/suffix索引SHA与首恢复batch ID均固定。它只声明 `checkpoint_state_and_sample_order_exact`；原run未保存Python/NumPy/CUDA/worker RNG，因此明确标 `rng_bitwise_unprovable`。bounded smoke 已实现为精确 `28000→28016`、一个完整 optimizer boundary，并补齐checkpoint深层schema、source-tree内容闭包、Python3.10早期门禁与artifact语义复验；但独立复审证明同一执行方仍可构造一套零训练却自洽的 `production_gpu` receipt。formal 当前不得把这种自签 receipt 当批准，必须引入独立 reviewer/control-plane 的批准信任根后才可部署恢复。
- Ego-only epoch2 当前 formal 数仍为 `0`。三次唯一 fresh smoke 均 fail-closed 且未生成可复用 receipt：第一次暴露 machine-output 文件通道/路径/覆盖合同，第二次在模型构造时暴露 `val_cfg` 与空 dataloader/evaluator 不一致，第三次先越过 validation 修复又暴露 DINO binding 重命名破坏官方 8-hex basename parser；这些均已在隔离 worktree 中 TDD 修复并独立复审。最新一次基于 clean `9f149fcb655bd23c37c555c6f801dff41e8c04da` 已成功构造模型与 86,319-item dataloader、验证 DINO nonalias binding，并恢复 checkpoint 到 `epoch=1, iter=21584`，随后在首次 forward 前被10项 MessageHub log scalar mismatch拦截。noGPU逐数组诊断确认 checkpoint、`Runner.resume()` 后、HIGHEST `before_train` 前三边界的 history/count dtype、shape、bytes完全一致；旧审计器只是对两个不同 `HistoryBuffer` 实例使用对象 `==` 而误报。下一步必须用内容语义 exact comparator（类型/max_length/history/count dtype/shape/bytes）替换该比较，并保留 runtime_info、resumed_keys、optimizer、scheduler 严格门禁；TDD和独立复审通过后才允许第四个唯一 fresh smoke。

### 9.2 Corrected epoch 与同事序列的同口径终测预注册

预注册 ID：`CORRECTED-EPOCH-MATRIX-FULL-V2-V1`。目的仅是报告 corrected checkpoint 在完整 test 上的结果，并与同事报告的 Ego stock epoch 序列核对；FULL-V2-TEST 结果不反向参与续训或选 epoch，续训决定仍只使用预注册 internal VAL。

| 项目 | 固定合同 |
| --- | --- |
| Ego2Exo test | `40,517` pairs；JSON SHA256 `86f5803758b3c8da16e0c1f373f001a1cd14633497b05a7ce33ae6ddfcc490de` |
| Exo2Ego test | `46,515` pairs；JSON SHA256 `b05faed68c0bea40df02d3fd81b0b25af9e3fbb6e4eb8e7e2b8cb9349de34b96` |
| checkpoint 行 | official Exo init（epoch 0）、corrected Ego-only epoch 1/2/3、corrected Joint epoch 1/2/3；只有实际通过 gate 并生成的 epoch 才运行 |
| 指标 | 同一 `SegMetricFull` 同时输出 stock、per-object、per-pair；与同事序列只比较 Ego2Exo stock，其余列独立报告 |
| 完整性 | 每方向 exact pairs、全部对象、mismatch/missing/extra=`0`、finite/bounded；raw/summary/config/checkpoint 全部 SHA256 |
| 决策隔离 | test 数字标 `report_only / excluded_from_training_decision`；不得因 test 结果改变 epoch 2/3 的继续或停止规则 |

执行顺序：Ego-only epoch 1 落盘并完成内部 VAL 后，优先在释放出的 `.96` GPU 上跑 Ego2Exo full stock；Joint 继续在 Seeta 训练。随后再补 Exo2Ego 和其他通过 gate 的 epoch checkpoint，避免两条训练互相等待。

### 9.3 新域 Exo2Ego zero-shot：DROID / BehaviorSim

预注册 ID：`CROSSVIEW-ZEROSHOT-EXO2EGO-V1`。这条轨道只测试跨数据域 zero-shot，不训练、不参与 corrected epoch 的继续/停止决策。

| 数据集 | 固定 revision / relation SHA256 | Exo2Ego scope | 路径与适配合同 |
| --- | --- | ---: | --- |
| `Travor278/droid-crossview-seg` | revision `f0a2e41ecfd195053b40b73d58d30e5f32f7c6df`; `exo2ego.json` `3e231bc7266e009fc873cbc3d19bd7ef5e19dca983177bd7986e5661fd3cdd32` | `2,162` pairs / `2,162` objects | JSON路径已含`data/`；每条top/object target均为同一单帧；Ego=`ego`，Exo=`exo1|exo2` |
| `Travor278/behaviorsim-crossview-seg` | revision `ff0e77c9d37f43b9b10a8ef063e777bb34d5627a`; `exo2ego.json` `9e6ee051887e8020dabacb033a6a22f4e69e99df94b44a716ffdc212d7ae7dbf` | `243` pairs / `243` objects | JSON路径不含`data/`；177条top-level `video_path`不是单帧、其中2条首帧也不是object target；只允许data-only改写为精确的`objects.0.video_path`单元素列表，其他字段逐字保持；Ego=`head_gaze`，Exo=`exo0..exo6` |

Single Expert 主结果固定使用论文定义的 C/Fusion 与官方 Exo2Ego checkpoint `fusion_exo2ego_full.pth`（SHA256 `0e6e2589e2055cb64b92c2967c7c5796bb9530a720531c6511cce82cfa012de3`）。B/Visual `visual_exo2ego_full.pth`（SHA256 `1fbdbba5294a29c36c32897263b9e8e006dfaa623c04227dc0c6bd3675b38f87`）只作辅助 individual-expert diagnostic，不冒充论文 Single Expert。

统一报告 stock/per-object/per-pair IoU 与 Dice、exact pairs/objects/evaluated、mismatch/missing/extra；因为两集均为每pair一个object，三种IoU和三种Dice必须分别在`1e-12`内相等，否则适配或聚合直接判失败。先做每数据集/专家10-pair smoke，review后才跑全量。所有数字标为 `new-domain zero-shot`，不是 paper mIoU。

Multi-Experts 只接受 A+B+C 的非GT PCCS/Cycle-Points最终选择结果。selector必须在完全移除/任意改变target GT后仍产生相同selected mask，且A/B/C原始mask和最终mask均落盘后再由独立metric读取。现有按target GT IoU择优的 SaveMask 永远标作oracle；行为合同未通过前，Multiple/PCCS保持`BLOCKED`而不填伪数字。

2026-08-14 对 `.96` clean HEAD `8b65a9c2c6c3ceaedee4967d5da8b24981be840c` 的完整行为审计进一步确认：TripleDecoder 能产生 A/VP、B/sparse、C/fusion 三路候选与 back-cycle points，但候选前向仍通过 `gt_masks[0].shape` 取得输出尺寸；正常非空 prompt 的 Cycle 选择按 point-in-prompt-bbox count、平局 cycle distance，不直接读 GT 内容，但缺 GT 会直接不输出，空 prompt fallback 更明确按 GT IoU 最大值选择。活动 config 实际使用 `SegMetric_TripleDecoder_SaveMask`，它全程按 GT IoU 选 decoder，是 oracle。仓库不存在独立 `pccs_selector.py` 或其测试，Cycle 也不生成/落盘 canonical `pred_mask_selected`，现有离线工具只保存三候选 RLE，无法从最终 selected mask 独立复算。

资产审计还确认 A 没有独立 checkpoint（原始 SAM2 decoder + DINO），B/C 分别使用已冻结的 Visual/Fusion checkpoint；但旧 Triple 的 Expert12 runtime `constr_prompt_fcs.0.weight=(256,256)` 与 B checkpoint `(256,512)` 必然 size mismatch，matcher 也只有 6 keys 对正式 B 的 60 keys；Expert3 与顶层加载继续使用 `strict=False`，C 的 `sparse_correspondence.*` 到 runtime `sparse_correspondence_backward.*` 没有闭合证明。因此正式 Multiple 必须先去除 GT-shape 与 GT fallback、加入纯 GT-free selector、严格核 A/B/C checkpoint、生成并原子保存 canonical selected mask，证明真实/随机/全零/移除 target GT 时候选、selector evidence、selected expert 与 final mask 逐位相同，再由独立 metric 只读该 mask。此前 131 个旧 work dir 无 receipt、raw/summary 或 selected-mask artifacts，不构成正式 PCCS 结果。

`.166`空间清理已于2026-08-13完成：精确删除 `/home/dell/datasets/robointer_droid_dual_exo` 下8个原始/中间缓存目录，释放 `919,193,792,512` B（约856 GiB）；根目录保留 `dataset.json`、`shards.jsonl` 与 `DOWNLOAD.md`，下载说明固定上述HF revision并链接DROID/RoboInter上游。`robointer_triview_1000`（约69.3 GiB）和`robointer_wrist_labelstudio`（约3.1 GiB）均保留；清理后根分区可用约1.39 TiB。

两个冻结快照现已部署到`.166`的`/home/dell/datasets/v2sam-crossview-zero-shot/`：DROID `3.1G`（7,788个原下载文件，3,893份HF revision metadata），BehaviorSim `148M`（2,248个原下载文件，1,123份metadata）；每份metadata首行均与对应固定revision一致，两个`exo2ego.json` SHA均复核通过。跨机归档SHA分别为DROID `67dea6cd7b53330661115a8f73ee042f6f621b5d21c46e1c2db109725d757e47`、BehaviorSim `4af0eadcc9c3c40e72189c92bfd32837b639ec745c745c54bb135cee61d43fe1`；远端deployment receipt SHA为`7fbe9089015d8c36dfae937974131cc8515adc2d6e6720b8298c1e8c21c4d506`。

#### 9.3.1 DROID C/Fusion 10-pair formal smoke（已通过）

正式结果来自`.166`的reviewed commit `a7e5140ca553a9e5e6a5f6aeeb00d71a707dca4c`，仅选择`DROID × C/Fusion × smoke`；方向为Exo2Ego，使用官方Fusion Exo2Ego checkpoint。run root：`/home/dell/codex_runs/crossview-zero-shot/formal/droid-C-smoke-retry5-a7e5140-20260813T170933Z`（目录名中的UTC时间对应北京时间2026-08-14 01:09）。

| scope | pairs / objects | stock IoU / Dice | per-object IoU / Dice | per-pair IoU / Dice | accounting |
| --- | ---: | ---: | ---: | ---: | --- |
| DROID C smoke | `10 / 10` | `49.5672 / 55.3125` | `49.5672 / 55.3125` | `49.5672 / 55.3125` | evaluated=`10`; mismatch/missing/extra=`0/0/0` |

DROID每pair精确一个object，因此三种聚合逐值相等；独立纯Python重算与正式summary精确一致。strict load为true，missing/unexpected/shape mismatch均为0，external-DINO闭包与post-test unchanged均通过；15个bound images、10条relation均通过发布后审计，published symlink/special/hidden-staging string均为0。run manifest及独立aggregate字节完全一致，SHA256均为`de39116d48bc4902c22fcc8cb4de2318fa96b13abd3921f15a5774f5ffb1663b`；final audit receipt SHA256为`4399ba03fae6cca8a8b8b5392ed3e24fdb69a695b6b1d0ee99d6332de15f2bee`；raw SHA256为`29fc6bbb…31ae`，summary SHA256为`cdfa49b5…bdbc`。该10-pair结果只证明执行与指标合同，不作为full zero-shot结论。

#### 9.3.2 BehaviorSim C/Fusion 10-pair formal smoke（已通过）

正式结果来自`.166`的reviewed commit `2e4b1df09c472fcb04773ba2746d3b0616c1691a`，public `dataset_id=BehaviorSim`、冻结的`runtime_scope=behaviorsim`，仅选择`BehaviorSim × C/Fusion × smoke`。run root：`/home/dell/codex_runs/crossview-zero-shot/formal/behaviorsim-C-smoke-2e4b1df-20260813T180300Z`（北京时间2026-08-14 02:03）。

| scope | pairs / objects | stock IoU / Dice | per-object IoU / Dice | per-pair IoU / Dice | accounting |
| --- | ---: | ---: | ---: | ---: | --- |
| BehaviorSim C smoke | `10 / 10` | `49.0793 / 53.2278` | `49.0793 / 53.2278` | `49.0793 / 53.2278` | evaluated=`10`; mismatch/missing/extra=`0/0/0` |

BehaviorSim full source为`243` pairs / `243` objects，data-only规范化精确改变`177`条top-level target；smoke是规范化后full的canonical前10条，含15张bound images。每pair一个object，故三种聚合精确相等。strict=True，missing/unexpected/shape mismatch均为0，external-DINO闭包完整且post-test state unchanged；独立aggregate、逐artifact SHA和raw重算均通过。run manifest SHA256 `042fb0d1c8f851bd4a7af869f0c851affbe76b1dbf4ccd6edf6e433ad3679ffe`；receipt `7c339ccf3c60d7caa97c26fab3791dade9deb24fb7b5955f8ce9888e6762d5eb`；job manifest `5ce19916cebd049b0129cc333695db854fc4b0fddaf67ec2568b0e9dca27732f`；raw `7240c9420345bedec89f2c274c60c5d48d8b0aee89b5b8de037fc82b8ca723f8`；summary `8566489d66486b609b8861847933ec50b4eb162b2913c955b90d63d75093b6df`。该结果同样只证明smoke合同，不是full zero-shot成绩。

在该正式成功run之前，若干GPU前部署/preflight失败和三次完成10/10后被发布审计拒绝的run均未产生最终receipt/run manifest，相关IoU/Dice只保留为diagnostic、不得进入正式表。它们依次暴露并修复了run-root `.`重定位、trusted containment root、hardlink导致的合法ctime变化，以及formal artifact closure遗漏；最终成功run使用全新run root且未复用任何失败产物。

#### 9.3.3 C/Fusion full zero-shot（正式结果）

两套 full 均来自 `.166` 的 reviewed commit `326f128e87ad2999eca471a61206d79b77aa9cb7`，每个数据集在同一唯一 run 内依次执行并绑定 `smoke→full`。两次运行都以 `rc=0` 终止、原子发布 final、staging 为 0；官方 `validate-aggregate`、独立 canonical aggregate、逐 artifact SHA 与独立 raw 重算均通过。由于两套数据都是每 pair 精确一个 object，stock、per-object、per-pair 三种 IoU/Dice 逐值完全相等。

| scope | pairs / objects | stock IoU / Dice | per-object IoU / Dice | per-pair IoU / Dice | accounting |
| --- | ---: | ---: | ---: | ---: | --- |
| DROID C full | `2,162 / 2,162` | `50.5893 / 55.1002` | `50.5893 / 55.1002` | `50.5893 / 55.1002` | evaluated=`2,162`; mismatch/missing/extra=`0/0/0` |
| BehaviorSim C full | `243 / 243` | `45.3399 / 50.8436` | `45.3399 / 50.8436` | `45.3399 / 50.8436` | evaluated=`243`; mismatch/missing/extra=`0/0/0` |

DROID run root 为 `/home/dell/codex_runs/crossview-zero-shot/formal/droid-C-smokefull-326f128-20260813T203000Z-1a5a30`；full raw/summary/receipt/job-manifest SHA256 分别为 `210ede6046a447681e5d6f4efa6654e4150c0a917785aad80eef49e45af1e736`、`e966b0fc1b546728dbf664eb64450f4256035480db2661500547048d9c530451`、`537e6e4bf6cb5fbadf0289f58e8a59171c89f914c619a14a76a3da458f602abd`、`44b403c7633a5c1635fe48885e43e75230d0d1d6d2196baf0f29812dffc2feed`；run manifest SHA256 为 `f8104c19c45492c63da6ac88f2325675a76e794747f2d837f39ba5c2bb45919b`。BehaviorSim run root 为 `/home/dell/codex_runs/crossview-zero-shot/formal/behaviorsim-C-smokefull-326f128-20260813T210609Z-fbf711`；full raw/summary/receipt/job-manifest SHA256 分别为 `2477b0237554de1bfee08565150864299f3693bda6ff1b551febfa603e999307`、`18010ea0fe82edac4c2f78dab0c4c615d7cb437abc1093b5ffde13a975a08340`、`36f1bd013b16d3dcfc137359dad18dfc80714b1a15616faba23a370d8f55777c`、`a9622baae962b3ec22246817e19db70ff7fc5987dede4d1e2520f5a3240f52f8`；run manifest SHA256 为 `2d306fa96bc7567b57df5304f9094b226790226cfda65d2e44f495957b5b6e3f`。两次 strict load 的 missing/unexpected/shape mismatch 均为 0，external-DINO 与 post-test unchanged 闭包均通过。

与论文标尺比较时必须保留 scope 标签：论文 C/Fusion Single Expert 在 Ego-Exo4D 上报告 `44.5 / 47.3`（Ego2Exo / Exo2Ego mIoU），ABC Multi-Experts + PCCS 报告 `46.3 / 49.6`；HANDAL-X 的 zero-shot Single/Multi 为 `66.4 / 77.2 IoU`，DAVIS-17 则报告不同指标 `J&F=78.8`。因此 DROID C 的 `50.5893` 在数值上高于论文 Exo2Ego Single Expert `47.3`，BehaviorSim C 的 `45.3399` 略低，但这些是不同数据集、不同对象分布与自建 relation scope，不能据此宣称超过论文或跨 benchmark 排名。它们可接受为经审计的**外域 Exo2Ego zero-shot baseline**：DROID 属强结果，BehaviorSim 属中等可用结果；二者都不替代原论文 split 的复现。

同一 pair 上对 B/C raw IoU/Dice 做 `100,000` 次 paired bootstrap（seed=`20260814`）后，DROID 的 C−B 为 `+7.4777/+8.4883` pp，95% CI 分别为 `[+5.7277,+9.2236]` / `[+6.6805,+10.3186]`，稳定高于 B；BehaviorSim 的 C−B 为 `+1.6744/+1.6713` pp，95% CI 为 `[-3.6800,+6.9718]` / `[-3.8372,+7.1741]`，区间跨 0。因此 BehaviorSim 的 C 数值本身是有效正式结果，但“C 稳定优于 B”目前只能写成点估计，不能写成统计上已确认。

#### 9.3.4 B/Visual full zero-shot（辅助 individual-expert）

B 首次 DROID formal 在推理前 fail-closed：Visual config 经 `projects/v2sam` symlink 错误导入 Fusion/DINO 模型类，而 B 合同正确要求 DINO 为 null；该失败无指标。TDD 修复与独立复审 CLEAN 后，reviewed commit `8649bc30e91c0feab00fa204637178e82605dbba` 只把 Visual config 的模型/runner import 指回 `projects.v2sam_visual.models`；真实官方 B checkpoint 在 `.166` 上为 `964/964` keys，missing/unexpected/shape=`0/0/0`，runtime/spec/attestation 均无 DINO。

| scope | pairs / objects | stock IoU / Dice | per-object IoU / Dice | per-pair IoU / Dice | accounting |
| --- | ---: | ---: | ---: | ---: | --- |
| DROID B full | `2,162 / 2,162` | `43.1116 / 46.6119` | `43.1116 / 46.6119` | `43.1116 / 46.6119` | evaluated=`2,162`; mismatch/missing/extra=`0/0/0` |
| BehaviorSim B full | `243 / 243` | `43.6655 / 49.1722` | `43.6655 / 49.1722` | `43.6655 / 49.1722` | evaluated=`243`; mismatch/missing/extra=`0/0/0` |

DROID B run root 为 `/home/dell/codex_runs/crossview-zero-shot/formal/droid-B-smokefull-8649bc3-20260813T220100Z-b864c1`，run manifest/full raw/full summary/full receipt SHA256 分别为 `b30c311c218d783b5b8398ffbe0db4183a28b1507a081a972e32f76f682a80c3`、`be27616e17025a9b9c289741be8184550d1ee25a753037774d31990f3685cd6d`、`9ad3f3d0b6257b8497dfdab2e9eb1a7575538f1cd8fb59d9eec3c43ee5faed3e`、`e2a7c25b4e1349a0f62c10c76d43b767afec383a6e453025b947ffc9c5dc341a`。BehaviorSim B run root 为 `/home/dell/codex_runs/crossview-zero-shot/formal/behaviorsim-B-smokefull-8649bc3-20260813T221600Z-bb8649`，run manifest/full raw/full receipt SHA256 分别为 `4b1a064151bf7f84318f7318958313f49df28198cfe174791ce200801b7bd87e`、`4603bca963fb142587ccd6faec2ebcdf1593302af7131466b4fe4e13f5d70314`、`b36cb9ebbc58955e1d85f0de301d5d063e77ce4f7539b9231cd63d3c88259881`；独立审计 receipt SHA256 为 `c702fffc5f4318fbc333f16f9fe46029f7e87e25994286f576aed23790ed1d08`。两次均 rc=0、原子发布、staging=0，三口径 exact，独立 raw 与 aggregate 重算通过。

在同口径 full 上，论文 Single Expert C/Fusion 相对辅助 B/Visual：DROID IoU/Dice 高 `+7.4777/+8.4883` pp；BehaviorSim 高 `+1.6744/+1.6713` pp。因此当前两套新域 zero-shot 的单专家首选均为 C/Fusion。Seeta `connect.westd.seetacloud.com:39257` 当前 root 仍拒绝已有全部公钥，因此执行在 `.166` 串行完成。Multiple/PCCS 仍保持 `BLOCKED`。

## 10. 当前“最好结果”应该怎样说

- **论文最佳**：ABC Multi-Experts + PCCS，paper mIoU `46.3 / 49.6`。
- **本次官方 checkpoint 同方向最佳（PROBE200 per-pair）**：Fusion Ego→Ego `50.18`；Fusion Exo→Exo `51.36`。
- **当前跨方向初始化最佳**：Fusion Exo checkpoint→Ego2Exo `36.47` per-pair；这正是 corrected Ego adaptation 的起点。
- **当前有效 Ego 专用训练后候选**：corrected POSTINIT Ego-only epoch 1，在 FULL-V2-TEST 上 stock IoU `40.72`、per-object `37.61`、per-pair `41.78`；这是 report-only 结果，不反向决定续训。Joint-natural epoch 1 尚未完成，故双向综合最佳仍未定。
- **当前新域 Exo2Ego zero-shot 最佳（full、每pair一object）**：论文 Single Expert C/Fusion 在 DROID 为 IoU/Dice `50.5893/55.1002`，在 BehaviorSim 为 `45.3399/50.8436`；两者均高于同域辅助 B/Visual。Multi/PCCS 因上述 GT/selector/checkpoint/artifact Critical blocker 尚无可报告数字。

## 11. 更新规则

每产生一个 gate / epoch，只有满足以下条件才写入正式结果区：

1. experiment ID、代码 commit、config SHA、data JSON SHA、initialization SHA 全部固定；
2. checkpoint 可加载，meta iter/epoch 与 optimizer state 合法；
3. step-0 POSTINIT attestation 为 `verified`；
4. 评测 pair/object 数精确，mismatch/missing/extra 均为 0；
5. stock / per-object / per-pair 分列，paper mIoU 不混入；
6. raw metrics、summary、decision、checkpoint 都记录路径、size 和 SHA256；
7. 明确 `valid / invalid / diagnostic / oracle / in-progress`，不得只凭 loss 正常宣布成功。

机器账本当前含 30 条已验证记录；本文会随着 corrected init、iter 500、iter 2000、epoch 1/2/3 和官方 VAL/PCCS 结果及时更新。


