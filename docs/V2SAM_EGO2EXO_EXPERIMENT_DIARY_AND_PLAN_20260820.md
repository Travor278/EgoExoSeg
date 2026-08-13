# V2-SAM Ego2Exo 复现实验日记与下一阶段计划

> 快照日期：2026-08-23（Asia/Shanghai）<br>
> 远端实验日期：2026-08-18 至 2026-08-22（UTC）<br>
> 范围：Ego2Exo 数据、权重、官方推理、从头训练、私有 NewMatcher 链路及后续刷榜实验<br>
> 原则：机器产物、哈希和完整日志优先于聊天记录；本文中的“论文值”“本次实测”“推断”和“待验证假设”严格分开。

## 1. 一页结论

### 1.1 结论：分差来自训练出的 SAM2 主分支状态，不是 NewMatcher 换了更强的 matcher

public epoch 2 为 `0.1468 / 0.1845`，v10 NewMatcher epoch 2 全对象为 `0.36248 / 0.40940`，相差约 `+0.2157 IoU / +0.2249 Dice`。源码核对后的结论是：**这不是两套 matcher 算法的能力差，而是两次独立训练在主 `prompt -> SAM2 mask` 分支上进入了质量完全不同的优化轨迹。**

public `vp_matcher.py` 与 v10 `vp_matcher_new.py` 的活跃 `__init__`、`_normalize_mask`、`forward` AST 完全一致；`vp_matcher_new.py` 后面多出的另一版实现全部被注释，没有参与运行。两边都执行同一套 mask encoder、Q/K/V cross-attention、spatial gate、transformer、残差 MLP、FiLM/mask prior 和粗 mask decoder。`RegionPooling`、`SAM2TrainRunner.inject_language_embd`、DINOv3 correspondence 的主 `forward` 也相同。因此，类名中的 “NewMatcher” **不代表这里真的启用了一个数学上更强的新 matcher**。

两次训练的关键配置同样对齐：同一个 `Ego2Exo_FullTrain.json` 和图像根目录、batch `16`、accumulation `4`、AdamW `4e-5`、Linear warmup `0→1.2 epoch`、随后 24-epoch cosine horizon；两边均设置 `frozen_sam2_decoder=False`，即 SAM2 mask decoder 的可训练部分会被更新。两次 epoch-2 结果因而应理解为：**近似相同模型图与优化超参下得到的两个不同 checkpoint**，而不是“旧网络对新网络”的结构性比较。

### 1.2 两边实际训练的计算图

```text
prompt VP = RegionPooling(prompt feature, prompt GT mask)
target VP = RegionPooling(target feature, target GT mask)       # 只用于训练监督

predict VP, coarse mask = VPFeatureMatcher(prompt VP,
                                            prompt mask,
                                            target feature)
coarse-mask VP = RegionPooling(target feature, coarse mask)
language prompt = MLP([predict VP, coarse-mask VP])
sparse points = DINOv3Correspondence(prompt image,
                                     target image,
                                     prompt mask)

final mask = SAM2(target feature,
                  language_embd=language prompt,
                  point_inputs=sparse points)

loss = 10 * (final-mask CE + final-mask Dice)
       + coarse-mask CE/Dice
       + contrast-scale * VP contrast loss
```

这个图有两个输出层级：

1. `VPFeatureMatcher` 直接产生 `coarse mask`，由 `small_loss_mask/small_loss_dice` 监督；
2. matcher 产生的 embedding 与 coarse-mask VP 经过 `constr_prompt_fcs` 形成 language prompt，再和 DINOv3 稀疏点一起送入 SAM2 mask decoder，产生最终榜单所用的 `final mask`，由主 `loss_mask/loss_dice` 监督。

因此，即使 matcher/coarse mask 几乎相同，只要稀疏点或 language prompt 让 SAM2 decoder 落入不同的数值状态，最终 mask 仍可以相差很大。尤其这里每个对象最多只保留一个 correspondence point；`sparse_correspondence.py` 还通过 `np.random.choice` 从候选点中抽样，而两次任务都保留 `deterministic=False`。同一个 seed 并不能保证独立 DDP 进程、DataLoader worker 和 NumPy 抽样消费完全相同的随机序列。单个正点位置的变化会直接改变 SAM2 `_forward_sam_heads` 的 point embedding、multimask 选择与 mask logits。

### 1.3 分项 loss 已经把首次分叉定位到 SAM2 主输出

| iter | 实现 | 主 `loss_mask` | 主 `loss_dice` | `small_loss_mask` | `small_loss_dice` | `loss_contr` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | public V2SAM | 12.5364 | 4.8748 | 2.0178 | 0.4940 | 459.2842 |
| 10 | v10 NewMatcher | 3.8476 | 2.9977 | 2.0178 | 0.4940 | 458.7539 |
| 20 | public V2SAM | 12.1776 | 4.8651 | 2.0188 | 0.4947 | 455.1787 |
| 20 | v10 NewMatcher | 3.3434 | 2.9422 | 2.0188 | 0.4947 | 454.7520 |
| 50 | public V2SAM | 12.0659 | 4.8063 | 2.0185 | 0.4945 | 453.8809 |
| 50 | v10 NewMatcher | 3.0729 | 2.6978 | 2.0185 | 0.4945 | 454.2314 |

这组日志给出了比最终分数更强的定位证据：

- `small_loss_mask/dice` 在相同 iter 逐项相同到日志精度，说明 matcher 的粗 mask 输出与监督行为基本一致；
- `loss_contr` 也处于同一量级且非常接近，说明 predicted VP 与 target VP 的对齐没有发生足以解释 21.6 个 IoU 点的结构性分叉；
- 唯一从最初十个 iter 就显著分叉的是 SAM2 最终输出的主 `loss_mask/dice`。这时训练还远不足以让一个“新 matcher”学出巨大能力差，分叉只能位于 `language prompt + sparse point -> SAM2 decoder -> final mask` 这段执行与其初始数值轨迹。

主 CE/Dice 又被乘以 `10` 加入总 loss。public 从一开始承受约 `12.5 + 4.9` 的主分割损失，v10 约为 `3.8 + 3.0`；这一差异会立即改变 SAM2 decoder、prompt MLP 和 matcher 收到的反向梯度。此后即使喂入相同数据，两者也会沿不同梯度方向继续放大差异：v10 很早进入“prompt 能稳定驱动 SAM2”的好盆地，public 则需要先从更差的 SAM2 mask 状态中恢复。public 同一轨迹到 epoch 4 曾升至 `0.2937 / 0.3513`，也符合“收敛更慢、盆地更差”，而不是“模型结构上只能到 0.1468”。

### 1.4 能严谨归因到哪里

可以确认：

- **不是 NewMatcher 公式带来 21.6 个 IoU 点。** 活跃 matcher 数学实现相同；v10 的主要价值是把批数据、图像路径、稀疏提示、SAM2 注入和训练状态保存合同稳定下来。
- **不是 evaluator 制造的分差。** 这里引用的两个数都是独立全对象口径；训练早期的主 loss 也已在 evaluator 之前显示出同方向分叉。
- **不是 public 模型容量的硬上限。** public 轨迹继续训练会明显上升；兼容的高质量 checkpoint 在修复后的 public-compatible 与 NewMatcher-compatible 推理链上可以对齐。
- **直接数值原因是主 SAM2 prompt-to-mask 分支学出的 checkpoint 不同。** matcher 辅助输出近似一致，但 public 的 SAM2 final-mask loss 从训练伊始就显著更差，并通过 10 倍权重持续主导优化。

不能严谨声称：某一行名为 `vp_matcher_new.py` 的代码单独贡献了 `+0.2157 IoU`。这两次运行不是锁定所有初始张量、DataLoader 顺序、稀疏点抽样和 CUDA kernel 的逐位确定性 A/B；`deterministic=False` 加上每对象单点随机抽样，足以让 SAM2 这种提示敏感的 decoder 从第一个 logging window 就分叉。若要把剩余原因压到某一个 runtime 张量，唯一充分的实验是保存同一初始化，在同一 batch 上固定 correspondence points，逐层比较 language embedding、SAM2 mask logits 和梯度。

因此，最准确且有解释力的一句话是：**public 与 v10 使用的是同一 Fusion 思路和同一活跃 matcher；public epoch 2 低，是该次非确定性训练中 sparse/language prompt 驱动的 SAM2 主分支从开局就处于更差状态，10 倍主损失又把差异持续放大；v10 epoch 2 高，是同一模型图训练出了更好的 SAM2 prompt-to-mask checkpoint，而不是 NewMatcher 另有一套更强匹配算法。**

## 2. 证据等级与指标口径

### 2.1 证据等级

| 标签 | 含义 |
| --- | --- |
| `VERIFIED` | 有完整运行、哈希、结构或独立复算证据 |
| `MEASURED` | 本次实际运行得到，但尚未形成不可变 receipt |
| `INFERRED` | 由多项证据支持，但没有端到端证明 |
| `FALSIFIED` | 已被实验否定的假设 |
| `OPEN` | 尚待实验回答 |

### 2.2 指标必须分列

| 指标 | 聚合方式 | 备注 |
| --- | --- | --- |
| stock / legacy | 公开 evaluator 的原始聚合 | 最接近本次官方 checkpoint 的历史 headline |
| per-object | 所有对象等权 | 多对象 pair 权重更大 |
| per-pair | 先在 pair 内平均对象，再让 pair 等权 | 不能与 stock 混写 |
| paper mIoU | 论文协议主表 | 只有协议完全一致时才能直接比较 |

本文所有小数均为 `[0, 1]` 范围；换算为百分数时乘以 100。

## 3. 关键身份与不可变产物

### 3.1 基础资产

| 资产 | SHA256 / 身份 | 状态 |
| --- | --- | --- |
| SAM2 Hiera Large | `7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b` | `VERIFIED` |
| DINOv3 ViT-L/16 | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` | `VERIFIED` |
| 原 Fusion checkpoint | `f0c986c0296c3eee9a64ee5fef9f48c60e0b1de28a94cf711f6037824da6eddb` | `VERIFIED` |
| 原 Visual checkpoint | `80b3a2ab59b9453734aac2060f9c8a2f530b548b6f312f87efd7e9545aa9e3eb` | `VERIFIED` |
| Ego2Exo test JSON | `86f5803758b3c8da16e0c1f373f001a1cd14633497b05a7ce33ae6ddfcc490de` | 40,517 records |

### 3.2 安全转换与权重等价性

| 检查 | 结果 |
| --- | --- |
| Fusion safetensors SHA256 | `901dab88af1a90571fd60008cac53d8c92e41bc7e23f184e56ca15b105cec6d0` |
| safetensors 张量数 | 1,335 |
| 逻辑张量字节数 | 2,125,559,976 |
| 源 checkpoint 与 safetensors 逐张量源字节比较 | 1,335/1,335 通过 |
| safetensors structure SHA256 | `bb5ce1af90bd6beeea5c0c5f646e1653a41d5afaca1f04096bab354bd35a0a6a` |
| 安全转换报告 | `SAFE_VS_SOURCE_BYTES=PASS` |

新旧发布权重的逐张量审计：

| 专家 | 旧文件 SHA | 新文件 SHA | 精确相同张量 | 张量内容摘要 |
| --- | --- | --- | ---: | --- |
| Fusion | `f0c986...` | `b865d5659a18cec43d7e98c0177a08d3f3bfc73134de4a6cc6e10378295dae0e` | 1,335/1,335 | `bdeac5eae9610000cf5aa486421723b2675c4115db1e990b382349b652d345f3` |
| Visual | `80b3a2...` | `0feb0137674c56dabeeda9a23303277600e9858b97a6a32fd91532c19704e41c` | 964/964 | `8c2bcf0816998da1120cc77a255e5bc334e339174918157b0430da574c1720e6` |

审计报告：

- 远端：`$REPRO/ego2exo_weight_tensor_equivalence.json`
- 报告 SHA256：`3a91e64f453e0a90686460bbf4e3b1000b8c2af8b94f2d4107306e1b13aa181d`
- 结论：文件 SHA 不同是 wrapper/metadata 等封装差异，不是模型张量差异。

### 3.3 安全加载器

安全加载器已完成以下核心门禁：

- 禁止 `torch.load`、`pickle.load`、`pickle.loads` 等运行时反序列化入口；
- 校验 safetensors 文件 SHA；
- 校验源 checkpoint、inventory、structure、tensor count 元数据；
- 校验实际张量数；
- `strict=True` 加载真实 `torch.nn.Module`，缺 key 必须失败；
- 已生成可运行的 safetensors 测试入口并完成 4-GPU smoke。

因此，后续低指标不能归因于安全加载器修改了权重。

## 4. 实验日记

### 4.1 2026-08-18：环境、checkpoint 和结构门禁

最初直接从 overlay 导入 MMEngine 时遇到：

```text
ModuleNotFoundError: No module named 'mmengine.dist'
```

根因是加载了 overlay 内不完整/不匹配的 MMEngine 路径。固定 `PYTHONPATH` 到已知可用的 `$REPRO/V2sam/mmengine` 后，运行时身份为：

- MMEngine `0.11.0rc0`；
- PyTorch `2.3.1+cu121`；
- CUDA wheel `12.1`；
- GPU `NVIDIA H100 80GB HBM3`。

随后模型构造因 `loss_mask=None` 失败。使用作者 loss 配置后完成 NewMatcher 结构探针：

| 项目 | 结果 |
| --- | ---: |
| expected / actual state entries | 1,335 / 1,335 |
| missing / unexpected | 0 / 0 |
| shape / dtype mismatch | 0 / 0 |
| 参数量 | 531,315,988 |
| 可训练参数量 | 7,682,082 |
| key/shape SHA | `3de8f7c3b9d242f9fbc23472c17a4fddcf35c3df42bad71367056bb584d3fe4b` |

这证明 checkpoint 与该 NewMatcher 类的**状态结构兼容**，但没有证明 forward 语义等价。

### 4.2 2026-08-18：安全 checkpoint 转换

采用静态检查 PyTorch ZIP storage、禁止 pickle 执行的方式，把 Fusion checkpoint 转为 safetensors。转换完成后又直接从源 ZIP storage 读取每个张量的逻辑字节，与 safetensors 中的张量逐一比较。

最终：

```text
checked_tensors: 1335
checked_storages: 1335
checked_logical_bytes: 2125559976
SAFE_VS_SOURCE_BYTES=PASS
```

该结果比“能成功 load”更强：它证明两个容器内的模型张量字节完全一致。

### 4.3 2026-08-18：NCCL 二进制错配

Python 包元数据显示 `nvidia-nccl-cu12==2.20.5`，但直接调用 `libnccl.so.2` 得到版本 `22907`，说明包 metadata 与实际动态库不一致。强制重装后：

- `ncclGetVersion` 返回 `22005`；
- 4-rank all-reduce 均得到 `sum=10.0`；
- 日志中四个 rank 的输出发生粘连，简单 `grep` 一度只数到两个 PASS，这是假阴性，不是 NCCL 仍失败。

结论：多卡通信环境已恢复，后续训练差异不能简单归因于 NCCL 无法工作。

### 4.4 2026-08-18：trusted pickle 与 safetensors 对照

在同一 8-record smoke scope 上：

| 加载路径 | Mean IoU | Mean Dice |
| --- | ---: | ---: |
| trusted pickle | 0.3725 | 0.5413 |
| safetensors | 0.3725 | 0.5412 |

随后在 40,517-record test JSON 上并行运行：

| 路径 | records | Mean IoU | Mean Dice | errors |
| --- | ---: | ---: | ---: | ---: |
| trusted pickle | 40,517 | 0.1725506309 | 0.2131154827 | 0 |
| safetensors | 40,517 | 0.1724805634 | 0.2130128645 | 0 |

固定 `seed=530358027`、8 GPU 再跑 safetensors：

```text
Mean IoU  = 0.17269247010249295
Mean Dice = 0.21320835745203878
```

三次结果接近，说明 `0.172x` 是稳定的该执行路径结果，不是随机 seed 或安全格式引入的偏差。

### 4.5 2026-08-18：切回高分 NewMatcher-compatible 参考路径

当时使用的目录名和入口曾被记作“公开 Fusion overlay”，但后续源码审计确认：该运行时实际激活的 matcher 与 `vp_matcher_new.py` 逻辑等价，checkpoint 内嵌 `model.type` 也是 `V2SAM_NEWMATCHER`。因此下面结果只能标为 **NewMatcher-compatible reference**，不能归因于原生 public `V2SAM`：

```text
Iter(test) [5065/5065]
Mean IoU  = 0.4469
Mean Dice = 0.5097
```

完整 evaluator 给出：

| Fusion checkpoint + NewMatcher-compatible runtime | IoU | Dice |
| --- | ---: | ---: |
| stock | 0.446970 | 0.509795 |
| per-object | 0.402292 | 0.460900 |
| per-pair | 0.449375 | 0.508969 |

计数：

- image-pairs：40,517；
- objects：100,223；
- raw per-pair JSON：约 4.0 MB。

这说明 checkpoint、本地测试数据和 NewMatcher-compatible 推理链可以组合出合理的作者量级结果；它**不证明**原生 public Fusion forward 能得到同样结果。作者侧使用原生 public Fusion 的复现同样偏低，与本次 public `V2SAM` 基线方向一致，因此高分与低分的核心差异应归因于 matcher/forward 语义，而不是把两条链都统称为 public `V2SAM`。

### 4.6 2026-08-18：公开官方 Visual 路径

公开 Visual checkpoint 在同一 Ego2Exo 测试集上：

| Visual official | IoU | Dice |
| --- | ---: | ---: |
| stock | 0.367035 | 0.428027 |
| per-object | 0.330651 | 0.385719 |
| per-pair | 0.382888 | 0.438850 |

旧 stock 日志显示 `0.3666 / 0.4276`，差异来自具体 evaluator/打印精度，量级一致。

### 4.7 2026-08-18：否定“last-object 指标导致 0.172”

对上述 NewMatcher-compatible reference 的 `raw_per_pair.json` 独立取每个 pair 的最后一个对象再平均：

```text
pairs: 40517
legacy_last_object_IoU:  0.44139143677711373
legacy_last_object_Dice: 0.5018778901705068
```

它与 `0.1726 / 0.2131` 完全不匹配。因此：

- `FALSIFIED`：低值是 evaluator 只取最后对象造成；
- `OPEN`（当时状态）：低分 NewMatcher overlay 与高分 NewMatcher-compatible reference 的 forward、prompt 生成、mask decoder 或运行配置究竟在哪一步分叉。

### 4.8 2026-08-18 至 2026-08-19：训练数据恢复

官方 Mini payload 共八段：

- part00 至 part06：每段 5,000,000,000 bytes；
- part07：2,879,173,120 bytes；
- 合计：37,879,173,120 bytes。

part03 最后约 225 MB 长时间停滞，最终用 HTTP Range 小段恢复并校验：

```text
part03 bytes  = 5000000000
part03 SHA256 = c76c2e6f83080ead8d2cbe0d2f0a7deb5e905e7237a4e696cf599820bd7f7cd6
```

Mini 分段并非可直接顺序 `cat | tar` 的规范 tar，流式提取出现大段零区和 `tar_rc=2`。不能仅凭 `tar` 警告断言 JPEG 已损坏；必须以 JSON 引用闭包和 JPEG 首尾字节为准。

Full 包的前若干段实际包含一个嵌套 ZIP：

| 项目 | 值 |
| --- | ---: |
| outer member | `data_segswap/data_segswap.zip` |
| zip offset | 1,024 |
| zip size | 119,567,081,059 |
| ZIP entries | 871,261 |

前 `aa` 至 `al` 已连续覆盖整个嵌套 ZIP，因此不需要等 Full 的所有后续分段才能读取其中 JPEG。

训练 JSON：

| JSON | bytes | SHA256 | records | unique image refs |
| --- | ---: | --- | ---: | ---: |
| Ego2Exo FullTrain | 457,116,335 | `bb83cac92179b21833d2465ce29d5b0cf81668d8443d569279313c891472175a` | 110,118 | 220,236 |
| Exo2Ego FullTrain | 496,995,598 | `212e7291990453b087f91dd848dd8a8a8f11a7a0a110fb257be760ca9c8c3058` | 123,381 | 246,762 |

两个方向的 union：

```text
union_referenced_images: 249642
pre_repair_missing: 6431
post_repair_missing: 0
post_repair_bad_jpeg: 0
DUAL_FULLTRAIN_FINAL_CLOSURE=PASS
```

最终 closure 报告 SHA256：

```text
ad6bc9dec706532e94b1b4bbd0b5a43579e6e6ce767e06479c780558803b140f
```

注意：本地初次 Mini 提取缺 6,431 张图，可能来自非规范分段的恢复/提取方式，不能仅凭该现象断言上游 Mini 仓库本身缺文件。修复后的本地训练目录已通过完整引用闭包。

Hugging Face repack 已完整上传到 `Travor278/V2SAM-EgoExo-Train-Mini-Complete`，本地 11 个载荷文件与远端逐项闭环核验为 `missing=[]`、`unexpected=[]`（远端另含平台生成的 `.gitattributes`）。2026-08-22 又补充了完整 dataset card，并在同一提交中同步更新 `SHA256SUMS`；远端回读哈希一致，提交为 `de7daed32c0a6db3317ba2ff54ceeaf62c0b6a98`。仓库当前按用户决定设为 public；README 明确提示该公开状态不授予任何额外的数据权利，使用者仍需遵守 Ego-Exo4D 的原始条款。

### 4.9 2026-08-19：Fusion 与 Visual 从头训练

两套训练均固定：

- 4 GPU；
- seed `530358027`；
- per-device batch `16`；
- gradient accumulation `4`；
- AdamW，初始 LR `4e-5`；
- BF16/官方 scheduler 框架；
- Ego2Exo FullTrain JSON；
- 修复后的训练图像闭包。

若 batch 语义没有额外缩放，则每次 optimizer step 的 nominal effective batch 为 `4 × 16 × 4 = 256`。该数值必须在后续 receipt 中明确记录，不能只写“batch 16”。

#### Fusion 训练

前 40 iterations 的日志显示总 loss 约从 `479` 到 `474`，其中 `loss_contr` 约 `453~459`，明显支配总 loss。epoch 2：

| 评测方式 | IoU | Dice |
| --- | ---: | ---: |
| 训练内 validation | 0.1465 | 0.1841 |
| epoch-2 checkpoint 独立 test | 0.1468 | 0.1845 |

训练内外一致，说明不是 validation hook 的统计错误。该轨迹继续到 epoch 4 时，训练内 validation 曾达到 `0.2937 / 0.3513`，说明 public Fusion 并非完全学不会、epoch 2 也不是其上界；但后续恢复段在约 `iter 8700` 时解析到接近 `4e-10` 的异常低 LR，且旧 checkpoint 没有持久化 contrast step，因此这条历史 run 不能作为干净 24-epoch public Fusion 的上界证据。

但该 run 实际解析到的模型类为：

```text
projects.v2sam.models.v2sam.V2SAM
```

而官方 checkpoint meta 内嵌配置是：

```text
projects.v2sam.models.V2SAM_NEWMATCHER
```

所以这次 Fusion run 只能标为“公开 V2SAM 训练基线”，不能标为“官方 NewMatcher 严格复现失败”。

#### Visual 训练

| Epoch | IoU | Dice |
| ---: | ---: | ---: |
| 1 | 0.1203 | 0.1522 |
| 2 | 0.2171 | 0.2679 |
| 3 | 0.2997 | 0.3583 |
| 4 | 0.3149 | 0.3721 |
| 5 | 0.3220 | 0.3804 |
| 6 | 0.3260 | 0.3836 |
| 7 | 0.3395 | 0.3989 |
| 8 | 0.3390 | 0.3963 |
| 9 | 0.3431 | 0.4018 |
| 10 | 0.3427 | 0.4010 |
| 11 | 0.3462 | 0.4051 |
| 12 | **0.3468** | **0.4057** |

结论：

- 学习过程稳定、总体仍在缓慢上升；
- 相对官方 Visual stock `0.3666~0.3670`，IoU 尚差约 `0.0198~0.0202`；
- epoch 12 时原 cosine LR 已接近/达到 0；
- checkpoint 设置为 `save_optimizer=False`，因此不能精确恢复原 optimizer/scheduler 状态；
- 所谓“续训”应定义为从 epoch-12 权重启动一条新的低 LR fine-tune，而不是无缝 resume。

### 4.10 2026-08-20 至 2026-08-21：Visual 低 LR 与可续训 24-epoch 训练

低 LR 12-epoch 路线完成后得到 `0.3469 / 0.4059`，没有超过原始 12-epoch 基线，也没有追平官方 checkpoint。随后启动了保存 optimizer、scheduler、message hub 和持久化 contrast step 的 24-epoch 从头训练，并对多个里程碑做独立全量测试：

| checkpoint | IoU | Dice | 相对官方 Visual `0.367035 / 0.428027` |
| --- | ---: | ---: | --- |
| `iter_20800.pth` | 0.3550 | 0.4144 | `-0.012035 / -0.013627` |
| `iter_27600.pth` | **0.3683** | **0.4305** | **`+0.001265 / +0.002473`** |
| `iter_34400.pth` | 0.3658 | 0.4275 | `-0.001235 / -0.000527` |
| `iter_41376.pth` | 0.3650 | 0.4260 | `-0.002035 / -0.002027` |

最佳权重为：

`$REPRO/runs/train-visual-fromscratch-24epoch-resumable-seed530358027-20260821-v1/retained_eval_candidates/iter_27600.pth`

结论：

- 按本次公开 evaluator 的 stock 口径，最佳点已略高于官方公开 Visual checkpoint；
- 最佳点相对论文表中 Visual Ego2Exo mIoU `0.362` 高 `0.0063`，但论文值只有在协议完全一致时才能直接比较，因此不据此宣布超过论文；
- 训练后段发生约 `0.0033` IoU 回落，后续应缩短评测间隔、保留 top-k，并以 held-out validation 选择 checkpoint；
- 最终 checkpoint 写入完整可恢复状态，但 cosine scheduler 的 `eta_min=0` 在终点触发 MMEngine `loss_factor should be larger than zero` 断言。checkpoint 本身可用；新配置必须令终点 LR 严格大于零。

### 4.11 2026-08-20 至 2026-08-21：Fusion/NewMatcher 修复与 2-epoch 诊断

NewMatcher 相对早期私有分支已经修复/验证的关键合同包括：SAM2 sparse prompt 透传、batch>1 聚合、sparse correspondence 病理循环、可持久化 contrast step、离线路径与 sampler/config 合同。修复后的 first-32 推理与 NewMatcher-compatible reference 逐对象完全一致：

| first-32 指标 | NewMatcher-compatible reference | NewMatcher | 最大绝对差 |
| --- | ---: | ---: | ---: |
| per-object IoU | 0.735047133308 | 0.735047133308 | 0 |
| per-object Dice | 0.763872641108 | 0.763872641108 | 0 |

2-epoch 诊断 checkpoint 的两次独立原生全量评测均为 `IoU 0.3834 / Dice 0.4327`。修复 evaluator 后，首轮完整 `40,517`-pair / `100,223`-object 评测得到：

| 聚合口径 | IoU | Dice |
| --- | ---: | ---: |
| per-object（正式全对象） | 0.362473582955 | 0.409392593508 |
| per-pair | 0.394539910507 | 0.443716221019 |
| legacy last-object | 0.383425743432 | 0.432728857493 |

旧 `0.3834 / 0.4327` 与新 evaluator 独立重算的 legacy last-object 结果逐值一致，因此它**不能**直接与 NewMatcher-compatible reference `0.446970 / 0.509795` 或论文 Fusion mIoU `0.445` 比较，原因有两项：

1. NewMatcher 原生 evaluator 的对象循环缩进错误，只把每个 pair 的最后一个对象加入结果；
2. 该 run 曾在 stock MMEngine 中途恢复，存在 replay/游标偏移风险，不是干净的严格 2-epoch 训练。

截至 2026-08-21，本轮审计已完成并通过：

- GT mask 原值、全零、随机和缺失反事实的预测一致性；
- checkpoint strict load 与 missing/unexpected key 合同；
- batch16 训练一次覆盖 16 个样本、每样本 5 个对象，共 80 个对象；
- optimizer、scheduler、message hub、contrast step 和 Adam moments 的严格断点续训；
- 23 项 overlay、1 项 sparse prompt、4 项 contrast state，共 28 项回归测试；
- 首轮全量 raw 输出与 test JSON 的每-pair 对象数序列逐项一致，合计恰为 `100,223` 个对象。

第二次独立全量复测也已完成：

| 完整全对象复测 | per-object IoU | per-object Dice | per-pair IoU | per-pair Dice | legacy IoU / Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第一次 | 0.362473582955 | 0.409392593508 | 0.394539910507 | 0.443716221019 | 0.383425743432 / 0.432728857493 |
| 第二次 | 0.362487187750 | 0.409403154414 | 0.394546017031 | 0.443721089678 | 0.383427493012 / 0.432727472011 |

两次评测的每-pair 对象数序列均与 test JSON 完全一致，且均为 `40,517` pairs / `100,223` objects。per-object 聚合差仅为约 `1.36e-5` IoU、`1.06e-5` Dice。因为评测配置明确使用 `randomness.deterministic=False`，不能宣称逐对象 raw 逐字节相同；这里的结论是**完整计数与聚合结果可重复**。

最终整合门于 2026-08-21 全部通过：28 项新鲜回归、GT 反事实、六层 train/test 零精确重叠、batch16 的 80/80 对象语义、strict checkpoint state load 和两次完整全量复测均为 PASS。审计版发布为 `v2sam-newmatcher-training-overlay-v11`；24-epoch 正式 run 已于 `2026-08-21T13:38Z` 在 GPU 0--3 启动，batch `16`、accumulation `4`、AdamW LR `4e-5`，每 400 iter 保存 optimizer、scheduler、message hub 和持久化 contrast step。这里的“state load”只证明状态张量完整；2026-08-22 进一步确认 stock epoch loop 不保存 mid-epoch dataloader cursor，故游标严格恢复另由 v12 处理。

数据泄漏审计已经完成：train `110,118` 条、test `40,517` 条；top-level key、video id、clip UUID、target path、prompt path 和 canonical-record hash 的精确 overlap 均为 `0`。仅 basename overlap=`1,314`，样本是不同视频中常见的帧号文件名，不能据此认定泄漏。

### 4.12 关键训练配置台账（复现时以本节为准）

以下只记录已经由配置、checkpoint 或运行日志证明的字段。这里的“从头”表示**不加载此前的 Ego2Exo 训练 checkpoint**；SAM2 Hiera Large 与 DINOv3 ViT-L/16 基础预训练权重仍按模型定义加载，不等于所有参数随机初始化。

共同数据与硬件合同：

- train：Ego2Exo FullTrain，`110,118` records，修复后引用闭包 `missing=0 / bad_jpeg=0`；
- test：完整 `40,517` pairs / `100,223` objects；
- seed：`530358027`，CUDA 仍为 `deterministic=False`；
- 正式多卡训练：4 GPU，per-device batch `16`，gradient accumulation `4`，nominal effective batch `256`；
- 混合精度：BF16；正式 checkpoint 必须包含 model、optimizer、scheduler、message hub 与持久化 contrast step。

| 实验 | 初始化与模型 | 优化器 / 调度 | 训练长度与保存 | 评测与结论 |
| --- | --- | --- | --- | --- |
| Public Fusion baseline | public `V2SAM`；不加载 Ego2Exo checkpoint | AdamW，LR `4e-5`；其余沿冻结配置 | epoch 2 独立 test；历史轨迹继续到 epoch 4 | epoch 2 为 `0.1468 / 0.1845`，epoch 4 训练内 validation 为 `0.2937 / 0.3513`；后续恢复段 LR/contrast 状态不可靠，只作为公开 Fusion 基线，不能称为官方 NewMatcher 复现 |
| Visual 12e baseline | public Visual `V2SAM`；从头任务 | AdamW `4e-5`；5% warmup，即 Linear `0→0.6 epoch`，Cosine `0.6→12` | 12 epoch；历史 checkpoint 未保存 optimizer，不能无缝续跑 | 最佳/最终 epoch 12：`0.3468 / 0.4057` |
| Visual resumable 24e | public Visual `V2SAM`；从头任务 | AdamW `4e-5`；Linear `0→1.2 epoch`，Cosine `1.2→24` | 24 epoch，完整恢复状态；里程碑独立全量评测 | `iter_27600=0.3683/0.4305` 最佳；`iter_41376=0.3650/0.4260`，证明最后点不是最佳点 |
| Visual EMA800 | 从 Visual `iter_27600` warm-start；即时权重与 EMA 双状态 | AdamW，沿用 checkpoint 实际 LR `1.0946120568e-5`；batch `8`、accumulation `8`；EMA momentum `0.002` | 800 iter；100 optimizer steps；严格状态审计通过 | 全量 `0.3690 / 0.4311`，当前 Visual 最佳 |
| Visual hybrid Muon400 | 从 Visual `iter_27600` warm-start | 17 个 matcher/prompt 二维矩阵用 Muon `4e-5`，其余 166 个状态用 AdamW `1e-6`；batch `8`、accumulation `8` | 400 iter | `0.3623 / 0.4226`，不晋升 |
| NewMatcher v10 diagnostic | `V2SAM_NEWMATCHER`；历史恢复链曾从 `iter_1800` 续到精确 2-epoch 边界 `iter_3448`，**不是新的干净从头 run** | AdamW `4e-5`；作者原始 Linear `0→1.2 epoch`、Cosine `1.2→24` | 2-epoch 诊断 checkpoint | 两次全对象：`0.362474/0.409393`、`0.362487/0.409403`；legacy 为 `0.3834/0.4327` |
| NewMatcher v11 | 审计代码；`load_from=None, resume=False`，从头启动 | **错误配置：Linear `0→24` 与 Cosine `1.2→24` 重叠**；`iter_5200` 实际 LR 仅约 `4.95e-6` | 到 `iter_5532`，最后稳定 `iter_5200` | `iter_4000` 全对象 `0.349390/0.395382`；该轨迹不能代表作者原始 scheduler |
| NewMatcher 正式 24e（Fresh24） | v12 审计源码树；`load_from=None, resume=False`；只加载 SAM2/DINOv3 基础权重 | AdamW `4e-5`；作者原始 Linear `0→1.2 epoch`、Cosine `1.2→24` | 24 epoch；每 epoch 保存完整状态；分布式任务 `job-bef509d3-de12-4244-b44c-733f4fb7f35b` | 从头训练的 epoch 2 全对象结果：per-object `0.359630 / 0.406739`，per-pair `0.3910 / 0.4400`，legacy-last-object `0.3801 / 0.4292`；40,517 pairs / 100,223 objects 独立复算通过，正式训练继续 |

作者调度的来源不是本次手工调参：原仓库配置使用 `warmup_ratio=0.05`，因此 12e 对应 `0.6` epoch warmup，24e 对应 `1.2` epoch warmup；官方 Fusion checkpoint 的内嵌 cfg 也记录了 24e、AdamW `4e-5`、Linear `0→1.2` 与 Cosine `1.2→24`。

## 5. 官方 checkpoint 的真实训练谱系

### 5.1 checkpoint meta

官方 Fusion checkpoint 包含：

| 字段 | 值 |
| --- | --- |
| epoch | 14 |
| iter | 24,136 |
| seed | 530358027 |
| experiment_name | `v2sam_20251101_214431` |
| time | `20251103_044754` |
| mmengine_version | `0.10.67b88c299...` |
| model type | `projects.v2sam.models.V2SAM_NEWMATCHER` |
| pretrained_pth | `None` |
| batch_size | 16 |
| accumulative_counts | 4 |
| max_epochs | 24 |
| optimizer | AdamW, LR `4e-5` |

### 5.2 代码来源

- 公开仓库 `jaychempan/V2-SAM` 的历史中没有 `V2SAM_NEWMATCHER`；
- 私有/受限仓库 `jaychempan/V2-SAM-O` 的 `feat/MOE` 分支存在该实现；
- 当前最强 provenance 候选为 commit `7b88c29999e7ab180d0303f3d05914afe700a992`；
- 本地已生成该 commit 的隔离包与 provenance 文件；
- 但 `feat/MOE` 当前活动 config 是 `V2SAM_DualDecoder_NEW`、SmallTrain、batch 8、12 epoch、带 pretrained，不能直接拿来当官方 checkpoint 的训练配置。

正确做法是：以 commit `7b88c299...` 的实现为代码根，以 checkpoint 内嵌 cfg 为训练合同，重构一份冻结的 runtime config。

### 5.3 已解决的路径标签混淆

同一官方 Fusion 权重曾产生以下表面矛盾：

- 被误标为公开 `V2SAM` 的 NewMatcher-compatible reference 得到 `0.4469 / 0.5097`；
- 当前 NewMatcher overlay forward 得到稳定的 `0.172x / 0.213x`；
- NewMatcher state structure 又与 checkpoint 1,335/1,335 完全兼容；
- checkpoint meta 明确声称训练类是 NewMatcher。

后续 pipeline trace 已确认高分 reference 的活跃 matcher 本身就是 NewMatcher 语义；作者使用原生 public Fusion 的结果也与本次低 public 基线同量级。故“public Fusion 高、NewMatcher 低”这一前提不成立。真正需要解释的是：早期低分 NewMatcher overlay 与正确 NewMatcher-compatible runtime 之间至少有一项未对齐：

1. checkout/commit 并非真正生成 checkpoint 的源码状态；
2. runtime config 中 prompt、matcher 或 decoder 参数未按嵌入 cfg 构造；
3. checkpoint key 虽全加载，但两个类对相同子模块的 forward 解释不同；
4. test pipeline/template/data mapping 不同；
5. 某个类变量或非 state-dict 状态没有保存在 checkpoint 中。

这是下一轮严格复现的 P0 问题。

## 6. 作者 H20 复现反思与代码风险点

以下来自作者/同学在 H20 上复现不稳定的截图和报告。它们与本地代码审计相互印证，但尚未全部通过单变量实验量化。

### 6.1 seed 与非确定性

- 历史配置出现 `seed=None, deterministic=False`；
- Transformer dropout、loss 点采样、对象采样、随机初始化仍会引入随机性；
- 同 seed 不等于绝对 bitwise deterministic，CUDA kernel、BF16/TF32 和 argmax 邻近值仍可能分叉。

当前 run 已固定 `seed=530358027` 和 `PYTHONHASHSEED`，但仍需记录每个 rank 的 sampler、首批 sample IDs 和 sparse point digest。

### 6.2 对比学习不是真正跨卡全局 batch

代码中的跨卡 `all_gather` 路径被注释，对比负样本主要来自单 rank。于是 GPU 数、每卡 batch 和 accumulation 即使得到相同 nominal effective batch，对比目标仍不等价。

### 6.3 `loss_contr` 存在 100 到 1 的硬切换

在 `_constr_step < 4000` 时对比损失乘 100，之后突然变成 1。风险包括：

- 计数是每个进程的 forward 次数，而不是全局 optimizer step；
- world size、batch、accumulation 或数据长度变化会改变切换发生的实际训练位置；
- `_constr_step` 是运行时类/实例状态，未确认会随 checkpoint 持久化；中断续跑可能从 0 重新计数；
- 当前早期 Fusion 日志确实由 `loss_contr` 支配。

### 6.4 对象采样可能有放回

`np.random.choice(n_objects, select_number)` 未显式写 `replace=False`。当需要选择多个对象时可能重复抽到同一对象，而对比 loss 又把非对角位置当负样本，形成 false negatives。

严格复现阶段必须保留原行为；改进实验再单独测试无放回采样和多正样本 label。

### 6.5 RegionPooling 对 sigmoid soft mask 使用 `nonzero()`

预测 mask 已过 sigmoid，绝大多数位置严格大于 0；对它调用 `nonzero()` 近似把整幅特征图都视为前景，再均匀抽点。这会放大噪声并使训练对微小数值变化敏感。

候选改进是 mask-weighted pooling 或明确阈值，但不能混入严格复现实验。

### 6.6 sparse correspondence 的 top-1 argmax 脆弱

当前每对象只保留很少点，DINO 相似度接近时，BF16/TF32 或 kernel 顺序差异可能交换 top-1，随后 SAM 收到完全不同的点提示。后续可测试：

- top-k 置信点；
- mutual nearest neighbor；
- 空间去重/最小距离；
- 固定缓存 sparse points，将 anchor 随机性与可训练网络随机性分离。

## 7. 已排除、已确认与仍待回答的问题

### 7.1 已排除

| 假设 | 结论 | 证据 |
| --- | --- | --- |
| safetensors 改坏了权重 | `FALSIFIED` | 1,335 张量逐字节一致，trusted/safe smoke 一致 |
| 下载到的新旧 checkpoint 内容不同 | `FALSIFIED` | Fusion 1,335/1,335、Visual 964/964 完全一致 |
| `0.1726` 是随机 seed 偶然值 | `FALSIFIED` | 多次及固定 seed 结果都约 `0.172x` |
| `0.1726` 是 last-object evaluator bug | `FALSIFIED` | 同 raw predictions 的 last-object 为 `0.4414` |
| 官方 Fusion 权重本身有问题 | `FALSIFIED` | NewMatcher-compatible reference 得到 `0.44697`；原生 public Fusion 偏低不能反推权重损坏 |
| 训练内 validation 算错导致 epoch-2 假低 | `FALSIFIED` | 独立 test `0.1468` 与训练内 `0.1465` 一致 |

### 7.2 已确认

| 事实 | 状态 |
| --- | --- |
| 官方 checkpoint meta 指向 NewMatcher | `VERIFIED` |
| 当前从头 Fusion 实际训练的是公开 V2SAM | `VERIFIED` |
| NewMatcher 与 checkpoint state structure 完全兼容 | `VERIFIED` |
| 原生 public V2SAM 结果偏低；早期 NewMatcher overlay 也因运行时未对齐而偏低 | `MEASURED`；两者均不能与正确 NewMatcher-compatible reference 混称为同一路径 |
| Visual 12 epoch 稳定上升但未追平官方 | `MEASURED` |
| Visual 24-epoch 最佳 `iter_27600` 为 `0.3683 / 0.4305`，EMA800 进一步达到 `0.3690 / 0.4311` | `MEASURED` |
| train/test 完整 image path 与 top-level key 均无精确重叠 | `VERIFIED` |
| 修复后 NewMatcher first-32 与 NewMatcher-compatible reference 逐值一致 | `VERIFIED` |
| NewMatcher 原生 evaluator 只统计每个 pair 的最后对象 | `VERIFIED_BUG` |
| NewMatcher 2-epoch 首轮全对象结果为 `0.362474 / 0.409393` | `MEASURED` |
| 旧 `0.3834 / 0.4327` 恰为 legacy last-object 聚合 | `VERIFIED_BUG_IMPACT` |
| NewMatcher GT 反事实、strict load、batch16 和 checkpoint 状态完整性 | `VERIFIED` |
| NewMatcher v11 stock mid-epoch dataloader cursor | `MISSING_RISK`；v12 cursor-aligned loop 已修复 |
| NewMatcher v11 共 28 项静态/运行时回归门禁 | `PASS` |
| 数据 union 引用闭包经修复后 missing=0、bad_jpeg=0 | `VERIFIED` |

### 7.3 待回答

1. 产生官方 checkpoint 的精确 NewMatcher source tree 是否就是 `7b88c299...`？
2. checkpoint 内嵌 cfg 中是否还有未被当前 runtime config 恢复的字段？
3. 第二次独立全对象 NewMatcher 评测能否逐对象、逐指标复现首轮结果？
4. 24-epoch 正式训练的 checkpoint cadence 如何兼顾严格续训与存储开销？
5. batch 1 与 batch 16 的推理逐样本数值是否还需扩展到完整随机子集逐值比较？
6. stock MMEngine 的 epoch 中途 resume 是否还需额外保存 sampler/cursor，避免非 accumulation-boundary 的 replay 或样本偏移？
8. Visual 的后续非 LR 消融能否稳定超过当前 EMA800 最佳 `0.3690 / 0.4311`？
9. HF repack 已完整上传、README/SHA256SUMS 已同步更新并通过远端回读验证；仍可选做一次从空目录回下载后的全载荷逐 SHA 验证。

## 8. 下一阶段实验计划

### P0：恢复严格 NewMatcher 合同

目标不是立刻跑 24 epoch，而是先证明“官方 checkpoint + 精确 NewMatcher runtime”可以复现高分 NewMatcher-compatible reference 的预测。

#### P0.1 冻结来源

- 固定 `V2-SAM-O@7b88c29999e7ab180d0303f3d05914afe700a992`；
- 保存完整 source manifest、每个 Python 文件 SHA 和工作树状态；
- 从官方 checkpoint 的 `meta.cfg` 生成 runtime config；
- 禁止使用分支当前的 DualDecoder/SmallTrain 活动配置；
- 固定 MMEngine、PyTorch、CUDA、NCCL 和 DINO/SAM2 SHA。

#### P0.2 推理等价 gate

在相同的 8、32、256 个 records 上同时运行：

1. 已复现 `0.4469` 的 NewMatcher-compatible reference；
2. 精确重构的 NewMatcher。

对每个 sample 保存：

- 输入 sample ID、query/target 路径；
- prompt mask 摘要；
- DINO sparse point 坐标、置信度与 SHA；
- matcher 输出摘要；
- decoder logits/pred mask SHA；
- 每对象 IoU/Dice。

从第一个不一致的节点定位根因。没有通过小样本 forward parity 前，不启动 24 epoch。

**Go 条件：** 32-record prediction 在预设容差内等价，且 checkpoint strict load 无 missing/unexpected。
**Stop 条件：** 同一输入在 sparse point 之前已经分叉；先定位 config/source，不再扩大数据集。

### P1：严格 NewMatcher 两 epoch 诊断训练

通过 P0 后运行 4-GPU gate：

- seed `530358027`；
- batch 16 / GPU；
- accumulation 4；
- AdamW LR `4e-5`；
- 24-epoch scheduler horizon，即使 gate 只先跑 2 epoch；
- 不修复原始采样、contrastive 或 pooling 行为。

必须额外导出：

| 类别 | 字段 |
| --- | --- |
| 优化 | optimizer step、micro step、LR、grad norm、loss scale |
| loss | `loss_mask`、`loss_dice`、`small_loss_*`、contrastive raw/scaled/weight |
| 状态 | `_constr_step`、4000-step 切换时刻、rank/world size |
| 数据 | sample IDs、对象数、重复对象率、每 rank batch digest |
| prompt | soft-mask min/max/mean、非零率、sparse point 数/置信度/hash |

总 loss 单独一条曲线不够，必须拆出 raw contrastive 与乘权后的 contrastive。

### P2：Visual 低 LR 短程续训

这是刷榜分支，不是严格复现分支。

基线方案：

- 初始化：Visual epoch-12 最佳权重；
- optimizer：新 AdamW；
- LR：`2e-6`；
- 新 warmup + cosine，2 至 4 epoch；
- 其余数据、seed、batch 和增强保持不变；
- 用独立 held-out validation 选点，test 只做最终一次确认。

建议 stop/go：

- 1 epoch 后 held-out IoU 提升至少 `0.003`：继续；
- 下降超过 `0.005` 或 loss/grad 异常：停止；
- 连续两次 validation 提升小于 `0.001`：停止；
- 不以反复查看 test set 来调 LR。

备选 LR 消融只做单变量：`1e-6`、`2e-6`、`5e-6`。先跑短 gate，再把最优一条延长。

### P3：训练稳定性改进消融

严格复现完成后，按以下顺序一次只改一个因素：

| 编号 | 改动 | 主要问题 | 优先级 |
| --- | --- | --- | --- |
| S1 | 记录并缓存 DINO sparse points | 分离 anchor 与网络随机性 | 高 |
| S2 | 对象采样 `replace=False` | 重复对象/false negatives | 高 |
| S3 | 多正样本 contrastive labels | 同对象重复被当负样本 | 高 |
| S4 | 跨 rank all-gather | 单卡负样本池不等价 | 高，但通信风险较大 |
| S5 | 100→1 平滑 schedule | loss 硬切换 | 高 |
| S6 | mask-weighted RegionPooling | soft mask `nonzero()` 近似全图 | 高 |
| S7 | top-k / mutual sparse points | top-1 argmax 脆弱 | 中高 |
| S8 | 更严格 deterministic 设置 | 定位重复性 | 诊断用途 |

每个改动都保留：基础 checkpoint、训练步数、样本序列、指标口径和 seed，仅改变目标因素。

### P4：Muon 可行性实验

Muon 不作为当前首选，原因：

- 当前 PyTorch `2.3.1` 没有可直接使用的原生 Muon；
- Muon 通常只适用于二维 hidden weight；
- bias、norm、embedding、卷积权重仍需要 AdamW 或明确 reshape/排除策略；
- 当前最大不确定性来自模型谱系和训练目标，而不是 AdamW 收敛速度。

如果 P0/P1 完成后仍要试 Muon，应采用 hybrid optimizer：二维线性层用 Muon，其余参数保持 AdamW，并先在固定 1,000 optimizer steps 的小实验比较：

- wall-clock；
- 显存；
- grad norm；
- 各分量 loss；
- held-out IoU；
- 是否产生 NaN/发散。

Muon 实验不得与采样、pooling 或 loss schedule 修复同时进行。

## 9. 资源与执行顺序

考虑 H100 成本，推荐顺序：

1. **CPU：** 完成 source/config diff、HF 远端闭包复核、loss 导出补丁和小样本 manifest；
2. **1 GPU：** 8/32-record 中间张量 trace；
3. **4 GPU：** NewMatcher 256-record parity 和 2-epoch gate；
4. **另 4 GPU（有预算时）：** Visual 低 LR 短程续训；
5. 只有 P0/P1 通过后才考虑完整 24 epoch；
6. Muon 和算法修复在严格基线之后排队。

## 10. 下一次开机前检查表

- [x] private HF repack 上传已结束，状态为 `complete` 且上传脚本输出 `HF_PRIVATE_UPLOAD=PASS`；
- [x] 远端 metadata/manifest 与载荷文件列表已核对，`missing=[]`、`unexpected=[]`；完整 tar 回下载逐 SHA 仍为可选的高成本复核；
- [ ] Ego2Exo/Exo2Ego union closure 再次得到 `249642 / missing=0 / bad_jpeg=0`；
- [ ] 固定 `7b88c299...` source tree，工作树无漂移；
- [ ] 从 checkpoint `meta.cfg` 生成配置，不引用活动 branch config；
- [ ] 记录 4 GPU、batch 16、accumulation 4 的真实 effective batch；
- [x] 官方 checkpoint 在 NewMatcher-compatible reference 路径复现 `0.4469 / 0.5097`；不得再标为原生 public V2SAM；
- [ ] NewMatcher 32-record trace 完成并定位首个分叉节点；
- [ ] loss 日志包含 raw/scaled contrastive 和 `_constr_step`；
- [ ] 训练支持在 optimizer accumulation 边界保存可恢复 checkpoint；
- [ ] Visual 续训使用独立 held-out validation，不反复调试 test。

## 11. 当前实验矩阵

### 11.1 官方 checkpoint 推理

| 专家/路径 | 权重 | scope | stock IoU | stock Dice | per-object IoU | per-pair IoU | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fusion NewMatcher-compatible runtime | official Fusion（checkpoint meta=`V2SAM_NEWMATCHER`） | 40,517 | 0.446970 | 0.509795 | 0.402292 | 0.449375 | `VERIFIED`；不可标成原始 public Fusion |
| Visual public V2SAM | official Visual | 40,517 | 0.367035 | 0.428027 | 0.330651 | 0.382888 | `VERIFIED` |
| NewMatcher overlay, trusted | same Fusion tensors | 40,517 | 0.172551 | 0.213115 | — | — | `MEASURED`, runtime mismatch |
| NewMatcher overlay, safetensors | same Fusion tensors | 40,517 | 0.172481 | 0.213013 | — | — | `MEASURED`, runtime mismatch |
| NewMatcher overlay, safetensors, fixed seed | same Fusion tensors | 40,517 | 0.172692 | 0.213208 | — | — | `MEASURED`, runtime mismatch |

### 11.2 从头训练

| 实验 | 模型类 | epoch | IoU | Dice | 判断 |
| --- | --- | ---: | ---: | ---: | --- |
| Fusion baseline | public `V2SAM` | 2 | 0.1468 | 0.1845 | 有效基线，但不是官方 NewMatcher 复现 |
| Fusion baseline（同一历史轨迹） | public `V2SAM` | 4 | 0.2937 | 0.3513 | 训练内 validation；证明 epoch 2 不是上界，但后续 LR/恢复合同异常，不能外推为干净 24e 结果 |
| Visual baseline | public Visual | 12 | 0.3468 | 0.4057 | 有效基线，仍低于官方约 0.020 IoU |
| Visual low-LR continuation | public Visual | 12 | 0.3469 | 0.4059 | 已完成，无提升 |
| Visual resumable from-scratch, best | public Visual | `iter_27600` | **0.3683** | **0.4305** | 从头训练轨迹最佳；较官方公开 checkpoint 分别约 `+0.001265 IoU / +0.002473 Dice` |
| Visual resumable from-scratch, final | public Visual | 24 | 0.3650 | 0.4260 | 后段回落，不是最佳 checkpoint |
| Visual hybrid Muon screen | public Visual + allowlisted Muon | `400 iter` | 0.3623 | 0.4226 | 从 `iter_27600` 启动；仅 17 个 matcher/prompt 二维隐藏矩阵用 Muon，其余 166 个训练参数仍用 AdamW；`iter_400` 严格续训审计通过，但完整评测低于当前最佳，不晋升 |
| Visual frozen-decoder screen | public Visual + frozen SAM2 decoder | `400 iter` | 0.3644 | 0.4260 | 修正双重加载后完整评测；低于 EMA800，不晋升 |
| Visual EMA screen | public Visual + AdamW + EMA | `800 iter` | **0.3690** | **0.4311** | 当前 Visual 最佳；完整 40,517-pair 评测，较官方约 `+0.001965 / +0.003073` |
| Visual Hybrid Muon Fresh24 | public Visual + hybrid Muon/AdamW | 2 | 0.0956 | 0.1235 | 真正从头、完整同口径验证；显著落后 AdamW+EMA，epoch-2 checkpoint 严格续跑审计通过后停止，不继续占用 4 卡 |
| Visual AdamW+EMA Fresh24 | public Visual + AdamW + EMA | 2 / 4 / 6 / 8 / 10 / 12 / 14 / 16 / 18 / 20 / 22 / 24（已完成） | 0.2251 / 0.3107 / 0.3301 / 0.3507 / 0.3585 / 0.3631 / 0.3659 / 0.3667 / **0.3702** / 0.3672 / 0.3698 / 0.3688 | 0.2800 / 0.3707 / 0.3905 / 0.4115 / 0.4192 / 0.4230 / 0.4265 / 0.4267 / **0.4300** / 0.4271 / 0.4295 / 0.4282 | 真正从头、完整同口径验证；epoch 18 为全轨迹最佳。最终 `epoch_24.pth`（`epoch=24, iter=82752, contrast=86751`）的 optimizer/scheduler/message hub/EMA、累计梯度边界和相邻 epoch 计数全部通过严格续跑审计，但指标未刷新，因此最佳仍为已按同 inode 固化的 `epoch_18.pth` |
| Visual AdamW+EMA 分层 LR Fresh24 | public Visual + AdamW + EMA；SAM2 可训练部分 LR 降为 `1e-5` | 2 | 0.1802 | 0.2211 | 真正从头；比同期统一 LR 主线低 `0.0449 / 0.0589`，epoch-2 checkpoint 严格恢复状态齐全后停止，不继续占用 GPU |
| Visual AdamW+EMA Fresh24 seed 2 | public Visual + AdamW + EMA；仅 seed=`530358028` | 2 / 4 / 6 / 8 / 10 / 12 / 14 / 16 / 18 / 20（epoch 20→24 分布式续训已提交） | 0.2338 / 0.3251 / 0.3412 / 0.3576 / **0.3671** / 0.3625 / 0.3597 / 0.3588 / 0.3581 / 0.3566 | 0.2893 / 0.3876 / 0.4031 / 0.4197 / **0.4297** / 0.4248 / 0.4200 / 0.4192 / 0.4185 / 0.4162 | 与主线保持相同数据、模型、优化器、调度、batch/accumulation 和 EMA，只改变 seed；epoch 12/14/16/18/20 连续较 epoch 10 回落，因此最佳仍为 epoch 10。`epoch_10.pth` 已通过严格审计并以同 inode 硬链接保存；`epoch_20.pth`（`epoch=20, iter=68960, contrast=72959`）的 optimizer/scheduler/message hub/EMA、累计梯度边界和相邻 epoch 计数也已通过严格续跑审计，未晋升。已提交独立 4×H100 分布式严格续训任务，当前排队，进入运行后首先核验恢复游标 |
| NewMatcher v10 diagnostic（全对象，repeat 1） | `V2SAM_NEWMATCHER` | 2 | 0.362474 | 0.409393 | 40,517 pairs / 100,223 objects；完整计数通过 |
| NewMatcher v10 diagnostic（全对象，repeat 2） | `V2SAM_NEWMATCHER` | 2 | 0.362487 | 0.409403 | 独立复测；聚合差约 1e-5 |
| NewMatcher v10 diagnostic（legacy） | `V2SAM_NEWMATCHER` | 2 | 0.3834 | 0.4327 | 只取每 pair 最后对象，非论文可比，非严格续训 |
| Strict NewMatcher v11 | `V2SAM_NEWMATCHER` | `iter_4000` | 0.349390 | 0.395382 | 全对象口径；虽从头启动，但误用 24-epoch Linear warmup，轨迹作废、不续跑 |
| Strict NewMatcher Fresh24 | `V2SAM_NEWMATCHER` | 2 / 6 / 8 / 24（运行中） | 0.359630 / **0.382618** / 评测中 / 待到达 | 0.406739 / **0.438121** / 评测中 / 待到达 | 真正从头；独立 4×H100 分布式任务。epoch 6 全对象门为 40,517 pairs / 100,223 objects，per-pair `0.4285/0.4851`、legacy-last-object `0.4215/0.4796`，较 epoch 2 明显上升；`epoch_6.pth` 与 `epoch_8.pth` 均已通过 checkpoint 门。独立 4090 已自动接续 epoch 8 全对象评测，正式训练不停机 |

Visual EMA checkpoint 的含义已按实际 MMEngine `EMAHook` 源码复核：保存时 EMA 权重进入主 `state_dict`，即时训练权重进入 `ema_state_dict`；严格 resume 时 hook 会交换回正确的两套状态，普通独立 test 则直接加载主 `state_dict` 中的 EMA 权重。因此训练内 EMA validation、独立 test 与可续跑 checkpoint 三者的语义一致。

### 11.3 论文数值参考

这些是论文/作者报告参考值，不是本文重新计算值：

| 组合 | Ego2Exo mIoU | Exo2Ego mIoU |
| --- | ---: | ---: |
| A / Anchor | 0.387 | 0.416 |
| B / Visual | 0.362 | 0.466 |
| C / Fusion | 0.445 | 0.473 |
| A+B / PCCS | 0.427 | 0.482 |
| A+B+C / PCCS | 0.463 | 0.496 |

当前只严格完成了 B 的公开 Visual 单专家推理，以及 C 的 NewMatcher-compatible 单专家推理。A、A+B 和 A+B+C 仍缺经过审计的 GT-free Anchor/PCCS 运行合同，不能用 oracle selector 冒充。

## 12. 关键路径索引

远端路径以：

```text
BASE=/inspire/ssd/project/luojianlan/zhubingwen-253108120125
REPRO=$BASE/v2sam-repro
HDD=/inspire/hdd/project/luojianlan/zhubingwen-253108120125
```

为根。

| 内容 | 路径 |
| --- | --- |
| 安全 Fusion 权重 | `$REPRO/fusion_ego2exo_full.safetensors` |
| 权重等价报告 | `$REPRO/ego2exo_weight_tensor_equivalence.json` |
| 最终训练数据闭包 | `$REPRO/ego2exo_exo2ego_fulltrain_union_closure.final.json` |
| Fusion full metric raw | `$REPRO/runs/official-fusion-fullmetric-ego2exo-20260818T182933Z/raw_per_pair.json` |
| Visual full metric raw | `$REPRO/runs/official-visual-fullmetric-ego2exo-retry-20260818T195622Z/raw_per_pair.json` |
| Fusion 从头训练 | `$REPRO/runs/train-fusion-ego2exo-seed530358027-20260819T123626Z/` |
| Visual 从头训练 | `$REPRO/runs/train-visual-ego2exo-seed530358027-20260819T124809Z-v2/` |
| Visual 24-epoch 最佳 checkpoint | `$REPRO/runs/train-visual-fromscratch-24epoch-resumable-seed530358027-20260821-v1/retained_eval_candidates/iter_27600.pth` |
| GitHub NewMatcher 审计分支 | `jaychempan/V2-SAM-O@travor`, commit `bd7ba86314b8a2916e4af0d55eba1ed1b37c271e` |
| NewMatcher v10 2-epoch 诊断 checkpoint | `$REPRO/runs/train-newmatcher-v10-strict-2epoch-seed530358027-20260821-v1/iter_3448.pth` |
| NewMatcher v11 错调度诊断 run | `$REPRO/runs/train-newmatcher-v11-strict-24epoch-seed530358027-20260821-v1/` |
| NewMatcher 正式 24-epoch run | `$REPRO/runs/train-newmatcher-v12-fresh24-fromscratch-seed530358027-20260822-v1/`；不得复用 v11 work dir 或 checkpoint |
| NewMatcher 最终整合门 | `$HDD/codex_v11_integrated_release_gate.json` |

本地补充文档：

- `V2SAM_EXPERIMENT_MATRIX_LIVE.md`：跨 Ego2Exo/Exo2Ego、PCCS 和其他实验的长期矩阵；
- `docs/superpowers/plans/2026-08-19-v2sam-ego2exo-remaining-reproduction.md`：上一阶段单专家推理计划；
- `V2SAM_O_WRZ_TRIPLE_DECODER_QUICKSTART.md`：内部多专家/TripleDecoder 代码入口说明。

## 13. 最终判断

目前不是“指标复现不出来”，而是已经把问题拆成了两部分：

1. **推理复现成功但实现标签已纠正：** 官方 Fusion 权重在 NewMatcher-compatible 路径、Visual 权重在公开 Visual 路径得到合理且稳定的结果；原先把 Fusion `0.446970 / 0.509795` 标成 public `V2SAM` 是错误的，因为 checkpoint meta 指向 `V2SAM_NEWMATCHER`，且该 overlay 的活跃 matcher 实现实际等价于 `vp_matcher_new.py`。作者侧使用原生 public Fusion 的结果也与我们的低 public 基线同量级，进一步排除了“原生 public Fusion 本来就能得到 0.446970”的解释。
2. **Visual 训练已达到官方公开 checkpoint 的量级：** 24-epoch 轨迹最佳 `iter_27600` 为 `0.3683 / 0.4305`，但最后 checkpoint 回落，说明后续刷榜要加强 checkpoint 选择而不是只延长训练。
3. **NewMatcher 的预测/评测审计已在当前声明范围内通过，正式 Fresh24 已从头启动并完成 epoch 2 全对象门：** 两次 v10 诊断 checkpoint 全对象结果为 `0.362474 / 0.409393` 和 `0.362487 / 0.409403`；旧 `0.3834 / 0.4327` 已证实是 last-object evaluator 口径。GT-free、六层数据泄漏、strict load、batch16、完整 evaluator 和回归门均未发现作弊证据。v11 的低轨迹已经定位为 Linear warmup 被错误延长到 24 epoch，不能续跑或用于评价作者调度。Fresh24 使用 `load_from=None, resume=False` 和作者原始 1.2-epoch warmup；epoch 2 checkpoint 的完整评测为 per-object `0.359630 / 0.406739`、per-pair `0.3910 / 0.4400`、legacy-last-object `0.3801 / 0.4292`，40,517 pairs / 100,223 objects、有限值和聚合均独立复算通过，训练保持连续运行。

交互式 Visual 主 seed 已完成全部 24 epoch，epoch 24 为 `0.3688/0.4282`，未超过 epoch 18 的 `0.3702/0.4300`；最终 checkpoint 已严格审计通过，但没有晋升。第二 seed 的 epoch 2/4/6/8/10/12/14/16/18/20 为 `0.2338/0.2893`、`0.3251/0.3876`、`0.3412/0.4031`、`0.3576/0.4197`、`0.3671/0.4297`、`0.3625/0.4248`、`0.3597/0.4200`、`0.3588/0.4192`、`0.3581/0.4185`、`0.3566/0.4162`，最佳仍为已固化的 epoch 10；epoch 20 checkpoint 已通过严格续跑审计，4×H100 分布式 epoch 20→24 严格续训任务已经提交、当前排队。四个跨 seed averaging checkpoint 已通过严格 key/shape/dtype 与 manifest 门；早先四路全量评测因交互实例回收仅运行到约 2.8k–3.0k/40,517，**没有产生最终指标**，恢复资源后必须从头补评测，不能与已完成的同轨迹 soup 混写。Hybrid Muon、分层学习率、Muon400、freeze-decoder400、同轨迹 soup、多档 LR continuation 与 EMA1200 均已形成完整负结果，不再重复。正式 NewMatcher Fresh24 使用独立 4×H100 分布式训练资源；epoch 6 全对象评测已完成为 `0.382618/0.438121`，epoch 8 评测已由独立 4090 自动接续，正式训练不停机。

## 14. 2026-08-22 交付与运行状态

### 14.1 审计代码交付

- 已将审计后的 NewMatcher 最小改动集推送到私有仓库 `jaychempan/V2-SAM-O` 的 `travor` 分支；
- GitHub `travor` 远端当前提交固定为 `bd7ba86314b8a2916e4af0d55eba1ed1b37c271e`，本地与远端 SHA 已交叉核验一致；该提交在原审计提交 `c7de25ff` 上补齐旧 checkpoint 缺失 `contrast_schedule.step` 时的严格兼容加载，已有该键的 checkpoint 不会被覆盖，其他无关 missing key 仍然报错；
- 提交包含严格 checkpoint 加载、完整对象 evaluator、持久化 contrast schedule、稀疏 prompt 修复、batch-16 聚合修复、稀疏点分层性能修复、可移植 24-epoch 配置和审计回归测试；
- 提交不包含训练数据、checkpoint、日志、token、服务器绝对路径或未完成的 safetensors 实验产物；
- 本地最终门禁为 19 项通过、6 项因缺少本地训练依赖而跳过；远端 CPU 动态回归为 37/37 通过。GPU 门禁随后确认 first-32 与 NewMatcher-compatible 参考逐值完全一致（46 个 IoU/Dice 值最大差为 0），真实 4 卡 batch-16 完成一次前反向并保存 `iter_1.pth`：`meta.iter=1`、`contrast_schedule.step=1`，optimizer、scheduler、message hub 均存在且日志无异常。门禁自动进入的完整验证不属于门禁合同，审计 checkpoint 后已终止并归还 GPU。

### 14.2 Mini 数据集与 Hugging Face dataset card

- 正确且闭包完整的 Mini 数据集已经上传至 Hugging Face 仓库 `Travor278/V2SAM-EgoExo-Train-Mini-Complete`；
- 上传状态为 `complete`；仓库当前按用户决定设为 public（`private=false`）；
- 本地 11 个载荷文件均已在远端闭环核验，`missing=[]`、`unexpected=[]`；远端另含 Hugging Face 自动生成的 `.gitattributes`；
- 数据包包含 4 个 tar 分片、Ego2Exo/Exo2Ego JSON、SHA256 清单和 provenance 闭包记录。
- HF README 已写清原 Mini 提取在修复前缺 6,431 条引用、修复后缺失/坏 JPEG 均为 0、两个方向的记录与唯一图像规模、四分片布局、下载/校验/解压命令及授权提示；README 与 `SHA256SUMS` 在提交 `de7daed32c0a6db3317ba2ff54ceeaf62c0b6a98` 中原子更新并通过远端回读哈希校验。

### 14.3 当前状态（精简快照）

> 本节只记录会变化的在线状态；已完成实验的配置、结果和失败原因统一保留在第 4、7、11 节，避免把进程日志当实验结论。

| 工作流 | 当前状态 | 固定合同 / 下一门 |
| --- | --- | --- |
| Visual Hybrid Muon Fresh24 | epoch 2 完整验证后已停止 | `0.0956 / 0.1235`，同期 AdamW+EMA 为 `0.2251 / 0.2800`；epoch-2 checkpoint 严格续跑状态完整，但该优化器路线不晋升 |
| Visual AdamW+EMA Fresh24 | 主 seed 已完成 24 epoch | epoch 2/4/6/8/10/12/14/16/18/20/22/24 为 `0.2251/0.2800`、`0.3107/0.3707`、`0.3301/0.3905`、`0.3507/0.4115`、`0.3585/0.4192`、`0.3631/0.4230`、`0.3659/0.4265`、`0.3667/0.4267`、`0.3702/0.4300`、`0.3672/0.4271`、`0.3698/0.4295`、`0.3688/0.4282`；`epoch_24.pth` 严格审计通过，最佳仍为已固化的 `epoch_18.pth` |
| Visual cross-seed soup | 四个 checkpoint 已生成；全量评测待从头补跑 | 主 e18 与 seed2 e10 按 `75/25`、`50/50`、`25/75` 平均，另测主 e18/e22/seed2 e10 的 `50/25/25`；四个 manifest/output SHA256 门均通过，平均 checkpoint 仅供评测、不可续训。先前四路评测在实例回收前仅到约 2.8k–3.0k/40,517，未产生最终指标；晋升门仍为完整指标超过 `0.3702/0.4300` |
| Visual AdamW+EMA 分层 LR Fresh24 | epoch 2 完整验证后已停止，GPU 0–3 已归还 | `0.1802 / 0.2211`，显著低于同期统一 LR 主线；稳定 epoch-2 checkpoint 含 optimizer/scheduler/message hub/EMA 与 183 个 optimizer groups，保留为负结果但路线不晋升 |
| Visual AdamW+EMA Fresh24 seed 2 | epoch 20→24 独立 4×H100 分布式续训排队中 | 除 seed=`530358028` 外与主线合同相同；epoch 2/4/6/8/10/12/14/16/18/20 为 `0.2338/0.2893`、`0.3251/0.3876`、`0.3412/0.4031`、`0.3576/0.4197`、`0.3671/0.4297`、`0.3625/0.4248`、`0.3597/0.4200`、`0.3588/0.4192`、`0.3581/0.4185`、`0.3566/0.4162`；最佳仍为已固化的 `epoch_10.pth`；`epoch_20.pth` 的 optimizer/scheduler/message hub/EMA、累计梯度边界和游标已严格审计通过，进入运行后下一门为 epoch 22 |
| NewMatcher Fresh24 | 独立 4×H100 分布式任务持续运行；已生成 `epoch_8.pth` | `load_from=None, resume=False`；AdamW `4e-5`；Linear `0→1.2 epoch`、Cosine `1.2→24`；epoch 6 全对象 `0.382618/0.438121`（40,517 pairs / 100,223 objects），per-pair `0.4285/0.4851`，legacy `0.4215/0.4796`；epoch 8 checkpoint 门通过且全对象评测已在独立 4090 启动，训练不停机 |
| NewMatcher 源码放行 | GitHub 远端已更新并核验为 `travor@bd7ba863` | CPU 回归 37/37、first-32 逐值 parity、真实 4 卡 batch-16 与完整 checkpoint 状态门均通过；正式 Fresh24 继续使用已冻结的 v12 源码/配置，不在运行中热替换 |
| CPU/4090 sidecar | epoch-6 已完成，epoch-8 全对象评测运行中 | epoch 6 原始输出通过 `40,517 records / 100,223 objects`、完整字段与有限值门，结果 `0.382618382571/0.438121441618`；自动链已校验 `epoch_8.pth` 并启动下一轮评测，正式 NewMatcher 训练全程不停 |

已冻结的运行/评测配置：

- Visual AdamW+EMA 分层 LR：`$REPRO/frozen_configs/visual_fromscratch24_adamw_ema_discriminative_lr_seed530358027_20260822_v1.py`，SHA256 `9c9ddedbddb276842789e68338fd31ba1c414bf7ab9c8cfc99af872caa2daa57`；`grounding_encoder=1e-5`，matcher/prompt 与其他任务层维持 `4e-5`；epoch-2 负结果后已停止；
- Visual AdamW+EMA seed 2：`$REPRO/frozen_configs/visual_fromscratch24_adamw_ema_seed530358028_20260822_v1.py`，SHA256 `e3159659b4a6e547078968f53ef378fb49bc33425e4efe3ab244f87b4215f2f2`；仅 seed 与主线不同，现使用 GPU 0–3 从头运行；
- NewMatcher Fresh24 epoch-2 独立全对象评测：`$REPRO/frozen_configs/newmatcher_v12_fresh24_epoch2_fullobject_eval_seed530358027_20260822_v1.py`，SHA256 `4717a215cd21da51518e58eb79d9b6a668de70c6e5f993df0a71caf1e428f7fd`；输出 `raw_per_pair.json`，并同时报告 per-object、per-pair 与 legacy-last-object。
- NewMatcher Fresh24 epoch-6 独立全对象评测：`$REPRO/frozen_configs/newmatcher_v12_fresh24_epoch6_fullobject_eval_seed530358027_20260823_v1.py`，SHA256 `6c3c3ec0a72818743f0ba0ab57987927be09983bff4b61c61452782ecec67025`；使用独立 4090 sidecar，不占用或停止正式训练。

新增了严格的仅评测 checkpoint averaging/SWA 工具 `$REPRO/tools/average_model_checkpoints.py`（SHA256 `6843f4c87944f2c70f30fadb1413f207842b1502ed8a34f128f2cf9b1bb458f5`）及回归测试 `$REPRO/tests/test_average_model_checkpoints.py`（SHA256 `f58ba8ddf2dbf667176a242a59cfdb288ba9ec0771a447c872be9d579d028ecc`）。工具要求所有输入 `state_dict` 的 key、shape、dtype 严格相同；浮点/复数张量用 float64/complex128 累加后恢复原 dtype，计数器等非浮点张量取最新 checkpoint；原子输出只含 `meta/state_dict`，写入输入/output SHA256 清单并明确 `resume_supported=false`，不可误作训练续跑点。测试按 RED→GREEN 执行，2/2 通过。

当前资源原则：

- 交互式 8×H100 继续保持并行利用：GPU 0–3 跑仅改变 seed 的第二条 AdamW+EMA；主 seed 完成后释放的 GPU 4–7 已立即切换为四路 cross-seed soup 全量评测；Muon 与分层 LR 路线均已基于完整结果淘汰；
- NewMatcher 使用独立分布式训练资源持续运行；epoch 2 checkpoint 到达后，优先由共享存储的 CPU/4090 sidecar 另起全对象 test；只有 4090 不满足运行合同或 ETA 明显不可接受时，才使用经 Visual epoch-2 筛选释放出的 4 张交互式 GPU，正式训练进程始终不停机；
- 任一 Visual 试验结束后立即核验完整指标、checkpoint 状态与日志，再决定释放 GPU 或继续下一项非 LR 消融；
- 正式 NewMatcher 的 epoch-2 全量结果必须使用 `40,517 pairs / 100,223 objects` 的全对象口径，同时另列 legacy-last-object，禁止混写。
