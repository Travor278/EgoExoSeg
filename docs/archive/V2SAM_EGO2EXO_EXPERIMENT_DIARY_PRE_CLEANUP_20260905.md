# V2-SAM Ego2Exo 复现实验日记与下一阶段计划

> 快照日期：2026-08-30（Asia/Shanghai）<br>
> 远端实验日期：2026-08-18 至 2026-08-30（UTC）<br>
> 范围：Ego2Exo/Exo2Ego 数据、权重、官方推理、从头训练、NewMatcher、Visual/Fusion 与 PCCS 收尾实验<br>
> 原则：机器产物、哈希和完整日志优先于聊天记录；本文中的“论文值”“本次实测”“推断”和“待验证假设”严格分开。

## 1. 一页结论

### 1.1 结论：历史 Public 的预训练 DINOv3 被二次初始化

public epoch 2 为 `0.1468 / 0.1845`，v10 NewMatcher epoch 2 全对象为 `0.36248 / 0.40940`，相差约 `+0.2157 IoU / +0.2249 Dice`。根因不是 NewMatcher 换了更强的 matcher，而是**历史 Public 在加载预训练 DINOv3 后，又被 MMEngine 的顶层 `init_weights()` 覆盖成随机初始化；v10 保留了预训练 DINO。**

错误 DINO 直接改变 sparse point，并通过额外 RNG 消费改变 matcher dropout 与 language prompt，最终使 SAM2 主 mask logits 从首 batch 就分叉。只阻止这次二次初始化后，12 个边界 trace 与 v10 全部逐张量一致，iter 1 和 iter 10 的 loss 也精确恢复到 v10。完整证据见 1.3；旧 evaluator 的 `iou_vp > 0` 过滤是另一个指标膨胀问题，不是训练根因。

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

这也解释了为何 coarse-mask 的 `small_loss_*` 可以相同，而主 `loss_mask/loss_dice` 相差巨大：历史 Public 的 DINO 被重置后，单个 correspondence point 与 language prompt 已经不同；两者随后共同改变 SAM2 `_forward_sam_heads` 的 point embedding、multimask 选择与 mask logits。

### 1.3 已定位根因：Public 在 MMEngine 初始化阶段把已加载的 DINOv3 重新随机初始化

> **一句话根因：** 历史 Public 把已加载预训练权重的 DINOv3 同时注册成 `V2SAM` 的顶层子模块和 sparse-correspondence 的子模块；MMEngine 随后对顶层 DINO 再调用一次 `init_weights()`，覆盖了预训练权重。v10 只把 DINO 保存在普通 `SparseCorrespondenceMatcher` 内部，因此预训练 DINO 得以保留。

#### 1.3.1 历史来源与最小代码差异

本结论不以“当前两个 overlay 看起来相同”代替历史事实。来源链已经回到历史材料：

- Aug17 源码归档 `v2sam-source-24ae5a-20260817.tar.gz` 的 SHA256 为 `d097991ccd8ce8dc80e33578eb4727b9883fdb5f0e6388573764269439b88b2c`；成员 `V2sam/projects/v2sam_fusion/models/v2sam.py` 含有下面的 Public 构造方式；
- 历史启动 config 实际导入 `projects.v2sam.models.V2SAM` 与 `SAM2TrainRunner`；`V2SAM` 内部使用 `VPFeatureMatcher`、`SparseCorrespondenceMatcher` 和 SAM2 runner；
- 使用该 config 和 Public 源码做的独立 4 卡短复现，在 iter 10 得到 `loss_mask=12.5365, loss_dice=4.8748`，与历史日志 `12.5364 / 4.8748` 一致。这证明下面审计的是历史执行行为，不是仅对当前 overlay 的猜测。

最小差异只有 DINO 的注册位置：

| 步骤 | 历史 Public | 正确 v10 |
| --- | --- | --- |
| 加载 DINOv3 | `self.dinov3_model = load_dinov3_model(...)` | `model = load_dinov3_model(...)` |
| 交给 sparse matcher | `dinov3_model=self.dinov3_model` | `dinov3_model=model` |
| 是否为 `V2SAM` 顶层 child | 是 | 否 |

MMEngine 的 `BaseModule.init_weights()` 会遍历直接 children，并对具有 `init_weights` 的 child 再执行初始化：

```python
for module in self.children():
    if hasattr(module, "init_weights"):
        module.init_weights()
```

Public 的顶层 `DinoVisionTransformer` 因而在 checkpoint 已加载后被重新初始化；v10 的 DINO 只位于普通 `SparseCorrespondenceMatcher` 下，而该 matcher 没有 `init_weights()`，所以 MMEngine 不会递归重置它。

历史 checkpoint 直接验证了这一点。对 5 个代表性 DINO tensor：

- v10 epoch 2 与原始 DINO asset **5/5 SHA256 相同**；
- Public epoch 2 与 asset **5/5 不同**，且与 Public 短复现的随机初始化值相同；
- 例如 `blocks.0.attn.qkv.weight` 的 asset/v10 标准差为 `0.03640248`、Public 为 `0.02000921`；asset/v10 的 `norm.weight` 均值/标准差为 `0.6614/0.4833`，Public 则是初始化值 `1/0`；
- Public checkpoint 还同时保存 `dinov3_model.*` 与 `sparse_correspondence.dinov3_model.*` 两套同值 key，证明它们是同一模块的两个注册路径。DINO 在训练中被冻结，因此该错误状态一直保留到 epoch 2。

#### 1.3.2 它如何只打坏主 SAM2 loss

DINO 二次初始化有两个直接后果：

1. sparse correspondence 使用错误的 DINO feature，选出的单个正点发生变化；
2. 二次初始化额外消耗 CUDA RNG；随后相同 matcher 的 Transformer dropout 使用不同 RNG，令 `predict_vp_embeds` 和 language prompt 小幅变化。

同 seed、同 batch、同初始化的边界 trace 给出的第一批证据如下：

| 边界 | Public vs v10 |
| --- | --- |
| batch 数据张量 | 相同；只在捕获的 CUDA RNG 状态上不同 |
| SAM2 image embeddings | 10/10 tensor 相同 |
| matcher coarse `pred_masks_tensor` | SHA256 相同 |
| DINO sparse point | Public `[545.625, 219.375]`；v10 `[511.875, 523.125]` |
| 送入 SAM2 的 point coords | Public `[582, 416]`；v10 `[546, 992]` |
| matcher `predict_vp_embeds` | 因 dropout RNG 不同而不同 |
| SAM2 选中 low-res logits 均值 | Public `-12.81343`；v10 `-24.14244` |
| iter 1 主 loss | Public `15.3671 / 4.8430`；v10 `5.7605 / 3.1070` |

这正好解释日志形态：

- `small_loss_mask/small_loss_dice` 监督的是 matcher 的 coarse mask；该 tensor 在 trace 中逐值相同，所以 auxiliary small losses 相同；
- `loss_contr` 使用 `predict_vp_embeds`，只受到 RNG/dropout 的次级影响，所以两边接近但不完全相同；
- 主 `loss_mask/loss_dice` 监督的是 `language prompt + sparse point -> SAM2 _forward_sam_heads -> low_res_masks`。单点位置和 language prompt 同时改变后，SAM2 主 logits 立即大幅分叉；主 loss 又乘以 10，随后主导不同的优化轨迹。

因此，首个语义张量分叉不在 optimizer、LR、辅助 loss 或 evaluator，而在 **DINO sparse point，以及由同一次错误初始化造成 RNG 偏移后的 language prompt**；它们共同进入 SAM2 主预测。

#### 1.3.3 历史日志与单变量反事实

| iter | 运行 | `loss_mask` | `loss_dice` |
| ---: | --- | ---: | ---: |
| 10 | 历史 Public | 12.5364 | 4.8748 |
| 10 | Public 精确短复现 | 12.5365 | 4.8748 |
| 10 | 历史 v10 | 3.8476 | 2.9977 |
| 100 | 历史 Public | 11.0120 | 4.7979 |
| 100 | 历史 v10 | 2.7519 | 2.9422 |

反事实保持历史 Public config、数据、seed、batch、matcher、sparse correspondence 和 SAM2 runner 不变，只在 `V2SAM` 构造完成后删除顶层 `dinov3_model` alias，使 MMEngine 看不到这个直接 child。冻结 config 与历史 config 的 diff 只有“导入该替代类”和“替换 `model.type`”两行。

结果：

- 日志确认 DINO 参数在 `V2SAMPublicPreserveDino.init_weights()` 前后全部保持不变；
- 与 v10 比较的 12 个边界文件全部 `mismatch_count=0`，包括 batch/RNG、sparse point、matcher 两个输出、language prompt、SAM2 head 全部输出和最终 loss，`first_different_boundary=null`；
- iter 1 精确恢复为 v10 的 `5.7605 / 3.1070`，iter 10 精确恢复为 `3.8476 / 2.9977`；small losses 和 `loss_contr` 也逐项恢复。

这个单变量实验说明：**不需要假设一套“更强 NewMatcher 算法”；仅保住预训练 DINO，就足以把 Public 的整个首 batch 张量链和前 10 iter loss 轨迹恢复到 v10。** 由于错误 DINO 在历史 Public 中被冻结并贯穿 epoch 2，这一根因足以解释为何两次训练从开局进入完全不同的主 SAM2 优化轨迹。尚未直接测量的是“修复 Public 后完整跑 2 epoch 是否精确得到 `0.36248 / 0.40940`”；因此不能把该终点数值写成已复现，但这不再妨碍根因归因。

主要 receipt 位于：

```text
/inspire/hdd/project/luojianlan/zhubingwen-253108120125/
  v2sam_codex_workspace/doc_evidence/root_cause_20260823/
  dist4_trace_20260824_v1/
    historical_dino_checkpoint_hashes.json
    v4/comparison.json
    public_preserve_dino_config.diff
    public_preserve_dino_vs_v10_v3/comparison.json
```

#### 1.3.4 evaluator 膨胀是另一个问题

旧 `projects/v2sam_fusion/evaluation/seg_metric_dualdecoder.py` 存在只保留 `iou_vp > 0` 的统计逻辑。它会删除零 IoU 对象并抬高报告值：v10 全部 `100,223` 个对象中有 `40,816` 个零 IoU，全对象 mean IoU 为约 `0.3624736`，只统计正 IoU 后约为 `0.6115136`；fresh v12 epoch 2 有 `41,284` 个零 IoU，全对象 mean IoU 约为 `0.3596304`。

这个过滤发生在训练之后，既不能改变 `loss_mask/loss_dice`，也不能解释 iter 1/10/100 已出现的训练分叉。它必须作为 evaluator 口径 bug 单独修复，不能与 DINO 初始化根因混写。

### 1.4 证据边界

本次已经证明的是：历史 Public 的实际 DINO 权重状态、该状态如何改变 sparse/language prompt 和 SAM2 主输出，以及只修复这一变量即可逐张量恢复 v10 的首 batch 与前 10 iter。当前 overlay 的表面一致性不作为历史证据；归因依赖 Aug17 归档、历史 checkpoint 哈希、历史日志精确短复现和单变量反事实。

仍未直接证明的是修复后的完整 2-epoch 终点会逐位等于 v10 的 `0.36248 / 0.40940`。如果目标只是确认根因，无需再完整复现两轮；如果目标是给“修复 Public 的 epoch-2 最终指标”建立独立数值 receipt，才需要再跑完整 2 epoch。

### 1.5 双向复现最终状态（2026-08-30）

本轮 V2-SAM 双向复现已经闭环。Ego2Exo 侧，公开 Visual、Strict NewMatcher Fresh24 和 WRZ 旧 PCCS 文件下的候选评测均已完成；Exo2Ego 侧，公开 Visual 12e 与公开 Fusion 24e 均已完成官方训练配置内置的 `SegMetric` 全量验证。关键最终结果为：

| 方向 / 模型 | 最佳 checkpoint | 指标口径 | IoU | Dice | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| Ego2Exo Visual AdamW+EMA Fresh24 | epoch 18 | 全量 `SegMetric` | 0.3702 | 0.4300 | 完成 |
| Ego2Exo Strict NewMatcher Fresh24 | epoch 16 | 全对象 / 作者 frame-level | 0.405197 / 0.450898 | 0.462907 / 0.509557 | 完成 |
| Ego2Exo PCCS（WRZ 旧指标文件） | epoch 16 配对 | Fusion-first triple decoder | 0.4635 | 0.5242 | 候选闭环 |
| Exo2Ego Visual | epoch 11 | 官方配置内置 `SegMetric` | 0.4426 | 0.5013 | 完成 |
| Exo2Ego Fusion | epoch 24 | 官方配置内置 `SegMetric` | **0.4820** | **0.5378** | 完成、exit code 0 |

作者在本轮结束时说明正在整理新的指标文件。因此，WRZ 现有 `seg_metric_fusionfirst_tripledecoder_newmetric.py` 的历史 PCCS 数值继续作为复现 receipt 保存，但不再被强行冻结成“作者最终新口径”。待新文件正式提供后，可在不重训 Visual/Fusion 的前提下直接加载已完成 checkpoint 重评；这属于后续 evaluator 对齐，不影响本轮训练与官方配置内置验证已经复现完成的结论。

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
| Visual EMA800 | 从 Visual `iter_27600` warm-start；即时权重与 EMA 双状态 | AdamW，沿用 checkpoint 实际 LR `1.0946120568e-5`；batch `8`、accumulation `8`；EMA momentum `0.002` | 800 iter；100 optimizer steps；严格状态审计通过 | 全量 `0.3690 / 0.4311`；Dice 为当前最高，但 IoU 低于 Fresh24 epoch 18 的 `0.3702`，因此只保留为 Dice 向 Pareto checkpoint |
| Visual hybrid Muon400 | 从 Visual `iter_27600` warm-start | 17 个 matcher/prompt 二维矩阵用 Muon `4e-5`，其余 166 个状态用 AdamW `1e-6`；batch `8`、accumulation `8` | 400 iter | `0.3623 / 0.4226`，不晋升 |
| NewMatcher v10 diagnostic | `V2SAM_NEWMATCHER`；历史恢复链曾从 `iter_1800` 续到精确 2-epoch 边界 `iter_3448`，**不是新的干净从头 run** | AdamW `4e-5`；作者原始 Linear `0→1.2 epoch`、Cosine `1.2→24` | 2-epoch 诊断 checkpoint | 两次全对象：`0.362474/0.409393`、`0.362487/0.409403`；legacy 为 `0.3834/0.4327` |
| NewMatcher v11 | 审计代码；`load_from=None, resume=False`，从头启动 | **错误配置：Linear `0→24` 与 Cosine `1.2→24` 重叠**；`iter_5200` 实际 LR 仅约 `4.95e-6` | 到 `iter_5532`，最后稳定 `iter_5200` | `iter_4000` 全对象 `0.349390/0.395382`；该轨迹不能代表作者原始 scheduler |
| NewMatcher 正式 24e（Fresh24） | v12 审计源码树；`load_from=None, resume=False`；只加载 SAM2/DINOv3 基础权重 | AdamW `4e-5`；作者原始 Linear `0→1.2 epoch`、Cosine `1.2→24` | 24 epoch 与计划内独立全对象评测均已完成；每 epoch 保存完整状态；训练和 e24 评测任务均已成功结束并释放资源 | 全对象 epoch 2/6/8/10/12/14/16/18/20/22/24 为 `0.359630/0.406739`、`0.382618/0.438121`、`0.384223/0.440600`、`0.381855/0.438381`、`0.397240/0.453889`、`0.401126/0.459220`、**`0.405197/0.462907`**、`0.394619/0.451620`、`0.397792/0.455131`、`0.398107/0.455471`、`0.397958/0.455139`；epoch 16 为最终最佳。e24 checkpoint 与 40,517-pair / 100,223-object raw 均已严格审计 |

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
| Visual 旧 24-epoch 最佳 `iter_27600` 为 `0.3683 / 0.4305`；EMA800 为 `0.3690 / 0.4311`；Fresh24 epoch 18 进一步达到 IoU 主榜最佳 `0.3702 / 0.4300` | `MEASURED` |
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
8. Visual 的后续非 LR 消融能否稳定超过当前 IoU 主榜最佳 Fresh24 epoch 18 `0.3702 / 0.4300`，并同时追平或超过 EMA800 的 Dice `0.4311`？
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
| Visual EMA screen | public Visual + AdamW + EMA | `800 iter` | 0.3690 | **0.4311** | 当前 Dice 最高、IoU 次优的 Pareto checkpoint；完整 40,517-pair 评测，精确值为 `0.368991317952 / 0.431145035264`，较官方约 `+0.001965 / +0.003073`，但 IoU 未超过 Fresh24 epoch 18。最佳点已以双硬链接固化为 `runs/train-visual-best27600-ema800-seed530358027-20260822-v1/best_checkpoints/iter_800_val_iou_0.3690_dice_0.4311_strict_resume.pth`，与原 `iter_800.pth` 同 inode（link count=`2`）。2026-08-24 复核为 `epoch=0, iter=800, contrast=32399`，optimizer、scheduler、message hub、顶层 `ema_state_dict` 均存在且位于 `accum=8` 边界，`STRICT_RESUME=PASS`；文件 SHA256=`abd6447dd179b67b0f105ffa8875d42dc9d7a7d74d9cfb3eb6defacc90778556`。 |
| Visual Hybrid Muon Fresh24 | public Visual + hybrid Muon/AdamW | 2 | 0.0956 | 0.1235 | 真正从头、完整同口径验证；显著落后 AdamW+EMA，epoch-2 checkpoint 严格续跑审计通过后停止，不继续占用 4 卡 |
| Visual AdamW+EMA Fresh24 | public Visual + AdamW + EMA | 2 / 4 / 6 / 8 / 10 / 12 / 14 / 16 / 18 / 20 / 22 / 24（已完成） | 0.2251 / 0.3107 / 0.3301 / 0.3507 / 0.3585 / 0.3631 / 0.3659 / 0.3667 / **0.3702** / 0.3672 / 0.3698 / 0.3688 | 0.2800 / 0.3707 / 0.3905 / 0.4115 / 0.4192 / 0.4230 / 0.4265 / 0.4267 / **0.4300** / 0.4271 / 0.4295 / 0.4282 | 真正从头、完整同口径验证；epoch 18 为全轨迹最佳。最佳点已以双硬链接固化为 `runs/train-visual-fromscratch24-adamw-ema-seed530358027-20260822-v1/best_checkpoints/epoch_18_val_iou_0.3702_dice_0.4300_strict_resume.pth`，与原 `epoch_18.pth` 同 inode（link count=`2`）。2026-08-24 复核为 `epoch=18, iter=62064, contrast=66063`，optimizer、两段 scheduler、message hub、EMA 均存在且位于 `accum=8` 边界，`STRICT_RESUME=PASS`；文件 SHA256=`5db7e0d6cfcdea1a4d413d53926160df22df988237c5f22dc6a8978a311b11ad`。最终 `epoch_24.pth`（`epoch=24, iter=82752, contrast=86751`）同样通过严格续跑审计，但指标未刷新，因此不晋升。 |
| Visual autoresearch AdamW+EMA control | public Visual + 原始对比损失 | `800 iter` | 0.0382 | 0.0583 | 从头 800-iter 同预算健康筛选；checkpoint 已生成，作为 Contrast-MP 的严格对照，不作为完整 24-epoch 排名点 |
| Visual autoresearch Contrast-MP v2 | public Visual + 按 `(video_id, object_id)` 合并同对象多正样本 | `800 iter` | 0.0382 | 0.0583 | 旧标签实验因实际仍调用 `get_contr_loss(..., idx=None)` 作废；v2 补齐 dataset→collate→实际 `projects.v2sam_visual.models.V2SAM`→loss 的 ID 传递，并通过 9 项功能/来源回归、4 项 contrast schedule 回归、真实 batch 数据门和四卡一迭代 checkpoint 门。真正 800-iter 任务完成完整 10,130-batch EMA 验证，结果与同预算原始 AdamW+EMA control `0.0382/0.0583` 完全持平，因此不续到 epoch 2。`iter_800.pth` 为 `meta.iter=800, contrast=4799`，optimizer、两段 scheduler、message hub、EMA 齐全，SHA256=`90d078a674e4993a94927dfc64871cb8b7bccb1675d849349ed402e77a810c85` |
| Visual autoresearch Raw-logit | public Visual + 辅助 small-mask 分支 raw-logit 修正 | `800 iter` | 0.0381 | 0.0583 | 修正 `vp_matcher` 概率再次进入 `use_sigmoid=True` loss 的双重 sigmoid 合同；40,517-pair 全对象评测已完成，结果与 control 的 `0.0382/0.0583` 实质持平，说明该修正单独使用不能带来可测增益，本轮不晋升 |
| Visual autoresearch Aux-off | public Visual + 关闭辅助 small-mask/small-dice 目标 | `800 iter` | 0.0381 | 0.0583 | 保持 AdamW、EMA、seed、初始化、batch、调度和 800-iter 预算不变；四卡 DDP 与严格续训 checkpoint 门通过，完整验证与 control/Raw-logit 持平，说明该辅助分支在此阶段既不是主要增益来源，也不是可单独消除的主要伤害源，不晋升；不再引用旧误标的 Contrast-MP 结果作比较 |
| Visual autoresearch Dense-mask | public Visual + 主 SAM2 BCE/Dice 全分辨率计算 | `800 iter` / epoch 2 / epoch 6（`iter 20688`，已完成） | 0.0512 / 0.2450 / **0.3310** | 0.0769 / 0.3049 / **0.3912** | 仅把主 SAM2 BCE/Dice 从随机点采样改为全分辨率计算；从 `iter_800.pth` 精确恢复到 epoch 2，再从完整 `iter_6896.pth` 精确恢复到 epoch 6，AdamW、EMA、seed、batch/accumulation 与 scheduler 均不变。epoch 2 较同进度 Fresh24 e2 提升 `+0.0199/+0.0249`，但 epoch 6 全量 10,130-batch 验证仅为 `0.3310/0.3912`，较 Fresh24 e6 `0.3301/0.3905` 只高 `+0.0009/+0.0007`，且远低于主榜最佳 Fresh24 e18 `0.3702/0.4300`，说明早期增益未在长程保持，路线不再延长。`iter_20688.pth` 稳定大小 `1,891,240,666` bytes，SHA256=`23fc9645f3e6ed6882170cde3f916e4e7757526c1d2ad0bab5267a16e9eb7c6a`；optimizer、scheduler、message hub、独立 EMA、`contrast_schedule.step=24687` 与累计梯度边界均通过严格续跑审计 |
| Visual autoresearch Dense-loss Anneal | Dense-mask epoch 6 完整状态 → 逐步降低 Dense 主损失权重 | `iter 20688→21488`（800 iter，已完成） | **0.3314** | **0.3917** | 从已审计 Dense-mask `iter_20688.pth` 以游标对齐 loop 精确恢复，不回放 epoch；Dense 权重按 forward `24688→38480` 从 `1→0` 线性退火，本段末约为 `0.9426`，其余 optimizer、EMA、seed、batch/accumulation 与 scheduler 不变。`iter_21488.pth` 为 `meta.iter=21488`、`contrast_schedule.step=25487`，optimizer、两段 scheduler、message hub 与 EMA 齐全，SHA256=`bfcafd0f5deb17617c1fc001eca4aa69db59f28d97aea06ab27e9211170673be`。显式剥离 MMEngine EMA 的 `steps`/`module.` 后导出 965-key 推理 checkpoint（SHA256=`7e14c3924d82c9eee0924bb2e71c19ced95d26e466b4b6f20418c5a850fd0347`），任务 `job-324a6545-83f4-44ba-b15d-1f502928df41` 完成 EMA 的 10,130-batch 全量评测，结果为 `0.3311/0.3913`；配对任务 `job-43ad9d3f-96ec-48a2-aff3-b9b352772188` 对原始 `state_dict` 完成同口径全量评测，结果为 `0.3314/0.3917`。raw 仅比 EMA 高 `+0.0003/+0.0004`、比 Dense e6 高 `+0.0004/+0.0005`，仍显著低于 Fresh24 e18 `0.3702/0.4300`，排除“EMA 短程滞后掩盖有效增益”，不晋升、不继续退火 |
| Visual autoresearch Dense→Sample curriculum | Dense-mask epoch 2 完整状态 → 瞬时恢复原始 sampled BCE/Dice | epoch 2→6（已完成） | **0.3233** | **0.3825** | 从已审计 Dense-mask `iter_6896.pth` 严格恢复，只把 `loss_sample_points` 切回 `True`；AdamW、EMA、seed、batch/accumulation 和 scheduler 不变。任务 `job-41189cb1-1952-43aa-8acd-b8d974393e33` 完成 10,130-batch 全量验证，较同进度 Fresh24 e6 `0.3301/0.3905` 低 `-0.0068/-0.0080`，也低于持续 Dense 的 `0.3310/0.3912`，路线淘汰、不再延长。最终 `iter_20688.pth` 稳定大小 `1,891,238,938` bytes，SHA256=`9e2813c86cddff1d508c56ec384cd1c1b622509a104702ab987121f755cc329f`；`meta.iter=20688`、`contrast_schedule.step=24687`，optimizer、两段 scheduler、message hub、顶层 EMA 与累计梯度边界均通过严格续跑门，故失败归因于方法本身而非恢复或保存故障。切换初期的目标冲击虽然后续消退，但没有保留 Dense e2 的早期收益 |
| Visual autoresearch Lovász | public Visual + per-mask binary Lovász hinge | `800 iter` | 0.0404 | 0.0652 | 保持 AdamW+EMA、seed、有效 batch 与调度不变，Lovász weight=`1.0`；9/9 回归、真实四卡两迭代、`iter_800.pth` 严格状态、800 条有限 loss 历史和 40,517-pair 全量验证均通过。仅略高于 control、低于 Dense-mask，远低于完整 Fresh24，不晋升 |
| Visual autoresearch Muon-v2 | public Visual + 扩展 Hybrid Muon + EMA | `800 iter` | 0.0364 | 0.0579 | 相对旧 Muon 补齐 EMA，并把 Muon 从 17 个 matcher/prompt 二维矩阵扩展到经 allowlist 审计的 68 个隐藏矩阵与中间卷积；其余 115 个 bias/norm/token/input-output head 参数仍由 AdamW 更新。真实四卡训练与完整 10,130-batch 验证完成；`iter_800.pth` 中两组 68/115 参数均有 100 次有限 optimizer 状态，scheduler、EMA、message hub 与 contrast counter 全部通过。结果低于同预算 AdamW+EMA control `0.0382/0.0583`，不晋升，也不延长到 epoch 2 |
| Visual AdamW+EMA 分层 LR Fresh24 | public Visual + AdamW + EMA；SAM2 可训练部分 LR 降为 `1e-5` | 2 | 0.1802 | 0.2211 | 真正从头；比同期统一 LR 主线低 `0.0449 / 0.0589`，epoch-2 checkpoint 严格恢复状态齐全后停止，不继续占用 GPU |
| Visual AdamW+EMA Fresh24 seed 2 | public Visual + AdamW + EMA；仅 seed=`530358028` | 2 / 4 / 6 / 8 / 10 / 12 / 14 / 16 / 18 / 20 / 21 / 22 / 23 / 24（已完成） | 0.2338 / 0.3251 / 0.3412 / 0.3576 / **0.3671** / 0.3625 / 0.3597 / 0.3588 / 0.3581 / 0.3566 / 0.3590 / 0.3558 / 0.3584 / 0.3585 | 0.2893 / 0.3876 / 0.4031 / 0.4197 / **0.4297** / 0.4248 / 0.4200 / 0.4192 / 0.4185 / 0.4162 / 0.4188 / 0.4156 / 0.4183 / 0.4184 | 与主线保持相同数据、模型、优化器、调度、batch/accumulation 和 EMA，只改变 seed；epoch 20→24 已从 `epoch_20.pth` 精确恢复并完成，尾部四点均未刷新，最佳仍为 epoch 10。epoch 22 的 optimizer、两段 scheduler、message hub、独立 `ema_state_dict`、累计梯度边界与游标门均已通过；分布式 wrapper 的最终非零退出不改变四个完整评测结果 |
| Visual cross-seed soup | 主 seed e18/e22 与 seed 2 e10 的权重平均 | `75/25`、`50/50`、`25/75`、三点 `50/25/25` | 0.1219 / 0.0725 / **0.1518** / 0.1222 | 0.1498 / 0.0917 / **0.1851** / 0.1500 | 四路均完成 40,517-pair 全量评测，全部远低于主 seed e18 `0.3702/0.4300`，不晋升。独立 seed 的可训练头不在可直接线性插值的同一参数盆地；跨 seed 参数平均会破坏已学习表示，路线关闭 |
| NewMatcher v10 diagnostic（全对象，repeat 1） | `V2SAM_NEWMATCHER` | 2 | 0.362474 | 0.409393 | 40,517 pairs / 100,223 objects；完整计数通过 |
| NewMatcher v10 diagnostic（全对象，repeat 2） | `V2SAM_NEWMATCHER` | 2 | 0.362487 | 0.409403 | 独立复测；聚合差约 1e-5 |
| NewMatcher v10 diagnostic（legacy） | `V2SAM_NEWMATCHER` | 2 | 0.3834 | 0.4327 | 只取每 pair 最后对象，非论文可比，非严格续训 |
| Strict NewMatcher v11 | `V2SAM_NEWMATCHER` | `iter_4000` | 0.349390 | 0.395382 | 全对象口径；虽从头启动，但误用 24-epoch Linear warmup，轨迹作废、不续跑 |
| Strict NewMatcher Fresh24 | `V2SAM_NEWMATCHER` | 2 / 6 / 8 / 10 / 12 / 14 / 16 / 18 / 20 / 22 / 24（训练与独立评测均已完成） | 0.359630 / 0.382618 / 0.384223 / 0.381855 / 0.397240 / 0.401126 / **`0.405197`** / 0.394619 / 0.397792 / 0.398107 / 0.397958 | 0.406739 / 0.438121 / 0.440600 / 0.438381 / 0.453889 / 0.459220 / **`0.462907`** / 0.451620 / 0.455131 / 0.455471 / 0.455139 | 真正从头；独立 4×H100 分布式任务。epoch 16 为最终最佳，独立复算精确均值 `0.405197334183/0.462906654638`。epoch 18、20、22、24 分别回落到 `0.394619303449/0.451619830646`、`0.397792125659/0.455130562513`、`0.398107312407/0.455471255935`、`0.397957776288/0.455138922929`，均不晋升。epoch 24 的作者 frame-level / per-pair 为 `0.446263857157/0.504549730257`，也低于 e16 的 `0.450898164298/0.509557001338`。e24 raw 通过 40,517 pairs / 100,223 objects、完整字段和有限值门，SHA256=`e0124b076cf4f7f14192a100dcd392366eed5495715f4141938002dacee4340f` |

2026-08-24 14:39（北京时间），Fresh24 已稳定生成 `epoch_18.pth`。独立 CPU watcher 审计得到 `meta.epoch=18, meta.iter=31032, contrast_schedule.step=31032`，optimizer、两段 scheduler、message hub 均存在，且处于 `accumulative_counts=4` 边界；SHA256=`4457de3c28682c5d7aac52f78ce1241e2c98cbc4a3d5b31ee9fd901cfba77d12`，`STRICT_RESUME=PASS`。4×H100 全对象评测任务 `v2sam-newmatcher-v12-e18-fullobject-ddp4-v1` 已提交并排队，评测与正式训练解耦，epoch 18 指标在完整 40,517-pair 输出通过独立复算前不提前填写。

该 e18 评测随后完整结束并由独立 CPU watcher 复算通过：40,517 pairs / 100,223 objects；全对象等权 `0.394619303449/0.451619830646`，作者 frame-level / per-pair `0.443412291779/0.501695669026`，legacy-last-object `0.434724880931/0.493986925142`；raw SHA256=`a1ba0ea7e5d5ef06cdf374d1d9f5db0f0541d403aa73352de9e3324266b80b26`。相对 e16，frame-level 同样回落 `-0.007485872519/-0.007861332312`；e18 frame-level IoU 也比论文 C / Fusion `0.445` 低 `0.001587708221`。因此只能保留 e16 为当前最佳，不能用 e18 替换或平均掩盖回落。

2026-08-25 05:43（北京时间），Fresh24 训练完整到达 epoch 24 并稳定保存 `epoch_24.pth`。独立 CPU watcher 审计得到 `meta.epoch=24, meta.iter=41376, contrast_schedule.step=41376`；optimizer、两段 scheduler、message hub 均存在且处于 `accumulative_counts=4` 边界，文件大小 `2,194,727,917` bytes，SHA256=`5ffa5b296cd7f6938bc8d039d13b3ce9bbffe727010459ae4e41836dbb9a7de6`，`STRICT_RESUME=PASS`。训练任务随后进入 10,130-batch 的内置 stock `SegMetric` 验证；该 evaluator 不写 `raw_per_pair.json`，不能冒充全对象审计。e24 独立 `SegMetricFull` 任务 `job-2b3fd445-e8a4-4c73-a488-20772634fabb` 已提交排队；在它的 raw 完整写盘和独立三口径复算完成前，不把 e24 晋升为最佳。

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

口径必须分开记录：上方 11.2 的 Strict NewMatcher Fresh24 主表采用**逐对象逐帧实例等权**聚合，因此 epoch 16 为 `0.405197334183/0.462906654638`；对同一个 `raw_per_pair.json` 先在每个 frame/pair 内平均对象、再让 40,517 个 frame/pair 等权，得到作者采用的 **frame-level** 聚合 `0.450898164298/0.509557001338`。2026-08-25 结合作者结果展示最终对齐：`0.405197` 是 per-object，较高的 `0.450898` 是 frame-level；两者只是对象数不同的 frame 权重不同，不是复现矛盾。与论文 C / Fusion `0.445` 的 frame-level IoU 同口径比较，epoch 16 高 `0.005898`（约 **0.59 IoU 点**）。per-object 数值继续作为 100,223 个对象完整覆盖的审计指标并列保留，不与 frame-level 混写。论文表没有给出可直接对齐的 Dice。

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
| Hugging Face Ego2Exo 模型包 | 私密仓库 `Travor278/V2-SAM`，revision `07ee27c5b227a6e7a0fdaa2100d2b617773a50a6`；结构与 `wangzeze/V2-SAM` 相同 |
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
2. **Visual 训练已达到并略超官方公开 checkpoint 的量级：** 旧 24-epoch 轨迹最佳 `iter_27600` 为 `0.3683 / 0.4305`；真正从头 AdamW+EMA Fresh24 在 epoch 18 达到当前 IoU 主榜最佳 `0.3702 / 0.4300`，EMA800 则保留当前最高 Dice `0.4311`。两条轨迹最后 checkpoint 均有回落，说明后续刷榜要加强 checkpoint 选择而不是只延长训练。
3. **NewMatcher 的预测/评测审计已在当前声明范围内通过，正式 Fresh24 已从头启动并完成 epoch 2 全对象门：** 两次 v10 诊断 checkpoint 全对象结果为 `0.362474 / 0.409393` 和 `0.362487 / 0.409403`；旧 `0.3834 / 0.4327` 已证实是 last-object evaluator 口径。GT-free、六层数据泄漏、strict load、batch16、完整 evaluator 和回归门均未发现作弊证据。v11 的低轨迹已经定位为 Linear warmup 被错误延长到 24 epoch，不能续跑或用于评价作者调度。Fresh24 使用 `load_from=None, resume=False` 和作者原始 1.2-epoch warmup；epoch 2 checkpoint 的完整评测为 per-object `0.359630 / 0.406739`、per-frame/per-pair `0.391043 / 0.440002`、legacy-last-object `0.380095 / 0.429172`，40,517 frame/pairs / 100,223 objects、有限值和聚合均独立复算通过，训练保持连续运行。作者另一次标为 frame-level 的 epoch-2 结果 `0.3936 / 0.4439` 与我们的 per-frame/per-pair 口径仅差 `+0.002557 / +0.003898`，而不是相对 per-object 口径表面上的 `+0.033970 / +0.037161`；因此两者已基本对齐，后续比较必须明确写出聚合口径。作者结果还报告 Location/Shape，说明其 evaluator 并非当前公开 `SegMetric` 的逐字同一实现；在取得其 evaluator/raw 文件前，不把剩余约 `0.26/0.39` 个百分点归因于训练配置。

交互式 Visual 主 seed 已完成全部 24 epoch，epoch 24 为 `0.3688/0.4282`，未超过 epoch 18 的 `0.3702/0.4300`；第二 seed 也已完成至 epoch 24，最佳仍为 epoch 10 的 `0.3671/0.4297`。四个跨 seed averaging checkpoint 全部显著退化，证明独立 seed 参数不可直接平均，该路线关闭。Hybrid Muon、分层学习率、Muon400、freeze-decoder400、同轨迹 soup、多档 LR continuation、EMA1200、Lovász、Muon-v2、Dense-mask 长程、Dense→Sample 与 Dense-loss Anneal 均已有完整负结果，不再重复；Dense-loss Anneal 从 Dense e6 精确续 800 iter 的显式 EMA/原始权重配对全量结果分别为 `0.3311/0.3913` 与 `0.3314/0.3917`，排除 EMA 滞后解释，不晋升。Visual 主榜最佳仍是 Fresh24 epoch 18 `0.3702/0.4300`，EMA800 `0.3690/0.4311` 只作为 Dice 向 Pareto 点。

首轮 autoresearch 的原始 AdamW+EMA control、Raw-logit 与 Aux-off 已完成 800-iter 全量筛选，分别为 `0.0382/0.0583`、`0.0381/0.0583`、`0.0381/0.0583`。此前标为 Contrast-MP 的 `0.0381/0.0583` 经源码复核证实仍调用 `get_contr_loss(..., idx=None)`，dataset/collate/model 与 Visual v5 相同，因此只是误标的原始 loss 重复实验。真正 Contrast-MP v2 补齐 `(video_id, object_id)` 从 dataset、collate、实际 Visual model 到 loss 的完整传递；修正一次“补丁落在 `projects/v2sam`、配置实际加载 `projects/v2sam_visual`”的运行时来源错误后，9 项功能/来源回归、4 项 contrast schedule 回归、真实 batch 数据门和实际 Visual 四卡一迭代 smoke 全部通过。候选已在 SHA 与 checkpoint 门后发布为 `$BASE/v2sam-visual-contrast-mp-overlay-v2`。真正 800-iter 任务 `job-39e31b00-5171-4f9a-bb3d-da63d768c888` 完成训练、严格状态审计和 10,130-batch EMA 验证，结果为 `0.0382/0.0583`，与严格同预算 control 完全持平；因此该方法在当前预算没有可测增益，不续到 epoch 2。`iter_800.pth` 为 `meta.iter=800, contrast=4799`，optimizer、两段 scheduler、message hub、EMA 齐全，SHA256=`90d078a674e4993a94927dfc64871cb8b7bccb1675d849349ed402e77a810c85`。

正式 NewMatcher Fresh24 的 24 epoch 训练与所有计划中的独立全对象评测均已完成；epoch 16 全对象 `0.405197/0.462907`、作者 frame-level `0.450898/0.509557` 为最终最佳。epoch 20、22、24 全对象分别为 `0.397792/0.455131`、`0.398107/0.455471`、`0.397958/0.455139`，frame-level 分别为 `0.446678/0.505285`、`0.446318/0.504745`、`0.446264/0.504550`，均不替换 e16。e24 checkpoint 严格状态门与 raw 的 40,517 pairs / 100,223 objects、完整字段、有限值及 SHA 门均通过。

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

### 14.3 Ego2Exo checkpoint 模型包

- 已将 `wangzeze/V2-SAM@50fd5a9a7e67d3fdaadab1cd0726b82896f89e02` 服务端复制为私密模型仓库 `Travor278/V2-SAM`，保留原仓库 8 个顶层文件的布局；
- 仅替换 Ego2Exo 的两个同名权重：`fusion_ego2exo_full.pth` 为 Strict NewMatcher Fresh24 epoch 16，`vp_ego2exo_full.pth` 为 Visual resumable from-scratch `iter_27600`；
- README 并列标注 NewMatcher per-object `0.405197334183/0.462906654638` 与作者 frame-level `0.450898164298/0.509557001338`，以及 Visual `0.3683/0.4305`，避免聚合口径混写；
- 最终 revision 为 `07ee27c5b227a6e7a0fdaa2100d2b617773a50a6`。Windows 端独立回读确认仓库保持 private、文件集合精确为 8 项，6 个 LFS 对象的大小与 SHA256 全部通过。
- HF README 已写清原 Mini 提取在修复前缺 6,431 条引用、修复后缺失/坏 JPEG 均为 0、两个方向的记录与唯一图像规模、四分片布局、下载/校验/解压命令及授权提示；README 与 `SHA256SUMS` 在提交 `de7daed32c0a6db3317ba2ff54ceeaf62c0b6a98` 中原子更新并通过远端回读哈希校验。

### 14.3 当前状态（精简快照）

> 本节只记录会变化的在线状态；已完成实验的配置、结果和失败原因统一保留在第 4、7、11 节，避免把进程日志当实验结论。

| 工作流 | 当前状态 | 固定合同 / 下一门 |
| --- | --- | --- |
| Visual Hybrid Muon Fresh24 | epoch 2 完整验证后已停止 | `0.0956 / 0.1235`，同期 AdamW+EMA 为 `0.2251 / 0.2800`；epoch-2 checkpoint 严格续跑状态完整，但该优化器路线不晋升 |
| Visual AdamW+EMA Fresh24 | 主 seed 已完成 24 epoch | epoch 2/4/6/8/10/12/14/16/18/20/22/24 为 `0.2251/0.2800`、`0.3107/0.3707`、`0.3301/0.3905`、`0.3507/0.4115`、`0.3585/0.4192`、`0.3631/0.4230`、`0.3659/0.4265`、`0.3667/0.4267`、`0.3702/0.4300`、`0.3672/0.4271`、`0.3698/0.4295`、`0.3688/0.4282`；`epoch_24.pth` 严格审计通过，最佳仍为已固化的 `epoch_18.pth` |
| Visual cross-seed soup | 四路 40,517-pair 全量评测均完成并淘汰 | 主 e18/seed2 e10 的 `75/25`、`50/50`、`25/75` 为 `0.1219/0.1498`、`0.0725/0.0917`、`0.1518/0.1851`；主 e18/e22/seed2 e10 的 `50/25/25` 为 `0.1222/0.1500`。均远低于 e18 `0.3702/0.4300`，不晋升、不再续跑 |
| Visual autoresearch | Contrast-MP v2、Dense-mask、Dense→Sample、Lovász、Muon-v2 与 Dense-loss Anneal 均已闭环且未晋升 | 旧“Contrast-MP 800”实际仍为 `idx=None` 原始 loss，结果作废；最终 v2 修正版通过 9 项功能/来源回归、4 项 contrast schedule 回归、8 sample / 40 object slots / 15 unique IDs / 25 duplicate slots 的 ID-mask 对齐门和实际 Visual 四卡一迭代 checkpoint 门。真正 800 配置 `visual_autoresearch_contrast_mp_genuine_v2_800_seed530358027_20260825_v1.py`（SHA256=`8f951a190778e7fe630db9830e7e286a993f155ec087e3bf1344ca08f2ccceed`）保持 batch8/accum8/AdamW `4e-5`/seed/EMA 与 control 一致，最终 `0.0382/0.0583` 与 control 持平，故不续到 epoch 2；严格 checkpoint SHA256=`90d078a674e4993a94927dfc64871cb8b7bccb1675d849349ed402e77a810c85`。Dense-mask epoch 6 为 `0.3310/0.3912`，Dense→Sample epoch 6 为 `0.3233/0.3825`，Lovász=`0.0404/0.0652`、Muon-v2=`0.0364/0.0579`，均不晋升；Dense-loss Anneal 的 EMA/raw 配对全量结果为 `0.3311/0.3913` 与 `0.3314/0.3917`，证明渐退火未带来可测增益且并非 EMA 滞后。Visual IoU 主榜最佳仍为 Fresh24 epoch 18 `0.3702/0.4300`，下一候选必须进入未覆盖的 GT-free 推理集成或其他新轴，不能重复既有负路线 |
| Visual AdamW+EMA 分层 LR Fresh24 | epoch 2 完整验证后已停止，GPU 0–3 已归还 | `0.1802 / 0.2211`，显著低于同期统一 LR 主线；稳定 epoch-2 checkpoint 含 optimizer/scheduler/message hub/EMA 与 183 个 optimizer groups，保留为负结果但路线不晋升 |
| Visual AdamW+EMA Fresh24 seed 2 | 已完成至 epoch 24 | 除 seed=`530358028` 外与主线合同相同；最佳仍为 epoch 10 `0.3671/0.4297`。尾部 epoch 21/22/23/24 为 `0.3590/0.4188`、`0.3558/0.4156`、`0.3584/0.4183`、`0.3585/0.4184`，均未晋升；任务从 `epoch_20.pth` 精确恢复，恢复状态与游标门已通过 |
| NewMatcher Fresh24 | 24 epoch 训练与 e24 独立全对象评测均已完成 | `load_from=None, resume=False`；AdamW `4e-5`；Linear `0→1.2 epoch`、Cosine `1.2→24`。epoch 16 全对象 `0.405197334183/0.462906654638`、作者 frame-level `0.450898164298/0.509557001338` 为最终最佳。e24 全对象 `0.397957776288/0.455138922929`、frame-level `0.446263857157/0.504549730257`、legacy `0.438106250775/0.497298368520`，没有刷新。e24 checkpoint 为 `epoch=24, iter=41376, contrast=41376`，optimizer、scheduler、message hub 与累计梯度边界完整，SHA256=`5ffa5b296cd7f6938bc8d039d13b3ce9bbffe727010459ae4e41836dbb9a7de6`；raw SHA256=`e0124b076cf4f7f14192a100dcd392366eed5495715f4141938002dacee4340f` |
| NewMatcher 源码放行 | GitHub 远端已更新并核验为 `travor@bd7ba863` | CPU 回归 37/37、first-32 逐值 parity、真实 4 卡 batch-16 与完整 checkpoint 状态门均通过；正式 Fresh24 继续使用已冻结的 v12 源码/配置，不在运行中热替换 |
| CPU/4090 sidecar | epoch 8/10/12/14/16/18/20/22/24 全对象评测及独立复算均已完成 | epoch 24 三口径为：全对象 `0.397957776288/0.455138922929`、作者 frame-level `0.446263857157/0.504549730257`、legacy `0.438106250775/0.497298368520`；40,517 pairs / 100,223 objects、有限值与 raw SHA256=`e0124b076cf4f7f14192a100dcd392366eed5495715f4141938002dacee4340f` 均通过。NewMatcher 尾部评测闭环完成，最佳保持 e16 |

已冻结的运行/评测配置：

- Visual AdamW+EMA 分层 LR：`$REPRO/frozen_configs/visual_fromscratch24_adamw_ema_discriminative_lr_seed530358027_20260822_v1.py`，SHA256 `9c9ddedbddb276842789e68338fd31ba1c414bf7ab9c8cfc99af872caa2daa57`；`grounding_encoder=1e-5`，matcher/prompt 与其他任务层维持 `4e-5`；epoch-2 负结果后已停止；
- Visual AdamW+EMA seed 2：`$REPRO/frozen_configs/visual_fromscratch24_adamw_ema_seed530358028_20260822_v1.py`，SHA256 `e3159659b4a6e547078968f53ef378fb49bc33425e4efe3ab244f87b4215f2f2`；仅 seed 与主线不同，现使用 GPU 0–3 从头运行；
- NewMatcher Fresh24 epoch-2 独立全对象评测：`$REPRO/frozen_configs/newmatcher_v12_fresh24_epoch2_fullobject_eval_seed530358027_20260822_v1.py`，SHA256 `4717a215cd21da51518e58eb79d9b6a668de70c6e5f993df0a71caf1e428f7fd`；输出 `raw_per_pair.json`，并同时报告 per-object、per-pair 与 legacy-last-object。
- NewMatcher Fresh24 epoch-6 独立全对象评测：`$REPRO/frozen_configs/newmatcher_v12_fresh24_epoch6_fullobject_eval_seed530358027_20260823_v1.py`，SHA256 `6c3c3ec0a72818743f0ba0ab57987927be09983bff4b61c61452782ecec67025`；使用独立 4090 sidecar，不占用或停止正式训练。

新增了严格的仅评测 checkpoint averaging/SWA 工具 `$REPRO/tools/average_model_checkpoints.py`（SHA256 `6843f4c87944f2c70f30fadb1413f207842b1502ed8a34f128f2cf9b1bb458f5`）及回归测试 `$REPRO/tests/test_average_model_checkpoints.py`（SHA256 `f58ba8ddf2dbf667176a242a59cfdb288ba9ec0771a447c872be9d579d028ecc`）。工具要求所有输入 `state_dict` 的 key、shape、dtype 严格相同；浮点/复数张量用 float64/complex128 累加后恢复原 dtype，计数器等非浮点张量取最新 checkpoint；原子输出只含 `meta/state_dict`，写入输入/output SHA256 清单并明确 `resume_supported=false`，不可误作训练续跑点。测试按 RED→GREEN 执行，2/2 通过。

当前资源原则：

- 训练资源保持解耦：Strict NewMatcher Fresh24 使用独立 4×H100 分布式任务；Visual seed 2 的 epoch 20→24 使用另一条独立 4×H100 分布式任务。交互式 8×H100 实例重启排队期间不承载正式训练，恢复后优先补跑主 seed 奇数 epoch 与 cross-seed soup 的完整评测；Muon 与分层 LR 路线均已基于完整结果淘汰；
- NewMatcher 使用独立分布式训练资源持续运行；epoch 2 checkpoint 到达后，优先由共享存储的 CPU/4090 sidecar 另起全对象 test；只有 4090 不满足运行合同或 ETA 明显不可接受时，才使用经 Visual epoch-2 筛选释放出的 4 张交互式 GPU，正式训练进程始终不停机；
- 任一 Visual 试验结束后立即核验完整指标、checkpoint 状态与日志，再决定释放 GPU 或继续下一项非 LR 消融；
- 正式 NewMatcher 的 epoch-2 全量结果必须使用 `40,517 pairs / 100,223 objects` 的全对象口径，同时另列 legacy-last-object，禁止混写。

### 14.4 下一条 Visual 候选（仅冻结设计，尚未获批、未实现、未占 GPU）

Dense→Sample abrupt switch 已在 epoch 6 得到 `0.3233/0.3825`，低于 Fresh24 e6 `0.3301/0.3905`，没有形成晋升。下一候选因此冻结为 **Dense loss 逐步退火到 sampled loss**，避免再次瞬时切换目标：

- 唯一合法起点是已审计的 Dense epoch-2 `iter_6896.pth`；不得从当前 abrupt-switch 轨迹继续；
- `iter=6896` 时权重为 `dense=1, sampled=0`；在 `iter 6896→13792`（epoch 2→4）之间线性退火到 `dense=0, sampled=1`；`iter 13792→20688`（epoch 4→6）保持纯 sampled；
- 权重只由 checkpoint 已保存的 `runner.iter` 推导，不增加新的可丢失计数器；恢复到任一 iteration 后必须得到相同权重；
- 两个端点必须短路未使用分支，既保持与现有 Dense/sampled loss 数值逐位一致，也避免无意义的双倍计算；中间点才同时计算两支并按权重相加；
- optimizer、EMA、seed、batch、accumulation、scheduler、数据顺序和其他 loss 全部不变，确保只检验 loss curriculum；
- RED/GREEN 门包括：Dense 端点等价、sampled 端点等价、中点加权等式、有限梯度、state-dict key 不增加、恢复前后权重一致和真实四卡 DDP smoke；
- epoch-6 全对象 gate：相对 Fresh24 epoch 6 的 `0.3301/0.3905` 至少提升 `+0.003/+0.003` 才延长；若只重复 Dense epoch 6 的约 `+0.001` 微弱增益，立即停止，不消耗完整 24 epoch；
- 即使通过短门，最终晋升仍必须超过 Visual IoU 主榜 `0.3702/0.4300`，或在 IoU 不低于 `0.3690` 时刷新 Dice `0.4311`。

本节只是为了冻结单变量实验合同；在用户明确批准前不修改训练源码、不生成运行配置、不启动任务。

## 15. 2026-08-26 Exo2Ego 与 PCCS 收尾

本节只保留已经由完整日志、checkpoint 或独立汇总文件证明的结果；仍在运行的 Fusion 不提前写成最终结论。

### 15.1 Ego2Exo PCCS（Fusion-first Triple Decoder）

评测使用 WRZ 的 `seg_metric_fusionfirst_tripledecoder_newmetric.py` 合同，样本数固定为 `40,517`。Visual 与 Fusion checkpoint 按同一组候选 epoch 配对；e18 已结束，因此本轮 Ego2Exo PCCS 候选全部闭环。

| epoch | Best IoU | Best Dice | Location ↓ | Shape / Boundary F | VP / Sparse / Fusion 选中数 | VP / Sparse / Fusion 专家 IoU | Cycle 次数 |
| ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 12 | 0.4579 | 0.5174 | 0.0552 | 0.6123 | 5,635 / 7,860 / 27,022 | 0.3667 / 0.3861 / 0.4352 | 7,282 |
| 14 | 0.4583 | 0.5189 | 0.0552 | 0.6129 | 5,798 / 7,904 / 26,815 | 0.3672 / 0.3859 / 0.4375 | 7,322 |
| **16** | **0.4635** | **0.5242** | **0.0535** | **0.6211** | **5,340 / 7,420 / 27,757** | **0.3668 / 0.3863 / 0.4446** | **6,729** |
| 18 | 0.4576 | 0.5187 | 0.0538 | 0.6161 | 5,622 / 7,981 / 26,914 | 0.3668 / 0.3866 / 0.4328 | 7,149 |
| 作者截图参考 | 0.4622 | 0.5232 | 0.0564 | 0.6164 | 5,506 / 8,266 / 26,745 | 0.3616 / 0.3859 / 0.4443 | 7,128 |

最终最佳为 **epoch 16**：相对作者截图参考，IoU `+0.0013`、Dice `+0.0010`、Location 降低 `0.0029`、Shape 提升 `0.0047`。完整对比表位于：

```text
$HDD/V2SAM_Ego2Exo_202608/jobs/pccs_wrz_fusionfirst_e10_e18_20260825_v1/pccs_completed_epochs_comparison.tsv
```

### 15.2 Exo2Ego Visual

公开 Visual 的 12-epoch 官方训练轨迹已经全部完成；指标为原生 `SegMetric` 的 frame-level 口径：

| epoch | IoU | Dice | epoch | IoU | Dice |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2013 | 0.2360 | 7 | 0.4274 | 0.4853 |
| 2 | 0.2814 | 0.3312 | 8 | 0.4336 | 0.4928 |
| 3 | 0.3593 | 0.4165 | 9 | 0.4383 | 0.4968 |
| 4 | 0.3819 | 0.4386 | 10 | 0.4403 | 0.4988 |
| 5 | 0.4036 | 0.4602 | **11** | **0.4426** | **0.5013** |
| 6 | 0.4075 | 0.4666 | 12 | 0.4421 | 0.5007 |

最佳 checkpoint 为 `epoch_11.pth`，SHA256=`7ddf55d1e93e9f83fcd657c7913b7ae20118f49649d747a3679e998ac2cf78a7`；正式 selector 已按 `(mean_IoU, mean_Dice)` 选中它，而不是机械选择最后一个 epoch。

### 15.3 Exo2Ego Fusion（已完成）

Fusion 使用公开 24-epoch 训练合同和 4×H100 分布式任务：AdamW `4e-5`，Linear warmup `0→1.2 epoch`，随后 Cosine `1.2→24 epoch`，每卡 batch `16`、`accumulative_counts=4`，总训练步数 `46,272`，偶数 epoch 做完整 `11,629/11,629` 验证。旧任务在 iter `18,922` 写 MMEngine scalar JSON 时遇到 `Disk quota exceeded`；清理本实验可重建文件后，从已验证的 `epoch_9.pth` 严格恢复到 `epoch=9, iter=17,352`，没有从头重训或回放已完成 epoch。

可从最终续训任务完整聚合日志直接恢复的偶数 epoch 轨迹如下；早期 e2/e4 数值来自同一训练轨迹在旧任务中的已保存 receipt：

| epoch | IoU | Dice | 证据 |
| ---: | ---: | ---: | --- |
| 2 | 0.2201 | 0.2600 | 旧任务完整验证 receipt |
| 4 | 0.3757 | 0.4311 | 旧任务完整验证 receipt |
| 10 | 0.4564 | 0.5125 | `11,629/11,629` |
| 12 | 0.4569 | 0.5129 | `11,629/11,629` |
| 14 | 0.4637 | 0.5197 | `11,629/11,629` |
| 16 | 0.4751 | 0.5314 | `11,629/11,629` |
| 18 | 0.4812 | 0.5375 | `11,629/11,629` |
| 20 | 0.4792 | 0.5352 | `11,629/11,629` |
| 22 | 0.4816 | 0.5374 | `11,629/11,629` |
| **24** | **0.4820** | **0.5378** | **`11,629/11,629`，最终最佳** |

最终续训任务为 `v2sam-exo2ego-fusion-official24-resume-e9-20260827-v2`（job `job-d5526555-2d10-42fd-a286-5db113735c4a`）。日志在 2026-08-29 23:51:45（北京时间）记录 `Saving checkpoint at 24 epochs`；最终验证在 2026-08-30 01:01:05 完成，并报告 `0.4820/0.5378`。任务随后输出 `processExitCode: 0`、`success`、`exit code 0`，平台状态为“已成功”，全日志未出现 Traceback、OOM、No space 或新的 quota 错误。按 `(mean_IoU, mean_Dice)` 规则，最终 Fusion 最佳点为 `epoch_24.pth`。

### 15.4 Exo2Ego PCCS（WRZ 旧指标文件下的候选筛选，已完成）

使用 WRZ 原生 `seg_metric_fusionfirst_tripledecoder_newmetric.py` 的 Fusion-first Triple Decoder 口径，固定 Exo2Ego 全量 `46,515` 个对象；Fusion checkpoint 固定为当时的候选 `epoch_4.pth`，只改变 Visual checkpoint。e9/e10/e11/e12 均已 `exit 0` 并完整统计全部对象。这组结果用于验证三专家推理链与筛选 Visual 候选，不再冒充作者正在整理的新指标文件的最终口径。

| Visual epoch | Best IoU | Best Dice | Location ↓ | Shape / Boundary F | VP / Sparse / Fusion 选中数 | VP / Sparse / Fusion 专家 IoU | Cycle 次数 |
| ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| **9（当前最佳）** | **0.4635** | 0.5165 | 0.0832 | 0.5399 | 16,505 / 15,224 / 14,786 | 0.4382 / 0.4156 / 0.3783 | 21,498 |
| 10 | 0.4626 | 0.5159 | 0.0835 | 0.5400 | 16,335 / 15,321 / 14,859 | 0.4404 / 0.4157 / 0.3779 | 21,331 |
| 11 | 0.4628 | 0.5161 | 0.0832 | 0.5400 | 16,405 / 15,296 / 14,814 | 0.4420 / 0.4159 / 0.3777 | 21,520 |
| 12 | 0.4631 | **0.5166** | **0.0830** | **0.5403** | 16,429 / 15,291 / 14,795 | 0.4416 / 0.4157 / 0.3781 | 21,446 |

按预先冻结的 `(mean_IoU, mean_Dice)` 字典序选择规则，e9 以 IoU `0.4635` 成为这组 Visual 候选的临时最佳；它相对 e12 的 IoU 高 `0.0004`，但 Dice 低 `0.0001`。四个候选之间差距很小，因此这里只冻结“固定 Fusion e4 时的 Visual checkpoint 候选筛选”这一历史事实。作者随后说明正在整理新的指标文件，所以本轮不再用旧 evaluator 对 Visual e9 与 Fusion e24 追加一次名义上的“最终 PCCS”并把它误写为新口径；新文件发布后可直接复用已完成权重重评，无需重训。

### 15.5 收尾结论

- Ego2Exo Visual、Strict NewMatcher、三种聚合口径和 WRZ 旧 PCCS 候选已经闭环；
- Exo2Ego Visual 12e 与 Fusion 24e 均使用公开训练合同完成，最终完整验证与任务退出状态已经落证；
- Exo2Ego 最佳单专家 checkpoint 为 Visual `epoch_11.pth` 与 Fusion `epoch_24.pth`；
- 旧 PCCS 实现的历史结果保留，但作者的新指标文件尚未发布，因此不再新增可能混淆口径的旧 evaluator 结果；
- 当前没有仍需监督的 V2-SAM 训练或评测任务。本轮复现目标于 2026-08-30 完成；后续只在作者新 evaluator 发布时另开指标对齐任务。

## 16. 作者新 Frame/Object 与 PCCS 标准最终整理（2026-09-02）

本节是后续发布与对比的唯一权威入口。最终标准固定为私有/公开分支 `jaychempan/V2-SAM-O:v2sam-pccs`、提交 `0e3bc33dec3e202ffbb86cec01038e60b18c162a`：

- object-level：所有对象等权聚合；
- frame-level：先在每个 image pair 内平均对象，再对 pair 等权聚合；
- PCCS：使用作者新分支的推理合同；Ego2Exo 为学习式选择器，Exo2Ego 为 Fusion-first 规则；
- Anchor：原始 SAM2 decoder，与对应 Visual checkpoint 共用 wrapper 加载合同；
- `Cont.A` 越高越好，`Loc.E` 越低越好；
- 全量评测固定 `batch=1`；Ego2Exo 为 `40,517 pairs / 100,223 objects`，Exo2Ego 为 `46,515 pairs / 109,253 objects`。

### 16.1 八个权重的完整矩阵（统一总表）

为避免“八个方法”和“八个 expert checkpoint”混写，本节保留两张互补表：§16.1.1 是最终发布用的两个方向 × 四方法矩阵；§16.1.2 是官方/重训 Visual、Fusion 共八个 expert checkpoint 的控制矩阵。

#### 16.1.1 新标准四方法最终矩阵（两个方向 × PCCS/Visual/Anchor/Fusion）

`—` 表示尚无可发布的完整结果。阶段值可用于监督，但不得冒充最终 checkpoint。

| Direction | Method / Weight | 来源 / epoch | Pairs | Objects | object IoU / Dice / Cont.A / Loc.E | frame IoU / Dice / Cont.A / Loc.E | 状态 |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| Ego2Exo | PCCS | 官方 Visual/Anchor/Fusion；`v2sam-pccs` | 40,517 | 100,223 | 0.4420 / 0.4985 / 0.5828 / 0.0798 | 0.4867 / 0.5435 / 0.6213 / 0.0716 | **完成**；0 skipped；exit 0 |
| Ego2Exo | Visual | 官方 `vp_ego2exo_full.pth`；同次新标准 evaluator | 40,517 | 100,223 | 0.3259 / 0.3819 / 0.4748 / 0.1024 | 0.3770 / 0.4341 / 0.5219 / 0.0927 | **完成** |
| Ego2Exo | Anchor | 原 SAM2 decoder + 官方 Visual checkpoint | 40,517 | 100,223 | 0.3726 / 0.4161 / 0.4809 / 0.1202 | 0.4034 / 0.4482 / 0.5063 / 0.1152 | **完成**；0 skipped；exit 0 |
| Ego2Exo | Fusion | 官方 `fusion_ego2exo_full.pth`；同次新标准 evaluator | 40,517 | 100,223 | 0.4037 / 0.4623 / 0.5509 / 0.1044 | 0.4511 / 0.5106 / 0.5957 / 0.0881 | **完成** |
| Exo2Ego | PCCS | 重训 Visual e19 / Fusion e20 + Anchor；作者最新 `v2sam-pccs@0e3bc33` 原生 `pccs_exo2ego.py`（`fusion_first`，无 selector） | 46,515 | 109,253 | 0.4896 / 0.5377 / 0.5706 / 0.0822 | 0.5384 / 0.5860 / 0.6145 / 0.0783 | **完成**；timeout7200 v6；0 skipped；exit 0 |
| Exo2Ego | Visual | 修复版重训 e19；最终 PCCS wrapper 同次评测 | 46,515 | 109,253 | 0.4505 / 0.5057 / 0.5347 / 0.0880 | 0.4994 / 0.5529 / 0.5765 / 0.0864 | **完成**；e19 按训练验证 frame IoU/Dice 选中 |
| Exo2Ego | Anchor | 原 SAM2 decoder + 重训 Visual e19 checkpoint | 46,515 | 109,253 | 0.3691 / 0.4003 / 0.4205 / 0.1735 | 0.4090 / 0.4421 / 0.4580 / 0.1665 | **完成**；与 PCCS 同次全量评测 |
| Exo2Ego | Fusion | 修复版重训 e20（选中）；最终 PCCS wrapper 同次评测 | 46,515 | 109,253 | 0.4805 / 0.5338 / 0.5684 / 0.0749 | 0.5274 / 0.5793 / 0.6104 / 0.0738 | **完成**；frame-primary 选 e20；e22 为训练验证 object/Loc.E Pareto 控制点 |

这里的 `fusion_first` 不是旧 WRZ evaluator，也不是本实验自行定义的替代 PCCS。远端最新分支头仍为 `0e3bc33dec3e202ffbb86cec01038e60b18c162a`；其中 `tools/pccs_eval.py:36-38` 明确规定 `ego2exo -> learned_selector`、`exo2ego -> fusion_first`，`projects/v2sam_pccs/configs/pccs_exo2ego.py:1,15-16` 又明确写为 deterministic fusion-first 且 `selector_model_path=None`。同分支 README 还把开发名 `SegMetric_FusionFirst_NewMetric` 正式映射为发布名 `PCCSFrameLevelMetric`。因此本次 Exo2Ego 的 Fusion-first 正是作者最新 PCCS 分支的方向专用正式实现。

#### 16.1.2 八个 Visual/Fusion expert checkpoint 控制矩阵

这张表回答“官方权重与重训权重分别测了什么”。Ego2Exo 重训 e18/e16 的最终 `v2sam-pccs` wrapper aggregation 已从共享 JSON 只读导入；旧 expert-only helper 数值仅保留在历史记录中，不再冒充最终严格口径。

| Direction | Expert weight | 来源 / epoch | object IoU / Dice / Cont.A / Loc.E | frame IoU / Dice / Cont.A / Loc.E | 严格口径状态 |
| --- | --- | --- | --- | --- | --- |
| Ego2Exo | Visual | 重训 e18 | 0.3317 / 0.3865 / 0.4832 / 0.1015 | 0.3798 / 0.4358 / 0.5265 / 0.0938 | 最终 `v2sam-pccs` wrapper；40,517 / 100,223；0 skipped |
| Ego2Exo | Fusion/NewMatcher | 重训 e16 | 0.4063 / 0.4641 / 0.5544 / 0.1143 | 0.4519 / 0.5103 / 0.5957 / 0.0944 | 最终 `v2sam-pccs` wrapper；40,517 / 100,223；0 skipped |
| Exo2Ego | Visual | 修复版重训 e19 | 0.4533 / 0.5081 / 0.5361 / 0.0880 | 0.5023 / 0.5554 / 0.5779 / 0.0864 | 新双层训练验证完成；最终候选 |
| Exo2Ego | Fusion | 修复版重训 e20（选中） | 0.4804 / 0.5336 / 0.5683 / 0.0753 | 0.5272 / 0.5791 / 0.6101 / 0.0742 | e24 已完成；按 frame-primary 规则最终选择 e20 |
| Ego2Exo | Visual | 官方 `vp_ego2exo_full.pth` | 0.3259 / 0.3819 / 0.4748 / 0.1024 | 0.3770 / 0.4341 / 0.5219 / 0.0927 | 最终 `v2sam-pccs` 同次全量评测 |
| Ego2Exo | Fusion | 官方 `fusion_ego2exo_full.pth` | 0.4037 / 0.4623 / 0.5509 / 0.1044 | 0.4511 / 0.5106 / 0.5957 / 0.0881 | 最终 `v2sam-pccs` 同次全量评测 |
| Exo2Ego | Visual | 官方 `vp_exo2ego_full.pth` | 0.4500 / 0.5051 / 0.5363 / 0.0871 | 0.4981 / 0.5510 / 0.5772 / 0.0866 | 已审计兼容的完整控制评测 |
| Exo2Ego | Fusion | 官方 `fusion_exo2ego_full.pth` | 0.4638 / 0.5164 / 0.5509 / 0.0813 | 0.5144 / 0.5669 / 0.5973 / 0.0780 | 已审计兼容的完整控制评测 |

结论：Ego2Exo 官方组合与重训组合均已完成严格 `v2sam-pccs` 全量评测。重训 e18/e16 组合的 PCCS object/frame=`0.4434/0.4994/0.5838/0.0827`、`0.4878/0.5446/0.6212/0.0742`；Anchor=`0.3727/0.4161/0.4809/0.1202`、`0.4034/0.4482/0.5063/0.1152`。对应 aggregation JSON、完整计数和零跳过门均通过。

### 16.2 与作者截图的 frame-level 对照

| Direction / Method | 作者参考 IoU / Dice / Cont.A / Loc.E | ours | ours − author | 备注 |
| --- | --- | --- | --- | --- |
| Ego2Exo PCCS | 0.4868 / 0.5437 / 0.6215 / 0.0717 | 0.4867 / 0.5435 / 0.6213 / 0.0716 | −0.0001 / −0.0002 / −0.0002 / −0.0001 | 最终 |
| Ego2Exo Visual | 0.3777 / 0.4348 / 0.5227 / 0.0931 | 0.3770 / 0.4341 / 0.5219 / 0.0927 | −0.0007 / −0.0007 / −0.0008 / −0.0004 | 最终 |
| Ego2Exo Anchor | 0.4034 / 0.4482 / 0.5062 / 0.1152 | 0.4034 / 0.4482 / 0.5063 / 0.1152 | −0.0000 / −0.0000 / +0.0001 / −0.0000 | 最终 |
| Ego2Exo Fusion | 0.4511 / 0.5106 / 0.5957 / 0.0883 | 0.4511 / 0.5106 / 0.5957 / 0.0881 | −0.0000 / −0.0000 / −0.0000 / −0.0002 | 最终 |
| Exo2Ego PCCS | 0.5353 / 0.5834 / 0.6118 / 0.0795 | 0.5384 / 0.5860 / 0.6145 / 0.0783 | +0.0031 / +0.0026 / +0.0027 / −0.0012 | **最终**；四项均优于作者参考 |
| Exo2Ego Visual | 0.4997 / 0.5528 / 0.5768 / 0.0865 | 0.4994 / 0.5529 / 0.5765 / 0.0864 | −0.0003 / +0.0001 / −0.0003 / −0.0001 | **最终 PCCS wrapper 同次评测** |
| Exo2Ego Anchor | 0.4090 / 0.4422 / 0.4582 / 0.1662 | 0.4090 / 0.4421 / 0.4580 / 0.1665 | −0.0000 / −0.0001 / −0.0002 / +0.0003 | **最终**；与作者基本一致，Loc.E 高 0.0003 |
| Exo2Ego Fusion | 0.5155 / 0.5681 / 0.5983 / 0.0776 | 0.5274 / 0.5793 / 0.6104 / 0.0738 | +0.0119 / +0.0112 / +0.0121 / −0.0038 | **最终选中 e20**；四项均优于作者参考 |

### 16.3 证据与当前状态

- 新标准根目录：`/inspire/hdd/project/luojianlan/zhubingwen-253108120125/V2SAM_Bidirectional_Refresh_20260831/pccs_standard_20260831/`；
- 机读总表：[V2SAM_NEW_METRICS_20260830.json](D:/Code/Work/EgoExoSeg/docs/V2SAM_NEW_METRICS_20260830.json)，SHA256=`f22f0e7abb1f14f5d452552c2aecbf5e769050494a97459ea3eb5095c704f2b4`；
- Ego2Exo 完成日志：`output/playwright/ego2exo_pccs_anchor_platform_completion_20260902.log`，SHA256=`b9e0d696ddba6fdcc30db9360d9603823ba100642958846bdb0950aefc037ed0`；
- Ego2Exo 正式 job `job-245f03ea-a31d-4694-bf68-59b861f99ffd`：`40517/40517`、`40,517/100,223`、0 skipped、fatal scan 通过、exit 0；成功保留资源已释放，节点占用 0；
- Ego2Exo 重训专家严格补评任务 `v2sam-ego2exo-retrained-pccs-anchor-4gpu-20260902-v1`（`job-37776b8a-2387-4061-b1aa-6592f34fa5e6`）：`10130/10130`、`40,517 pairs / 100,223 objects`、zero skipped、fatal scan passed、exit 0；共享 aggregation JSON 已于 2026-09-04 从 4090 只读导入。PCCS object/frame=`0.443373/0.499417/0.583804/0.082747`、`0.487821/0.544627/0.621182/0.074240`；Anchor=`0.372666/0.416052/0.480887/0.120176`、`0.403391/0.448192/0.506276/0.115181`。配置/launcher SHA256 分别为 `664f5fe134193ea11ff29b87a89422c30d4a63e017b2641a8e018d70d9870303`、`ab559f677aadee6afacd727f0f0e2669f4bf334feff0d4a01538ef7af5628f41`；
- Exo2Ego Visual24 job `job-bbe75677-373d-4749-a338-afc00b65d607`：24 epoch 成功，e19 为 IoU/Dice 选择结果，e24 亦保留；
- Exo2Ego Fusion24 源 job `job-4ae97608-56a3-4acb-9b1c-48131f16cac7` 于三天上限前保存 e21；最终 v5 `job-a2e6d4e2-e06d-4b02-9269-8b7728f5e845` 从 e21/iter40488 严格恢复到 e24，`FUSION_E21_TO_E24_COMPLETE=PASS`、exit 0，并已及时释放成功保留资源。同一冻结 config/model/data/optimizer/schedulers、world size/batch/accumulation 未变，仅 `work_dir/load_from/resume` 变化；launcher SHA256=`531d086cabead9d67dd971d88ad86d11ddfe2d02c005c0693103af54cbb67867`；
- Fusion e22 object/frame=`0.4809/0.5346/0.5686/0.0751`、`0.5266/0.5788/0.6094/0.0739`；e24=`0.4802/0.5337/0.5681/0.0755`、`0.5257/0.5776/0.6085/0.0748`。e24 未刷新八项中的任何一项；最终按作者 frame-level 对照主目标选择 e20（frame IoU/Dice/Cont.A 最高），e22 保留为 object 四项及 frame Loc.E 的 Pareto 控制点。
- Hugging Face 发布：已用去 optimizer、模型 `state_dict` 不变的发布副本替换 `Travor278/V2-SAM` 的 `vp_exo2ego_full.pth`（Visual e19，965 keys，926,619,220 bytes，LFS SHA256=`fd2e1b8f342c4468b45c962c61d4c914b9f6d71b6c6085014fb9a3ed2924b65f`）与 `fusion_exo2ego_full.pth`（Fusion e20，1336 keys，2,136,421,714 bytes，LFS SHA256=`8373b17180c198898fe9f9e39d09fe72991150ce3053b207b19e81986c60cab1`）。远端 `main` 已核验为 commit `8b8310e9ee13023cb8711c2417e18cc7361dc3aa`；完整工作区与回执位于 `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/hf_publish_exo2ego_20260903/`。4090 完成 LFS 后最终 commit POST 曾断连，本机 API 随后复用已上传 blob 原子提交成功；因重试凭据曾被私有 Terminal 2 回显，必须轮换该 Hugging Face token 并清理/关闭该终端。
- Exo2Ego 修复代码发布：`jaychempan/V2-SAM-O/travor@27e28e71a1bc8276deb306efc86c958c12168bf3`，新增 `EXO2EGO_FIXES/README.md`、精确 public-base patch 和 `TRAVOR_EXO2EGO_AUDIT.md`，根 README 同步改为双方向范围。源分支核心回归 7/7、Python compile、`git diff --check`、五个 launcher `bash -n` 均通过；patch 对 public `50e7c05` 的正向 apply 门通过；远端 `refs/heads/travor` 已回读等于本地提交。
- Exo2Ego PCCS/Anchor v1 `job-bef48599-c293-49ac-9eb3-dff70f477140` 因 rank 尾部差触发 600 秒 `ALLGATHER` timeout，局部内存结果不可安全续跑。最终 v6 `v2sam-exo2ego-selected-pccs-anchor-4gpu-timeout7200-20260904-v6`（`job-c31a76d1-5aed-4847-ba7b-a59edf8d2b6a`）只把 process-group timeout 改为 7200 秒，Visual e19/Fusion e20/Anchor、Fusion-first、数据、指标和 batch=1 均不变；2026-09-04 08:51 完成 `11629/11629`、`46,515 pairs / 109,253 objects`、zero skipped、fatal scan passed、`EXO2EGO_TIMEOUT7200_4GPU_FULL_GATE=PASS`、exit 0。精确 aggregation JSON 位于 `results/retrained_pccs_exo2ego_4gpu_aggregation_20260904_timeout7200_v6.json`；配置/launcher SHA256 分别为 `4c5bda2eb6fc5b19536eb1c1830546317fb1eaadae254be1a32701fc11a7be95`、`a6dbaec63273290d3bbac15de5a25db1d05ca2d4d1241006fff1e45acdebf599`。

### 16.4 实施审计与异常记录（精简）

#### 16.4.1 Exo2Ego 与公开代码的差异（根因审计）

##### 16.4.1.1 一句话结论

本次 Exo2Ego 没有发明新的 Visual/Fusion 网络：SAM2、VP matcher、sparse correspondence、loss 定义和数据变换沿用 public 模型图；发布补丁的精确基线是 `V2-SAM@50e7c05`，它相对历史 `31c3bab` 只先合入了最小 DINO 所有权修复。其余差异集中在三个会决定“是否能正确训练和比较”的合同：**持久 contrast schedule、Fusion 的 DINO 注册位置、方向/训练/评测配置**。Visual 只涉及第一和第三项；Fusion 三项都涉及。最终权重提升不能笼统归因于“多训到 24e”，尤其 Fusion 的主要修复是先保住正确的预训练 DINO。

##### 16.4.1.2 最小代码差异与触发链

| 层面 | 历史 public / 默认行为 | 本次 Exo2Ego 正式行为 | 为什么重要 |
| --- | --- | --- | --- |
| Visual/Fusion contrast 计数 | `V2SAM._constr_step` 是 Python 类属性；首次 forward 从 0 加到 1，`<4000` 时把 `loss_contr` 乘 100；不进 `state_dict`，进程重启/新实例会丢失 | `self.contrast_schedule` 是注册 buffer；本轮 fresh 合同从 3999 开始，首个 forward 原子前进到 4000、scale=1；`contrast_schedule.step` 随 checkpoint 保存和恢复 | public fresh 会让前 3,999 次 forward 使用 100× 对比损失，且断点续训无法证明步数连续；本轮把首步和续训状态都冻结 |
| Fusion DINO 注册 | 历史 public `31c3bab` 使用 `self.dinov3_model = load_dinov3_model(...)`，再把同一对象传给 `self.sparse_correspondence`；DINO 同时是顶层 child 和 matcher child | public `50e7c05` 及本轮都改为局部变量加载，只由普通 `SparseCorrespondenceMatcher` 持有，不再保留顶层 alias | MMEngine `BaseModule.init_weights()` 会再次遍历直接 child。历史顶层 DINO 在已加载官方权重后被二次初始化，再被冻结；`50e7c05` 是本轮补丁的正确 DINO 基线 |
| Visual 默认方向与 epoch | public Visual config 默认指向 `Ego2Exo_FullTrain.json`，`max_epochs=12`，`randomness.seed=None` | 显式 Exo2Ego train/test JSON；24e；seed=`530358027`；fresh 非空目录拒绝启动 | public Visual config 不能不改路径就冒充 Exo2Ego；本轮同时固定可复现身份和完整 24e 选择轨迹 |
| Fusion 训练合同 | public 已是 Exo2Ego/24e、AdamW `4e-5`、batch 16/GPU、accumulation 4、warmup `0→1.2e`、cosine `1.2→24e`，但含上述 DINO/contrast 问题，seed 未固定 | 保留相同优化器、有效 batch、LR 和 scheduler；只修状态/初始化，seed 固定；偶数 epoch 完整验证 | 排除了“换超参/换 matcher 算法”这一混杂因素 |
| evaluator | public stock `SegMetric`、旧 WRZ last-object 和旧 triple-decoder 结果曾并存；旧 Fusion 还有把输出接错到 Visual mask 的风险 | 训练期使用 `author_metric.CombinedLevelSegMetric(strict=True)`；最终统一由 `v2sam-pccs@0e3bc33` 原生入口输出 object/frame 两套指标 | 训练根因和指标口径分开；旧 evaluator 值不再冒充新标准 |
| PCCS 推理 | 旧 WRZ 候选只作为历史回执 | 新标准 Exo2Ego 固定 Fusion-first；Visual e19、Fusion e20、原 SAM2 Anchor 由同一 wrapper/全量门评测 | 保证最终四方法矩阵中的专家选择、路由和聚合口径一致 |

公开 contrast 代码的关键行为是：

```python
if not hasattr(V2SAM, "_constr_step"):
    V2SAM._constr_step = 0
V2SAM._constr_step += 1
_contr_scale = 1.0 if V2SAM._constr_step >= 4000 else 100.0
```

本轮 overlay 改为实例级、可保存状态：

```python
self.contrast_schedule = ContrastScheduleState()  # INITIAL_STEP = 3999
loss_contr_raw = loss_contr
_contr_scale, _constr_step = self.contrast_schedule.advance()
loss_contr = loss_contr_raw * _contr_scale.to(
    device=loss_contr_raw.device,
    dtype=loss_contr_raw.dtype,
)
```

Fusion 的功能图没有改变；只改变 DINO 的模块所有权：

```python
# historical public <= 31c3bab：顶层 alias 会被 MMEngine 再次 init
self.dinov3_model = load_dinov3_model(...)
self.sparse_correspondence = SparseCorrespondenceMatcher(
    dinov3_model=self.dinov3_model, ...)

# corrected：只由普通 matcher 持有
dinov3_model = load_dinov3_model(...)
self.sparse_correspondence = SparseCorrespondenceMatcher(
    dinov3_model=dinov3_model, ...)
```

##### 16.4.1.3 张量与运行证据

- 用未修复 public 代码得到的 Exo2Ego Fusion checkpoint 中，DINO 共 368 个 tensor：只有 25 个仍与官方 DINO asset 相同，343 个已经不同；顶层 `dinov3_model.*` 与 matcher 内别名 368/368 逐张量相同，证明不是 checkpoint key 映射错误，而是同一 DINO 在初始化阶段被重置后冻结；
- 作者公开 Fusion checkpoint 的对应 DINO 为 368/368 与官方 asset 相同；本轮 corrected Fusion 在 MMEngine 初始化前后也保持 368/368 相同；
- Visual 不含 DINO，不能把 Fusion 根因硬套给 Visual。Visual 的直接证据是启动门 `3999→4000`、首步 scale=1、训练 `loss_contr` 为正常个位数，并且 checkpoint 新增唯一持久 key `contrast_schedule.step`；
- 正式 overlay 与对应审计 patched variant diff 为 0，6 项源码/状态测试通过；Visual checkpoint 加载为 matched 964、unexpected 0；Fusion 在 PCCS wrapper 中 matched 1335、因 wrapper 命名差异 remapped 371、missing/unexpected 0；
- Fusion e21→e24 严格续训继续使用相同 config/model/data/optimizer/schedulers、world size、batch 和 accumulation，仅允许 `work_dir/load_from/resume` 变化，证明后半程不是重新开跑或改变训练合同。

##### 16.4.1.4 修复收益与证据边界

| Expert | 控制权重 frame IoU / Dice / Cont.A / Loc.E | corrected 最终选择 | corrected − control |
| --- | --- | --- | --- |
| Visual | 0.4981 / 0.5510 / 0.5772 / 0.0866 | e19：0.5023 / 0.5554 / 0.5779 / 0.0864 | +0.0042 / +0.0044 / +0.0007 / −0.0002 |
| Fusion | 0.5144 / 0.5669 / 0.5973 / 0.0780 | e20：0.5272 / 0.5791 / 0.6101 / 0.0742 | +0.0128 / +0.0122 / +0.0128 / −0.0038 |

这里的 control 是同一新指标兼容口径下的作者公开权重控制评测，而不是早期 stock/legacy 数值。Visual 的提升较小且 object Cont.A/Loc.E 并非全面占优；Fusion 的 object/frame 八项均明显改善，符合 DINO 初始化修复是主要因素的预期。尚不能声称每一点提升都由单一代码行产生：Visual 同时从 12e 扩到 24e 并固定 seed；Fusion 则保留了 public 24e 超参，因而其因果归因更干净。

##### 16.4.1.5 当前公开性边界

public `V2-SAM` 的 `50e7c05` 已含最小 DINO 所有权修复，但不含本轮持久 contrast、严格旧权重兼容、冻结 Exo2Ego 配置和回归门。2026-09-03 已把完整 Exo2Ego 审计包推到私密仓库 `jaychempan/V2-SAM-O` 的 `travor@27e28e71a1bc8276deb306efc86c958c12168bf3`：`EXO2EGO_FIXES/` 内的 91 KB patch 以 public `50e7c05` 为精确基线，SHA256=`758cb8d25a1d700a346ac85cbf5adc17b205fa9b392f3a5a99bceaf56dde0cbe`，已在干净 detached worktree 通过正向 `git apply --check`。它独立交付 source/config/launch/tests，没有覆盖现有 Ego2Exo `projects/v2sam`。Hugging Face README 仍需改为指向该不可变 commit。

#### 16.4.2 有效结果与失败回执

| 项目 | 结果 |
| --- | --- |
| Ego2Exo 官方四方法 | 完成；frame PCCS/Visual/Anchor/Fusion 分别为 `0.4867/0.5435`、`0.3770/0.4341`、`0.4034/0.4482`、`0.4511/0.5106`；完整四指标见 §16.1 |
| Exo2Ego Visual24 | e19 训练验证 object/frame `0.4533/0.5081/0.5361/0.0880`、`0.5023/0.5554/0.5779/0.0864`；最终 PCCS wrapper 同次全量为 `0.4505/0.5057/0.5347/0.0880`、`0.4994/0.5529/0.5765/0.0864` |
| Exo2Ego Fusion24 | e24 已完成；最终选 e20；PCCS wrapper 同次全量 object/frame `0.4805/0.5338/0.5684/0.0749`、`0.5274/0.5793/0.6104/0.0738` |
| Hugging Face Exo2Ego 发布 | `main@8b8310e9ee13023cb8711c2417e18cc7361dc3aa`；`vp_exo2ego_full.pth`=Visual e19，`fusion_exo2ego_full.pth`=Fusion e20；远端大小和 LFS SHA256 均通过独立复核；token 轮换/Terminal 2 清理待用户确认 |
| Exo2Ego 修复代码发布 | `jaychempan/V2-SAM-O/travor@27e28e71a1bc8276deb306efc86c958c12168bf3`；不覆盖 Ego2Exo tree；精确 patch/config/launch/tests/audit 已推送并通过远端 SHA 回读 |
| Visual EMA screen | Ego2Exo object/frame `0.3324/0.3890/0.4856/0.0945`、`0.3816/0.4394/0.5310/0.0896`；保留为 Pareto 控制点，不进入八方法主表 |
| Exo 官方补评 attempt 1 | 裸 `python` 不在 PATH，exit 127；未进入正式推理 |
| Exo 官方补评 attempt 2 | 主体推理后 launcher 第 121 行未闭合引号，exit 2；不作为重训矩阵输入，失败保留资源已释放 |
| Ego 4090 首次尝试 | PID 376259 在 `36040/40517` 后随交互实例消失；production JSONL 未提交，拒绝聚合；后由 H100 完整替代 |

早期逐 epoch 进度、重复 ETA 与临时排队状态已从主日记移除；完整机器日志、阶段 receipt 和远端 audit 文件仍保留，因此精简不影响可追溯性。

#### 16.4.3 Hugging Face `Travor278/V2-SAM` 文件审计（2026-09-03）

远端 `main@8b8310e9ee13023cb8711c2417e18cc7361dc3aa` 共 8 个顶层文件。结论是：**四个 expert 权重目前都不需要再次替换；README 必须更新；公开代码/config 需要补齐后再宣称可完整复现。** Anchor 没有独立训练 checkpoint，Exo2Ego PCCS 是 Fusion-first 规则，也不应凭方法名额外上传一个伪“PCCS 权重”。

| 远端文件 | 当前身份 | 是否更新 | 结论 |
| --- | --- | --- | --- |
| `vp_ego2exo_full.pth` | 982,288,288 bytes；SHA256=`bc94b285f5421359702b2c42ef2aec83ffee6787b301dea5821628ccdca8d0c4` | **否** | 与 2026-08-25 已选 Visual `iter_27600` 发布矩阵完全一致 |
| `fusion_ego2exo_full.pth` | 2,192,404,525 bytes；SHA256=`7a75826ffb7f065e7db9c20322567eb6058c9697e079180f77cda9ad4097d5b3` | **否** | 与 Strict NewMatcher Fresh24 e16 发布矩阵完全一致 |
| `vp_exo2ego_full.pth` | 926,619,220 bytes；SHA256=`fd2e1b8f342c4468b45c962c61d4c914b9f6d71b6c6085014fb9a3ed2924b65f` | 已完成 | corrected Visual e19，去 optimizer 的发布副本；远端 LFS hash 已复核 |
| `fusion_exo2ego_full.pth` | 2,136,421,714 bytes；SHA256=`8373b17180c198898fe9f9e39d09fe72991150ce3053b207b19e81986c60cab1` | 已完成 | corrected Fusion e20，去 optimizer 的发布副本；远端 LFS hash 已复核 |
| `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | SHA256=`8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` | **否** | 与作者固定 DINO asset 逐字节一致，正是正确 Fusion 所需基础权重 |
| `json.tar.gz` | SHA256=`1852df7d8aa6bc53f15fb05d129105131a67d53f49ca92705639ee32119dc74f` | **暂不替换** | 与作者 pinned 仓库逐字节一致；新标准测试 JSON/selector 应以新增、带版本文件发布，不能无证据覆盖原 archive |
| `.gitattributes` | 1,519 bytes | **否** | LFS 跟踪合同仍适用 |
| `README.md` | 3,147 bytes | **必须更新** | 当前仍写“只替换 Ego2Exo、两个 Exo2Ego 与作者仓库逐字节相同”，在本次 commit 后已成为错误陈述；标题、权重来源、哈希、代码链接和指标表均需刷新 |

README 下一版至少应写明：

1. 四个 expert 分别来自哪个方向、实现和选中 epoch/iter；
2. Exo2Ego Visual e19/Fusion e20 的 object/frame 四指标及与作者值差异；PCCS/Anchor 在最终全量任务完成前留空，不提前填阶段值；
3. 新发布文件的大小、LFS SHA256、optimizer 已剥离但 `state_dict` 保持不变；
4. 公开 `V2-SAM@50e7c05` 已修 DINO 顶层所有权，但仍缺本轮持久 contrast、冻结配置、严格兼容和完整回归门，不能单独代表这两个新 Exo2Ego 权重的训练闭包；
5. 指向 corrected Exo2Ego 审计包 `V2-SAM-O/travor@27e28e71`，以及 `v2sam-pccs@0e3bc33` 的评测入口；
6. `Cont.A` 越高越好、`Loc.E` 越低越好，object/frame 不混算。

建议但非立即必需的新增文件是一个小型 `MODEL_MANIFEST.json`（记录四权重 SHA、来源、epoch、对应代码 commit 和指标口径）以及 Exo2Ego Visual/Fusion 的 frozen config/加载示例。它们应新增而不是覆盖 `json.tar.gz`。本节只是审计结论；除已授权的两个 Exo2Ego checkpoint 外，本轮未修改其他 Hugging Face 文件。

#### 16.4.4 剩余工作

1. Fusion e21→e24 四卡严格续训已完成并释放资源；最终 Fusion 选 e20，Visual 选 e19。
2. Exo2Ego timeout7200 v6 已完成并回填 §16.1.1/§16.2；释放成功保留资源，并把最终值同步到 Hugging Face README/manifest（需用户授权修改 README）。
3. Ego2Exo 重训专家 aggregation 已导入 §16.1.2 和机读 JSON。
4. 当前实验矩阵已闭环；仅 Hugging Face README/manifest 更新和已回显 token 的轮换/Terminal 2 清理仍需用户授权。
