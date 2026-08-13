# V2-SAM 可恢复双四卡训练设计

## 1. 目标与实验身份

本阶段同时建立两条彼此隔离、可审计、可从断点恢复的 Ego2Exo 训练链路：

| GPU | 实验标签 | 模型实现 | 训练起点 | 训练周期 |
| --- | --- | --- | --- | ---: |
| 0–3 | Fusion/NewMatcher | `projects.v2sam.models.V2SAM_NEWMATCHER` | epoch 0 | 24 epoch |
| 4–7 | B/Visual | `projects.v2sam.models.V2SAM`（Visual 源码） | epoch 0 | 12 epoch |

这里不能把两条任务都标为 `V2SAM_NEWMATCHER`。当前已审计的
`V2SAM_NEWMATCHER` 固定构造 DINO sparse-correspondence 分支，是 Fusion
实现；它和 Fusion 权重都有 1,335 个张量。Visual 权重只有 964 个张量，
不含 `sparse_correspondence` 的 371 个张量，对应公开 Visual `V2SAM`。
把同一个 NewMatcher 类部署到两组 GPU 只会得到两条 Fusion 训练，不会得到
论文 B/Visual。

先前 4–7 上完成的 Visual epoch-12 checkpoint 只保留为历史基线。它不能作为
本轮严格训练的 resume 点，因为旧配置没有保存完整 optimizer 状态，学习率已经
走到调度末端，而且把它接到 NewMatcher 会产生模型谱系错误。

## 2. “严格”和“可续跑”的定义

本设计中的“严格训练”表示复现实验合同固定且可核验：

- seed 固定为 `530358027`，并设置相同的 `PYTHONHASHSEED`；
- 每条任务使用 4 GPU、每卡 batch 16、gradient accumulation 4；
- nominal effective batch 为 `4 × 16 × 4 = 256`；
- optimizer 为 AdamW，初始 LR 为 `4e-5`；
- Fusion 使用 24-epoch scheduler horizon，Visual 使用 12-epoch horizon；
- 使用同一个已通过闭包检查的 Ego2Exo FullTrain JSON 和图像根目录；
- 不在严格基线中混入 Muon、采样修复、跨卡对比学习或 pooling 修改。

“可续跑”表示恢复后以下状态连续：

1. model state；
2. AdamW 参数和一、二阶矩；
3. LR scheduler 的阶段与位置；
4. AMP/optim wrapper 状态；
5. runner 的 epoch、iteration 和 message hub；
6. 控制 contrastive loss `100 → 1` 切换的 forward 计数；
7. distributed sampler 的 epoch。

训练继续采用作者合同中的 `deterministic=False`。因此恢复运行不承诺和不中断运行
逐 bit 相同；CUDA kernel、worker 随机流和 BF16 仍可产生细微分叉。验收标准是
训练状态和损失分支语义连续，而不是 checkpoint 文件或后续浮点轨迹逐字节相同。

## 3. 隔离架构

不得修改已经用于官方权重推理复现的两个 overlay。实现时从已验证来源复制并发布
两个新的、名称明确的训练 overlay：

- NewMatcher 源：`v2sam-o-ego2exo-7b88c299-overlay-v1`；
- Visual 源：`v2sam-visual-official-24ae5a-overlay-v2`；
- 新 NewMatcher 训练 overlay：`v2sam-newmatcher-resumable-7b88c299-overlay-v1`；
- 新 Visual 训练 overlay：`v2sam-visual-resumable-24ae5a-overlay-v1`。

复制采用 staging 目录、源码哈希校验和原子发布。已有目标目录身份不一致时必须停止，
不能覆盖。每个 overlay 只包含该模型所需的恢复状态改动和测试，不互相链接
`projects/v2sam`。

启动前生成 immutable manifest，至少记录：

- Git revision 或 source archive SHA；
- 活动模型文件及 `models/__init__.py` SHA256；
- 解析后的 Python class、模块名和源码绝对路径；
- runtime config SHA256；
- 两个训练 JSON、数据闭包报告、SAM2 和 DINO 资产 SHA256；
- PyTorch、CUDA、NCCL、MMEngine 和 XTuner 版本；
- seed、world size、per-device batch、accumulation 和 effective batch。

Visual manifest 还必须证明模型 state 中不存在
`sparse_correspondence.*`；Fusion manifest 必须证明该前缀存在。这个结构门禁防止
Visual/Fusion 软链接再次接反。

## 4. Contrastive 状态持久化

当前 `V2SAM` 和 `V2SAM_NEWMATCHER` 都把 `_constr_step` 存成 Python class
attribute。新进程启动时它会回到 0，使已经跨过第 4,000 次 forward 的训练重新将
`loss_contr` 乘 100。此前错误 resume 中 `loss_contr` 从约 1.7 跳到约 175，已经
验证了这个风险。

两个隔离 overlay 都引入一个小型 `ContrastScheduleState` 模块。它包含一个
persistent `torch.long` buffer，并提供一次 forward 对应一次的计数/缩放接口。
边界语义严格保留原实现：

- forward 1 至 3,999 使用 scale 100；
- forward 4,000 及以后使用 scale 1；
- DDP 每个 rank 各自持有同值计数；
- buffer 随 model state 保存和恢复，不参与梯度计算。

日志额外输出 `contr_raw`、`contr_scale` 和 `constr_step`。这些字段名不能包含
`loss`，避免 MMEngine 的 `parse_losses` 把诊断字段再次计入总 loss。

本阶段只修复状态持久性，不把硬切换改成平滑 schedule，也不把 forward 次数改成
optimizer step；后两项属于后续算法消融。

## 5. Checkpoint 与恢复合同

正式配置按 epoch 保存 checkpoint，并设置 `save_optimizer=True`。每个正式
checkpoint 必须包含：

- `state_dict`，且含 contrast schedule 的 persistent step；
- `optimizer`（MMEngine 将 OptimWrapper 的 state dict 保存在此键）；
- `param_schedulers`；
- `meta.epoch`、`meta.iter`、`meta.seed`；
- `message_hub`。

所有 epoch checkpoint 均保留，避免最新文件损坏时没有上一个完整边界可退。恢复只
允许显式指定通过检查的 checkpoint；禁止裸 `--resume` 自动猜测文件。

恢复时始终使用原始、可执行的 canonical runtime config，而不是 work-dir 中
MMEngine dump 出来的配置。后者曾把 `template_map_fn` callable 序列化成字符串并导致
`TypeError: 'str' object is not callable`。

恢复前的 validator 必须拒绝：

- 缺 `optimizer` 或 `param_schedulers` 的 checkpoint；
- 缺 contrast step 的新格式 checkpoint；
- checkpoint model class、源码 SHA、config SHA 或数据 SHA 不匹配；
- checkpoint epoch/iter 与 scheduler 状态不一致；
- `template_map_fn` 不是 callable；
- 目标 work-dir 属于另一实验或仍有活进程。

恢复后的第一条训练日志必须证明：epoch/iter 等于 checkpoint，LR 等于 scheduler
在该 iteration 的预期值，contrast step 没有回零，contrast scale 位于正确分支。
任一项失败立即终止四个 rank。

## 6. 配置与数据流

两条训练都读取已验证的 Ego2Exo FullTrain JSON：

```text
Ego2Exo_FullTrain.json
bytes: 457116335
sha256: bb83cac92179b21833d2465ce29d5b0cf81668d8443d569279313c891472175a
records: 110118
```

图像根目录使用修复后的 Mini 提取目录。最终 union closure 必须保持：

```text
union_referenced_images: 249642
missing_images: 0
bad_jpeg: 0
report_sha256: ad6bc9dec706532e94b1b4bbd0b5a43579e6e6ce767e06479c780558803b140f
```

Fusion/NewMatcher 使用 DINOv3 与 SAM2 资产，并以官方 checkpoint 内嵌 cfg 为训练
合同，不使用 `feat/MOE` 分支当前的 DualDecoder/SmallTrain 活动配置。Visual 只使用
Visual `V2SAM` 合同；不得为了满足名称要求人为挂接 DINO 分支。

训练 run 目录使用不同前缀、不同 master port、独立 PID/log/manifest 文件。0–3 和
4–7 之间不共享 work-dir 或 latest-run 指针。

## 7. 测试与启动门禁

实现遵循 TDD，按以下顺序验证。

### 7.1 单元级 RED/GREEN

1. 先证明当前 class attribute 不进入 `state_dict`，重建对象后计数回零；
2. 再实现 persistent state，使 state-dict round trip 保留第 3,999/4,000 步边界；
3. 验证诊断字段不会改变 MMEngine 计算的总 loss；
4. 验证 resume validator 拒绝旧的 `save_optimizer=False` checkpoint；
5. 验证 model identity gate 能区分 1,335-tensor Fusion 与 964-tensor Visual。

### 7.2 实际四卡中断/恢复 smoke

每个 overlay 使用自己的小型训练 scope，并把日志间隔设为 1：

1. 使用正式的 4 GPU、batch 16/GPU、accumulation 4 跑到至少两个 optimizer steps；
2. 保存带 optimizer 的测试 checkpoint；
3. 正常终止所有 rank；
4. 从该 checkpoint 重启并再跑至少一个 optimizer step；
5. 比较重启前后 epoch/iter、LR、contrast step、contrast scale 和 optimizer step。

NewMatcher 先前曾在 4 GPU 上超过 20 分钟仍没有第一条 `Iter(train)`。因此 smoke
设置启动超时：模型和 dataloader 已完成初始化后 10 分钟内仍没有第一个 iteration，
判为性能门禁失败，捕获进程、GPU、I/O 和 stack 证据，不直接启动两天训练。

### 7.3 模型谱系门禁

Visual 可用官方 Visual checkpoint 做小样本 sanity check，其公开路径已得到约
`0.3670 / 0.4280`。

NewMatcher/Fusion 仍有一个已知 P0 风险：相同官方 Fusion 张量在公开 `V2SAM`
路径得到约 `0.4469 / 0.5097`，在当前 NewMatcher runtime 得到约
`0.1727 / 0.2132`。因此在启动 24 epoch 前，必须用冻结配置完成小样本 forward
contract 检查并解释这一分叉。若同一官方 checkpoint 仍在 NewMatcher 路径产生低值，
0–3 只用于诊断，不把该代码标为官方严格训练，也不盲跑完整周期。

## 8. 正式调度与监控

所有门禁通过后并发启动：

- GPUs 0–3：Fusion/NewMatcher，24 epochs；
- GPUs 4–7：B/Visual `V2SAM`，12 epochs。

每条任务至少记录：

- 每 10 iterations 的 LR、总 loss、各 mask/dice 分量、scaled contrastive、
  `contr_raw`、`contr_scale`、`constr_step` 和 grad norm；
- 每个 epoch 的 validation stock IoU/Dice；
- 每个 checkpoint 的 SHA256、大小、epoch、iter 和恢复合同检查结果；
- 进程退出码、最终 GPU 释放状态和结果 JSON。

监控只报告最新 iteration、吞吐、ETA、loss 分量和错误数，不用大量模型参数日志淹没
终端。实例回收后，先运行 checkpoint validator，再用显式路径恢复；不得重复此前的
机械 `--resume`。

## 9. 失败处理与结果命名

- source/config/data 任一身份不匹配：不启动 GPU；
- 四卡任一 rank 失败：终止同一任务的全部 rank，另一组任务可继续；
- LR、optimizer 或 contrast step 恢复不连续：将 checkpoint 标为不可恢复并退回
  上一个通过检查的 epoch；
- NewMatcher forward contract 未通过：结果只能标为“NewMatcher candidate
  diagnostic”，不能标为官方复现；
- Visual 结果始终标为 `V2SAM Visual`，不得标成 `V2SAM_NEWMATCHER Visual`；
- warm-start、低 LR continuation、Muon 和任何算法修改都建立新实验标签与 work-dir，
  不覆盖本轮严格基线。

## 10. 验收标准

开始正式训练前必须同时满足：

1. 两个 overlay 的源码和模型身份门禁通过；
2. 两套单元测试经历预期 RED 后全部 GREEN；
3. 两套四卡中断/恢复 smoke 证明 optimizer、scheduler 和 contrast step 连续；
4. 数据闭包、资产和 config 哈希通过；
5. Visual 官方 checkpoint sanity check 通过；
6. NewMatcher 的官方 checkpoint/runtime 分叉得到解释或被明确降级为诊断实验。

正式运行完成的最低交付物为：两条 immutable run manifest、完整日志、逐 epoch
checkpoint receipt、validation 历史、最终指标 JSON，以及一次真实实例中断后的恢复
证据。只有这些证据齐全，才能声明训练“可续跑”或结果“严格复现”。
