# Ego2Exo / Exo2Ego Joint Training 复现指南

本文只讨论**任务层面的双向联合训练**：在同一次训练中，同时使用
`ego2exo` 和 `exo2ego` 两个方向的数据，共享同一个 ObjectRelator 模型和优化器。

这里的 joint training 不表示每个训练 step 都会严格配对一条 `ego2exo` 和一条
`exo2ego` 样本，也不表示模型中存在两个方向专属的 head。当前实现的本质是：

```text
ego2exo dataset ─┐
                 ├─ ConcatDataset ── shuffle / batch ── shared ObjectRelator
exo2ego dataset ─┘
```

相关入口：

- 训练说明：[docs/Train_Evaluation.md](docs/Train_Evaluation.md)
- 数据生成：[datasets/build_egoexo.py](datasets/build_egoexo.py)
- 联合数据模块：[objectrelator/train/train_ObjectRelator.py](objectrelator/train/train_ObjectRelator.py)
- 数据集与 batch collator：[objectrelator/train/train_datasets.py](objectrelator/train/train_datasets.py)
- 参数定义：[objectrelator/mask_config/data_args.py](objectrelator/mask_config/data_args.py)
- 启动脚本：[scripts/train_ObjectRelator.sh](scripts/train_ObjectRelator.sh)

## 1. 双向任务的数据定义

`datasets/build_egoexo.py` 使用同一个数据构建流程，通过 `--task` 决定 query
视角和 target 视角：

| 任务 | Query / visual prompt | Target image | 分割监督 |
|---|---|---|---|
| `ego2exo` | ego 图像及其对象 mask | exo 图像 | exo 对象 mask |
| `exo2ego` | exo 图像及其对象 mask | ego 图像 | ego 对象 mask |

构建后的每条 JSON 数据主要包含：

```text
image               目标视角图像
anns                目标视角的分割标注
first_frame_image   查询视角图像
first_frame_anns    查询视角的 visual prompt mask
instruction         查询对象对应的文本描述
```

两个任务最终使用完全相同的数据字段和 `EgoExo_Dataset_train` 类。模型不需要额外的
task ID 来区分方向；方向已经由 query/target 字段的组织方式决定。

## 2. 生成 ego2exo 和 exo2ego 数据

以下命令假设 Ego-Exo4D 已经按照 [docs/DATASET.md](docs/DATASET.md) 完成视频帧和
标注预处理。

先生成不含文本描述的双向训练 JSON：

```bash
python datasets/build_egoexo.py \
    --root_path /path/to/ego-exo4d/data_root \
    --save_path /path/to/save/ego2exo_train_visual.json \
    --split_path /path/to/ego-exo4d/data_root/split.json \
    --split train \
    --task ego2exo

python datasets/build_egoexo.py \
    --root_path /path/to/ego-exo4d/data_root \
    --save_path /path/to/save/exo2ego_train_visual.json \
    --split_path /path/to/ego-exo4d/data_root/split.json \
    --split train \
    --task exo2ego
```

`build_egoexo.py` 会在每个方向上分别完成以下处理：

1. 根据任务方向设置 query camera 和 target camera。
2. 找出两个视角中同时可见的对象。
3. 为 query 图像生成 `first_frame_anns`。
4. 为 target 图像生成 `anns`。
5. 保留两个视角中对象的对应顺序，并检查两侧对象数量一致。

然后按照 [docs/DATASET.md](docs/DATASET.md) 中的流程，分别对两份 JSON 使用外部
LLaVA `gen_text.py` 生成对象描述。例如：

```bash
# 在提供 gen_text.py 的 LLaVA 环境中执行
python gen_text.py \
    --image_path /path/to/ego-exo4d/data_root \
    --json_path /path/to/save/ego2exo_train_visual.json \
    --save_path /path/to/save/ego2exo_train_visual_text_tmp.json

python gen_text.py \
    --image_path /path/to/ego-exo4d/data_root \
    --json_path /path/to/save/exo2ego_train_visual.json \
    --save_path /path/to/save/exo2ego_train_visual_text_tmp.json
```

最后使用仓库内的 `build_text.py` 整理文本字段：

```bash
python datasets/build_text.py \
    --text_path /path/to/save/ego2exo_train_visual_text_tmp.json \
    --save_path /path/to/save/ego2exo_train_visual_text.json

python datasets/build_text.py \
    --text_path /path/to/save/exo2ego_train_visual_text_tmp.json \
    --save_path /path/to/save/exo2ego_train_visual_text.json
```

训练前建议检查：

```python
import json

for path in [
    "/path/to/save/ego2exo_train_visual_text.json",
    "/path/to/save/exo2ego_train_visual_text.json",
]:
    data = json.load(open(path))
    print(path, len(data))
    assert len(data) > 0
    assert all("image" in x for x in data)
    assert all("anns" in x for x in data)
    assert all("first_frame_image" in x for x in data)
    assert all("first_frame_anns" in x for x in data)
    assert all("instruction" in x for x in data)
```

还应抽查每条数据是否满足：

```python
len(sample["anns"]) == len(sample["first_frame_anns"])
```

这不仅影响分割监督，也影响仓库中对象级跨视角对齐损失对对象顺序的假设。

## 3. 开启 joint training

联合训练使用以下三个参数：

```text
joint_training=True
joint_json_ego2exo=/path/to/ego2exo_train_visual_text.json
joint_json_exo2ego=/path/to/exo2ego_train_visual_text.json
```

`scripts/train_ObjectRelator.sh` 已经提供了两个 JSON 参数，但当前脚本没有显式开启
`joint_training`。建议将相关部分配置为：

```bash
deepspeed --master_port=29526 objectrelator/train/train_ObjectRelator.py \
    --joint_training True \
    --joint_json_ego2exo "/path/to/ego2exo_train_visual_text.json" \
    --joint_json_exo2ego "/path/to/exo2ego_train_visual_text.json" \
    --image_folder "/path/to/ego-exo4d/data_root" \
    --seg_task "region" \
    ...
```

也可以按照原文档的做法，将
`objectrelator/mask_config/data_args.py` 中的默认值改为：

```python
joint_training: bool = True
```

仅设置两份 `joint_json_*` 路径不会自动开启 joint training。如果
`joint_training=False`，训练入口只会读取 `region_json_path`，默认脚本中的该路径
是 ego2exo 数据。

## 4. 任务数据的合并方式

开启 joint training 后，`make_unify_datamodule_joint()` 执行：

```python
egoexo_dataset = EgoExo_Dataset_train(
    json_path=data_args.joint_json_ego2exo,
    ...
)
exoego_dataset = EgoExo_Dataset_train(
    json_path=data_args.joint_json_exo2ego,
    ...
)

datasets = [egoexo_dataset + exoego_dataset]
```

PyTorch Dataset 的 `+` 返回 `ConcatDataset`，因此联合训练集的长度为：

```text
N_joint = N_ego2exo + N_exo2ego
```

随后这个 `ConcatDataset` 被放入 `UnifyDatasetSingleDatasetForBatch`。注意外层
`datasets` 列表只有一个元素，所以 `UnifyDatasetSingleDatasetForBatch` 中“每 16
条样本切换数据集”的逻辑并不会在 ego2exo 和 exo2ego 之间切换。

## 5. 实际任务采样行为

`LLaVATrainerSSL` 没有重写 train sampler，训练使用 Hugging Face Trainer/Accelerate
的常规随机或分布式采样逻辑。采样对象是已经拼接好的联合数据集。

因此一个 epoch 中两个任务的期望采样比例是：

```text
P(ego2exo) = N_ego2exo / (N_ego2exo + N_exo2ego)
P(exo2ego) = N_exo2ego / (N_ego2exo + N_exo2ego)
```

当前实现具有以下行为：

- batch 可以同时包含两个方向的样本；
- batch 也可能只包含其中一个方向；
- 不保证每个 step 都是 1:1 配对；
- 不保证整个训练中的两个任务权重为 1:1；
- `data_ratio` 没有参与这条联合训练路径的双向任务采样；
- `bs=16` 只属于外层数据集切换设计，对当前的内部 `ConcatDataset` 不起双向平衡作用。

如果两份 JSON 长度相差很大，较长任务会获得更高训练权重。复现仓库原始行为时应保留
这种按数据量加权的采样。若实验要求严格 1:1，需要额外实现 balanced sampler、对较短
数据集重复采样，或者修改联合数据模块；这不属于当前仓库的默认实现。

在分布式训练下，全局 batch size 为：

```text
per_device_train_batch_size
× GPU 数量
× gradient_accumulation_steps
```

`dataloader_drop_last=True`，最后一个不完整 batch 会被丢弃。分布式 sampler 还可能为
保证各 rank 长度一致而补齐索引，因此精确 step 数应以 Trainer 日志为准。

## 6. 两个任务如何共享模型训练

两个方向使用相同的数据类、collator、模型 forward 和损失函数。模型不会收到
`ego2exo`/`exo2ego` 字符串，也没有方向专属参数。

对任意一个 batch：

1. `DataCollatorForCOCODatasetV2` 将两个方向的样本按相同字段进行 padding 和堆叠。
2. `LLaVATrainerSSL.compute_loss()` 只调用一次 `model(**inputs)`。
3. ObjectRelator 根据每条样本中已经组织好的 query/target 图像计算目标分割。
4. batch 内所有样本的分割和对象对应损失共同组成一个标量 `loss`。
5. 对该统一 loss 执行一次反向传播和一次共享优化器更新。

对两个任务而言，区别只在数据语义上：

```text
ego2exo sample:
    ego prompt + exo target → exo segmentation loss

exo2ego sample:
    exo prompt + ego target → ego segmentation loss
```

模型层面完全共享：

- vision encoder；
- multimodal projector；
- language model；
- region/text fusion 模块；
- pixel decoder；
- Mask2Former decoder；
- region projector 和分割 query。

当前代码没有：

- 两套模型；
- 两个 optimizer；
- 两个方向专属 head；
- task-specific loss weight；
- ego2exo/exo2ego 交替训练循环；
- 在 Trainer 中单独记录两个方向的 loss。

所以任务层面的 joint training 可以写成：

```text
L_joint(θ)
  = Σ L(sample_i; θ),  sample_i ∈ shuffled(D_ego2exo ∪ D_exo2ego)
```

两个任务通过同一组参数 `θ` 相互影响。

## 7. 与两阶段训练结合

`joint_training` 与 `first_stage` 是两个独立开关：

- `joint_training`：决定训练集是否包含两个方向；
- `first_stage`：决定当前阶段哪些模型参数可训练，以及是否缩小数据集。

### Stage 1

建议配置：

```text
joint_training=True
first_stage=True
pretrained_model_path=/path/to/base_pretrained_model
output_dir=/path/to/objectrelator-stage1-joint
```

Stage 1 会：

- 分别读取 ego2exo 和 exo2ego JSON；
- 对每份数据随机保留约 `1/20`；
- 合并两个方向的子集；
- 冻结除参数名包含 `fuse_model` 以外的参数；
- 使用两个方向的分割监督训练 MCFuse。

Stage 1 依赖 `condition="multi-condition"`，因为 `fuse_model` 只在 multi-condition
配置下创建。

### Stage 2

建议配置：

```text
joint_training=True
first_stage=False
pretrained_model_path=/path/to/objectrelator-stage1-joint
output_dir=/path/to/objectrelator-stage2-joint
```

Stage 2 会重新构造完整的双向联合训练集，并从 Stage 1 checkpoint 加载模型。关闭
`first_stage` 后，不再执行“只训练 `fuse_model`”的冻结逻辑；但 vision backbone 默认
仍会因为 `train_backbone=False` 而被冻结。

若 Stage 1 使用 joint training，而 Stage 2 忘记开启 `joint_training`，Stage 2 将退化为
只训练 `region_json_path` 指向的单方向任务。

## 8. 配置读取陷阱

仓库中部分模块没有直接使用训练入口解析后的参数，而是在模块导入时重新创建默认参数：

```python
# objectrelator/train/train_datasets.py
training_args = TrainingArguments()

# ObjectRelator_decoder.py
data_args = DataArguments()
```

这会导致：

- `--joint_training True` 可以正常控制训练入口的数据模块选择；
- 仅通过命令行传入 `--first_stage True` 时，模型冻结逻辑会生效，但
  `train_datasets.py` 中的 `1/20` 子采样不一定生效；
- 仅通过命令行修改 `condition` 时，decoder 内部可能仍使用
  `data_args.py` 中的默认 condition。

为了严格复现仓库文档描述的行为，建议每个阶段同时：

1. 在启动脚本中显式传递 `--joint_training`、`--first_stage` 和两份 JSON 路径；
2. 将 `objectrelator/mask_config/data_args.py` 中的 `first_stage` 与 `condition`
   默认值同步为当前阶段需要的值；
3. Stage 1 完成后，将 `first_stage` 默认值改回 `False` 再启动 Stage 2。

更稳健的长期修复方式，是把训练入口解析后的 `training_args` 和 `data_args` 显式传入
数据集与 decoder，而不是在模块导入时重新实例化参数。

## 9. 复现检查清单

训练前：

- [ ] 已用 `--task ego2exo` 生成正向 JSON。
- [ ] 已用 `--task exo2ego` 生成反向 JSON。
- [ ] 两份 JSON 都已完成 LLaVA 文本生成和 `build_text.py` 处理。
- [ ] 两份 JSON 的路径分别传给 `joint_json_ego2exo` 和 `joint_json_exo2ego`。
- [ ] 已确认 `joint_training=True`。
- [ ] `image_folder` 指向两份 JSON 共同使用的 Ego-Exo4D 根目录。
- [ ] 已记录两份 JSON 的样本数，并确认是否接受按数据量采样。
- [ ] Stage 1 时 `condition="multi-condition"`、`first_stage=True`。
- [ ] Stage 2 从 Stage 1 checkpoint 加载，并设置 `first_stage=False`。
- [ ] 两个阶段均保持 `joint_training=True`。

训练开始后：

- [ ] 日志中的 `total unify dataset number` 与预期数据量一致。
- [ ] 非 Stage 1 时应接近 `N_ego2exo + N_exo2ego`。
- [ ] Stage 1 时应接近两份数据各自 `1/20` 后的总和。
- [ ] 检查 checkpoint 是否写入预期的 `output_dir`。
- [ ] 检查 Trainer 日志中的 mask、dice、region 和跨视角对齐损失是否为有限值。

## 10. 一句话总结

当前仓库的 ego2exo/exo2ego joint training 是一种**数据集拼接式多任务训练**：
先独立生成两个方向的数据，再将其拼接、随机采样，并让所有样本共享同一个
ObjectRelator 和统一损失进行参数更新；任务比例由两份数据的长度决定，而不是由显式
任务调度器或固定的双向配对策略决定。
