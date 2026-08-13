# V2-SAM Ego2Exo + Exo2Ego 混训流程

## 1. 目标

使用同一个 V2-SAM Fusion 单专家模型，同时训练 Ego2Exo 和 Exo2Ego。

本方案不修改模型结构和原有 loss，只新建联合训练配置，并使用官方
`fusion_exo2ego_full.pth` 作为初始化权重进行 4 个 epoch 的微调。

## 2. 数据准备

训练数据：

```text
data_segswap/
ego2exo_train.json    110118 条
exo2ego_train.json    123381 条
```

测试数据：

```text
data_segswap_test/
ego2exo_test.json
exo2ego_test.json
```

当前服务器参考路径：

```text
/ssd1/zhangzheng/v2samo/datasets/Ego-Exo4D-Seg/work/yuqian_fu/Ego/data_segswap
/ssd1/zhangzheng/v2samo/datasets/Ego-Exo4D-Seg/hf_train/ego2exo_train.json
/ssd1/zhangzheng/v2samo/datasets/Ego-Exo4D-Seg/hf_train/exo2ego_train.json
```

## 3. 下载初始化权重

在 V2-SAM 项目根目录执行：

```bash
cd /path/to/V2-SAM

HF_ENDPOINT=https://hf-mirror.com \
hf download jaychempan/V2-SAM \
  weights/fusion_exo2ego_full.pth \
  --local-dir .
```

权重信息：

```text
文件：weights/fusion_exo2ego_full.pth
大小：2126037244 bytes
SHA-256：0e6e2589e2055cb64b92c2967c7c5796bb9530a720531c6511cce82cfa012de3
```

建议下载后校验：

```bash
sha256sum weights/fusion_exo2ego_full.pth
```

## 4. 新建联合训练配置

从原始 Fusion 配置复制一份：

```text
projects/v2sam/configs/v2sam_joint_from_exo2ego_4ep.py
```

核心参数：

```python
pretrained_pth = 'weights/fusion_exo2ego_full.pth'

batch_size = 16
accumulative_counts = 4
max_epochs = 4
lr = 1e-5

work_dir = './work_dirs/v2sam_joint_from_exo2ego_4ep'

randomness = dict(seed=42, deterministic=False)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
)
```

分别定义两个训练数据集：

```python
ego2exo_dataset = dict(
    type=VideoObjectRelatorDataset,
    sam2_folder=train_data_root,
    expression_file=ego2exo_train_json,
    template_map_fn=dict(
        type=template_map_fn_factory,
        template=prompt_template,
    ),
    max_length=max_length,
    lazy=True,
    mode='short',
    repeats=1,
    extra_image_processor=extra_image_processor,
    sampled_frames=1,
    select_number=5,
)

exo2ego_dataset = dict(
    type=VideoObjectRelatorDataset,
    sam2_folder=train_data_root,
    expression_file=exo2ego_train_json,
    template_map_fn=dict(
        type=template_map_fn_factory,
        template=prompt_template,
    ),
    max_length=max_length,
    lazy=True,
    mode='short',
    repeats=1,
    extra_image_processor=extra_image_processor,
    sampled_frames=1,
    select_number=5,
)
```

使用 `ConcatDataset` 拼接两个方向：

```python
train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        ego2exo_dataset,
        exo2ego_dataset,
    ],
)
```

两个训练集分别包含 110118 和 123381 条数据，比例约为 `47:53`，不需要额外设置任务采样权重。

## 5. Loss 机制

混训不需要增加 task-specific loss。Ego2Exo 和 Exo2Ego 的方向由 prompt 图像与 target 图像的互换表达，两个方向共用原模型 loss。

总 loss 为：

```text
loss_mask
+ loss_dice
+ small_loss_mask
+ small_loss_dice
+ loss_contr
```

原代码中的权重关系：

```text
主 mask BCE：配置权重 2.0，结果再乘 10
主 Dice：配置权重 0.5，结果再乘 10
辅助 mask BCE：配置权重 2.0
辅助 Dice：配置权重 0.5
contrastive loss：前 4000 iter 乘 100，之后乘 1
```

需要注意，4 个 epoch 共约 14608 iter，因此第一个 epoch 和第二个 epoch 开头主要由 contrastive loss 主导。这是原模型代码中的机制，不是联合训练配置额外引入的逻辑。

优化器使用 BF16 AMP，并设置梯度裁剪：

```python
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=1, error_if_nonfinite=False),
    accumulative_counts=4,
    loss_scale='dynamic',
    dtype='bfloat16',
)
```

## 6. 启动四卡混训

```bash
tmux new-session -d -s v2sam_joint '
set -o pipefail
cd /path/to/V2-SAM &&
export PATH=/path/to/conda/env/bin:$PATH &&
export PYTHONPATH=.:./mmengine:${PYTHONPATH:-} &&
export CUDA_VISIBLE_DEVICES=0,1,2,3 &&
export HF_HOME=/path/on/data-disk/cache/huggingface &&
export TORCH_HOME=/path/on/data-disk/cache/torch &&
export TMPDIR=/path/on/data-disk/tmp &&
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" \
  work_dirs/v2sam_joint_from_exo2ego_4ep &&
PORT=29518 bash tools/dist.sh train \
  projects/v2sam/configs/v2sam_joint_from_exo2ego_4ep.py 4 \
  2>&1 | tee work_dirs/v2sam_joint_from_exo2ego_4ep/train_console.log
'
```

查看训练：

```bash
tmux attach -t v2sam_joint
```

查看日志：

```bash
tail -f work_dirs/v2sam_joint_from_exo2ego_4ep/train_console.log
```

每个 epoch 保存一个 checkpoint：

```text
epoch_1.pth
epoch_2.pth
epoch_3.pth
epoch_4.pth
```

多日训练建议保存优化器状态：

```python
checkpoint = dict(
    type=CheckpointHook,
    save_optimizer=True,
    by_epoch=True,
    interval=1,
)
```

如果使用 `save_optimizer=False`，训练中断后只能重新加载模型参数，无法严格恢复优化器和学习率调度状态。

## 7. Ego2Exo 验证

当前方案在每个 epoch 后验证 Ego2Exo：

```text
数据目录：data_segswap_test
标注文件：ego2exo_test.json
```

使用修正后的完整指标：

```python
custom_imports = dict(
    imports=[
        'projects.v2sam.evaluation',
        'v2sam_metric_fix.seg_metric_full',
    ],
    allow_failed_imports=False,
)

val_evaluator = dict(
    type='SegMetricFull',
    iou_metrics=['IoU', 'Dice'],
)
```

`SegMetricFull` 输出三种聚合：

```text
stock IoU/Dice：复现原指标的截断行为
per-object IoU/Dice：统计全部 object mask
per-pair IoU/Dice：先对每个图像对求均值，再跨图像对求均值
```

当前实验的 Ego2Exo stock IoU：

```text
epoch 1：25.74
epoch 2：29.45
epoch 3：30.83
epoch 4：30.87
```

## 8. Exo2Ego 测试

训练结束后需要将验证 JSON 切换为：

```text
exo2ego_test.json
```

单卡测试示例：

```bash
cd /path/to/V2-SAM

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=.:./mmengine:${PYTHONPATH:-} \
python tools/test.py \
  projects/v2sam/configs/v2sam_joint_from_exo2ego_4ep.py \
  --checkpoint work_dirs/v2sam_joint_from_exo2ego_4ep/epoch_4.pth \
  --cfg-options \
  val_dataloader.dataset.datasets.0.expression_file=/path/to/exo2ego_test.json \
  test_dataloader.dataset.datasets.0.expression_file=/path/to/exo2ego_test.json
```

建议同时测试两组权重：

```text
1. 原始 fusion_exo2ego_full.pth
2. 联合训练后的 epoch_4.pth
```

通过对比两组 Exo2Ego 指标，可以判断联合训练是否造成旧方向能力遗忘。

## 9. 注意事项

1. 当前方案从 Exo2Ego 专用权重初始化，再联合训练两个方向。
2. Ego2Exo 和 Exo2Ego 必须分别报告，不能只报告混合平均值。
3. 每个 epoch 使用正式 test JSON 验证只适合内部排查。论文实验应划分独立验证集，最终测试集只使用一次。
4. `SegMetricFull` 的主要参考指标建议使用 `per-object` 或 `per-pair`，同时保留 stock 指标用于和旧结果对齐。
5. 多卡训练前应确认 GPU 空闲、数据路径有效、初始化权重 SHA-256 正确。
6. 训练启动后应在日志中确认出现：

```text
Load pretrained weight from weights/fusion_exo2ego_full.pth
```

否则说明初始化权重没有真正加载。
