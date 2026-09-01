# New PCCS Multi-Expert Metric Run Changes

## 目标

把学长新版 PCCS 多专家评价策略同步到服务器，并用同一模型、同一数据重新跑 5 个 frame-level setting，看新版策略对最终指标的影响。

## 服务器改动

服务器代码目录：

`/ssd1/zhangzheng/v2samo/code/V2-SAM-O/projects/v2sam/evaluation/`

新增/同步了 3 个 evaluator 文件：

- `pccs_metric.py`
- `frame_level_seg_metric.py`
- `obj_level_seg_metric.py`

修改了：

- `__init__.py`

改动内容：

- 注册 `PCCSMetric`
- 注册 `PCCSFrameLevelMetric`
- 注册 `ObjLevelSegMetric`
- 注册 `FrameLevelSegMetric`

同步前已备份原注册文件：

`/ssd1/zhangzheng/v2samo/code/V2-SAM-O/projects/v2sam/evaluation/__init__.py.bak_20260831_191128`

## 兼容性改动

学长新版 metric 默认读取这些 key：

- `pred_masks_visual`
- `pred_masks_anchor`
- `pred_masks_fusion`
- `points_visual_in_query`
- `points_anchor_in_query`

服务器旧 V2-SAM-O 三专家模型实际输出的是：

- `pred_masks_vp`
- `pred_masks_sparse`
- `pred_masks_fusion`
- `points_vp_in_query`
- `points_sparse_in_query`

所以我只加了 key alias，映射关系是：

| 新版命名 | 服务器旧命名 |
|---|---|
| `visual` | `vp` |
| `anchor` | `sparse` |
| `fusion` | `fusion` |

同时补了旧保存脚本需要的接口：

- `get_decoder_info()`

这个接口只是返回新版的 `selection_info`，方便继续用原来的 `tools/test_with_decoder_save_ddp.py` 保存专家选择信息。

## 没有改动的东西

没有改模型结构。

没有改 checkpoint/权重。

没有改数据 JSON。

没有改 dataloader 采样逻辑。

没有改 IoU/Dice 公式。

这次重跑主要只改变最终多专家选择策略和 evaluator 统计方式。

## 新版 PCCS 策略理解

这次跑的是：

`PCCSFrameLevelMetric`

策略大致是：

1. 先检查 fusion mask 质量。
2. 如果 fusion 质量通过，优先选 fusion。
3. 如果 fusion 质量不过关，再在 visual/anchor 两个专家之间用 cycle consistency 选。
4. 最终按 frame-level 方式汇总指标。

## 跑的 5 个 setting

输出根目录：

`/ssd1/zhangzheng/v2samo/new_pccs_metric_runs/framelevel_20260831_192303`

5 个子目录：

- `egoexo4d_exo2exo_same_random`
- `droid_ego2exo`
- `droid_exo2ego`
- `behaviorsim_ego2exo`
- `behaviorsim_exo2ego`

每个目录里都有：

- `console.log`
- `decoder_info_newpccs.json`
- `decoder_info_newpccs_dedup.json`
- `pccs_diag.jsonl`
- `status.txt`

注意：`decoder_info_newpccs.json` 里 DDP sampler padding 会多 1-2 条重复记录，所以我额外生成了去重版 `decoder_info_newpccs_dedup.json`，这个数量和最终日志里的 evaluated object count 对齐。

## 新指标结果

| Setting | 旧 IoU / Dice | 新 IoU / Dice | 变化 |
|---|---:|---:|---:|
| EgoExo4D exo2exo same/random | 37.32 / 41.37 | 36.38 / 40.50 | -0.94 / -0.87 |
| DROID ego2exo | 61.92 / 64.99 | 63.66 / 66.71 | +1.74 / +1.72 |
| DROID exo2ego | 61.14 / 65.23 | 63.75 / 66.93 | +2.61 / +1.70 |
| BehaviorSim ego2exo | 48.89 / 53.54 | 49.35 / 54.17 | +0.46 / +0.63 |
| BehaviorSim exo2ego | 47.72 / 51.74 | 52.61 / 56.63 | +4.89 / +4.89 |

## 专家选择分布

| Setting | visual / anchor / fusion |
|---|---:|
| EgoExo4D exo2exo | 164 / 325 / 605 |
| DROID ego2exo | 480 / 913 / 769 |
| DROID exo2ego | 254 / 1126 / 782 |
| BehaviorSim ego2exo | 14 / 70 / 159 |
| BehaviorSim exo2ego | 31 / 71 / 141 |
