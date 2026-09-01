# Exo2Exo 六个 Baseline 实验配置与对齐说明

生成时间：2026-08-30

本文档用于说明 Exo2Exo zero-shot segmentation 表格中六个 baseline 的代码来源、权重来源、实际运行配置，以及相对于 Runze 学长当时给出的 HANDAL-X baseline 说明有哪些严格对齐、哪些地方为了 Exo2Exo 数据做了必要适配。

## 1. 总体结论

整体上，六个方法的模型来源与 Runze 学长给出的 baseline 方向是对齐的：XSegTx、SEEM、PSALM、ObjectRelator 均使用对应官方仓库或学长指定的 baseline 权重；V2-SAM 与 V2-SAM-O 使用 V2-SAM 系列代码与 Ego-Exo4D 方向权重。

但本次任务不是复现 HANDAL-X，而是把同一批模型评测到我们新构建的 Exo2Exo 数据上。因此我们没有直接使用 HANDAL-X 的 json，而是新增了若干 evaluation wrapper，用于读取 Exo2Exo json、解码 RLE mask、跑模型推理并统一计算 IoU/Dice。这些 wrapper 不重新训练模型，也不修改 baseline 的核心网络结构。

需要特别说明的一点：当前 PSALM 行使用的是 `zamling/PSALM` 官方仓库与 official PSALM 权重，不是 `YuqianFu/ObjectRelator-Original` 这一份 Hugging Face baseline 包。如果后续要求完全对齐 Runze 当时上传的 PSALM 复现包，需要额外再跑一版。

## 2. 评测数据

本次六模型表格使用的 Exo2Exo 数据来自人工筛选后的 exo prompt mask，并构造成“一帧 prompt mask 对另一 exo 视角 target GT mask”的 frame-level json。

| 文件 | 用途 | 样本数 |
|---|---|---:|
| `/ssd1/zhangzheng/v2samo/datasets/Ego-Exo4D-Seg/json/human_select_processed/exo2exo_selected_sam2_prompt_pair_same_random.json` | V2-SAM / V2-SAM-O / XSegTx | 1094 |
| `/ssd1/zhangzheng/v2samo/datasets/Ego-Exo4D-Seg/json/human_select_processed/exo2exo_selected_sam2_prompt_pair_same_random_objectrelator_v2.json` | ObjectRelator / PSALM / SEEM | 1093 |

数据组成：

- 547 条 temporal strictly matched：prompt frame 与 target frame 相同。
- 547 条 temporal mismatched：每个 prompt 额外随机采样一个存在同一物体的 target frame。

ObjectRelator / PSALM / SEEM 使用 1093 条，是因为转换到它们需要的输入格式时，有 1 条 target mask 为空或不可用；V2-SAM、V2-SAM-O、XSegTx 使用完整 1094 条。

## 3. Baseline 来源与实际配置

| 方法 | Runze 给的参考 | 实际代码路径 | 实际权重路径 | 对齐情况 |
|---|---|---|---|---|
| XSegTx | `EGO4D/ego-exo4d-relation/tree/main/correspondence`，以及 `YuqianFu/Xseg-Baseline` | `/ssd1/zhangzheng/v2samo/baselines/ego-exo4d-relation/correspondence/SegSwap` | `/ssd1/zhangzheng/v2samo/baselines/weights/xseg/correspondence/SegSwap/model/model.pth` | 对齐。代码来自 EGO4D 官方 correspondence 仓库，权重来自学长给的 XSeg baseline 包。 |
| SEEM | `UX-Decoder/Segment-Everything-Everywhere-All-At-Once` | `/ssd1/zhangzheng/v2samo/baselines/SEEM` | `/ssd1/zhangzheng/v2samo/baselines/weights/seem/seem_focall_v0.pt` | 对齐。只做推理，没有重新训练，符合 Runze 提到的 SEEM 只测试推理。 |
| PSALM | `zamling/PSALM`，另有学姐 HF baseline 包说明 | `/ssd1/zhangzheng/v2samo/baselines/PSALM` | `/ssd1/zhangzheng/v2samo/baselines/weights/psalm_official` | 部分对齐。代码对齐 official PSALM；当前结果使用 official PSALM 权重，不是 `YuqianFu/ObjectRelator-Original`。 |
| ObjectRelator | `lovelyqian/ObjectRelator` | `/ssd1/zhangzheng/v2samo/baselines/ObjectRelator` | `/ssd1/zhangzheng/v2samo/baselines/weights/objectrelator_joint` | 对齐。代码来自 ObjectRelator 官方仓库，权重使用 Ego-Exo4D joint 版本。 |
| V2-SAM Single-Expert/Fusion | `jaychempan/V2-SAM` | `/ssd1/zhangzheng/v2samo/code/V2-SAM` | `/ssd1/zhangzheng/v2samo/code/V2-SAM/weights/fusion_exo2ego_full.pth` | 对齐 V2-SAM 旧版单专家 fusion。为了本地 import，把 `v2sam_fusion` 改名/复制成 `projects/v2sam`。 |
| V2-SAM-O PCCS Multi-Experts | V2-SAM-O 内部仓库 `wrz` 分支 | `/ssd1/zhangzheng/v2samo/code/V2-SAM-O` | VP: `/ssd1/zhangzheng/v2samo/code/V2-SAM-O/weights/vp_exo2ego_full.pth`; Fusion: `/ssd1/zhangzheng/v2samo/code/V2-SAM-O/weights/fusion_exo2ego_full.pth` | 对齐内部 V2-SAM-O 代码。使用 PCCS/cycle-consistency 选择专家，不使用 GT oracle 选专家。 |

仓库版本：

| 仓库 | remote | commit / 状态 |
|---|---|---|
| XSegTx / ego-exo4d-relation | `https://github.com/EGO4D/ego-exo4d-relation.git` | `d6b7443` |
| SEEM | `https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git` | `7b2e76d` |
| PSALM | `https://github.com/zamling/PSALM.git` | `d5dbfda` |
| ObjectRelator | `https://github.com/lovelyqian/ObjectRelator.git` | `25ecbc0` |
| V2-SAM | `https://github.com/jaychempan/V2-SAM.git` | `24ae5a0`，本地有 `projects/v2sam` 适配目录 |
| V2-SAM-O | 内部压缩包 / `wrz` 分支代码 | 非 git 目录 |

## 4. 我们新增或修改过的内容

这些修改的目的都是让不同 baseline 能读取同一个 Exo2Exo 数据格式，或者把结果保存成统一可统计的 json。除 V2-SAM-O 权重加载兼容外，没有改各 baseline 的核心模型结构。

### XSegTx

新增脚本：

`/ssd1/zhangzheng/v2samo/baselines/ego-exo4d-relation/correspondence/SegSwap/eval_xseg_pair_exo2exo.py`

主要适配：

- 读取 Exo2Exo json。
- 解码 prompt mask 和 target GT mask。
- 使用 SegSwap 原始模型与原始 `MASKThresh` 阈值。
- 如预测 mask 尺寸与 GT 不一致，使用 nearest resize 对齐。
- 输出 raw per-sample json 与统计文件。

### SEEM

新增脚本：

`/ssd1/zhangzheng/v2samo/baselines/SEEM/eval_seem_pair_exo2exo.py`

主要适配：

- 读取 ObjectRelator/SEEM 兼容格式的 Exo2Exo json。
- 使用 `configs/seem/focall_unicl_lang_demo.yaml` 与 `seem_focall_v0.pt`。
- 将 prompt mask 作为 reference visual prompt。
- SEEM logits 经过阈值化生成 binary mask。
- 输出 raw per-sample json 与统计文件。

### PSALM

新增脚本：

`/ssd1/zhangzheng/v2samo/baselines/PSALM/psalm/eval/eval_psalm_pair_exo2exo.py`

主要适配：

- 读取 ObjectRelator/PSALM 兼容格式的 Exo2Exo json。
- 使用 `psalm_video` / Mask2Former 配置。
- 如果模型输出多个 instance mask，选择模型分数最高的 instance，不用 GT 挑 best。
- 使用 `float32` 推理，避免部分环境下 `float16` 数值或算子不稳定。
- 输出 raw per-sample json 与统计文件。

注意：这里的 PSALM 权重是 official PSALM，不是 `YuqianFu/ObjectRelator-Original`。

### ObjectRelator

新增脚本：

`/ssd1/zhangzheng/v2samo/baselines/ObjectRelator/convert_v2sam_exo2exo_to_objectrelator_json.py`

`/ssd1/zhangzheng/v2samo/baselines/ObjectRelator/objectrelator/eval/eval_objectrelator_pair_exo2exo.py`

主要适配：

- 将 V2-SAM 风格的 Exo2Exo json 转成 ObjectRelator/PSALM/SEEM 更容易读取的格式。
- 读取 prompt image、target image、prompt mask 和 target GT。
- 如果模型输出多个 instance mask，选择模型分数最高的 instance，不用 GT oracle。
- 输出 raw per-sample json 与统计文件。

### V2-SAM Single-Expert/Fusion

本地适配：

- 将旧版仓库中的 `projects/v2sam_fusion` 按前面讨论改成 `projects/v2sam`，方便配置文件和 import 与现有测试脚本对齐。
- 使用旧版 V2-SAM 的 fusion single-expert checkpoint：`fusion_exo2ego_full.pth`。
- 使用本地保存 mask/逐样本结果的 metric 与 `tools/test.py` 运行。

没有改 fusion 模型的主体 forward 逻辑。

### V2-SAM-O PCCS Multi-Experts

实际使用：

- 配置：`/ssd1/zhangzheng/v2samo/code/V2-SAM-O/projects/v2sam/configs/v2sam.py`
- evaluator：`SegMetric_CYCLE_TripleDecoder`
- 测试入口：`/ssd1/zhangzheng/v2samo/code/V2-SAM-O/tools/test_with_decoder_save_ddp.py`

主要适配：

- 使用 PCCS/cycle-consistency 进行三专家选择。
- 没有使用 `SegMetric_TripleDecoder_SaveMask` 这类按照 GT IoU 选 best decoder 的 oracle metric。
- Expert 1/2 加载 VP checkpoint 时，过滤了一个 shape 不匹配的 key：`constr_prompt_fcs.0.weight`。其余权重正常加载。
- DDP 推理脚本支持保存 decoder info。后续 cross-view DROID 任务额外加了绝对 `collect_dir`，用于避免 MMEngine DDP 最后汇总 `.dist_test/tmp...` 时 rank 间路径不共享的问题。

V2-SAM-O 还依赖：

- SAM2: `/ssd1/zhangzheng/v2samo/code/V2-SAM-O/weights/sam2/sam2_hiera_large.pt`
- DINOv3: `/ssd1/zhangzheng/v2samo/code/V2-SAM-O/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`

## 5. 实际运行命令模板

下面命令中的路径是本次 Exo2Exo 表格实际使用的路径。GPU 编号可按服务器空闲情况调整。

通用变量：

```bash
DATA_ROOT=/ssd1/zhangzheng/v2samo/datasets/Ego-Exo4D-Seg/work/yuqian_fu/Ego/data_segswap_test
V2SAM_JSON=/ssd1/zhangzheng/v2samo/datasets/Ego-Exo4D-Seg/json/human_select_processed/exo2exo_selected_sam2_prompt_pair_same_random.json
OR_JSON=/ssd1/zhangzheng/v2samo/datasets/Ego-Exo4D-Seg/json/human_select_processed/exo2exo_selected_sam2_prompt_pair_same_random_objectrelator_v2.json
```

XSegTx：

```bash
cd /ssd1/zhangzheng/v2samo/baselines/ego-exo4d-relation/correspondence/SegSwap
CUDA_VISIBLE_DEVICES=0 python eval_xseg_pair_exo2exo.py \
  --expression-json "$V2SAM_JSON" \
  --data-root "$DATA_ROOT" \
  --ckpt-path /ssd1/zhangzheng/v2samo/baselines/weights/xseg/correspondence/SegSwap/model/model.pth \
  --segswap-root /ssd1/zhangzheng/v2samo/baselines/ego-exo4d-relation/correspondence/SegSwap \
  --output work_dirs/xseg_pair_same_random_1094/raw_xseg_pair_same_random_1094.json
```

SEEM：

```bash
cd /ssd1/zhangzheng/v2samo/baselines/SEEM
CUDA_VISIBLE_DEVICES=0 python eval_seem_pair_exo2exo.py \
  --image-folder "$DATA_ROOT" \
  --json-path "$OR_JSON" \
  --checkpoint /ssd1/zhangzheng/v2samo/baselines/weights/seem/seem_focall_v0.pt \
  --config configs/seem/focall_unicl_lang_demo.yaml \
  --output work_dirs/seem_focall_v0_exo2exo_pair_1093_fp32/raw_seem_focall_v0_exo2exo_pair_1093_fp32.json
```

PSALM：

```bash
cd /ssd1/zhangzheng/v2samo/baselines/PSALM
CUDA_VISIBLE_DEVICES=0 python psalm/eval/eval_psalm_pair_exo2exo.py \
  --image-folder "$DATA_ROOT" \
  --model-path /ssd1/zhangzheng/v2samo/baselines/weights/psalm_official \
  --json-path "$OR_JSON" \
  --output work_dirs/psalm_official_exo2exo_pair_1093/raw_psalm_official_exo2exo_pair_1093.json \
  --model-dtype float32 \
  --batch-size 1 \
  --num-workers 0 \
  --topk 1
```

ObjectRelator：

```bash
cd /ssd1/zhangzheng/v2samo/baselines/ObjectRelator
CUDA_VISIBLE_DEVICES=0 python objectrelator/eval/eval_objectrelator_pair_exo2exo.py \
  --image-folder "$DATA_ROOT" \
  --model-path /ssd1/zhangzheng/v2samo/baselines/weights/objectrelator_joint \
  --json-path "$OR_JSON" \
  --output work_dirs/objectrelator_joint_exo2exo_pair_1093_fp32/raw_objectrelator_joint_exo2exo_pair_1093_fp32.json \
  --model-dtype float32 \
  --batch-size 1 \
  --num-workers 0 \
  --topk 1
```

V2-SAM Single-Expert/Fusion：

```bash
cd /ssd1/zhangzheng/v2samo/code/V2-SAM
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/ssd1/zhangzheng/v2samo/code/V2-SAM \
torchrun --nproc_per_node=4 tools/test.py \
  projects/v2sam/configs/v2sam_joint_egoexo_exo2ego.py \
  --launcher pytorch \
  --checkpoint /ssd1/zhangzheng/v2samo/code/V2-SAM/weights/fusion_exo2ego_full.pth \
  --work-dir /ssd1/zhangzheng/v2samo/code/V2-SAM/work_dirs/fusion_exo2ego_full_on_exo2exo_pair_same_random_1094_4gpu \
  --cfg-options \
    test_dataloader.dataset.datasets.0.sam2_folder="$DATA_ROOT" \
    test_dataloader.dataset.datasets.0.expression_file="$V2SAM_JSON" \
    test_dataloader.num_workers=0
```

V2-SAM-O PCCS Multi-Experts：

```bash
cd /ssd1/zhangzheng/v2samo/code/V2-SAM-O
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/ssd1/zhangzheng/v2samo/code/V2-SAM-O \
torchrun --nproc_per_node=2 tools/test_with_decoder_save_ddp.py \
  projects/v2sam/configs/v2sam.py \
  --launcher pytorch \
  --work-dir /ssd1/zhangzheng/v2samo/code/V2-SAM-O/work_dirs/pccs_exo2exo_pair_1094_ddp2 \
  --save-decoder-info \
  --decoder-info-output /ssd1/zhangzheng/v2samo/code/V2-SAM-O/work_dirs/pccs_exo2exo_pair_1094_ddp2/decoder_info_pccs_1094_ddp2.json \
  --cfg-options \
    test_dataloader.dataset.datasets.0.sam2_folder="$DATA_ROOT" \
    test_dataloader.dataset.datasets.0.expression_file="$V2SAM_JSON" \
    test_dataloader.num_workers=0 \
    test_evaluator.type=SegMetric_CYCLE_TripleDecoder
```

## 6. 结果文件位置

| 方法 | 结果目录 |
|---|---|
| XSegTx | `/ssd1/zhangzheng/v2samo/baselines/ego-exo4d-relation/correspondence/SegSwap/work_dirs/xseg_pair_same_random_1094` |
| SEEM | `/ssd1/zhangzheng/v2samo/baselines/SEEM/work_dirs/seem_focall_v0_exo2exo_pair_1093_fp32` |
| PSALM | `/ssd1/zhangzheng/v2samo/baselines/PSALM/work_dirs/psalm_official_exo2exo_pair_1093` |
| ObjectRelator | `/ssd1/zhangzheng/v2samo/baselines/ObjectRelator/work_dirs/objectrelator_joint_exo2exo_pair_1093_fp32` |
| V2-SAM Single/Fusion | `/ssd1/zhangzheng/v2samo/code/V2-SAM/work_dirs/fusion_exo2ego_full_on_exo2exo_pair_same_random_1094_4gpu` |
| V2-SAM-O PCCS Multi | `/ssd1/zhangzheng/v2samo/code/V2-SAM-O/work_dirs/pccs_exo2exo_pair_1094_ddp2` |

## 7. 指标是否一致

核心指标是一致的：所有方法最终都用预测 binary mask 与同一个 target GT mask 计算 IoU 和 Dice，并对样本做平均。

但完整协议不是完全逐行一致，因为各方法生成 mask 的方式不同：

- XSegTx 使用 SegSwap 自带阈值 `MASKThresh`。
- SEEM 使用 reference visual prompt 后对 logits 阈值化。
- PSALM / ObjectRelator 可能输出多个 instance，选择模型分数最高的 instance，不使用 GT oracle。
- V2-SAM Single 直接评测 fusion mask。
- V2-SAM-O PCCS 先得到 VP / Sparse / Fusion 三个 mask，再通过 PCCS/cycle-consistency 选择最终 mask。

因此，论文中建议表述为：所有 baseline 使用各自标准推理与后处理流程生成最终 binary mask，然后在同一 Exo2Exo GT 上用统一 IoU/Dice 公式评测。

如果后续要进一步消除评测脚本差异，建议把所有模型的最终预测 mask 统一保存为 RLE，然后再用一个独立 `eval_saved_masks.py` 对六个方法的 RLE 重新计算 IoU/Dice。

## 8. 当前六模型 Exo2Exo 结果

所有数值均为百分比。`Zero IoU` 越低越好，其余列越高越好。

| Method | Training / Pretrain Data | N | IoU | Dice | Same-frame IoU | Random-mismatch IoU | Zero IoU | IoU >= 50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XSegTx | COCO / Self-Gen. Pairs | 1094 | 7.95 | 10.92 | 8.11 | 7.80 | 69.38 | 6.12 |
| SEEM | COCO Panoptic, RefCOCO/+g | 1093 | 13.50 | 16.68 | 14.74 | 12.26 | 68.34 | 12.90 |
| PSALM | COCO Panoptic, RefCOCO/+g, etc. | 1093 | 9.04 | 10.87 | 8.79 | 9.28 | 60.66 | 8.42 |
| ObjectRelator | Ego-Exo4D | 1093 | 27.54 | 31.19 | 29.70 | 25.37 | 46.39 | 28.45 |
| V2-SAM Single-Expert/Fusion | Ego-Exo4D | 1094 | 34.93 | 39.30 | 35.87 | 34.00 | 43.69 | 37.57 |
| V2-SAM-O PCCS Multi-Experts | Ego-Exo4D | 1094 | 37.32 | 41.37 | 38.71 | 35.92 | 39.40 | 39.31 |
