# V2-SAM-O-wrz Triple-Decoder Quickstart


注意：新版 `TripleDecoder` 主要用于推理阶段组合三个专家，不是重新训练一个总模型。Expert2/Expert3 的权重仍然来自旧版单专家训练结果。

## 1. 新版和旧版的分工

新版代码目录：

```bash
NEW_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-O-wrz
```

旧版资源目录：

```bash
OLD_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-main
```

分工：

- `V2-SAM-main`：旧版单专家训练，主要产出 `v2sam_visual/vision` 和 `v2sam_fusion` 权重。
- `V2-SAM-O-wrz`：新版三专家推理，加载 Expert1/Expert2/Expert3，并用 evaluator/PCCS 选择最终结果。

不要把旧版 `tools/test.py` 和新版 `projects/v2sam/configs/v2sam.py` 混用。新版测试就用新版目录里的 `tools/test.py`。

## 2. 三个专家和权重对应关系

```text
Expert1 / Sparse / Anchor
= DINOv3 + original SAM2
= 不训练，只加载原始 SAM2 和 DINOv3 权重

Expert2 / Visual / VP
= 旧版 v2sam_visual / v2sam_vision
= 在新版里由 Expert12 wrapper 加载

Expert3 / Fusion
= 旧版 v2sam_fusion
= 在新版里由 Expert3 wrapper 加载

PCCS / cycle consistency
= 推理阶段的专家选择策略
= 不需要训练
```

所以：

- 缺 Expert1：通常不是缺训练权重，而是 SAM2/DINOv3 原始权重路径没配好。
- 缺 Expert2：需要旧版 `v2sam_visual` 训练出的 ckpt。
- 缺 Expert3：需要旧版 `v2sam_fusion` 训练出的 ckpt。
- `TripleDecoder` 本身不需要一个额外的总 ckpt。

## 3. 关键硬编码路径

当前新版代码里有一些写死的路径，最省事的方式是创建软链接。

代码里会找：

```bash
/gemini/code/V2-SAM
/gemini/code/V2-SAM/third_parts/sam2/sam2_configs/sam2_hiera_large.pt
/gemini/space/DINOv3_ckp/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
/gemini/space/refsam_weights/pretrained/refsam_weights/epoch_12_exo2ego.pth
/gemini/code/ckp/fusion_exo2ego_full.pth
```

其中：

- `epoch_12_exo2ego.pth` 是 Expert2/Visual 权重的硬编码入口。
- `fusion_exo2ego_full.pth` 是 Expert3/Fusion 权重的硬编码入口。
- 即使跑 `ego2exo`，如果不改源码，也仍然把 `ego2exo` 对应权重链接到这两个硬编码文件名上。

## 4. 准备基础软链接

每换一台机器都要重新检查这些软链接。

```bash
NEW_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-O-wrz
OLD_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-main

cd $NEW_ROOT

mkdir -p /gemini/code
ln -sfn $OLD_ROOT /gemini/code/V2-SAM

mkdir -p /gemini/code/V2-SAM/third_parts/sam2/sam2_configs
ln -sfn $OLD_ROOT/weights/sam2/sam2_hiera_large.pt \
  /gemini/code/V2-SAM/third_parts/sam2/sam2_configs/sam2_hiera_large.pt

mkdir -p /gemini/space/DINOv3_ckp
ln -sfn $OLD_ROOT/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  /gemini/space/DINOv3_ckp/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

mkdir -p /gemini/space/refsam_weights/pretrained/refsam_weights
mkdir -p /gemini/code/ckp
```

检查：

```bash
readlink -f /gemini/code/V2-SAM
ls -lh /gemini/code/V2-SAM/third_parts/sam2/sam2_configs/sam2_hiera_large.pt
ls -lh /gemini/space/DINOv3_ckp/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

## 5. 选择 Exo2Ego 权重

Exo2Ego 测试时，使用：

```bash
OLD_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-main

VISION_CKPT=$OLD_ROOT/work_dirs/v2samvision_exo2ego_full_4gpu_bs16_acc2_val4/epoch_12.pth
FUSION_CKPT=$OLD_ROOT/work_dirs/v2fusion_exo2ego_full_4gpu_bs16/epoch_24.pth

ln -sfn $VISION_CKPT /gemini/space/refsam_weights/pretrained/refsam_weights/epoch_12_exo2ego.pth
ln -sfn $FUSION_CKPT /gemini/code/ckp/fusion_exo2ego_full.pth

readlink -f /gemini/space/refsam_weights/pretrained/refsam_weights/epoch_12_exo2ego.pth
readlink -f /gemini/code/ckp/fusion_exo2ego_full.pth
```

测试 JSON：

```bash
TEST_JSON=$OLD_ROOT/datasets/EgoExo4D_Relation_Test/exo2ego_test.json
```

## 6. 选择 Ego2Exo 权重

Ego2Exo 测试时，使用：

```bash
OLD_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-main

VISION_CKPT=$OLD_ROOT/work_dirs/v2samvision_ego2exo_full_4gpu_bs16_acc2_val4/epoch_12.pth

# 二选一：
# 1. 用自己训练的 Fusion 权重
FUSION_CKPT=$OLD_ROOT/work_dirs/v2fusion_ego2exo_full_4gpu_bs16_acc2_val4/epoch_24.pth

# 2. 或者用官方/学长给的 Fusion 权重
# FUSION_CKPT=$OLD_ROOT/weights/official_v2sam/weights/fusion_ego2exo_full.pth

ln -sfn $VISION_CKPT /gemini/space/refsam_weights/pretrained/refsam_weights/epoch_12_exo2ego.pth
ln -sfn $FUSION_CKPT /gemini/code/ckp/fusion_exo2ego_full.pth

readlink -f /gemini/space/refsam_weights/pretrained/refsam_weights/epoch_12_exo2ego.pth
readlink -f /gemini/code/ckp/fusion_exo2ego_full.pth
```

测试 JSON：

```bash
TEST_JSON=$OLD_ROOT/datasets/EgoExo4D_Relation_Test/ego2exo_test.json
```

## 7. 检查 config 是否是 TripleDecoder

```bash
cd /inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-O-wrz

PYTHONPATH=.:../V2-SAM-main/mmengine:$PYTHONPATH \
python - <<'PY'
from mmengine.config import Config
cfg = Config.fromfile("projects/v2sam/configs/v2sam.py")
print("model:", cfg.model["type"])
print("evaluator:", cfg.val_evaluator)
PY
```

期望看到模型是 `V2SAM_TripleDecoder`。

当前 config 里 evaluator 可能默认是：

```python
val_evaluator = dict(type='SegMetric_TripleDecoder_SaveMask', ...)
```

正式跑 PCCS/循环一致性时，命令行覆盖成：

```bash
val_evaluator.type=SegMetric_CYCLE_TripleDecoder
test_evaluator.type=SegMetric_CYCLE_TripleDecoder
```

说明：

- `SegMetric_CYCLE_TripleDecoder`：用循环一致性/PCCS 选择专家，正式测试优先用这个。
- `SegMetric_TripleDecoder_SaveMask`：按 GT IoU 选最优专家，并保存 mask/decoder 信息，更像分析上界，不是纯推理 PCCS 指标。

## 8. 权重加载 smoke test

先不跑数据，只确认 Expert12 和 Expert3 能加载。

```bash
cd /inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-O-wrz

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=.:../V2-SAM-main/mmengine:$PYTHONPATH \
python - <<'PY'
from projects.v2sam.models import (
    V2SAM_Expert_12_DualDecoder,
    V2SAM_Expert_3_NewMatcher,
    SAM2TrainRunner,
    SAM2Expert3,
)

print("loading Expert12...")
e12 = V2SAM_Expert_12_DualDecoder(
    grounding_encoder=dict(type=SAM2TrainRunner)
)
print("Expert12 OK")

print("loading Expert3...")
e3 = V2SAM_Expert_3_NewMatcher(
    grounding_encoder=dict(type=SAM2Expert3)
)
print("Expert3 OK")
PY
```

看到：

```text
Expert12 OK
Expert3 OK
```

说明基础权重路径基本通了。

如果看到 `Warning: No checkpoint specified`，不一定是错。TripleDecoder 的专家权重是在 `Expert_12_DualDecoder.py` 和 `Expert_3_NewMatcher.py` 里各自加载的，不靠 `tools/test.py --checkpoint`。

## 9. 检查测试 JSON

```bash
python - <<'PY'
import json
p="/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-main/datasets/EgoExo4D_Relation_Test/ego2exo_test.json"
data=json.load(open(p))
print(type(data), len(data))
if isinstance(data, dict):
    k=next(iter(data))
    print("first key:", k)
    print(data[k].keys())
else:
    print(data[0].keys())
PY
```

新版 dataset 已确认可以直接用这种 dict 格式：

```text
dict_keys(['video_id', 'video_path', 'prompt', 'objects'])
```

## 10. 单卡跑 PCCS 测试

以 `ego2exo` 为例：

```bash
cd /inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-O-wrz

OLD_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-main
CONFIG=projects/v2sam/configs/v2sam.py
TEST_JSON=$OLD_ROOT/datasets/EgoExo4D_Relation_Test/ego2exo_test.json
TEST_ROOT=$OLD_ROOT/datasets/EgoExo4D_Relation_Test/data_segswap_test/work/yuqian_fu/Ego/data_segswap_test

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=.:../V2-SAM-main/mmengine:$PYTHONPATH \
python tools/test.py $CONFIG \
  --work-dir work_dirs/triple_ego2exo_cycle \
  --cfg-options \
  val_dataloader.dataset.datasets.0.sam2_folder=$TEST_ROOT \
  val_dataloader.dataset.datasets.0.expression_file=$TEST_JSON \
  test_dataloader.dataset.datasets.0.sam2_folder=$TEST_ROOT \
  test_dataloader.dataset.datasets.0.expression_file=$TEST_JSON \
  val_evaluator.type=SegMetric_CYCLE_TripleDecoder \
  test_evaluator.type=SegMetric_CYCLE_TripleDecoder
```

以 `exo2ego` 为例，只改：

```bash
TEST_JSON=$OLD_ROOT/datasets/EgoExo4D_Relation_Test/exo2ego_test.json
```

以及提前把 Expert2/Expert3 的软链接切到 exo2ego 权重。

## 11. 四卡跑 PCCS 测试

不要用 `tools/dist.sh` 追加 `--cfg-options`，这个脚本不会转发额外参数。直接用 `torchrun`：

```bash
cd /inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-O-wrz

OLD_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-main
CONFIG=projects/v2sam/configs/v2sam.py
TEST_JSON=$OLD_ROOT/datasets/EgoExo4D_Relation_Test/ego2exo_test.json
TEST_ROOT=$OLD_ROOT/datasets/EgoExo4D_Relation_Test/data_segswap_test/work/yuqian_fu/Ego/data_segswap_test

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=.:../V2-SAM-main/mmengine:$PYTHONPATH \
torchrun --nproc_per_node=4 --master_port=29541 \
  tools/test.py $CONFIG \
  --launcher pytorch \
  --work-dir work_dirs/triple_ego2exo_cycle_4gpu \
  --cfg-options \
  val_dataloader.dataset.datasets.0.sam2_folder=$TEST_ROOT \
  val_dataloader.dataset.datasets.0.expression_file=$TEST_JSON \
  test_dataloader.dataset.datasets.0.sam2_folder=$TEST_ROOT \
  test_dataloader.dataset.datasets.0.expression_file=$TEST_JSON \
  val_evaluator.type=SegMetric_CYCLE_TripleDecoder \
  test_evaluator.type=SegMetric_CYCLE_TripleDecoder
```

## 12. 保存 decoder 选择信息

如果要保存每个样本/对象最后选了哪个专家，用：

```bash
cd /inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-O-wrz

OLD_ROOT=/inspire/qb-ilm/project/space-intelligence-multimodality/liuzhenyang-240108540154/zzz/V2-SAM-main
CONFIG=projects/v2sam/configs/v2sam.py
TEST_JSON=$OLD_ROOT/datasets/EgoExo4D_Relation_Test/ego2exo_test.json
TEST_ROOT=$OLD_ROOT/datasets/EgoExo4D_Relation_Test/data_segswap_test/work/yuqian_fu/Ego/data_segswap_test

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=.:../V2-SAM-main/mmengine:$PYTHONPATH \
python tools/test_with_decoder_save_ddp.py $CONFIG \
  --save-decoder-info \
  --annotation-json $TEST_JSON \
  --decoder-info-output work_dirs/triple_ego2exo_cycle/decoder_info.json \
  --work-dir work_dirs/triple_ego2exo_cycle \
  --cfg-options \
  val_dataloader.dataset.datasets.0.sam2_folder=$TEST_ROOT \
  val_dataloader.dataset.datasets.0.expression_file=$TEST_JSON \
  test_dataloader.dataset.datasets.0.sam2_folder=$TEST_ROOT \
  test_dataloader.dataset.datasets.0.expression_file=$TEST_JSON \
  val_evaluator.type=SegMetric_CYCLE_TripleDecoder \
  test_evaluator.type=SegMetric_CYCLE_TripleDecoder
```

## 13. 训练怎么理解

新版 `TripleDecoder` 不建议直接训练。真正要训练的是专家：

- Expert2/Visual：旧版 `v2sam_visual` 或 `v2sam_vision` 训练得到。
- Expert3/Fusion：旧版 `v2sam_fusion` 训练得到。
- Expert1/Sparse：不用训练，只依赖 SAM2 原始权重和 DINOv3。

训练好 Expert2/Expert3 后，把 ckpt 链到新版硬编码路径：

```bash
ln -sfn $VISION_CKPT /gemini/space/refsam_weights/pretrained/refsam_weights/epoch_12_exo2ego.pth
ln -sfn $FUSION_CKPT /gemini/code/ckp/fusion_exo2ego_full.pth
```

PCCS 是推理时的选择策略，不需要训练。

如果确实要在新版仓库里训练单专家，需要先把 `projects/v2sam/configs/v2sam.py` 的 `model` 从 `V2SAM_TripleDecoder` 切回单专家块，例如：

- Fusion/Expert3：`V2SAM_NEWMATCHER + SAM2Expert3`
- 旧版视觉/Expert2：通常还是建议走旧仓库的 `v2sam_visual`

不要用 `V2SAM_TripleDecoder` 开始训练。

## 14. 常见报错

`sam2_hiera_large.pt` 找不到：

```bash
ls -lh /gemini/code/V2-SAM/third_parts/sam2/sam2_configs/sam2_hiera_large.pt
```

`DINOv3_ckp` 找不到：

```bash
ls -lh /gemini/space/DINOv3_ckp/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Expert2 权重找不到：

```bash
ls -lh /gemini/space/refsam_weights/pretrained/refsam_weights/epoch_12_exo2ego.pth
readlink -f /gemini/space/refsam_weights/pretrained/refsam_weights/epoch_12_exo2ego.pth
```

Expert3 权重找不到：

```bash
ls -lh /gemini/code/ckp/fusion_exo2ego_full.pth
readlink -f /gemini/code/ckp/fusion_exo2ego_full.pth
```

`ModuleNotFoundError: sklearn`：

```bash
pip install scikit-learn
```

不要用：

```bash
pip install sklearn
```

`IndexError: cuda_visible_devices[local_rank]`：

GPU 数量和启动参数不一致。先单卡：

```bash
CUDA_VISIBLE_DEVICES=0
```

多卡时，`CUDA_VISIBLE_DEVICES` 数量要和 `torchrun --nproc_per_node` 一致。

指标比单专家低：

- 先确认用的是 `SegMetric_CYCLE_TripleDecoder`，不是 `SegMetric_TripleDecoder_SaveMask`。
- 再确认 Expert2/Expert3 软链接是否切到了同一个方向的权重。
- 最后和旧版单专家修复指标的 `perobject_IoU` 对齐比较，不要混用旧截断口径。

## 15. 最小执行顺序

```text
1. 进入 V2-SAM-O-wrz。
2. 建 /gemini 软链接。
3. 按方向链接 Vision ckpt 和 Fusion ckpt。
4. smoke test 加载 Expert12 / Expert3。
5. 单卡跑 SegMetric_CYCLE_TripleDecoder。
6. 单卡没问题再四卡。
7. 需要分析专家选择时，再跑 test_with_decoder_save_ddp.py 保存 decoder_info。
```
