# RoboInter-Data 注意点实测记录

日期: 2026-07-09

本目录用于复核 `Robotic_CrossView_Dataset_Survey_20260709.md` 中 RoboInter-Data 的两个必须实测的问题:

1. segmentation mask 是否覆盖 `primary + wrist` 两路视角。
2. 打包视频只有 180x320 时，是否还能按 episode/video ID 映射回 DROID / RH20T 原始高清视频。

## 本地样本

已下载/生成:

- `VideoID_2_SegmentationNPZ.json`: 官方 video id 到 segmentation npz 的映射。
- `val_video.json`: 官方 validation video id 划分。
- `All_Keys_of_Primary.json`: 官方 Primary 质量等级列表。注意这里的 `Primary` 是 annotation quality，不是 `observation.images.primary` 相机。
- `meta/lerobot_droid_anno/info.json`、`episodes.jsonl`
- `meta/lerobot_rh20t_anno/info.json`、`episodes.jsonl`
- `data/lerobot_droid_chunk-000.tar` 和解包后的 1000 个 DROID parquet。
- `data/lerobot_rh20t_chunk-000.tar` 和解包后的 1000 个 RH20T parquet。
- `caveat_check_summary.json`: 由 `inspect_robointer_caveats.py` 生成的完整统计。

复现命令:

```powershell
python RoboInter-Data\inspect_robointer_caveats.py
```

## 问题 1: mask 是否两路视角都标注

结论: **不能把 RoboInter-Data 直接当作 wrist 与 primary 双路 mask 真值使用。**

实测依据:

- 官方映射文件是 `VideoID_2_SegmentationNPZ.json`，语义是 **episode video ID 到 segmentation NPZ**，不是 episode pair 到双视角 masks。
- 全量 235,920 个 key 中，198,194 个有 npz，37,726 个为 `null`。
- key 命名里没有 `wrist`、`primary`、`hand`、`image_in_hand`，只有:
  - DROID: `exterior_image_1_left` 76,500 个，`exterior_image_2_left` 76,500 个。
  - RH20T: `RH20T_cfg...camera_serial` 82,920 个。
- DROID chunk-000 的 parquet 实测:
  - 1000 个 episode parquet。
  - `camera_view`: `exterior_image_1_left` 500 个，`exterior_image_2_left` 500 个。
  - `annotation.segmentation` / `Q_annotation.segmentation` 全为空串，mask 不内嵌在 parquet，而是外部 raw npz。
  - 835/1000 个 episode_name 在映射文件中有 npz。
- RH20T chunk-000 的 parquet 实测:
  - 1000 个 episode parquet。
  - `camera_view` 是相机 serial，如 `750612070853`、`035622060973`。
  - `annotation.segmentation` / `Q_annotation.segmentation` 全为空串。
  - 887/1000 个 episode_name 在映射文件中有 npz。

对 DROID 的额外发现:

- DROID 的两个外部视角可以做外部视角之间的双视角筛选，但这不是 wrist 与 exo 的配对。
- 全量 `exterior_image_1_left` / `exterior_image_2_left` 成对统计:
  - 完整外部双视角 pair: 76,500
  - 两路都有 npz: 56,566, 73.94%
  - 只有一路有 npz: 15,734, 20.57%
  - 两路都没有 npz: 4,200, 5.49%
- validation 中的 DROID 外部成对统计更好:
  - 完整外部双视角 pair: 1,860
  - 两路都有 npz: 1,678, 90.22%
  - 只有一路有 npz: 175, 9.41%
  - 两路都没有 npz: 7, 0.38%

判定:

- **wrist 与 primary 评测集**: RoboInter 现成 mask 至少不能证明两路都有；按当前元数据，应视为 **primary/selected video 一路 mask**。需要自建 wrist 侧真值，或下载 demo/raw segmentation 后做更深层可视化检查。
- **DROID 外部视角之间的评测集**: 可行，可以过滤出两路 external camera 都有 npz 的 pair。

## 问题 2: 180x320 与映射回原始高清视频

结论: **打包 LeRobot 视频确实是低清；episode/video ID 保留了回原始数据的线索，但高分辨率视频不在 RoboInter 的 LeRobot 包内，需要走 DROID/RH20T 原始数据接口。**

实测依据:

- `meta/*/info.json` 对 DROID 与 RH20T 都写明:
  - `observation.images.primary`: shape `[180, 320, 3]`, h264, 10 fps。
  - `observation.images.wrist`: shape `[180, 320, 3]`, h264, 10 fps。
- parquet 的 `annotation.origin_shape`:
  - DROID chunk-000: 1000/1000 为 `[320, 180]`。
  - RH20T chunk-000: 1000/1000 为 `[640, 360]`。
- video/annotation id 保留了原始来源结构:
  - DROID 示例: `10010_exterior_image_2_left` -> `/OXE_DROID/data/ann_human/0/sam_mask/10010_exterior_image_2_left.npz`
  - RH20T 示例: `RH20T_cfg1_task_0001_user_0001_scene_0001_cfg_0001_035622060973` -> `/RH20T/data/ann_human/0/sam_mask/...npz`
- 官方数据卡也说明 LeRobot 包是 parquet + MP4 videos，`videos/` 下分 `observation.images.primary/` 与 `observation.images.wrist/`；高分辨率视频需要通过 DROID hr_video_reader 和 RH20T API 获取。

判定:

- 低清问题成立: 不能把 RoboInter LeRobot 包直接作为最终高质量评测集画面。
- 映射回原始数据的可行性: **ID 足够作为映射起点**，但还需要单独实测 DROID/RH20T 原始视频读取器，把一个 `episode_name` 成功还原到原始帧，再验证 mask resize / SAM2 refine 的质量。

## 当前建议

- 若目标是 wrist 与 exo: 用 RoboInter 中有 npz 的 video 作为 query/source mask 候选，但 target wrist mask 需要自跑 SAM2/人工复核；不要默认官方有 wrist 侧真值。
- 若目标可接受外部视角之间的配对: DROID 可直接筛 `exterior_image_1_left` 与 `exterior_image_2_left` 两路都有 npz 的 56,566 个 pair，validation 里也有 1,678 个 pair 可起步。
- 下一步最小实测: 下载少量 `Annotation_raw/segmentation_npz` 或 demo subset，渲染 3-5 个 mask overlay；同时用 DROID/RH20T 官方 reader 对同一 ID 拉原始高分辨率帧，验证坐标缩放/refine 流程。

## 官方来源

- Hugging Face 数据卡: https://huggingface.co/datasets/InternRobotics/RoboInter-Data
- RoboInter 论文: https://arxiv.org/abs/2602.09973
- DROID: https://droid-dataset.github.io/
- RH20T: https://rh20t.github.io/
