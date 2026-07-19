# DROID 跨视角实测记录

检查日期: 2026-07-10

## 简短结论

DROID 对同一条 episode / 同一次物体交互，**确实存在多相机视角**，且是本次调研里最贴合 ego-exo 式任务的结构。

- 每条 episode 有 **3 路同步相机**：2 个可调外部 ZED 2(`exterior_image_1` / `exterior_image_2`)+ 1 个腕部 ZED Mini(`wrist_image`)。
- 这正好构成 “1 腕部 : 2 外部” = **1 ego : 2 exo**，与 Ego-Exo4D 的 take 结构(1 ego + 多 exo)同构，json 格式迁移最自然。
- 已用真实下载样本确认:见 `samples/` 下的三视角 montage,同一个 “NUTRI GRAIN” 能量棒 + 金属托盘在 `exterior_1` / `exterior_2` / `wrist` 三路里被同一时刻拍到。

## 本目录文件

- `raw/`: 从 `lerobot/droid_100` 下载的真实样本
  - `exterior_image_1_left/file-000.mp4`、`exterior_image_2_left/file-000.mp4`、`wrist_image_left/file-000.mp4`:三路相机的视频(100 条 episode 拼接在一条 mp4 里,共 32,212 帧)。
  - `data/file-000.parquet`:每帧的 `episode_index` 等元数据,用来切分 episode 边界。
  - `meta/`:`info.json`、`tasks.parquet` 等。
- `samples/`: 生成的三视角 montage(`episode_000/001/002_crossview.png`),每张按 episode 内 start/mid/end 抽 4 帧。
- `crossview_manifest.json`: 已检查内容的机器可读摘要。
- `inspect_droid.py`: 可复用的本地检查脚本。

## 下载样本证据

样本来源(公开、免 gate,约 2 GB):

`https://huggingface.co/datasets/lerobot/droid_100`

观察到的结构:

| 项 | 值 |
|---|---|
| 相机 | `exterior_image_1_left`、`exterior_image_2_left`、`wrist_image_left` |
| 每路分辨率(此镜像) | 180×320(立体相机的**左目**,已降采样) |
| 帧率 | 15 fps |
| episode 数 | 100(总 32,212 帧) |
| 示例 episode 长度 | ep0=166 帧、ep1=238 帧、ep2=142 帧 |

可视化样本:

- `samples/episode_001_crossview.png`:同一能量棒/托盘交互,在 2 个外部视角 + 1 个腕部视角下的画面。

> 关键点:`wrist` 路是抓取近景(能看清被操作物体),`exterior_1/2` 路是工作台全景(物体较小但可见)——这正是 wrist↔external 的跨视角对应任务需要的“同一物体、不同视角”。

## 相机配置(已核验,官方论文 / 官网)

| 相机 | 类型 | 说明 |
|---|---|---|
| `exterior_image_1` | 外部(exo) | 可调三脚架上的 ZED 2 立体相机 |
| `exterior_image_2` | 外部(exo) | 第二个可调 ZED 2 立体相机 |
| `wrist_image` | 腕部(wrist) | 装在末端执行器手腕上的 ZED Mini 立体相机 |

- 原始 DROID 为**立体 + 1280×720 @ 15Hz**,提供每个相机的内参 + 外参立体标定;共 **1,417 个相机视点**(官网称 “view points”)。
- 2025-04 官方在 HuggingFace 补发了 36k episodes 的**改进版标定**;2024-12 更新语言标注(95% 成功 episode 各 3 条,共 75k episodes)。

## 数据集异常与注意事项(本次核验修订)

1. **本镜像是降采样单目**:`lerobot/droid_100` 只暴露每个立体相机的**左目**,分辨率 180×320。要做 benchmark 交付质量,需回到原始 DROID 取立体 + 1280×720 高清帧。
2. **HuggingFace 不是“全量镜像”**:官网只在 HF 上放**补充的标注/标定文件**(`KarlP/droid`),轨迹数据本体在公开 GCS bucket `gs://gresearch/robotics`(`droid` RLDS 1.7 TB / `droid_100` 样本 2 GB / `droid_raw` 立体 8.7 TB、非立体 5.6 TB)。之前把 HF 称为“镜像”不准确。
3. **License 是 CC-BY 4.0**:官网页面本身**没写** license,但论文正文(arXiv 2403.12945)明确 “We open source the full DROID dataset under CC-BY 4.0 license”。之前“license 官网未标、需 repo 确认”应更新为 CC-BY 4.0。
4. **无 segmentation mask**:DROID 自带只有 RGB 立体 + 立体深度 + 语言标注,无 mask。可用 **RoboInter-Data**(SAM2 + 人工复核)补齐,其覆盖 DROID 152,986 episodes。
5. **DROID paper 的 arXiv id 是 2403.12945**(RSS 2024),之前报告未直接给出。

## 复现方式

```powershell
python D:\Code\Work\EgoExoSeg\DROID\inspect_droid.py --episodes 0 1 2 --frames 4
```

脚本会读取 `data/file-000.parquet` 切分 episode 边界,对每条 episode 抽帧,从三路 mp4 的**同一全局帧号**解码(三路已同步),写出 `samples/` 下的 montage 和 `crossview_manifest.json`。

## 官方来源

- 官网: https://droid-dataset.github.io/
- 论文(RSS 2024): https://arxiv.org/abs/2403.12945
- 本次样本镜像: https://huggingface.co/datasets/lerobot/droid_100
- 全量数据: `gs://gresearch/robotics`(RLDS/TFDS)
- mask 补齐: https://huggingface.co/datasets/InternRobotics/RoboInter-Data
