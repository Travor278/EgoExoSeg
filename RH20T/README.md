# RH20T 跨视角实测记录

检查日期: 2026-07-10

## 简短结论

RH20T 对同一次物体交互，**确实存在大量多相机视角**——而且视角密度是本次所有候选里最高的。

- 官方配置(已核验论文原文):每采集平台 **8–10 个全局 RGBD 相机 + 1–2 个腕部(in-hand)相机**,全部标定到机器人基座系、时间同步。
- 已用真实下载样本确认:见 `samples/episode_000000_global8views.png`,**同一个 UR5 场景**(绿色台垫 + 黑色杯子 + 记号笔)在 **8 个全局相机**里被同一时刻拍到。
- 这使 RH20T 成为唯一能把“视角差”做成**可控变量**的数据集:一条轨迹可组合出 C(8,2)≈28 个 exo-exo pair,并可按视角夹角分桶做 geometric ambiguity 分析。

## ⚠️ 关于本目录使用的社区镜像(重要)

本目录样本来自社区镜像 [`hainh22/rh20t`](https://huggingface.co/datasets/hainh22/rh20t)(免 gate),它**内部不一致**,使用前必须知道:

- **8 个 serial 编号相机**(`cam_036422060909` 等,360×640)是**一条真实 RH20T 场景**的多个全局视角 —— 可直接用,montage 见 `episode_XXXXXX_global8views.png`。
- **3 个命名相机** `cam_front_view` / `cam_side_view` / `cam_eye_in_hand`(256×256)来自**另一个来源、另一段场景**(红色底座 + 绿色插销的 sim 风格画面),**不是同一条 take**。证据 montage 见 `episode_XXXXXX_mirror_mismatch.png`。
  - 硬证据:同一 episode 下 serial 相机是 353 帧,而命名相机是 157 帧,长度都对不上。
- 因此这个镜像给不了“同场景的 wrist↔exo 配对”。真正的 wrist(in-hand)视角要走**官方 RH20T 发布**(Google Drive / 百度云)才能拿到与全局相机对齐的腕部流。

## 本目录文件

- `raw/`: 从 `hainh22/rh20t` 下载的 episode 0 与 episode 1、全部 11 个相机文件夹的 mp4。
- `samples/`:
  - `episode_000000_global8views.png` / `episode_000001_global8views.png`:**主交付物**,8 个全局相机同场景多视角。
  - `episode_000000_mirror_mismatch.png` / `episode_000001_mirror_mismatch.png`:镜像不一致的取证图(3 个命名视角是另一场景)。
- `crossview_manifest.json`: 机器可读摘要,含 mirror caveat。
- `inspect_rh20t.py`: 可复用检查脚本。

## 相机配置(已核验,官方论文原文)

来自 arXiv 2307.00595 Sec. III:
> “Each platform contains a robot arm with force-torque sensor, gripper and **1–2 inhand cameras, 8–10 global cameras**, 2 microphones, a haptic device, a pedal ...”
> “**all the cameras are calibrated with respect to the robot's base frame**, and all the recorded data are **synchronized in the temporal domain**.”

| 项 | 值(核验后) |
|---|---|
| 全局相机 | 8–10 个 RGBD(每机位 RGB + Depth + 双目 IR) |
| 腕部相机 | 1–2 个 in-hand |
| 分辨率 | RGB / Depth / 双目 IR 均 **1280×720 @ 10Hz**;另有**两个**降采样版 **640×360** 与 **320×180** |
| 标定/同步 | 全部相机标定到机器人基座系,时间同步 |

## 规模与机器人配置(已核验)

| 项 | 值 |
|---|---|
| 规模 | 110k+ 机器人序列 + 等量人类演示 / 50M+ 帧(>40M robot + >10M human) |
| 任务 | 147 任务(48 RLBench + 29 MetaWorld + 70 自提),共 42 skills |
| 机器人配置 | **4 种机械臂(Flexiv Rizon、UR5、Franka、KUKA iiwa)× 4 种夹爪 × 3 种力矩传感器 = 7 种配置(Cfg1–Cfg7)** |

> 修订:之前把“7 种配置”写成“7 种机器人/robot”不准确 —— 实际是 **4 种臂**组合出 **7 种 Cfg**,不是 7 个不同机器人。

## License 与下载(已核验)

- **分段 license**:RH20T-C(scene 0001–0005)= CC BY-SA 4.0;RH20T-NC(scene 0006–0010)= CC BY-NC 4.0。做 benchmark 尽量落在 RH20T-C 以避开非商用条款。
- 官方下载:Google Drive + 百度云(按 7 个 config 分别提供),另有全分辨率与降采样两个版本。
- **无 mask**(模态为 RGB/depth/IR/力觉/音频/本体感知);RoboInter-Data 覆盖其 82,894 条(仅 primary+wrist 两路)。

## 复现方式

```powershell
python D:\Code\Work\EgoExoSeg\RH20T\inspect_rh20t.py --episode 0 --frames 3
```

脚本自动把 serial 全局相机(主 montage)与命名视角(取证 montage)分开渲染,并把 mirror caveat 写进 `crossview_manifest.json`。

## 官方来源

- 官网: https://rh20t.github.io/
- 论文(ICRA 2024): https://arxiv.org/abs/2307.00595
- 本次样本镜像(社区,注意不一致): https://huggingface.co/datasets/hainh22/rh20t
- mask 补齐: https://huggingface.co/datasets/InternRobotics/RoboInter-Data
