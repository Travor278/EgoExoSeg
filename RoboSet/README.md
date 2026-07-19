# RoboSet 跨视角实测记录

检查日期: 2026-07-10

## 简短结论

RoboSet 对同一次物体交互，**确实存在多相机视角**，而且视角布局是本次所有候选里最规整的。

- 每帧 **4 路同步 RealSense D455**:3 个固定外部相机(`top` / `left` / `right`)+ 1 个腕部相机(装在末端执行器上方)。
- wrist↔exo 与 exo↔exo 两种配对都能组:结构比 DROID 多一路固定外部相机。
- 已用真实下载样本确认:见 `samples/` 下的四视角 montage,同一套厨房桌面场景(烤面包机、砧板、蓝色盘子上的水果)在 top/left/right/wrist 四路里被同一时刻拍到。

## 本目录文件

- `raw/baking_slide_in_bowl_scene_2_20230329-102845.h5`:从 HF 镜像下载的真实样本(约 431 MB,一个 h5 里打包了 10 条 trajectory)。
- `samples/`: 生成的四视角 montage(`Trial1/11/13_crossview.png`),每张按 trajectory 内 start/mid/end 抽 4 帧。
- `crossview_manifest.json`: 机器可读摘要。
- `inspect_roboset.py`: 可复用检查脚本。

## 下载样本证据

样本来源(公开、免 gate,MIT):

`https://huggingface.co/datasets/jdvakil/RoboSet-Teleoperation`

实测到的 HDF5 结构(每个 `TrialN/data/` 下):

| 数据键 | 含义 | shape |
|---|---|---|
| `rgb_top` / `rgb_left` / `rgb_right` | 3 个固定外部相机 RGB | (T, 240, 424, 3) |
| `rgb_wrist` | 腕部相机 RGB | (T, 240, 424, 3) |
| `d_top` / `d_left` / `d_right` / `d_wrist` | 对应 4 路深度 | (T, 240, 424) |
| `ctrl_arm` / `qp_arm` / ... | 动作 / 本体感知 | (T, 7) 等 |

- 示例 trajectory 长度约 42 帧(5Hz)。
- 该 HF 镜像分辨率为 **240×424**(已降采样),原始 RoboSet 更高。

可视化样本:

- `samples/Trial1_crossview.png`:同一厨房场景在 top / left / right / wrist 四个视角下的画面。

## 相机配置(已核验,RoboAgent 论文原文)

> “three fixed cameras (top, left, right), and a wrist camera mounted above the end-effector. **The four Realsense D455 camera views provide complementary perspectives of the workspace.**”

| 相机 | 类型 | 说明 |
|---|---|---|
| `top` / `left` / `right` | 外部(exo) | 3 个固定 RealSense D455 |
| `wrist` | 腕部(wrist) | 装在末端执行器上方的 D455 |

## 规模与获取(已核验)

| 项 | 值 |
|---|---|
| 规模 | 官网口径 **28,500** 条(9,500 遥操作 via Oculus Quest 2 + 19,000 kinesthetic 回放);论文 full 版 **98,050** 条 |
| embodiment | 单一 Franka Emika Panda(4 个物理 setup,不是 4 个机器人) |
| 采集频率 | **5Hz**(时序密度低,对“视频”benchmark 不友好) |
| 活动 | 官网 gallery 列 **7 个**厨房活动:`baking_prep`、`clean_kitchen`、`heat_soup`、`make_tea`、`make_toast`、`serve_soup`、`stow_bowl`(与 HF 目录结构一致) |
| License | MIT;官网 `dl.fbaipublicfiles.com/RoboSet/...` 直链 tar.gz;HF 镜像 425 GB |

## 数据集异常与注意事项(本次核验修订)

1. **无 mask**:论文里 SegmentAnything 只用于**训练时离线语义增广**(MT-ACT),**不是**发布的标注。要做 object-level cross-view segmentation 必须另建 mask。
2. **活动数 7 vs 6 的口径差异**:官网 gallery 是 7 个活动(含 `make_toast`),但论文 Table 2 只列 6 个(省了 `make_toast`,把 `clean_kitchen` 记为 `cleaning_up`)。“7 个活动”以官网/HF 目录为准。
3. **规模口径内部不一致**:landing page 写 28,500,teleoperation 子页写 30,050。引用时注明来源。
4. **5Hz + HDF5 自解析**:每个 h5 约 450 MB、需自己解析提取各路视频,原始 HF 镜像连 README/viewer 都基本是空的。性价比低于 DROID / RH20T。

## 复现方式

```powershell
python D:\Code\Work\EgoExoSeg\RoboSet\inspect_roboset.py --trials 3 --frames 4
```

脚本读取 `raw/` 下的 h5,对前几条 trajectory 抽帧,写出 `samples/` 下的四视角 montage 和 `crossview_manifest.json`。

## 适配结论

wrist↔exo + exo↔exo 结构干净,是很好的补充;但 5Hz 低帧率 + 需自解析 HDF5 + 无 mask,建议在 benchmark 里降为 **backup**(见主报告 §2.2)。

## 官方来源

- 官网: https://robopen.github.io/roboset/
- 论文(RoboAgent, ICRA 2024): https://arxiv.org/abs/2309.01918
- 本次样本镜像: https://huggingface.co/datasets/jdvakil/RoboSet-Teleoperation
