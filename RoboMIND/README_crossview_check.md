# RoboMIND 跨视角实测记录

检查日期: 2026-07-09

## 简短结论

RoboMIND 对同一条 episode / 同一次物体交互，**确实存在多相机视角**，但只存在于特定子集。

- 已用真实下载样本确认: `pick_apple_into_drawer_h5` 包含两条 Franka episode，每个 `trajectory.hdf5` 的 `observations/rgb_images` 下都有 `camera_left`、`camera_top`、`camera_right`，三路均为 600 帧。
- 官方格式文档也确认 `h5_agilex_3rgb` 有 `camera_front`、`camera_left_wrist`、`camera_right_wrist`。
- 单 RGB 子集不提供跨视角 pair，例如 `h5_franka_1rgb`、`h5_ur_1rgb`、`h5_tienkung_gello_1rgb`、`h5_tienkung_xsens_1rgb`。

重要细节: RoboMIND 不提供物体 instance ID、segmentation mask 或 bounding box。对同一个 HDF5 轨迹而言，多路相机是同一物理场景/任务/物体交互的不同视角；但跨 episode 或跨 robot embodiment 时，只能默认是同任务/同类别物体，不能默认是同一个物体实例，除非人工做视觉核验。

## 本目录文件

- `pick_apple_into_drawer_h5_2.zip`: ModelScope 官方示例压缩包，约 180 MB。
- `pick_apple_into_drawer_h5/`: 解压后的样本，包含两条 HDF5 episode。
- `samples/`: 生成的 left/top/right 三视角 contact sheet，抽取帧为 0、300、599。
- `crossview_manifest.json`: 已检查 HDF5 文件的机器可读摘要。
- `robomind.yaml`、`all_robot_h5_info.md`、`all_robot_h5_info_v1.2.md`: 从 ModelScope 下载的 RoboMIND 官方静态文档副本。
- `static/`: 从 ModelScope/Hugging Face 的官方 `static/` 路径补齐的静态资料，包括 `robomind.yaml`、`all_robot_h5_info*.md`、`franka_3rgb_2cam_paths.md`、`RoboMIND_v1_2_instr.csv`、`images/`、`language_description_annotation_json/`。
- `inspect_robomind_crossview.py`: 可复用的本地检查脚本。

## 下载样本证据

样本下载地址:

`https://www.modelscope.cn/datasets/X-Humanoid/RoboMIND/resolve/master/example_data/pick_apple_into_drawer_h5%202.zip`

观察到的 HDF5 结构:

| Episode | HDF5 路径 | RGB 相机 | 帧数 |
|---|---|---:|---:|
| `0923_164719` | `pick_apple_into_drawer_h5/success_episodes/0923_164719/data/trajectory.hdf5` | `camera_left`, `camera_top`, `camera_right` | 600 |
| `0923_164743` | `pick_apple_into_drawer_h5/success_episodes/0923_164743/data/trajectory.hdf5` | `camera_left`, `camera_top`, `camera_right` | 600 |

可视化样本:

- `samples/0923_164719_frame_0300_cameras.jpg`
- `samples/0923_164743_frame_0300_cameras.jpg`

这两张图显示了同一抽屉/苹果交互在左侧、顶部、右侧三个视角下的画面。

## 官方子集证据

来自 `robomind.yaml`:

| 子集 | 官方相机 | 跨视角状态 |
|---|---|---|
| `h5_franka_3rgb` | `camera_top`, `camera_left`, `camera_right` | 可以做跨视角，属于外部视角-外部视角。个别轨迹例外，只有 left/right 两路。 |
| `h5_agilex_3rgb` | `camera_front`, `camera_left_wrist`, `camera_right_wrist` | 可以做跨视角，包含一个前置外部视角和两个腕部视角。它是 RoboMIND 中最适合 wrist-exo pair 的子集。 |
| `h5_franka_1rgb` | `camera_top` | 不支持跨视角。 |
| `h5_ur_1rgb` | `camera_top` | 不支持跨视角。 |
| `h5_tienkung_gello_1rgb` | `camera_top` | 不支持跨视角。 |
| `h5_tienkung_xsens_1rgb` | `camera_top` | 不支持跨视角。 |

来自 `all_robot_h5_info.md` 和 `all_robot_h5_info_v1.2.md`:

- `h5_agilex_3rgb/10_packplate/.../trajectory.hdf5`: `camera_front`、`camera_left_wrist`、`camera_right_wrist`，每路 shape 为 `(762,)`。
- `benchmark1_1_release/h5_agilex_3rgb/20_takecorn_2/.../trajectory.hdf5`: 同样包含三路相机，每路 shape 为 `(539,)`。
- `h5_franka_3rgb/241021_close_trash_bin_1/.../trajectory.hdf5`: `camera_left`、`camera_right`、`camera_top`，每路 shape 为 `(122,)`。
- `benchmark1_1_release/h5_franka_3rgb/apples_placed_on_a_ceramic_plate/.../trajectory.hdf5`: 同样包含三路相机，每路 shape 为 `(98,)`。
- `h5_franka_3rgb/2024_09_20_close_cabinet/.../trajectory.hdf5`: 个别轨迹例外，只包含 `camera_left` 和 `camera_right`，每路 shape 为 `(200,)`。

## 数据集异常与注意事项

这套数据的公开仓库和论文版本有几处容易误读的地方，使用前需要按“实测优先”处理。

1. `static/` 不是空目录，但本地之前只有一个占位式 `static/all_robot_h5_info.md`。官方有效路径是 `static/all_robot_h5_info.md`、`static/all_robot_h5_info_v1.2.md`、`static/robomind.yaml` 等；根路径下直接请求 `robomind.yaml` 或 `all_robot_h5_info.md` 会 404。现在本地 `static/` 已补齐官方公开的主要静态资料，包括图片与语言标注 JSON。
2. Hugging Face repo 是 gated，Dataset Viewer API 返回 401；未登录授权前，只能检查 repo tree、README 和公开示例 zip，不能把全量样本统计当作已验证事实。
3. 版本口径混杂: 当前 arXiv v3 / README 主文使用 107k trajectories、479 tasks、96 object classes；README 末尾仍保留 Version 1.0 的 55K trajectories、279 tasks、69 object classes。旧 arXiv v1 里又是 55k / 279 / 61 的口径。报告里引用规模时必须注明版本。
4. `franka_3rgb_2cam_paths.md` 明确列出 675 条 Franka 3RGB 只有 left/right 两路的轨迹；因此筛选 `h5_franka_3rgb` 时不能默认三路相机齐全。
5. 当前公开 `static/` 中没有可用的 `RoboMIND_intrinsics.md`，该文件在 ModelScope 路径下返回 404，HF tree 也显示相关文件已被删除过。跨视角几何核验不能默认有完整内参/外参。
6. RoboMIND 仍然没有 object instance id、mask、box。它适合做多视角/多 embodiment 数据候选池，但如果要做 object-level cross-view segmentation benchmark，必须另建 mask 与身份关联。

## 复现方式

运行:

```powershell
python D:\Code\Work\EgoExoSeg\RoboMIND\inspect_robomind_crossview.py --root D:\Code\Work\EgoExoSeg\RoboMIND
```

脚本会把可视化 contact sheet 写入 `samples/`，并把汇总结果写入 `crossview_manifest.json`。
