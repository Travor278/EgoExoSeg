# RoboInter mask 可视化样例(2026-07-18/19)

> **画廊**:`gallery.html`(本目录,离线可开)—— DROID 50 条(按指令去重,覆盖 ≥38 个官方 scene_id、12 个 lab)+ RH20T 50 takes(22 种指令,cfg1 单一平台)。
> 每卡:两路外部 GT mask(绿)+ wrist 列 SAM2 预测(橙,contact-frame 自动提示+全程传播,97/100 成功,非真值)。
> 生成:`build_gallery.py all` → `sam2_wrist_auto.py`(GPU)→ `enrich_scenes.py`(GCS 拉官方 metadata,含相机外参,存 `droid_episode_meta.json`)→ `build_gallery.py render --force && html`。
> 分发包:仓库根 `RoboInter_gallery_20260719.zip`(144MB)。

回答 docx/0708 的问题:mask 标注是干嘛用的、为什么只标 exo↔exo、可视化到物体上、wrist 侧能否用 SAM2 补。
结论详见主报告 `Robotic_CrossView_Dataset_Survey_20260709.md` §0.7 / §10。

## 图片索引

| 文件 | 内容 |
|---|---|
| `{base}_crossview_mask.png` | 三列 = 相机A(+GT mask) / 相机B(+GT mask) / **wrist(+SAM2 预测,橙色)**,四行 = 均匀采样时刻。绿色 = RoboInter 官方 GT(仅外部/全局相机);橙色 = 我们用 SAM2 少量点击自建的 wrist 预测(**非官方真值**,tag 注明模型/点击数/prompt 帧) |
| `{base}_annotation_suite.png` | (仅 DROID)exterior_1 单列,叠加 RoboInter 完整逐帧"中间表征"套件:mask(绿) + object_box(黄) + gripper_box(青) + 10 步 trace(品红) + contact points(红十字) + placement(白),左下角放大 inset |

已渲染 4 组:

| base | 来源 | 任务 | 看点 |
|---|---|---|---|
| `10020` | DROID(val) | take the black lid | 小目标(~35px);**wrist f0 物体出画 → SAM2-b+ 全程传播正确输出空 mask**(与 GT 的可见性语义一致) |
| `10010` | DROID | move the bowl | ext1 远小 / ext2 底缘大 / wrist 满帧;**SAM2 wrist mask 连搅拌棒缝隙形态都与 ext2 GT 一致** |
| `0` | DROID | pulling the fabric | in-the-wild 卧室;大形变织物;遮挡下 GT 连贯 |
| `task_0001_user_0001_scene_0004_cfg_0001` | RH20T | press the button | **GT 为 360×640**(比打包视频高清);同 take 本地即有 6-7 路已标注全局相机;wrist(in-hand)全程正对目标,SAM2 预测紧贴按钮装置 |

## 实测确认(累计)

1. **mask = 单个被操作目标物体**:npz 仅 `masks` 键,shape `(1, T, 1, H, W)` 二值,T=episode 帧数;物体不可见帧全 0(现成 visibility 信号)。DROID 180×320,RH20T 360×640。
2. **腕部零 mask 是设计使然**:13 类中间表征全部锚定 primary(外部)相机像素系,服务 plan-then-execute VLA;zip 中央目录 198,194 个成员无一 wrist(与 mapping 非空数互证)。
3. **两路外部 mask 同一物体身份**:DROID 3 组 + RH20T 1 组均视觉确认。
4. **SAM2 自建 wrist mask 预演**(`sam2_wrist_preview.py`):
   - GPU(py312, torch2.10+cu128, RTX 5070)+ hiera-b+ 全视频传播,几秒/条;CPU 退化为 tiny+4 帧稀疏模式。
   - 每条只需 1–4 次点击;**点击落点差 10px 就会选错物体**(10010 首次点中碗里的搅拌棒、RH20T 首次一个正点落在桌布上导致 mask 爆帧)—— 这正是"必须人工复核 + 跨视角身份对齐"的实证,失败 prompt 已在脚本注释中留档。

## 复现

```powershell
# 1) 按需抽取 GT mask npz(免下 53GB 分卷 zip;首次缓存 ~23MB 中央目录索引)
python RoboInter-Data\fetch_segmentation_npz.py 10020_exterior_image_1_left 10020_exterior_image_2_left

# 2) 视频:下载对应 chunk tar 到 RoboInter-Data\videos\
#    lerobot_droid_chunk-000.tar (846MB) / lerobot_rh20t_chunk-000.tar (584MB)
#    HF: InternRobotics/RoboInter-Data / Annotation_with_action_lerobotv21/lerobot_{droid,rh20t}_anno/videos/chunk-000.tar
#    (tar 内 wrist 段在前、primary 段在后,脚本按需抽 mp4)

# 3) SAM2 wrist 预测(可选;GPU 用 py312 环境)
& D:\Dev\conda-envs\py312\python.exe RoboInter-Data\sam2_wrist_preview.py

# 4) 渲染
python RoboInter-Data\visualize_robointer_masks.py 10020 10010 0
python RoboInter-Data\visualize_robointer_masks.py --rh20t auto
```

⚠️ license:RoboInter-Data 数据卡声明全部数据/代码 **CC BY-NC-SA 4.0**;再分发含其 mask 的评测集需带 NC 条款。SAM2 权重(`V2sam/weights/sam2/`)遵循 Meta SAM2 license(Apache 2.0)。
