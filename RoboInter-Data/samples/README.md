# RoboInter mask 可视化样例(2026-07-18)

回答 docx/0708 的三个问题:mask 标注是干嘛用的、为什么只标 exo↔exo、把 mask 可视化到物体上。
结论详见主报告 `Robotic_CrossView_Dataset_Survey_20260709.md` §0.7。

## 图片索引

| 文件 | 内容 |
|---|---|
| `{base}_crossview_mask.png` | 三列 = exterior_1(+mask) / exterior_2(+mask) / wrist(无 mask),四行 = 均匀采样时刻。同一被操作物体在两路外部相机都有人工复核 mask,腕部零标注 |
| `{base}_annotation_suite.png` | exterior_1 单列,叠加 RoboInter 完整逐帧"中间表征"套件:segmentation mask(绿) + object_box(黄) + gripper_box(青) + 10 步 gripper trace(品红) + contact points(红十字) + placement_proposal(白),左下角放大 inset |

已渲染 3 个 DROID base(均为 chunk-000 内两路外部视角都有 npz 的 pair):

| base | 任务 | LeRobot episode(ext1 / ext2) | 看点 |
|---|---|---|---|
| `10020` | take the black lid from the top of the can | 000002 / 000004 | 小目标(瓶盖在 180×320 下仅 ~35px);validation 集样例 |
| `10010` | move the bowl to the right | 000003 / 000001 | 同一只碗:ext1 远景小目标 / ext2 画面底缘大目标 / wrist 占满整帧 —— exo↔exo 间尺度差已很大 |
| `0` | pulling the fabric to the left | 000000 / 000031 | in-the-wild 卧室场景;大形变织物;机械臂遮挡下 mask 连贯 |

## 三个实测确认

1. **mask = 单个被操作目标物体**:npz 只有 `masks` 键,shape `(1, T, 1, H, W)` 二值,T 与 episode 帧数逐条对齐;物体不可见的帧 mask 全 0(现成 per-frame visibility 信号)。DROID npz 180×320,RH20T npz 360×640。
2. **腕部零 mask 是设计使然**:mask 是 13 类逐帧中间表征之一,服务 plan-then-execute VLA;论文明确所有中间表征 "based on primary observation"(外部相机像素系),wrist 只是策略的额外观测输入。→ wrist↔exo 的腕部真值必须自建。
3. **两路外部 mask 是同一物体身份**:三个样例均视觉确认 ext1/ext2 的 mask 指向同一被操作物体 → exo↔exo pair 可直接当 cross-view correspondence 真值。

## 复现

```powershell
# 1) 按需抽取 mask npz(免下 53GB 分卷 zip;首次会缓存 ~23MB 中央目录索引)
python RoboInter-Data\fetch_segmentation_npz.py 10020_exterior_image_1_left 10020_exterior_image_2_left

# 2) 视频:下载 chunk tar(~846MB/1000 eps;tar 内 wrist 段在前、primary 段在后)
#    https://huggingface.co/datasets/InternRobotics/RoboInter-Data/resolve/main/Annotation_with_action_lerobotv21/lerobot_droid_anno/videos/chunk-000.tar
#    -> 存为 RoboInter-Data\videos\lerobot_droid_chunk-000.tar(脚本会按需抽 mp4)

# 3) 渲染
python RoboInter-Data\visualize_robointer_masks.py 10020 10010 0
```

⚠️ license 注意:RoboInter-Data 数据卡声明全部数据/代码 **CC BY-NC-SA 4.0**(非"follow DROID/RH20T");再分发含其 mask 的评测集需带 NC 条款。
