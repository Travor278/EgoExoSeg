# 腕部视角手工标注测试(labelme)

目的:实测"人工直接标腕部视角被操作物体"的质量与耗时,对比 SAM2 自动结果。
你判断的前提成立:机器人场景里目标物体明确(指令写着),标注员无需模型辅助定位。

## 一次性准备(已完成的部分)

- 帧已抽好:`frames/{base}/f{帧号}.png`(6 条 DROID episode × 6 帧,放大 2 倍便于描边;`meta.json` 记录指令与帧号)
- labelme 已装(`python -m pip install labelme`)

## 标注流程(预计 20–40 分钟/36 帧)

```powershell
cd D:\Code\Work\EgoExoSeg\RoboInter-Data\manual_wrist_kit
labelme frames\10020 --labels target    # 一次开一个 episode 目录
# labelme 7.x:不嵌图与自动保存已是默认行为(旧版的 --nodata/--autosave 不再需要)
```

1. 每张图用 **Create Polygons** 沿被操作物体描边(指令见该目录 meta.json,如 "take the black lid…" 就只标黑瓶盖);
2. 标签名随意(统一用 `target` 即可,转换脚本把所有多边形都算作目标);
3. 物体不可见的帧**直接跳过不画**(等价"空 mask");
4. `--autosave` 会在同目录生成 `f****.json`;
5. 建议记录每条 episode 的用时,评估人均标注成本。

## 转换 + 出对比图

```powershell
python convert_labelme.py
```

产出:
- `manual_masks/{base}_wrist_manual.npz`(与 SAM2 npz 同构)
- `../samples/{base}_manual_vs_sam2.png`:三列对比 —— exterior GT(绿) | wrist SAM2 自动(橙) | wrist 手工(蓝)

## 评估要点

- 边界质量:手工 vs SAM2(尤其近景大形变/遮挡帧);
- 耗时:如 36 帧 ×N 秒 → 推算 hundred-level 测试集(如 300 集 × 6 帧)的人力;
- 结论走向:若手工质量/成本可接受 → wrist 真值走"稀疏关键帧手工 + SAM2 插帧传播 + 人工复核"混合管线(手工帧同时充当传播 prompt,一举两得)。
