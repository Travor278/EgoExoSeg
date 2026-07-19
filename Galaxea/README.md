# Galaxea (Open-World) 跨视角实测记录

检查日期: 2026-07-13

## 简短结论

Galaxea R1 的采集台**确实是 wrist↔head 的多相机同步结构**(1 头部 external + 2 腕部 in-hand);但本次实测暴露一个对我们任务的**关键风险**:

- 在实测样本里,**head 相机是一个"看向整个房间"的宽视角**(能看到远处的桌子、显示器、木椅),被操作物体(橙色杯子)在里面**很小、很难定位**;
- 而 **wrist 相机是紧贴物体的近景**,杯子几乎占满画面(见 `samples/episode_000000_crossview.png` 的 frame 301);
- 两者**尺度差极大** → wrist↔head 的物体对应/分割**很难**,head 视角里可能根本切不出那个物体。

这正好坐实了 §9 里对 Galaxea 的头号 caveat:"物体在 wrist 与 head 同时可见"这一前提**不能默认成立**,是 go/no-go 级的问题。

## ⚠️ 关于本目录用的"社区替身"(重要)

**官方 Galaxea Open-World Dataset(720×1280,head + head_right + 双 wrist)在 HF 是 gated、ModelScope 需登录**,当前账号(Travor278)未接受条款 → 两条渠道都下不了(见下"获取官方数据")。

因此本目录用的是**同一台 Galaxea R1 机器人、同一套相机 rig 的社区免 gate 录制** [`s-tian/galaxea_r1_cup`](https://huggingface.co/datasets/s-tian/galaxea_r1_cup) 作替身:

| | 官方 Open-World | 本目录社区样本 |
|---|---|---|
| 相机 | head + head_right(外部)+ left/right_wrist | head(外部)+ left/right_wrist |
| 分辨率 | 720×1280 AV1 | **168×94(极小,训练用归一化 dump)** |
| gate | HF gated + ModelScope 登录 | **免 gate** |

替身**只用于验证 rig 结构与 co-visibility**,画质不代表官方;上面的"head 是宽视角、共视差"结论**需在官方 720×1280 数据上复核**(官方 head 帧构图可能更贴工作台,且 head_right 提供第二个外部视角)。

## 本目录文件

- `raw/{head,left_wrist,right_wrist}/episode_0000{00,09,18}.mp4`: 社区 R1 'cup' 样本,3 相机 × 3 episode(每个 <2MB)。
- `samples/episode_0000{00,09,18}_crossview.png`: head + 双 wrist 的跨视角 montage,每张抽 4 帧。
- `crossview_manifest.json`: 机器可读摘要(含官方 vs 替身对照 + caveat)。
- `inspect_galaxea.py`: 可复用脚本(用 crossview_viz 的 AV1-robust 解码,官方 AV1 也能解)。

## 相机配置(官方,已核验)

来自 arXiv 2509.00576 + HF 数据卡:R1-Lite 双臂移动机器人,每 episode **4 路 RGB**:
- `observation.images.head_rgb` + `observation.images.head_right_rgb` —— 头部**外部/exo**(两者疑为近立体对,作"两个不同 exo"较弱);
- `observation.images.left_wrist_rgb` + `observation.images.right_wrist_rgb` —— 双臂**腕部/in-hand**;
- 720×1280 AV1 @15fps,**严格时间同步(已核验)**;500+h / ~10万轨迹 / 2.87 TB;**无 object mask**(仅中英双语子任务语言)。

## 关键发现:wrist↔head 共视性(本次实测重点)

看 `samples/` 三张 montage(尤其 ep0 frame 301):
- **head 列**:整段都是宽视角房间(桌子/显示器/木椅),被操作物体在画面里小且易被杂物淹没;
- **wrist 列**:橙色杯子近景清晰、占画面主体。

含义:Galaxea R1 的 head 更像"机器人朝前的 egocentric 全景",不是俯拍工作台的固定第三人称 → **wrist↔head 是一个大尺度差、强视角差的困难 pair**,head 侧能否稳定切出目标物体存疑。要把 Galaxea 用作 wrist↔exo benchmark,**必须先在官方高清数据上逐帧核验共视率**(已列入主报告 §8 待办)。

## 获取官方数据(两条路,都需你一次授权)

1. **HF(推荐,最省事)**:登录 `Travor278` 访问 <https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset>,点 **"Agree and access repository"**(gate 是 auto,点一下即自动放行)。之后本机已存的 HF token 即可下载,我就能用官方 720×1280 数据重做本套(含 head_right)。
2. **ModelScope**:配置 ModelScope 登录 token(`modelscope login`),该镜像匿名下载被拒(报 "username cannot be empty")。

## 复现

```powershell
python D:\Code\Work\EgoExoSeg\Galaxea\inspect_galaxea.py --episodes 0 9 18 --frames 4
```

## 来源

- 官方论文: https://arxiv.org/abs/2509.00576 · 项目页: https://opengalaxea.github.io/G0/
- 官方数据(gated): https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset
- 本次替身样本(免 gate): https://huggingface.co/datasets/s-tian/galaxea_r1_cup
