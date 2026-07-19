# AIRoA MoMa 跨视角实测记录

检查日期: 2026-07-13

## 简短结论

AIRoA MoMa 是 **head(外部)+ hand(腕部)双路同步**结构,而且本次实测的**关键结论是正面的** —— 它通过了 Galaxea 没通过的那个共视性检验:

- **head 相机是"看向工作台/货架"的视角**(不是 Galaxea 那种朝前看整个房间的全景),被操作物体(货架上的饮料瓶)在 head 里**清晰可辨**;
- **hand 相机是鱼眼近景**,抓取时目标瓶子占满画面;
- → 同一个瓶子在 head 与 hand 里**确实同时可见**(见 `samples/task001/010_ep000000_crossview.png`)→ **wrist↔head 对应/分割是可行的**。

因此:**在"新的 wrist↔exo 可下载源"里,AIRoA 比 Galaxea 更值得跟进**(Galaxea 的 head 是宽视角房间图、物体尺度差太大、共视弱)。

## 关于本目录用的样本(比 Galaxea 那次更靠谱)

**官方 `airoa-org/airoa-moma` 在 HF 是 gated**(auto-gate,当前账号未接受条款 → 403)。但同一个 **官方 org 发布了一个免 gate 的 demo** [`airoa-org/corl2025_demo`](https://huggingface.co/datasets/airoa-org/corl2025_demo),而且:

| | 官方 MoMa | 本次 demo |
|---|---|---|
| 机器人/rig | Toyota HSR;hand + head | **同一 HSR;同一 hand + head** |
| 分辨率 | 640×480 | **640×480(全分辨率,未降采样)** |
| 帧率 | 30fps | 10fps |
| 来源 | 官方 | **同一官方 org** |
| gate | gated | **免 gate** |

→ 这个 demo 是**全分辨率、官方出品**,是 MoMa 很强的替身;甚至 demo 本身(34 个 task × ~5 episode)就可能够抽一个 hundred-level pilot,不一定非要官方全量。

## 本目录文件

- `raw/task{001,010}/{head,hand}/episode_0000{00,01}.mp4`: 官方 demo 样本,2 相机 × 2 task,全分辨率 AV1。
- `samples/task{001,010}_ep000000_crossview.png`: head + hand 跨视角 montage,每张抽 4 帧。
- `crossview_manifest.json`: 机器可读摘要。
- `inspect_airoa.py`: 可复用脚本(crossview_viz 的 AV1-robust 解码)。

## 相机配置(已核验)

- `observation.image.head` —— 头部相机,朝向工作台(**external/exo**);
- `observation.image.hand` —— 手部**鱼眼** in-hand 相机(**wrist**);
- 640×480,AV1,10fps(官方 30fps);**时间同步已核验**(官方论文 + HF 质量过滤丢弃 >0.3s 延迟 episode);
- **无 object mask**(仅分层语言标注 + 相机标定)。

## 关键发现:head↔hand 共视性 = 通过(本次实测重点)

对比 Galaxea(head 宽视角房间、物体几乎丢失):
- AIRoA 的 **head 俯看货架/工作台**,目标瓶子在 head 里中等大小、可辨认;hand 里目标瓶子近景占满 → **同物体两视角共视成立**,wrist↔head 分割任务可做。
- **caveat**:① hand 是**鱼眼**、畸变强,做 mask 对应需处理畸变;② 仍有尺度差(head 中等 / hand 满屏);③ 货架上**多个相似瓶子**(目标 vs 干扰项)→ 对应任务非平凡,但这对 benchmark 反而是有价值的难例;④ head 是机器人头戴(随平台移动),非固定第三人称。

## 获取官方全量(可选)

demo 已是全分辨率,pilot 未必需要官方。若要官方全量 30fps:登录 HF 账号打开 <https://huggingface.co/datasets/airoa-org/airoa-moma>,点 **"Agree and access repository"**(auto-gate)即可,之后本机 token 可下。

## 复现

```powershell
python D:\Code\Work\EgoExoSeg\AIRoA_MoMa\inspect_airoa.py --tasks 001 010 --episode 0 --frames 4
```

## 来源

- 官方论文: https://arxiv.org/abs/2509.25032
- 官方数据(gated): https://huggingface.co/datasets/airoa-org/airoa-moma
- 本次样本(免 gate,官方 org demo): https://huggingface.co/datasets/airoa-org/corl2025_demo
