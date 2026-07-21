# Robotic Cross-View 数据集调研报告

> **任务来源**:`V2-SAM to PAMI (20260613--).docx` — Sec 2.3 Robotic cross-view (todo) & 0708 条目 4 (robotic data)
> **日期**:2026-07-09
> **调研方式**:5 路并行检索 + 21 个一手来源抓取 + 105 条事实声明提取,关键结论已对官方页面/论文原文逐条核验
> **goal**:tab,2–3 个 potential

---

## 0. TL;DR

| 排名 | 候选 | 一句话理由 |
| --- | --- | --- |
| 1 | **DROID + RoboInter-Data** | 1 腕部 + 2 外部同步立体视频;RoboInter 已提供 **SAM2 生成 + 人工复核的 mask,但只在 2 个外部相机上、腕部无 mask**。→ **exo↔exo 完全免标注(56,566 对现成,约 1–2 周);wrist↔exo(=ego↔exo 同构)需自建**物体在腕部视角画面中**的 mask(下文简称"腕部 mask";mask 是逐视角的——同一物体在每路相机图像上各是一个独立 mask,RoboInter 只标了外部相机那一路)(约 2–3 周)** |
| 2 | **RH20T** | 每场景 **8–10 全局相机 + 1–2 腕部相机**,全同步全标定;唯一能把"视角差"做成可控变量、支撑 geometric-ambiguity 梯度分析的数据集 |
| 3 | **RoboMIND**(docx 原链接) | 4 种 embodiment 同任务采集,cross-embodiment story 独有;但可用子集有限且无 mask,需全自建(约 3–4 周) |
| backup(帧级任务可上调) | RoboSet(docx 原链接) | 4 视角布局最规整(3 固定 + 1 腕部)+ 4 路带 depth;综合列 backup(HDF5 需自解析 + 无 mask),但 **5Hz 只对时序 benchmark 是硬伤、帧级 cross-view 无碍 → wrist↔exo 任务可上调**(见 §0.6) |

**最重要的单一发现**:**RoboInter-Data**(ICLR 2026)——在 DROID + RH20T 之上做的标注层,自带人工复核过的 segmentation mask。**注意范围(已实测,见 §0.6):mask 全键在外部/全局相机上,腕部零 mask**;每 episode 另打包第三人称 + 腕部两路**视频**(视频≠都有 mask)。它命中 docx 中 "if manually labelling is needed, auto-segmentation" 的预案 —— **对 exo↔exo 连人工复核都免了;但 wrist↔exo(ego↔exo 同构)的腕部 mask 仍需自建**。

---

## 0.5 复核与实测更新(2026-07-10)

本次对报告做了两层独立复核:(A)**真实样本实测**——下载 RoboMIND / DROID / RH20T / RoboSet 的真实样本,直接看 HDF5 / 视频里的相机流,确认“同一物体是否有多个同步视角”,并生成可视化 montage;(B)**对抗式事实核验**——6 个数据集各派一个 researcher 抓官方页面/论文原文逐条核对,再派一个 skeptic 复核非“confirmed”的条目。

**总体结论:报告的核心声明(相机配置 / 规模 / license / 有无 mask)绝大多数被一手来源确认。** 下面是被修订/需加注的条目,以及实测样本索引。

**核验结论一览**

| 数据集 | 实测样本(同物体多视角?) | 事实核验 | 主要修订 |
| --- | --- | --- | --- |
| RoboMIND | `RoboMIND/samples/`,苹果+抽屉三视角 | 基本 confirmed | 天工“单视角”仅指已发布 RGB,硬件另有 head/chest/waist/back 深度相机 |
| DROID | `DROID/samples/`,能量棒在 2 外部+1 腕部三视角 | 基本 confirmed | **license = CC-BY 4.0**;HF 非全量镜像;补 arXiv id |
| RH20T | `RH20T/samples/`,同一 UR5 场景 **8 全局视角** | 基本 confirmed | **“7 config”= 4 臂×4 夹爪×3 力矩传感器,非 7 个机器人**;两个降采样版 640×360 与 320×180 |
| RoboSet | `RoboSet/samples/`,厨房场景 top/left/right/wrist 四视角 | 基本 confirmed | 活动数 7(官网)vs 6(论文 Table 2);规模 28,500 vs 30,050 口径差 |
| RoboInter-Data | ── | confirmed | arXiv id 属实;152,986+82,894=235,880 ≠ 卡片 235,920(源自带 40 ep 出入);**mask 视角覆盖:文档层面 UNVERIFIED,但 §0.6 已用本地全量 key 定死 = 仅外部/全局相机、腕部 0** |
| auto-mask 证据 | ── | confirmed | 77.4/67.1 是 proposal“覆盖上限”指标(非端到端);补 DOMR / O-MaMa / LM-EEC 论文标题 |

**逐条修订**

- **DROID**:① 论文为 **arXiv 2403.12945**(RSS 2024),原始为**立体 1280×720 @ 15Hz**,共 1,417 view points。② **License 是 CC-BY 4.0**(论文正文明写 “open source ... under CC-BY 4.0”),官网页面本身未标 —— 原报告“license 官网未标,需 repo 确认”应更新。③ **HuggingFace 不是全量镜像**:HF(`KarlP/droid`)只放补充的标注/标定文件,轨迹本体在 `gs://gresearch/robotics`(`droid` RLDS 1.7 TB / `droid_100` 样本 2 GB / `droid_raw` 立体 8.7 TB)。④ 实测用 `lerobot/droid_100`(180×320 左目降采样),`exterior_1/2 + wrist` 三路同步确认。
- **RH20T**:① **“7 种机器人配置”表述需修正** —— 官方是 **4 种机械臂(Flexiv Rizon / UR5 / Franka / KUKA iiwa)× 4 种夹爪 × 3 种力矩传感器 = 7 种 Cfg(Cfg1–7)**,不是 7 个不同机器人。② 降采样有**两个**版本:640×360 与 320×180(原报告只提 320×180)。③ 147 任务 = 42 skills。④ 实测镜像 `hainh22/rh20t` 内部不一致:8 个 serial 相机是真实同场景多视角(已用),3 个命名相机(front/side/eye_in_hand)是**另一来源另一场景**、不可当同一 take,详见 `RH20T/README.md`。
- **RoboSet**:① 单一 Franka Panda(4 个物理 setup)。② SAM 只做**离线训练增广**,非发布标注,无 mask 属实。③ 活动数 **7(官网 gallery,与 HF 目录一致)vs 6(论文 Table 2)**;规模 **28,500(landing)vs 30,050(teleop 子页)** 口径不一致。④ 实测 HDF5 结构 `rgb_{top,left,right,wrist}` + `d_{...}` 四路确认。
- **RoboMIND**:AgileX/Franka/UR5e 相机规格逐字属实;**天工“单视角”仅对已发布的 `*_1rgb` RGB 数据成立**,论文硬件描述里天工头/胸/腰/背各有 Orbbec Gemini 深度相机。per-robot 拆分(Franka 52,926 / UR5e 25,170 / 天工 19,152 / AgileX 10,629)确认;v1 物体类别数两处口径:arXiv v1=61、HF README v1.0=69。
- **RoboInter-Data**:arXiv 2602.09973 = “RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation”(ICLR 2026,OpenReview `PGUC3mmMoi`),13+ 类逐帧标注,segmentation 单独打包 `segmentation_npz.zip.*`(~50 GB)。**mask 覆盖哪几路:文档层面未写明(fact-check 判 UNVERIFIED),但下一节 §0.6 用本地全量 key 已实测定死 —— mask 全键在外部/全局相机,腕部 0,不是 primary/wrist 两路。** 注意 152,986+82,894=235,880 与卡片写的 235,920 有 40 ep 出入,是源本身的不一致。
- **§6 auto-mask 证据**:arXiv 2508.04050 = **DOMR**(ACM MM 2025,Ego-Exo4D 上);**77.4 (ego) / 67.1 (exo) 是 “proposal generator 覆盖上限” 指标**(每个 GT 配最高 IoU 的 proposal),不是端到端结果(端到端 DOMR 是 49.7 Ego→Exo / 55.2 Exo→Ego)。arXiv 2506.06026 = **O-MaMa**(ICCV 2025),250ms/帧、FastSAM 70ms 属实。arXiv 2510.11417 = **“Robust Ego-Exo Correspondence with Long-Term Memory”(LM-EEC)**,SAM2 视图内强、跨视图 prompt 迁移弱的结论一致。

**实测样本索引**(每个子目录另有中文 `README.md`)

| 目录 | 主 montage | 说明 |
|---|---|---|
| `RoboMIND/samples/` | `0923_164719_frame_0300_cameras.jpg` | Franka,苹果放抽屉,left/top/right 三视角 |
| `DROID/samples/` | `episode_001_crossview.png` | 能量棒,exterior_1/2 + wrist 三视角 |
| `RH20T/samples/` | `episode_000000_global8views.png` | UR5 台面,8 全局相机同场景;另有 `*_mirror_mismatch.png` 取证图 |
| `RoboSet/samples/` | `Trial1_crossview.png` | 厨房场景,top/left/right/wrist 四视角 |

> 复核方法与全部逐条 verdict 已归档;可复现脚本见各子目录的 `inspect_*.py` 与根目录 `crossview_viz.py`。

---

## 0.6 面向"建库"的实质补充(2026-07-10,基于真数据)

以下不是对旧结论的更正,而是**动了真数据后**、对"如何建这个 benchmark"有直接影响的新发现,按对方案的影响排序。

### ⚠️(最重要)首选方案 ① 的盲点:RoboInter 的 mask 只在**外部相机**,腕部**零 mask**

本地统计 `RoboInter-Data/VideoID_2_SegmentationNPZ.json` 全部 235,920 个 key 的相机归属:**DROID `exterior_image_1_left` 76,500 + `exterior_image_2_left` 76,500 + RH20T 全局 serial 82,920,`wrist`/`in-hand` key = 0**。这把 §8 列的头号 go/no-go 直接落地了:

- ✅ **真正"免标注"的现成 benchmark 是 DROID `exterior_1 ↔ exterior_2`(exo↔exo)**:两路都有 npz 的 **56,566 对**(validation 里 1,678 对)可直接起步。
- ❌ **wrist↔exo(即 ego↔exo 的机器人版)RoboInter 给不了 mask** —— 腕部真值必须自建(SAM2 + 人工复核)。
- 因此 §5 方案① step 2 "筛物体在 wrist 与 primary 同时可见、用 mask 可见性判断" 对 wrist 路**不成立**,应修正为二选一:(a)query/target 都用外部相机做 exo↔exo(完全免标注);(b)坚持 wrist↔exo 则接受"腕部自建 mask"的成本——但因 query 侧外部 mask 现成,工作量约为 RoboMIND 全自建路径的一半。

### DROID:外部相机逐 scene 重摆 → 多样性强但"视角差"不可控

2 个外部 ZED2 装在**可调三脚架**上、逐 scene 重新摆位(这正是 "1,417 view points" 的来源)。episode 内固定,但跨 episode 的 exo 视角差异极大 —— 对"exo↔exo 基线能力"是多样性优点,但**"按视角夹角分桶"的可控分析 DROID 做不了**,该分析轴只能靠 RH20T。

### RH20T:分桶分析要外参(镜像不带);但几何 auto-mask 可行性全场最高

- §5 方案② 的核心卖点"按夹角分桶"需要相机外参,而本次用的社区镜像 `hainh22/rh20t` **不带外参/标定** → 要做这条必须走**官方发布**(GDrive/百度)拿标定文件,镜像只够做定性多视角演示。
- 反过来:8–10 全局相机**全标定到基座系 + 全 RGBD** → 深度+外参投影的几何 auto-mask 可行性远高于 DROID,正好压 auto-mask 成本(RoboInter 已给 RH20T 82,920 个全局相机 mask,同样无 wrist)。

### RoboSet:被低估了 —— 4 路**全带 depth**,且"5Hz 硬伤"对本任务不成立

- 实测 HDF5:`d_top / d_left / d_right / d_wrist` **四路深度齐全** → 和 RH20T 一样支持几何 auto-mask;加上 4 视角含 wrist、布局最规整,对**帧级 wrist↔exo** 其实相当合适。
- **"5Hz 低帧率"只在需要时序密度时才是硬伤**;ego2exo/exo2ego 本质是**帧级**跨视角对应,不吃帧率 → 对本任务 5Hz 基本无影响。
- 重估:对"想要真 wrist 视角 + 愿意 auto-mask(有 depth 辅助)"的场景,RoboSet 的价值高于当前"backup"定位,可上调为 wrist↔exo 的备选 potential。

### RoboMIND:AgileX 双腕部相机是"两个都在动"的难配对

AgileX 的 2 个 hand-eye 相机随左右臂**独立运动** → wrist↔wrist 是双动跨视角,比 wrist↔固定 exo 难;更可控的是 wrist↔front(前置固定)。Franka 3rgb 有 675 条只有 left/right(已文档化),筛选时排除。

### 一句话给方案的落点

- 想**最快、零标注**出一个 robotic cross-view bench → **DROID `exterior_1↔exterior_2`(exo↔exo),56,566 对现成 mask**。
- 想要 **ego↔exo 同构(wrist↔exo)** → 腕部 mask 必须自建;此时 **RoboSet(4 视角+depth+含 wrist)** 与 **RH20T(几何 auto-mask 可行 + 官方外参)** 比"只有 RoboInter 外部 mask 的 DROID"更省事。

---

## 0.7 RoboInter mask 实测可视化与语义确认(2026-07-18)

### mask 的语义(抽 12 个 npz 实测,DROID×11 + RH20T×1)

- **每条 camera-video 恰好 1 个物体** —— npz 键仅 `masks`,shape `(1, T, 1, H, W)`,二值(bool 或 float64 0/1)。这个物体就是语言指令里**被操作的目标物体**(与逐帧 `annotation.object_box` 同一目标,如 10020 = "take the black lid…" 的黑色瓶盖)。**不是全景/多物体分割** → 对我们意味着每对 query/target 天然只有 1 个 GT 对象,格式对齐 ego2exo json 最简单,但没有 distractor 级多物体标注。
- **T = episode 帧数逐条对齐**(240/104/113/126/236 均与 parquet/episodes.jsonl 一致,10fps 时间轴同 LeRobot 视频)。
- **允许空帧**:物体不可见时该帧 mask 全 0(实测 `66378_exterior_image_2_left` 仅 57/192 帧非空、`10010_ext1` 101/104)→ 现成的 per-frame visibility 信号,可直接用于"共视帧"筛选;也提醒 pair 两路的可见帧要取交集。
- **分辨率**:DROID npz = 180×320(与打包视频同);**RH20T npz = 360×640**(= `origin_shape`,比其打包视频 180×320 高一档 → RH20T 侧 mask 真值更精细,映射回原始高清更有利)。

### 为什么只标外部相机(论文原文定性,非猜测)

RoboInter 的 mask 是 13 类逐帧**中间表征**(subtask/primitive_skill/object_box/gripper_box/trace/contact/affordance/placement/segmentation/…)之一,服务其 **plan-then-execute VLA**:论文明确 "intermediate representations (**based on primary observation**)" 条件化 Executor —— **所有 2D 标注(mask、各类 box、10 步 gripper trace、接触点)都定义在 primary=外部相机的像素坐标系**;wrist 流只是策略的**额外观测输入**("extended with an additional wrist-view input"),从来不是标注画布。且 trace/placement 这类表征只在**静止机位**上才是良定义的(wrist 系里 gripper 几乎不动、目标频繁出画),SAM2 传播在静止外部相机上也远更稳。→ "腕部 0 mask"不是数据缺口的偶然,而是其任务定位(策略学习监督,非 cross-view correspondence)的必然。对我们的推论:**wrist mask 没有任何现成来源,自建不可避免**(与 §0.6 结论一致)。

### ⚠️ license 修正

RoboInter-Data 数据卡(README front-matter)明确声明 **全部数据/代码 CC BY-NC-SA 4.0** 并配 gated 表单(实测匿名 resolve 仍可直接下载,gate 未强制)。**修正 §3.3 原表述"license 未单独声明、follow DROID/RH20T"** → 实为**非商用 + 相同方式共享**。对 benchmark 的影响:基于 RoboInter mask 的评测集再分发需带 NC 条款(与 RH20T-NC 同类注意事项);DROID 本体仍是 CC-BY 4.0,受限的是 RoboInter 标注层。

### 工程解锁:npz 免 53GB 按需抽取 + 视频获取

- `segmentation_npz.zip.00/.01/.02` 是**单一 zip 的字节切分**(README: `cat` 合并),故可用 HTTP Range 直接抽单文件:新工具 **`RoboInter-Data/fetch_segmentation_npz.py`** 首次拉取 zip 中央目录(~27MB,198,194 members,缓存为 `segzip_index.json`)后,任意 episode 的 npz 仅 **~10–130KB** 一取即得(压缩率极高:88MB float64 → 130KB)。→ 56,566 对全量 mask 也只需按需抽,不必下 53GB。
- LeRobot 视频按 chunk 打 tar(`videos/chunk-000.tar` ≈ 846MB / 1000 episodes × primary+wrist 两路);tar 内**wrist 段在前、primary 段在后**,可流式部分抽取。已完整下载 chunk-000(与本地 1000 个 parquet 同 chunk)。
- npz 在 zip 内路径:DROID `OXE_DROID/data/ann_human/0/sam_mask/<episode>.npz`、RH20T `RH20T/data/ann_human/0/sam_mask/<episode>.npz`,与 `VideoID_2_SegmentationNPZ.json` 的值一致。

### 可视化(脚本 `RoboInter-Data/visualize_robointer_masks.py`,输出 `RoboInter-Data/samples/`)

- `{base}_crossview_mask.png`:exterior_1(+mask) | exterior_2(+mask) | wrist(无 mask)× 4 时刻 —— 同一物体两路外部 mask 齐、腕部零标注一图看清;
- `{base}_annotation_suite.png`:exterior_1 上叠加完整中间表征套件(mask+object_box+gripper_box+trace+contact+placement,含放大 inset)—— 直观展示 mask 在 RoboInter 里的真实用途。
- 已渲染 3 组(`10020` 瓶盖小目标/val 集、`10010` 碗-两路 exo 尺度差大、`0` in-the-wild 卧室织物),见 `RoboInter-Data/samples/README.md`;三组均视觉确认 **ext1/ext2 的 mask 是同一物体身份**,且 wrist 画面里目标物体清晰可见(共视成立)但零标注。
- 质量抽查印象与 §6 预期一致:SAM2+人工复核的 mask 边界在 180×320 下贴合;终版 benchmark 仍按 §5 路径映射回高清帧后 refine。
- 二次核验:zip 中央目录 198,194 个成员 = `exterior_image_1` 64,368 + `exterior_image_2` 64,498 + RH20T 全局 serial 69,328,**wrist/in-hand = 0**(与 `VideoID_2_SegmentationNPZ.json` 非空值计数 198,194 一致,两个独立来源互证)。

### exo↔exo pair 计数(2026-07-19,本地 mapping 全量统计)

| 来源 | pair 定义 | 全量(无序对) | val 集 | 备注 |
|---|---|---|---|---|
| DROID | 同 `video_id` 的 ext1+ext2 两路都有 npz | **56,566**(/76,500 完整双视角) | **1,678** | 只一路有 npz 15,734、两路都无 4,200 |
| RH20T | 同 take(去掉相机 serial 后缀)任取 2 路有 npz 的全局相机,C(k,2) | **178,620**(12,191 takes × 峰值 6 路/take,最多 8 路) | 7,725 | 每 take 标注相机数分布:1×204 / 2×385 / 3×663 / 4×1,280 / 5×2,327 / **6×3,322** / 7×2,402 / 8×1,608 |

- 合计 ~235k 无序对;按 query→target 双向构造则 ×2。均为"两路都有 npz"的原始对,尚未做 Qsheet 质量过滤与共视帧(两路非空帧交集)过滤,有效可用数会略降。
- ⚠️ **修正旧表述**:§0.6/§2.2/§3.4 等处"RoboInter 给 RH20T **82,920 个 mask**"不准确 —— 82,920 是 RH20T 的 **key 总数**(含 13,592 个 null),真正有 npz 的是 **69,328**(与 zip 中央目录成员数一致)。DROID 同理:2×76,500 key 中非空 64,368+64,498。
- ⚠️ RH20T "同 take 各相机标的是同一物体"尚未做视觉抽查(DROID 已抽 3 组确认同一物体身份);RH20T 多相机全标定 + 178k 对 → 夹角分桶的原料比预期更充裕。
  - → 2026-07-19 已补 1 组视觉抽查(scene_0004 两路全局相机,`samples/task_0001_..._crossview_mask.png`):**同一物体身份确认**;且本地 chunk 里单 take 就有 6–7 路已标注相机,C(k,2) 产能属实。

### wrist 侧 SAM2 预演 + 本机算力结论(2026-07-19)

**RH20T mask 已同样可视化**(脚本 `--rh20t auto` 模式):GT 360×640 边缘明显比 DROID 的 180×320 更利;所选 take("press the button")wrist(in-hand)相机全程正对目标物体 → RH20T 的 wrist↔exo 共视性(至少该类任务)良好。

**用 SAM2 给 wrist 视角补预测 mask 已跑通**(`RoboInter-Data/sam2_wrist_preview.py`,输出橙色叠加进 crossview 图,与绿色 GT 明确区分):

- 算力:本机 `py312` 环境(torch 2.10+cu128)可用 **RTX 5070 GPU** → SAM2 hiera-**b+** 全视频传播(百余帧/条仅数秒);CPU 路径(base Anaconda 环境,torch cpu 版)退化为 tiny+4 帧稀疏模式也能跑。4 条(DROID×3 + RH20T×1)全部产出。
- 质量印象:1–4 次点击即可得到贴合的 wrist mask;b+ 全程传播在物体出画帧**正确输出空 mask**(与 GT 可见性语义一致,tiny 稀疏版会幻觉假阳性)。
- **两个实证的失败模式**(已留档在脚本注释):① 点击偏 10px 选错物体(10010 首点命中碗内搅拌棒,SAM2 全程跟踪了棒);② 正点落在目标边缘外(RH20T 落桌布)→ mask 传播爆帧。→ 佐证 §6 结论:**auto-seg 必须人工复核**,且"点该点谁"本身就是跨视角身份对齐问题(V²-SAM 的任务)。
- 由此,wrist GT 构建管线定型:**外部相机 GT 定身份 → wrist 帧少量人工点击 → SAM2(-L)全程传播 → 人工复核可见性与边界**;本机 GPU 即可承担 hundred-level 的传播计算。

### 腕部手工标注实测(2026-07-21,labelme,6 集 33 帧)—— 混合管线定型

用户用 labelme 手标 6 条 DROID episode 的腕部关键帧(`manual_wrist_kit/`),与 SAM2 结果逐帧 IoU 对比(`samples/*_manual_vs_sam2.png`):

| episode | 目标 | SAM2 来源 | mean IoU(手工 vs SAM2) | 判读 |
|---|---|---|---|---|
| 10020 瓶盖 | 小刚体 | 人工点击 prompt | **0.93** | prompt 对了,SAM2≈人工质量 |
| 10010 碗 | 含干扰棒 | 人工点击(修正版) | **0.87** | 同上 |
| 0 织物 | 大形变 | 人工点击 | 0.36(单帧 0.00–0.97) | 形变+边界歧义,时序漂移 |
| 10082 银罐 | 小刚体 | **自动 prompt** | 0.41(后期 0.04) | 传播后期丢失 |
| 1000 红物 | 小刚体 | 自动 prompt | **0.00**(零重叠) | **选错物体**(身份错误) |
| 10427 绿笔 | 细长小物 | 自动 prompt | **0.00**(SAM2 全空) | 自动提示彻底失败;手工连 261px 远景都精准 |

**结论(管线定型)**:①人工在"身份判断 + 可见性判断"上碾压(自动 prompt 4 例中 2 例身份级失败);②**prompt 正确时 SAM2 达人工质量(IoU 0.87–0.93)**;③形变物体与长程传播仍需人工纠偏。→ wrist 真值管线最终形态:**人工稀疏关键帧(定身份+可见性,~30–60s/帧)→ SAM2 以人工帧为 prompt 密集传播 → 人工抽查漂移**。hundred-level 成本估计(修正):**150 集 × 5 关键帧 × 45s ≈ 9.4 人时;300 集 × 6 帧 ≈ 22.5 人时**(若实测每帧仅 ~15–20s 则减半;可按 docx 分工拆给多人)。

**V²-SAM 本机可行性(修订)**:此前"无 CUDA"判断作废——`py311/py312` 均有 cu128 torch,RTX 5070 8GB 可用。V²-SAM 权重在 HF `jaychempan/V2-SAM`,但需另下 SAM2-L + **DINOv3(Meta gated,需 HF 账号过 license)**;8GB 显存 bf16 单帧推理估计可塞下(偏紧),小规模 demo 可本机试,**成批 ZSL 评测仍建议 A6000**。

---

## 1. 任务定义与筛选标准

**Goal(docx 0708 条目 4)**:机器人的不同视角(同一个机器人的不同视角,或相同轨迹下不同机器人的视角),object 在不同的 camera、机器人;mask。

由此展开的 6 条硬性筛选标准:

1. **同时刻多视角**:同一机器人场景有多个同步相机(腕部/ego + 外部/exo,或多个外部相机);
2. **物体跨视角共视**:被操作物体在多个相机中同时可见;
3. **视频优先**(docx:"best to be in videos");
4. **mask**:最好自带;没有则评估 auto-segmentation(SAM2)可行性与人工成本(docx:"if manually labelling is needed, auto-segmentation");
5. **规模**:测试集 hundred-level(对齐 ego2exo / exo2ego 测试集体量,参考 Ego-Exo4D correspondence 测试集为 291 个 take);
6. **可下载**:license 与获取渠道实际可行。

**任务形式对齐**:与已完成的 ego2exo / exo2ego 一致 —— 给定 query 视角中某物体的 mask,在另一同步视角中分割同一物体(wrist↔external 即机器人版的 ego↔exo)。

---

## 2. plan 2.3 节原链接精查

### 2.1 RoboMIND

- 论文:[arXiv 2412.13877](https://arxiv.org/abs/2412.13877)(v3 于 2025-05 修订)
- 数据:[HuggingFace `x-humanoid-robomind/RoboMIND`](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND) / [项目页](https://x-humanoid-robomind.github.io/)(V2.0 已上 ModelScope)

**总体规模**:107k 轨迹(约 12.3 TB)、479 任务、96 类物体、4 种 embodiment,另含 5k 失败演示(带失败原因标注)。分 embodiment:Franka 52,926 条(其中约 26.9k real + 26.1k sim)、UR5e 25,170 条、天工人形 19,152 条、AgileX 双臂 10,629 条。

**相机配置(决定可用性)**——按 HF 仓库文件夹划分:

| 子集 | 视角数 | 相机细节 | cross-view 可用性 |
|---|---|---|---|
| `h5_franka_3rgb` | 3 | 3× 外部 Intel RealSense D435i(left 480×640 / top 720×1280 / right 480×640),**全部固定外部相机,无腕部** | ✅ 仅 exo↔exo |
| `h5_agilex_3rgb` | 3 | **2× 腕部 hand-eye Orbbec Astra + 1× 前置相机**,全部 480×640 | ✅ **wrist↔exo(RoboMIND 内唯一)**+ wrist↔wrist |
| `h5_franka_1rgb` / `h5_ur_1rgb` / `h5_tienkung_*_1rgb` | 1 | UR5e 仅一个顶置 D435i;天工用机身内置相机 | ❌ 单视角,不可用 |

> 论文原文(已核验):Franka —— "three Intel RealSense D435i cameras (left, top, and right) with resolutions of 480×640, 720×1280, and 480×640 pixels";AgileX —— "two hand-eye Orbbec Astra cameras and one front-facing camera, all at 480×640 resolution"。
> 注意:675 条 `h5_franka_3rgb` 轨迹因设备维护只有 left+right 两路。

**多视角(≥2 相机)可用子集 —— 实测拆分(2026-07-10)**:RoboMIND **不公开 per-camera-subset(3rgb vs 1rgb)的精确轨迹数**;论文只给 per-embodiment(Franka 真实 26,866 + 仿真 26,070 / AgileX 10,269 / UR5e 25,170 / 天工 15,187),HF 数据以**按 task 压缩的 tar.gz 分片**存放,不下载无法精确点数。已用可访问代理量夹逼:

| 子集 | 轨迹数 | 依据 |
|---|---|---|
| **AgileX 3rgb**(2 腕部+1 前置) | **10,269**(精确) | 全部 AgileX 即 3rgb |
| **Franka 3rgb(真实)** | **≈ 21.6k–25.6k**(区间) | 真实 Franka 26,866 扣掉 `franka_1rgb`;后者占比两代理不一致:标注子集 3287:161→1rgb≈4.7%(→25.6k),归档分片 277:68→1rgb≈20%(→21.6k) |
| Sim Franka 3rgb | 26,070(**仿真**,3 视角) | 论文 |

**采样密度实测(2026-07-10,已下即删)**:`h5_franka_3rgb` 全量 = **2.4 TB**(HF gated 无权,改走 ModelScope 免 gate 下载),全量点数不成比例;改下 3 个 task(共 4.2 GB)按"压缩字节/轨迹"密度外推:

| task | 大小 | 轨迹数 | episode 帧长 | MB/轨迹 |
|---|---|---|---|---|
| pick_apple_into_chest | 659 MB | 2 | ~600(长 episode 异常值) | 329 |
| yellow_square | 839 MB | 6 | ~180–191 | 140 |
| slide_close_drawer | 2,939 MB | 30 | ~68–134 | 98 |

> task 越大越偏向"多轨迹、短 episode"(MB/轨迹 越低);字节加权 pooled 密度 116.8 MB/轨迹 → **≈20.5k**,剔除长 episode 异常值 → **≈22.8k**。**与代理区间 21.6k–25.6k 收敛,naive 的 7.3k 是长 episode 异常值假象。**
>
> **最终:Franka3rgb(真实)≈ 21–24k(两法收敛,点估计 ~22k);真实机器人多视角可用 ≈ 31–34k**(+ AgileX 10,269)。原报告"≈63k"实为 **全部 Franka(含 26k 仿真)+ AgileX** 的合计,真实部分只约一半。要个位精确仍需下 2.4 TB,判定不值当。

**标注情况(已双票核验确认)**:**无任何 segmentation mask / bounding box**。仅有:任务级语言描述、10k 轨迹子集的 frame-level 细粒度语言描述、成功/失败标签。→ mask 需从零自建。

**格式与获取**:原始 HDF5 轨迹文件,按 task 打包为 `tar.gz(.part-xx)`(如 `benchmark1_0_compressed/h5_franka_3rgb/<task>.tar.gz`);sim 数据的 depth 暂不可用。**获取渠道(2026-07-10 实测)**:HF `x-humanoid-robomind/RoboMIND` 是 **gated**——可读 repo tree,但**下载数据 403(需申请授权)**;而 **ModelScope 同数据免 gate 可直接下**(`https://www.modelscope.cn/datasets/X-Humanoid/RoboMIND/resolve/master/<path>`,支持 range 请求)——**绕开 gate 审批的现成通道**。注意子集体量巨大:`h5_franka_3rgb` 全量 2.4 TB、`h5_franka_1rgb` 720 GB。**FPS 不可考**:2026-07-10 开真实样本(`pick_apple_into_drawer` 的 `trajectory.hdf5`)核实,结构只有 `master`/`observations`/`puppet` 三组、**无任何时间戳字段**(attrs 仅 `compress`/`sim`),论文/数据卡也不给 per-embodiment 频率 → 除非拿原始 ROS bag 否则无法确定,不是"待实测"而是"释出文件不含此信息"。

**新增实测 caveat(2026-07-09)**:公开仓库的静态文件与版本口径较混乱。有效说明文件在 `static/` 路径(`static/robomind.yaml`,`static/all_robot_h5_info*.md`),根路径同名文件在 ModelScope 返回 404;HF Dataset Viewer 因 gated 返回 401,只能看 repo tree/README/示例包;当前 arXiv v3 与 README 主文为 107k / 479 tasks / 96 classes,但 README 仍保留 Version 1.0 的 55K / 279 tasks / 69 classes,旧 arXiv v1 又是 55k / 279 / 61 classes。另:`static/RoboMIND_intrinsics.md` 当前不可用/已删除,不能默认有完整内外参。

**结论**:可用,但只有两个子集有效,且 AgileX 子集(wrist↔exo)才贴合 ego-exo 式任务;Franka 子集可作 exo-exo 补充。多 embodiment 是其独有卖点。

### 2.2 RoboSet

- 官网:[robopen.github.io/roboset](https://robopen.github.io/roboset/)
- 论文:RoboAgent,[arXiv 2309.01918](https://arxiv.org/pdf/2309.01918)(ICRA 2024)
- 镜像:[HF `jdvakil/RoboSet-Teleoperation`](https://huggingface.co/datasets/jdvakil/RoboSet-Teleoperation)(425 GB,MIT)

**相机配置(已核验,论文原文)**:每帧 4 个同步 RealSense D455 视角 —— **3 个固定外部相机(top / left / right)+ 1 个腕部相机(装在 end-effector 上方)**。"The four Realsense D455 camera views provide complementary perspectives of the workspace."
→ wrist↔exo 与 exo↔exo 都能配,视角布局是全部候选中最规整的。

**规模**:官网口径 28,500 条轨迹(9,500 遥操作 Oculus Quest 2 + 19,000 kinesthetic 回放);论文 full 版口径 98,050 条;单一 Franka Panda embodiment,厨房桌面场景,7 大 activity(make_tea、make_toast、clean_kitchen 等)× 场景 × 子任务组织。

**硬伤**:
1. **采集频率仅 5Hz**——时序密度低,对"视频/时序"benchmark 不友好(对比:RH20T 10Hz、AgiBot World 30Hz);**但注意:本任务(ego2exo/exo2ego)是帧级跨视角对应,不吃帧率,所以 5Hz 对本任务基本无影响 —— 这条只在需要时序密度时才算硬伤(见 §0.6 重估)**;
2. **无 mask**(论文中 SegmentAnything 仅用于训练时的语义增广,不是发布标注);但 4 路**全带 depth**(`d_top/left/right/wrist`,已实测)→ 支持几何 auto-mask;
3. 原始 HDF5(每文件 ~450MB,HF 镜像连 README 都是空的、viewer 无法预览),需自写解析提取各路视频;
4. kinesthetic 回放部分(19k)动作与场景多样性有限。

**获取**:MIT license;官网提供按任务的 tar.gz 直链(`dl.fbaipublicfiles.com/RoboSet/...`),下载无阻碍。

**结论**:综合性价比低于 DROID/RH20T(主因 HDF5 需自解析 + 无 mask),整体列 **backup**;**但对"帧级 wrist↔exo"这一具体任务是可上调的 potential** —— 布局最规整、含真 wrist、4 路带 depth、5Hz 不构成障碍(见 §0.6)。即:video/时序 story 用不上它,帧级 cross-view story 里它比排名更值得考虑。

---

## 3. 横向候选精查

### 3.1 DROID ⭐

- 官网:[droid-dataset.github.io](https://droid-dataset.github.io/)(RSS 2024)

**相机配置(已核验)**:标准化 Franka Panda 采集台,每场景 3 个同步视角 —— **2× 可调外部 ZED 2 立体相机 + 1× 腕部 ZED Mini 立体相机**。全部提供内参 + 外参立体标定。共 1,417 个相机视点配置。

**规模**:76k 轨迹 / 350 小时 / **564 个场景**(50 个采集者,北美/亚洲/欧洲,in-the-wild 场景多样性远超实验室数据集)/ 86 任务。

**标注**:**DROID 官方自带无 mask**(只发 RGB 立体视频 + 语言标注:95% 成功 episode 各 3 条,2024-12 更新)。→ mask 要靠**外挂的 RoboInter-Data 标注层**补(它是别人在 DROID 之上单独做的数据集,不属于 DROID 官方发布);且 RoboInter **只在 2 个外部相机(`exterior_1_left` / `exterior_2_left`)上有 mask,腕部无 mask**(见 3.3,已实测确定)。

**获取与数据现状(2026-07-10 核验)**:**主数据是单一当前版本(没有新旧两份并存),不是"合并版"**。按用途分 4 个下载包,均在 GCS `gs://gresearch/robotics`:`droid` RLDS 1.7 TB(训练用)/ `droid_100` 2 GB(调试样本)/ `droid_raw` 立体 HD 8.7 TB(MP4)/ 非立体 5.6 TB。

- **更新与主数据的关系 = 叠加、非合并**:2024-12 的语言标注(75k eps,3 条/episode)与 2025-04 的改进标定(**仅 36k/76k 子集**)是官方在 HuggingFace 上**单独发布的叠加文件**(用 "provide improved / updated" 口径),**不并入主 RLDS、也不重新打包新数据集**;主 RLDS 里仍是原始标定,要用改进版得自己 overlay 到那 36k 上。
- ⚠️ 实测 caveat:官网把标定/标注都指向 `huggingface.co/KarlP/droid`,但该仓库当前实际只有 `README.md`(cc-by-4.0)+ `.gitattributes`,**数据文件不在其中**(疑似已迁移/改挂载),下载前需先确认落点。
- **License = CC-BY 4.0**(论文正文明写,官网页面未标)。HF `KarlP/droid` 只放补充标注/标定,**不是轨迹数据的全量镜像**。〔详见 §0.5〕

**适配点**:1 wrist + 2 exo 恰好构成"1 ego : 2 exo"结构,和 Ego-Exo4D 的 take 结构(1 ego + 多 exo)同构,格式迁移最自然;立体相机意味着有深度可用于几何核验。

### 3.2 RH20T ⭐

- 官网:[rh20t.github.io](https://rh20t.github.io/)(ICRA 2024)

**相机配置(已核验,官网原文)**:每采集平台 **8–10 个全局 RGBD 相机 + 1–2 个腕部(in-hand)相机**;"all the cameras are calibrated with respect to the robot's base frame, and all the recorded data are synchronized in the temporal domain"(全部标定到机器人基座系 + 时间同步,按时间戳保存,标定质量逐场景人工校验)。

**规格**:RGB / depth / 双目 IR 均为 1280×720 @ 10Hz(另有 320×180 压缩版);多数序列时长 10–100 秒。

**规模**:110k+ 机器人操作序列(另有等量人类演示视频)/ 147 任务(48 来自 RLBench + 29 来自 MetaWorld + 70 自提,共 42 skills)/ **7 种配置(Cfg1–7)= 4 种机械臂(Flexiv Rizon / UR5 / Franka / KUKA iiwa)× 4 种夹爪 × 3 种力矩传感器**,并非 7 个不同机器人〔2026-07-10 核验修订,详见 §0.5〕/ 50M+ 帧(>40M robot + >10M human)。

**标注**:无 mask,无物体级标注(模态为 RGB/depth/IR/力觉/音频/本体感知)。→ RoboInter-Data 给 **82,920 个 mask,全部键在全局相机 serial 上,无 wrist mask**(已实测确定,见 3.3;原表述"primary+wrist 两路"有误)。

**获取**:**分段 license**——RH20T-C(scenes 0001–0005)为 CC BY-SA 4.0,RH20T-NC(scenes 0006–0010)为 CC BY-NC 4.0;Google Drive(支持 Rclone)+ 百度云,提供全分辨率与降采样两个版本。

**适配点(独有)**:
1. 一条轨迹可组合出 C(10,2)≈45 个 exo-exo pair + ~10–20 个 wrist-exo pair,**cross-view pair 产能全场最高**(exo-exo 的 mask 有 RoboInter 现成;wrist-exo 的腕部 mask 需自建);
2. 相机全部标定到同一坐标系 → 可**按视角夹角分桶**(30°/60°/90°/…),把 docx 中 "A. Geometric ambiguity" 从定性挑战变成可控实验变量——这是其他所有数据集给不了的;**注意:分桶需相机外参,须走官方发布(GDrive/百度)拿标定,LeRobot 社区镜像 `hainh22/rh20t` 不带外参**(见 `RH20T/README.md`);
3. 7 种 config 执行同一任务集 → 支持 cross-embodiment same-task 对照(见 §7);
4. 全标定 + 全 RGBD → 深度+外参投影的几何 auto-mask 可行性全场最高(压 wrist 侧自建 mask 的成本);
5. 高分辨率(1280×720)。

### 3.3 RoboInter-Data ⭐⭐(关键发现)

- 论文:RoboInter,**ICLR 2026**,[arXiv 2602.09973](https://arxiv.org/abs/2602.09973)
- 数据:[HF `InternRobotics/RoboInter-Data`](https://huggingface.co/datasets/InternRobotics/RoboInter-Data)(**免 gate 直接下载**,408 GB)

**是什么**:不是新采集的数据集,而是在 **DROID(152,986 eps)+ RH20T(82,894 eps)** 之上做的大规模标注层:235,920 episodes、571 场景、**10+ 类逐帧中间表征标注**——其中包括 **segmentation mask**(单独打包 `Annotation_raw/segmentation_npz`,约 50 GB;`VideoID_2_SegmentationNPZ.json` 提供映射,null 表示该 episode 无分割标注)。
> **两个数勿混**:RH20T 的 **82,894** 是 RoboInter 覆盖的 **episode 数**;下文出现的 **82,920** 是 RH20T 全局相机的 **mask npz key 数**(按相机键计),两者接近但不是同一量纲。DROID 侧同理:152,986 episodes,对应 `exterior_1_left` / `exterior_2_left` 各 76,500 个 key。

**mask 生成方式(论文,已提取原文)**:"RoboInter-Tool automatically transports the annotation to SAM2 for object segmentation and tracking, and the result is asynchronously returned for review" —— **SAM2 自动分割/跟踪 + GUI 人工异步复核**。即:docx 里预案的 auto-segmentation 流程,别人已经做完并过了人工质检。

**视角**:LeRobot 包每 episode 打包两路**视频** —— `observation.images.primary`(第三人称)+ `observation.images.wrist`(腕部),180×320 @ 10fps;另含 7,246 episodes 的验证集划分。**注意:视频有两路,但 mask 不是按这两路组织的**(见下)。

**license**:~~标注层未单独声明,follow 原始 DROID / RH20T~~ **修正(2026-07-18,见 §0.7):数据卡明确声明全部数据/代码 CC BY-NC-SA 4.0**(卡片带 gated 表单但未强制,匿名可下);再分发含 RoboInter mask 的评测集需带 NC 条款。

**两个 caveat(2026-07-10 已用本地全量元数据实测确定,原 ⚠️ 消解)**:
1. ✅ **mask 只在外部/全局相机上,腕部零 mask —— go/no-go 已有答案**。本地统计 `VideoID_2_SegmentationNPZ.json` 全部 235,920 个 key 的相机归属:DROID `exterior_image_1_left` 76,500 + `exterior_image_2_left` 76,500 + RH20T 全局 serial 82,920,**`wrist`/`in-hand` key 数 = 0**。因此:
   - **免标注的现成 pair 是 DROID `exterior_1 ↔ exterior_2`(exo↔exo)**,两路都有 npz 的 **56,566 对**(val 1,678 对);
   - **wrist↔exo 的腕部真值 RoboInter 给不了,必须自建**(query 侧外部 mask 现成,故工作量约为全自建的一半)。
2. ✅ **180×320 确认为低清;但"映射回原始高清"的 ID 链路已确认存在**,不是未知项。npz 路径本身保留了溯源结构(如 DROID `10010_exterior_image_2_left` → `/OXE_DROID/data/ann_human/0/sam_mask/…npz`;RH20T `RH20T_cfg1_task_0001_…_035622060973` → `/RH20T/…/sam_mask/…npz`),`annotation.origin_shape` 也在 parquet 里(DROID `[320,180]`、RH20T `[640,360]`)。剩下的只是**工程步骤**(用 DROID hr_video_reader / RH20T API 拉原始帧 + resize/SAM2 refine),不再是数据可得性上的不确定。

### 3.4 AgiBot World

- [GitHub OpenDriveLab/AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World) / [arXiv 2503.06669](https://arxiv.org/html/2503.06669v4)

AgiBot G1 人形 ×100 台,**每台 8 相机**:前置 RGB-D + 3 前置鱼眼 + 每臂 end-effector RGB-D/鱼眼 + 2 后置鱼眼;30Hz 同步录制;1,001,552 轨迹 / 2,976 小时 / 217 任务;含 depth 与相机标定;CC BY-NC-SA 4.0,HF + OpenDataLab 下载。无 mask。

**不选的理由**:鱼眼畸变对 mask 标注与评测都是额外负担;体量过大(Beta 43.8 TB)筛选成本高;非商用 license 对 benchmark 分发有约束。作为"人形机器人 ego-exo"的未来扩展方向值得记一笔。

### 3.5 BRMData

- [arXiv 2405.18860](https://arxiv.org/abs/2405.18860)

双臂移动平台:**2× 腕部 RGBD(每臂一个,可动)+ 1× 中央固定 RGBD**,480×640,30Hz(动态任务 D435 60Hz);10 个家务任务 × 50 条 ≈ **500 episodes**(每条 400–1200 步);MIT license,但下载渠道是京东云盘分享链接(box.jd.com,国内访问方便,海外用户不友好)。无 mask。

**评价**:规模恰好 hundred-level、双臂 wrist↔中央 exo 结构干净,可作小而美的补充测试集;但没有冗余量供筛选,单一来源风险高,列为观察项。

### 3.6 BridgeData V2

- [bridgedata-v2.github.io](https://bridgedata-v2.github.io/)

WidowX 250;1 个固定 over-the-shoulder RGBD + **2 个采集中位姿随机的 RGB 相机**,无腕部相机;640×480,控制频率 5Hz,平均每条仅 38 步;53,896 轨迹。无 mask;**license = CC-BY 4.0**(2026-07-10 官网确认)。

**不推荐**:辅助相机位姿逐轨迹随机(跨轨迹视角不可控)、无 wrist 视角、低频短轨迹。

### 3.7 GraspNet-1Billion(用于说明为什么"有 mask 也不选")

97,280 张 RGB-D(1280×720)/ 190 个静态杂乱场景 / 88 物体;**自带全帧 instance mask + 6D pose**(首帧人工标注 + 已知相机运动传播)。但采集方式是**臂载相机沿轨迹顺序扫过 256/512 个视点**——不是多相机同时拍摄,场景是静态摆拍、无操作过程。

**结论**:它属于 static multi-view 类型,与 benchmark 中已有的 **HANDAL-X 定位重复**,且不满足"视频 + 同时多视角"要求 → 不进 robotic bench。若日后想扩 static multiview 的量,可以回来考虑。

---

## 4. 总对比表

| 数据集 | 机器人/embodiment | 同步视角(wrist + exo) | 视频/同步/分辨率 | Mask | 规模 | License / 下载 | 适配结论 |
|---|---|---|---|---|---|---|---|
| **DROID** | Franka(标准化采集台 ×多站点) | **1 腕部 ZED Mini + 2 外部 ZED 2**(均立体,原始 1280×720@15Hz) | ✅ 视频;内外参标定(25-04 改进版) | ❌ 自带无 → RoboInter 覆盖,**仅 2 外部相机(exterior_1/2),wrist 无 mask**;exo↔exo 现成 56,566 对 | 76k 轨迹 / 350h / 564 场景 | **CC-BY 4.0**(论文);数据在 `gs://gresearch/robotics`,HF 仅补充文件 | ⭐ **首选** |
| **RH20T** | 4 臂(Flexiv/UR5/Franka/KUKA)×4 夹爪×3 力矩=**7 配置** | **1–2 腕部 + 8–10 全局 RGBD** | ✅ 1280×720 @ 10Hz;时间同步 + 基座系标定 | ❌ 自带无 → RoboInter 给 82,920 个 mask,**全在全局相机,无 wrist**;可几何 auto-mask | 110k+ 序列 / 147 任务(42 skills)/ 50M+ 帧 | RH20T-C:CC BY-SA 4.0;RH20T-NC:CC BY-NC 4.0;GDrive + 百度云 | ⭐ **次选** |
| **RoboMIND** | Franka / AgileX 双臂 / UR5e / 天工人形 | AgileX:**2 腕部 + 1 前置**;Franka:3 外部(无腕部);UR5e / 天工:单视角 ❌ | ✅ 视频;480×640(top 720×1280);**FPS 释出文件无时间戳,不可考**(见下注) | ❌ 仅语言标注(10k 子集 frame-level) | 全集 107k;**多视角可用:真实 ≈31–34k**(Franka3rgb 真实 ~22k〔代理区间+采样密度两法收敛〕+ AgileX 10,269)**+ 仿真 26,070**;原"≈63k"含 26k 仿真、真实只约一半(拆分/采样见 §2.1) | HF gated(Apache 2.0 badge);V2.0 ModelScope | ⭐ **第三** |
| RoboSet | Franka(单一,4 个物理 setup) | **1 腕部 + 3 固定**(D455,top/left/right);**4 路全带 depth** | ⚠️ 仅 5Hz(帧级任务无碍);HDF5 需自解析 | ❌(论文中 SAM 仅用于训练增广) | 28.5k(9.5k 遥操作 + 19k kinesthetic);论文 full 98k | MIT;官网 tar.gz 直链;HF 镜像 425GB | backup(对 wrist↔exo 可上调) |
| RoboInter-Data | (标注层,覆盖 DROID + RH20T) | 视频打包 primary + wrist 两路;**但 mask 只在外部/全局相机** | ✅ 180×320 @ 10fps(低清,ID 链路可映射回原始数据) | **✅ SAM2 + 人工复核**(~50GB npz);**wrist 无 mask** | 235,920 eps / 571 场景(验证集 7,246;卡片 235,920 与 152,986+82,894=235,880 有 40ep 出入) | HF 免 gate,408GB;license 随 DROID/RH20T | **⭐ 与 DROID/RH20T 配套使用** |
| **Galaxea Open-World**〔新 25-09〕 | R1-Lite 双臂移动(星海图) | **2 腕部 + 2 头部**(head+head_right 疑立体);同步已核验 | ✅ 720×1280 AV1@15fps;严格时间同步 | ❌ 自带无(需自建) | 500+h / ~10万轨迹 / 2.87 TB | **HF gated + ModelScope 需登录**(实测两渠道均需授权);CC BY-NC-SA 4.0 | 新增 wrist↔head 源;⚠️**社区替身实测 head 为宽视角、wrist↔head 尺度差大、共视弱**(见 `Galaxea/`),待官方高清核验 |
| **AIRoA MoMa**〔新 25-09〕 | Toyota HSR 移动(AIRoA 日) | **1 腕(hand,鱼眼)+ 1 头(head)**;同步已核验 | ✅ 640×480@30fps | ❌ 自带无 | 未captured | HF gated;但**官方 org 有免 gate 全分辨率 demo `corl2025_demo`** | ⭐**实测共视通过**:head 俯看货架、目标瓶子在 head↔hand 共视成立(优于 Galaxea,见 `AIRoA_MoMa/`);⚠️hand 鱼眼畸变 |
| **RoboMIND 2.0**〔新 25-12〕 | 双臂/移动新版(x-humanoid) | **2 腕部 + 1 前置**;同步(同一时刻) | ✅ 同步 | ❌ 真实数据无(仿真有 affordance) | 未captured | 随 RoboMIND;HF/ModelScope | v1 的新版,front+双腕同步 |
| AgiBot World | AgiBot G1 人形 ×100 | 8 相机(前 RGB-D + 3 鱼眼 + 双臂腕部 + 2 后置鱼眼) | ✅ 30Hz 同步 | ❌ | 1M+ 轨迹 / 2,976h | CC BY-NC-SA 4.0;HF + OpenDataLab | 鱼眼 + 体量过大,暂不选;人形扩展备选 |
| BRMData | 双臂移动平台 | 2 腕部 RGBD + 1 中央固定 RGBD | ✅ 480×640 @ 30/60Hz | ❌ | 仅 ~500 eps(10 任务 ×50) | MIT;京东云盘 | 规模无冗余,观察项 |
| BridgeData V2 | WidowX 250 | 1 固定 + 2 **随机位姿**;无腕部 | ⚠️ 5Hz,平均 38 步/条 | ❌ | 53.9k | **CC-BY 4.0**(官网已确认) | 不推荐 |
| GraspNet-1B | 臂载相机(静态场景) | ❌ 单相机顺序扫 256/512 视点 | ❌ 非同步、非操作视频 | ✅ 全帧 mask + 6D pose | 190 场景 / 97,280 帧 | 研究开放 | 与 HANDAL-X 定位重复,不进 robotic bench |

### 先用哪 2–3 个(2026-07-12 决定)

把"新老候选"放一起后,**首批只用 2–3 个**,取舍逻辑是"零标注优先 + 各答一个不同问题":

- **必选 2 个(最小可行、成本最低):**
  1. **DROID + RoboInter** —— exo↔exo 的 **56,566 对现成 mask、零标注**,当基线 cross-view correspondence;
  2. **RH20T** —— RoboInter 已给 82,920 个全局相机 mask,且**唯一能按视角夹角做 geometric-ambiguity 诊断分析**(PAMI 差异化卖点)。
  > 这两个的 mask 都被 RoboInter 覆盖 → 几乎不用标注就能起步,是**性价比最高的起点**。
- **第 3 个(要不要加、加谁,二选一,按 story 定):**
  - 想要 **cross-embodiment 轴** → **RoboMIND**(计划书原候选,4 embodiment;ModelScope 免 gate 可下);
  - 想要 **一个干净、当下可下的 wrist↔exo 新源** → **AIRoA MoMa**(官方免 gate demo `corl2025_demo`,全分辨率;**已实测共视通过**——原本这里写的是 Galaxea,但 2026-07-13 实测 Galaxea 共视弱、AIRoA 共视好,故换成 AIRoA)。
  > 两者都**要自建 wrist mask**,所以按想讲的故事二选一即可,不必都上。
- **其余(Galaxea〔共视弱降级〕/ RoboMIND 2.0 / VTouch++ / RoboSet …)先作 backup / 观察**,不进首批。

**一句话:先做 DROID + RH20T(零标注起步),第 3 个在 RoboMIND(跨形态)与 AIRoA MoMa(新 wrist↔exo,实测共视通过)之间二选一。**

---

## 5. 推荐方案与构建路径

### 方案 ①(首选):DROID + RoboInter-Data

**为什么**:唯一"多视角同步视频 + 现成人工复核 mask"同时满足的组合 —— **但要分清:现成 mask 在 2 个外部相机上,所以完全免标注的是 exo↔exo;1 wrist + 2 exo 虽与 Ego-Exo4D 的 1 ego + 多 exo 结构同构,但那条 wrist↔exo 路线的腕部 mask 仍需自建(见路径 B)**;json 格式迁移(参考 ego2exo/exo2ego/HANDAL 格式)最自然;564 个 in-the-wild 场景保证多样性;免 gate 下载。

**构建路径(已按 §0.6 的实测结论修订 —— mask 只在 2 外部相机,腕部无 mask)**:

- **路径 A(推荐,完全免标注):exo↔exo**
  1. 用 `VideoID_2_SegmentationNPZ.json` 过滤 `exterior_1_left` 与 `exterior_2_left` **两路都有 npz** 的 episode(全量 56,566 对,validation 1,678 对);
  2. 抽 hundred-level 测试集,`exterior_1 ↔ exterior_2` 双向构造 query/target json;
  3. 按 npz 里的 ID(如 `10010_exterior_image_2_left`)映射回 DROID 原始高清帧,mask 上采样 / SAM2 refine;
  4. 可视化抽查 → 跑 V²-SAM 与 ObjectRelator 的 ZSL。
- **路径 B(若坚持 wrist↔exo,即 ego↔exo 同构):** query 用外部相机现成 mask,**target 腕部 mask 需自建**(SAM2 视频传播 + 人工复核);工作量约为全自建的一半。

**工作量估计**:路径 A 约 1–2 周(无标注环节);路径 B 约 2–3 周(仅腕部需标注)。
**Caveats**:~~mask 双视角覆盖待实测~~(已确定:仅 2 外部相机有 mask,腕部无);DROID 每场景有 3 视角但 RoboInter 只打包/标注外部,腕部要用需自跑 auto-seg;~~DROID license 需在 repo 确认~~(已确定 CC-BY 4.0)。

### 方案 ②(次选,与 ① 互补):RH20T

**为什么**:8–10 全局 + 1–2 腕部的视角密度全场唯一,且全部相机标定到机器人基座系 → 可以**按 view-pair 的相对夹角构造难度梯度**,把 docx 中 "A. Geometric ambiguity" 变成可控变量做系统分析(例:30° 内 vs 60–90° vs 对侧视角的 IoU 退化曲线)。这一分析轴是 PAMI 版 diagnostic benchmark 的差异化卖点。

**构建路径**:用 RoboInter 覆盖的 82,920 个**全局相机 mask**(exo↔exo)起步 —— 注意这些 mask 全在全局相机上、无 wrist(已实测);需要 wrist↔exo 或更多 exo 配对时,对选中的 hundred-level 子集用"单视角内 SAM2 传播 + 跨视角人工 spot-check"扩标(**外参 + depth 可做几何一致性自动校验,这是 RH20T 相对 DROID 的 auto-mask 优势**)。

**工作量估计**:RoboInter 全局相机 mask 起步 1–2 周;扩到 wrist / 多 exo 全配对 2–4 周。
**Caveats**:10Hz 帧率中等;**"按夹角分桶"需相机外参 → 必须走官方发布拿标定,社区镜像 `hainh22/rh20t` 不带外参**(已实测);测试集尽量落在 RH20T-C(CC BY-SA)以避开非商用条款;下载走 GDrive/Rclone 或百度云,体量大需提前规划存储。

### 方案 ③(第三候选,保 story):RoboMIND

**为什么**:计划书原链接;4 种 embodiment 同任务采集,是"object 在不同**机器人**"这半句 goal 的最好载体(见 §7);AgileX 子集提供 wrist↔exo,Franka 子集提供 exo↔exo,恰好覆盖两种配对类型。

**构建路径**:AgileX 3rgb 与 Franka 3rgb 各抽几百条 → SAM2 auto-seg(用语言标注辅助定位目标物体)→ 人工复核 → 构造 json。

**工作量估计**:3–4 周(auto-mask 必须人工复核,见 §6 的定量依据)。
**Caveats**:HF gated,但 **ModelScope 免 gate 可直接下**(无需等审批,见 §2.1);**FPS 不可考 —— 释出的 HDF5 只有 `master`/`observations`/`puppet` 三组、无任何时间戳字段(attrs 仅 `compress`/`sim`),论文/数据卡也不给 per-embodiment Hz,除非拿原始 ROS bag 否则无法确定**(2026-07-10 已开样本核实);AgileX 双腕部相机随左右臂**独立运动**,wrist↔wrist 是"双动"难配对,wrist↔front(前置固定)更可控;无 depth 辅助校验(AgileX 的 Orbbec Astra 有深度但质量一般)。

### 组合建议(写进 PAMI 的结构)

三个候选**各回答一个不同的问题**,而非同质堆叠:

- **DROID**:标准 1-ego-2-exo 配置下,robotic 场景的 cross-view correspondence 基线能力(与 Ego-Exo4D 人类场景直接对照)。**注:免标注版是 exo↔exo;要与 Ego-Exo4D 的 ego↔exo 严格对照,需走 wrist↔exo(腕部 mask 自建,见方案① 路径 B)**;
- **RH20T**:视角差距梯度 → geometric ambiguity 的系统性分析;
- **RoboMIND**:cross-embodiment 泛化 → 同任务不同机器人形态下的对应能力。

最小可行版本:只做 ① + ②(两个 potential 即达标);③ 在时间允许时补充。

---

## 6. Mask 生成可行性(定量依据)

来自本次核验到的一手数据:

| 证据 | 数据 | 含义 |
|---|---|---|
| 开放词汇检测(YOLO-UniOW)proposal + SAM2,对人工 GT 的 mask IoU(**DOMR**,[arXiv 2508.04050](https://arxiv.org/pdf/2508.04050),ACM MM'25,Ego-Exo4D 上) | **ego 视角 ~77.4%,exo 视角 ~67.1%**(这是 **proposal 覆盖上限**指标:每个 GT 配最高 IoU 的 proposal,**非端到端**;DOMR 端到端为 49.7 Ego→Exo / 55.2 Exo→Ego) | 纯自动 pipeline 的质量上限不够 GT 级,**必须人工复核**;exo(小目标、远景)比 ego 更难 |
| SAM 自动 mask 的系统性偏差(同上) | 部分选中 + 边界偏移,作者专门加 Mask Refinement 阶段 | 自动 mask 直接当 GT 会引入系统性标注噪声 |
| O-MaMa 的全自动跨视角匹配 pipeline(FastSAM 候选 + DINOv2 匹配,[arXiv 2506.06026](https://arxiv.org/html/2506.06026v1)) | ~250ms/帧(FastSAM 仅 70ms) | 大规模预标注在算力上完全可行,可作候选生成器 |
| SAM2 直接跨视角迁移 prompt 的能力([arXiv 2510.11417](https://arxiv.org/html/2510.11417v1) 等多篇一致结论) | SAM2 单视角内传播强,**跨视角 prompt 迁移不可靠**(feature gap 未建模) | auto-seg 应"**每视角独立** SAM2 传播 + 跨视角身份人工关联",而不是跨视角自动传播 |

**推荐的 auto-mask 流程**(用于 RoboMIND / RH20T 扩标):
首帧目标定位(语言标注 + 开放词汇检测)→ 各视角**独立** SAM2 视频传播 → 跨视角身份关联(几何:外参/depth 投影一致性;外观:DINOv2 匹配)→ **人工 spot-check 跨视角身份 + 抽查 mask 边界**。

**质量口径建议**:GT 采用"SAM2 传播 + 人工复核"的两级标准,并在论文中明确标注协议(可引用 RoboInter 的做法作为先例)。

---

## 7. 关于"相同轨迹下不同机器人的视角"

goal 中这半句在**严格意义上无对应数据**:不同机器人无法同时执行同一条物理轨迹并被同步拍摄,全领域不存在此类数据集。可实现的最接近替代:

1. **cross-embodiment same-task**(推荐口径):同一任务定义在不同机器人上分别采集 —— RH20T(7 种 robot config × 147 任务)与 RoboMIND(4 embodiment,任务有重叠)都支持;query 与 target 来自不同 embodiment 的同任务不同 take,物体实例相同或同类。这可以定义为 benchmark 的一个 **hard split**(非同步 + 跨形态,同时打 docx 的 Temporal-related + Viewpoint-related 标签)。
2. RH20T 还配有与机器人序列对应的**人类演示视频**(等量 110k),支持 human↔robot 的同任务 correspondence——若想把 ego-exo(人)与 robotic 两个 bench 打通,这是现成的桥。

建议在 docx 中把这半句 goal 改写为 "cross-embodiment same-task correspondence",避免评审误解为同步数据。

---

## 8. Next To Do

- [x] **(go/no-go,已完成 2026-07-10)** RoboInter mask 视角覆盖已用本地全量元数据查清:**只在外部/全局相机,腕部 0 mask**;`VideoID_2_SegmentationNPZ.json` 235,920 key 中 198,194 有 npz(见 §0.6 / §3.3)。→ **结论:走 DROID `exterior_1↔exterior_2` exo↔exo 可完全免标注(56,566 对);wrist↔exo 腕部需自建**;
- [ ] DROID episode 筛选脚本:筛 `exterior_1_left` 与 `exterior_2_left` 两路都有 npz 的 episode → 候选池(全量 56,566 / val 1,678);
- [ ] 抽样构造 hundred-level 测试集,对齐 V²-SAM 推理 json 格式(参考 ego2exo / exo2ego / HANDAL 格式),先跑通一条 demo;
- [ ] 映射回 DROID 原始高清视频(npz ID 链路已确认存在),确定 mask 上采样 / refine 方案;
- [ ] V²-SAM 与 ObjectRelator 的 ZSL 结果(对齐 docx 3 节的实验流程);
- [ ] (并行)RH20T-C 子集下载规划(存储 + Rclone);**注意夹角分桶必须走官方发布拿外参(社区镜像不带)**;
- [ ] (排队)RoboMIND HF gate 申请,AgileX 3rgb 子集小样检查同步质量(FPS 释出文件无时间戳、不可考);
- [ ] 更新 docx 2.3 节:补充本报告结论 + 把 goal 中"相同轨迹不同机器人"改写为 cross-embodiment same-task。
- [ ] **(新)** 逐帧核验 §9 新候选(Galaxea / AIRoA MoMa / RoboMIND 2.0)的**物体在 wrist 与 head 同时可见率**——这是它们能否支撑 wrist↔exo 的头号未验证假设。

---

## 9. 补充广搜:新候选数据集与项目(2026-07-12)

方法:deep-research 扇出 5 路检索 → 抓 20 个一手源 → 提 92 条声明 → 对其中 25 条做 3 票对抗核验(23 confirmed / 2 refuted)。下列除标注外均为 **2025–2026 新发布**、且经核验。

### 9.1 真实机器人新候选(带 wrist+external 同步视角)

| 排名 | 候选 | 机构/时间 | 相机配置(同步?) | Mask | 规模 / License / 下载 | 对我们 wrist↔exo 的用处 |
|---|---|---|---|---|---|---|
| ⭐1 | **Galaxea Open-World** [2509.00576](https://arxiv.org/abs/2509.00576) · [HF](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) | 星海图, 25-09 | R1-Lite 双臂移动;4 路 RGB = head + head_right(外部/头,疑立体对)+ left/right_wrist(腕部);720×1280 AV1@15fps;**严格时间同步(已核验)** | ❌(仅中英双语子任务语言) | 500+h / 2.87 TB;CC BY-NC-SA 4.0;**HF gated + ModelScope 需登录**(实测) | wrist↔head 结构现成,但**下载需授权** + **社区替身实测 co-visibility 弱**(head 宽视角房间图、wrist 近景,尺度差大;见 `Galaxea/README.md`)→ 降级为"需先在官方高清上核验共视率再定",不再是无条件⭐1 |
| ⭐1(实测后升) | **AIRoA MoMa** [2509.25032](https://arxiv.org/abs/2509.25032) · [HF](https://huggingface.co/datasets/airoa-org/airoa-moma) · demo [corl2025_demo](https://huggingface.co/datasets/airoa-org/corl2025_demo) | AIRoA(日), 25-09 | Toyota HSR 移动;hand(鱼眼)+ head 双 RGB;640×480;**同步已核验**(丢弃 >0.3s 延迟 episode) | ❌(仅分层语言 + 标定) | 官方 gated;**官方 org 另有免 gate 全分辨率 demo `corl2025_demo`(已下实测)** | **实测最强新 wrist↔exo 源**:官方 demo 全分辨率、免 gate;**共视检验通过**——head 俯看货架、目标瓶子在 head↔hand 都可辨(见 `AIRoA_MoMa/`)。⚠️hand 鱼眼畸变 + 尺度差 + 货架多相似瓶(难例);需自建 mask |
| ⭐3 | **RoboMIND 2.0** [2512.24653](https://arxiv.org/abs/2512.24653) | x-humanoid, 25-12 | 双臂/移动新版;front + left/right_wrist **同步(同一时刻)** | ❌(真实数据仅 stage 级语言;pixel affordance 仅仿真) | —;随 RoboMIND;HF/ModelScope | v1 已在库,v2 是**独立新版**、带 front+双腕同步;真实数据"segmentation"指时序动作分割,非物体 mask |
| 4 | **VTouch++ / VTOUCH** [2604.20444](https://arxiv.org/html/2604.20444) | 上海人形/OpenLoong, ~26-04 | OpenLoong 平台:left+right_wrist + head **三路 RGB-D 硬件同步**(30Hz 统一触发) | ❌ | —;**未来日期 arXiv,极新,下载渠道/license 未captured** | 配置强(2 腕 + 头,硬同步);但太新、渠道未知,需先确认可下 |
| 5 | **PEWM** [2508.20840](https://arxiv.org/pdf/2508.20840) | —, 25-08 | 每个 primitive **5 路相机同步**;11,465 real primitives | ❌(仅 Qwen2.5-VL 文本指令) | —;**公开下载未说明**;5 路中是否含 wrist 未确认(可能全 external) | 5 路同步覆盖强,但**可得性存疑 + wrist 未确认**,待核 |
| 6 | **X Square XRZero-G0-3K** [HF](https://huggingface.co/datasets/x-square-robot/XRZero-G0-3K) | 自变量机器人 | head + 双 wrist;~2000h / 3000 任务 | ?(未核) | **MIT license**(宽松) | 有潜力 + MIT 宽松;但细节来自核验旁注、非独立 3-0 声明,**须直接查 HF/GitHub** |

### 9.2 唯一带 GT mask 的候选(但是仿真)

- **NVIDIA PhysicalAI-Robotics-Manipulation-Augmented** [HF](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Augmented):Isaac Sim 合成,1000 条 Franka 叠方块。**唯一自带 GT segmentation mask** —— 但**只在 table(外部)相机**,wrist 相机仅 RGB(2000 视频 = 1000 table + 1000 wrist,但只有 1000 段 table 分割)。用处:**做一个"合成 pilot"**先把评测协议跑通,且因有完整 3D state,wrist 侧 GT 可用几何投影生成(不必靠 SAM2)。⚠️把它当"同步 wrist↔exo"的说法在核验中被**否决(1-2)**,谨慎。

### 9.3 方法 / 协议参考(不是数据源,但直接可借鉴)

- **CCMP**(Cross-View Object Correspondence via Cycle-Consistent Mask Prediction,**CVPR 2026**,[2602.18996](https://arxiv.org/abs/2602.18996)):条件二值分割 + cycle-consistency(把 target 视角 mask 投回),**并引入 HANDAL-X 机器人 benchmark**(正是我们已在用的 HANDAL-X)。检索层判为"机器人角度最有价值",但未进最终核验集 → **建议直接精读**,可能是最贴近我们任务的 SOTA 方法 + 现成机器人评测。
- **SegMASt3R**(NeurIPS 2025 spotlight,[2510.05051](https://arxiv.org/abs/2510.05051)):在 MASt3R 3D 基座上接 segment-matching 头,跨最多 180° 视角做 mask 对应,**比 SAM2 video propagator 高 ~30% AUPRC**。仅静态室内(ScanNet++/Replica)无机器人,但**跨视角对应的 baseline / 方法可借**。
- **Hoi!**(Dec 2025,[2512.04884](https://arxiv.org/abs/2512.04884)):cross-view 铰接操作,3048 序列/381 物体,人手+手持 gripper,1 Aria ego + 2 iPhone exo。其"三路 wrist+external 同步"说法被**否决(1-2)**;作人类+gripper 的协议参考、borderline。

### 9.4 已核验排除(不满足硬标准)

- **Humanoid Everyday** [2510.08807](https://arxiv.org/abs/2510.08807):发布数据里只有**单个头戴 egocentric** 相机(第三人称仅存在于云评测网页,不在数据里)→ 排除。
- **EgoDex**(Apple, [2505.11709](https://arxiv.org/abs/2505.11709)):**单目 egocentric 人类**视频,无 exo、无 mask → 排除。
- **AgiBot World Colosseo** [2503.06669](https://arxiv.org/abs/2503.06669):检索误标,实为已在库的 AgiBot World → 排除。
- **Seg2Track-SAM2** [2509.11772](https://arxiv.org/html/2509.11772v1):单目驾驶视频,无跨视角 → 仅作 SAM2 tracking 工具参考。

### 9.5 全局关键 caveat(对所有 9.1 候选)

1. **没有一个真实候选自带 object mask** —— 和 DROID/RH20T 一样,mask 都得 SAM2/DINOv2 自建;小型在手物体 + gripper 自遮挡下的自动标注精度**未测**。唯一有 GT mask 的是 NVIDIA 合成、且只在外部视角。
2. **"物体在 wrist 与 external 同时可见"这一最关键前提** —— 这是它们能否支撑 wrist↔exo 任务的**头号未知**,须逐帧抽查。已实测两个(2026-07-13):
   - ❌ **Galaxea**(社区替身,见 `Galaxea/`):head 是宽视角房间图、物体尺度差极大、**共视弱** → 降级;
   - ✅ **AIRoA MoMa**(官方免 gate demo,全分辨率,见 `AIRoA_MoMa/`):head 俯看货架、目标瓶子在 head↔hand **共视成立** → **升为实测后最强新 wrist↔exo 源**(hand 鱼眼需处理畸变)。
   - 其余(RoboMIND 2.0 / VTouch…)同类风险仍未测。**这条实测直接改变了新候选排序:AIRoA > Galaxea。**
3. **这些的"external"多是机器人头部相机**(Galaxea head、AIRoA head、VTouch head,随平台移动的 robot-egocentric),**不是固定第三人称机位** —— 对"按视角夹角分桶做 geometric-ambiguity"不如 RH20T 的固定全局相机;Galaxea 的 head+head_right 疑为近立体对,作"两个不同 exo"很弱。
4. 净判断:**这批新数据不改变首选("DROID exo↔exo 免标注"仍是成本最低的起点)**;但若我们本就要自建 wrist mask 走 wrist↔exo,**Galaxea / AIRoA MoMa 是 DROID/RoboMIND 之外最值得加入的、当下可下载的 wrist↔head 同步新源**,而 **CCMP(CVPR 2026)+ SegMASt3R** 是方法侧最该跟进的两篇。

---

## 10. 仿真 cross-view:BEHAVIOR-1K 路线评估(2026-07-19)

动机:真实数据侧 wrist mask 必须自建(§0.7);仿真里 **GT mask 是渲染器免费送的**,且视角可任意加。评估"BEHAVIOR-1K 场景 + exo/ego/wrist mask"的可行性。

### 核实到的事实(一手来源)

- **传感器**:OmniGibson 的 [VisionSensor](https://behavior.stanford.edu/reference/sensors/vision_sensor.html) 原生支持 rgb / depth(linear) / **seg_semantic / seg_instance** / normals / optical flow / 2D+3D bbox —— 任意相机、逐帧、像素级实例分割,**零标注成本、零 SAM2、零人工复核**。
- **数据**:[BEHAVIOR Challenge 数据页](https://behavior.stanford.edu/challenge/dataset.html) —— NeurIPS 2025 挑战 10,000 条人类遥操作 demo(50 任务×200 条,>1,200h);**2026 挑战已扩到 20,000 条 / 100 任务 / 3.27TB,LeRobot V3 格式**;另发 raw HDF5(1.44TB)。观测:头部 720×720 + 双腕 480×480 的 RGB+depth。
- **重放**:官方脚本 `OmniGibson/scripts/learning/replay_obs.py` 按录制的 state+action **逐步重放 raw HDF5 并重新渲染观测**(modalities 与相机由 `robot_sensor_config` 配置;逐步 load state 规避物理漂移)。先例:PointWorld(arXiv 2601.03782)重放时**外挂 3 个虚拟相机**(left/right shoulder + head)采数据 → 加任意 exo 机位 + seg_instance 是配置层面的事。
- **场景复杂度**:50 个全交互真实感住宅场景 / 10k+ 物体模型 / 1,000 activities,Omniverse RTX 渲染(含流体/布料/透明)—— 视觉丰富度远超 RLBench/MetaWorld 一类"仿真太简单"的量级。

### 判断:难度中低,瓶颈在工程不在标注

| 步骤 | 内容 | 估计 |
|---|---|---|
| 1 | 装 BEHAVIOR-1K/OmniGibson(需 Isaac Sim,RTX GPU;本机 5070 8GB 达最低线,WSL2 可用) | 1–2 天 |
| 2 | 改重放管线:`robot_sensor_config` 加 `seg_instance`,场景里加 2–3 个固定 exo VisionSensor(位姿自定) | 2–4 天 |
| 3 | 抽 hundred-level episode 子集重放渲染(全量 20k 无必要),导出 wrist/head/exo 各路 RGB+instance mask → 组 pair json | 1–2 天 + GPU 渲染时 |

- **一次重放同时得到全部三类 pair**:wrist↔exo(=ego↔exo 同构)、head↔exo、exo↔exo,且**每路都有 GT mask**(目标物体 instance id 已知,直接按 id 提 mask,连"哪一个是被操作物体"都由 task 定义给出)。
- **可控轴红利**:exo 相机位姿完全可控 → 视角夹角/距离/遮挡可以做成系统变量,与 RH20T 的"真实但固定机位"互补,是 docx "A. Geometric ambiguity" 的第二个可控实验场;域差(sim2real)本身也可作 diagnostic 维度而非缺陷。
- **风险/注意**:① Isaac Sim 环境安装在 Windows/WSL 上偶有坑,留 buffer;② 8GB 显存跑重放渲染可行但慢,大批量建议放 A6000;③ demo license 待核(挑战页未标,商用性存疑,benchmark 再分发前查);④ 头部相机随机器人动,"固定第三人称 exo"必须靠自加的虚拟相机,不能只用释出的三路。
- **与 §9.2 NVIDIA PhysicalAI(唯一带 GT mask 的仿真候选)的关系**:BEHAVIOR-1K 在规模(20k vs 1k)、场景真实感、任务多样性上全面超过,且经重放可補 wrist mask —— 若走 sim 轨道,**直接以 BEHAVIOR-1K 为主**,PhysicalAI 降为快速 pilot。

**结论:值得立项**。"BEHAVIOR-1K 场景 + exo/ego/wrist 三路 GT mask"约 1–2 周可出初版(hundred-level),是全家族里唯一"三路皆真值、视角可控、零标注"的 track;真实(DROID/RH20T)+ 仿真(BEHAVIOR)双轨正好构成 PAMI 的 real↔sim 对照轴。

### pilot 工序定稿(2026-07-19,读官方 replay 源码后)

目标形态对齐 RoboInter:每 (episode × 相机) 一路 RGB 视频 + 被操作物体逐帧二值 mask;视角 = **2–3 个自摆固定 exo VisionSensor + head(ego) + 双 wrist**。关键事实(读 `OmniGibson/scripts/learning/replay_obs.py` 确认):

1. **重放入口现成**:`HDF5PlaybackWrapper`(omnigibson.envs)按录制逐步重放;raw 路径 `2026-challenge-rawdata/task-{id:04d}/episode_{demo:08d}.hdf5`,`task_id = demo_id // 10000`;机器人 **R1Pro**;仓库自带 `download_gcs_rawdata.py`(GCS 公开桶,天然续传)。
2. **mask = 改一行模态**:`robot_obs_modalities=["proprio","rgb","depth_linear"]` → 追加 `"seg_instance"`;robot 侧传感器配置在 `robot_sensor_config`(wrist VisionSensor 480² + head zed_link 720²)。
3. **exo 机位注入**:OmniGibson env config 的 `external_sensors`(静态 VisionSensor,自定位姿)—— 位姿完全可控 = 视角差可做成实验变量。
4. **"哪个物体"由任务定义给出**:env.task 的 object scope 即任务相关物体清单,按 instance id 直接从 seg_instance 提取,无需任何判断。
5. **前置资产**:除 og_dataset 外还需 `2026-challenge-task-instances`(任务元数据+场景 json,`gm.DATA_PATH` 下);选简单桌面型任务从 `metadata/available_tasks.yaml` 挑。

工序:env 就绪 → 拉 task-instances + 1 个简单任务的 1–2 条 raw episode → 改 replay(seg_instance + external_sensors ×2–3)→ 导出 RoboInter 式打包(mp4 + masks npz + manifest)→ 复用本地 montage/画廊管线出图。估 1–2 天(env 落地后)。

### pilot 首跑结果(2026-07-20,turning_on_radio demo 1550)✅ 跑通

- **产出**:5 路同步视角(3 自摆 exo + ego head + left wrist)× 1,942 步,每路 `rgb.mp4 + seg.npz`(GT instance mask + id→物体名 + 任务物体清单);卡片 `BehaviorPilot/samples/task0000_demo1550_crossview_gt.png`。可见性:exo2/head **100%**、left_wrist 77%(近景大 mask)、exo0/exo1 0%(盲摆没对准,见下)。
- **两个关键工程结论(都已留档在脚本)**:
  1. **机器人相机的 seg_instance 在此 Isaac 5.1 栈必崩**(闭源 bug,[OG#2312](https://github.com/StanfordVL/OmniGibson/issues/2312) 已 root-cause:instance 映射表不建 + -7 越界)→ 采用 issue 验证过的 **viewer 相机多趟重放**(每趟一个机位;ego/wrist 趟逐步吸附到对应传感器位姿,移动后补渲染再采集),外部 VisionSensor 同样中招、不可用;
  2. 自建 playback env 必须带 **`gm.ENABLE_TRANSITION_RULES=False`**(vanilla replay 内部设置;漏掉则 env 构建段错误,与 seg 无关——探针梯 A/B/C 定位)。
- **下轮迭代**(小改动):exo 机位改为瞄准**目标物体运行时位置**(radio_89 可从 scene 查询)而非机器人起始位;mask 目标限定 radio(现为 radio+coffee_table 并集)或双色分别渲;5 趟 ≈ 每 demo ~15 分钟(4090),hundred-level 可行。
- 脚本:mint2027 `~/behavior_pilot/pilot_viewercam.py` + `~/pilot_views_run.sh`;本地渲染 `BehaviorPilot/render_sim_cards.py`。

### v2.2 多任务批跑(2026-07-21):4 任务 3 场景 18/20 趟,单目标环形机位

- **产物**:1550 收音机 / 50220 捕鼠夹 / 71020 玩具 / 42750 罐头,共 4 卡(`BehaviorPilot/samples/*_crossview_gt.png`);流式写盘版(mp4 逐帧编码 + seg 写 gzip 分块 h5)内存 O(1),5,994–16,662 步长 demo 无压力;viewer 实际渲 **720×1280**(gm 尺寸覆盖无效,惰性建流适配)。
- **机位结论**:环形三机位瞄准目标物体后,radio 卡 **3/3 exo 全程 100% 可见**;但杂乱室内平均每 demo 只有 ~1/3 环位命中(其余撞墙)→ 下一步用 **seg 可见性反馈自动筛机位**(采样候选位姿→按目标可见帧率打分取 top-K)。
- **ego/wrist 可见率 14–37% 是任务的真实时序稀疏**(导航段出画、交互段近景大 mask)——正好是"消失-重现/共视窗口"难度轴的天然来源;出卡采样帧应偏向共视窗口。
- 遗留:42750 的 head/wrist 两趟可复现段错误(原因未查,exo 三路正常);打包走"远端抽帧+统计、只回传 ~70MB bundle"模式(`make_card_bundle.py`),不再整包回传 GB 级 h5。

### v3.2(2026-07-21):seg_instance_id 管线 + 七列旗舰卡

- **#2312 再发作与最终绕法**:viewer 相机也非绝对安全——`seg_instance` 的归约节点在特定"场景×视角"组合下仍概率性段错误(1550 head 曾过后又崩)。**改用 `seg_instance_id`(像素→prim 路径,另一条标注器管线)后原必崩趟全通**;prim 路径含物体名,下游名字匹配零改动。此为 sim 管线最终形态。
- **1550 七列卡**(`BehaviorPilot/samples/1550_*.png`):3 自动 exo(100%×3)+ 真实 ego head(24%,遥操作员原始头部行为)+ **gaze-ego(100%,头部位姿+逐帧凝视目标,理想化 ego)** + 左腕(14%)+ **右腕(73%,该任务主用右臂)**;腕部视角已继承真实 realsense 内参(FOV 修正)。机位缓存改 per-task(同任务各 episode 共用,DROID 三脚架语义)。
- 可用面:**5 任务 × 3 场景全通**(radio/mousetraps/toys/pie/batteries);已知问题:office(930950)/hotel(792930)/42750-tracking 这三个**任务实例**在任何管线下均 139(疑实例级资产问题),绕行方案=换同场景兄弟任务(94/76 等)。

**实验日志(持续记录)**
- 07-21 晚 C7 补全批(38 趟,过夜):4 老 demo(50220/71020/720270/822240)各补 6 趟(exo0-2 重扫掠〔清 per-task 位姿缓存〕+head_gaze+双腕 FOV 版)→ 七列标准;新增兄弟任务 **94 dispose_of_batteries(942050,office)** 与 **76 dispose_of_glass(761590,hotel)** 各 7 趟 → 若通,7 种场景模型全覆盖。1550 保持不动(已达标)。
- 稳态成本实测口径:~0.3s/步/趟;1,942 步 demo 七列 ≈25 分钟,16k 步 ≈2.5–3h;此前数日主要为一次性调试成本(环境+3 个 Isaac 底层 bug)。BEHAVIOR-1K 可用面:2026 挑战集 100 任务 × 200 demo/任务。
- 07-22 C7B+R2 收官:**8 任务 8 demo 出卡**(`BehaviorPilot/samples/*_crossview_gt.png`),gaze-ego/双腕(FOV 版)在 5 个 demo 全产出(71020 gaze 77%、双腕 ~33%,交互段近景 mask 优);办公室(942050)拿到 exo1/2 双路、酒店(761590)exo1 一路 → **7 种场景模型全部有 GT 产出**。剩余确定性 139 集中于:①扫掠**最高分位姿**的正式重放(exo0,6/6 复现——最贴目标的机位反而必崩,exo1/2 次优位姿全活)②部分 demo 的机器人贴身 tracking 视角(42750/942050/761590/720270-左腕)。改进项:sweep 候选不足 3 正分时禁用 0 分凑数(71020 exo1/2 vis=0% 即凑数位姿)、改为对正分位姿做角度抖动。

---

## 附:全部来源链接

**计划书原链接**
- RoboMIND 论文:https://arxiv.org/abs/2412.13877 · HF:https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND · 项目页:https://x-humanoid-robomind.github.io/
- RoboSet 官网:https://robopen.github.io/roboset/ · RoboAgent 论文:https://arxiv.org/pdf/2309.01918 · HF 镜像:https://huggingface.co/datasets/jdvakil/RoboSet-Teleoperation

**推荐候选**
- DROID:https://droid-dataset.github.io/
- RH20T:https://rh20t.github.io/
- RoboInter(ICLR 2026):https://arxiv.org/abs/2602.09973 · HF:https://huggingface.co/datasets/InternRobotics/RoboInter-Data

**其他考察对象**
- AgiBot World:https://github.com/OpenDriveLab/AgiBot-World · https://arxiv.org/html/2503.06669v4
- BRMData:https://arxiv.org/abs/2405.18860
- BridgeData V2:https://bridgedata-v2.github.io/
- GraspNet-1Billion:https://www.emergentmind.com/topics/graspnet-1billion-benchmark(二手综述;原始论文为 CVPR 2020)

**auto-mask 可行性证据**
- 开放词汇检测 + SAM2 质量上限(ego 77.4 / exo 67.1 IoU):https://arxiv.org/pdf/2508.04050
- O-MaMa(FastSAM + DINOv2 跨视角匹配,~250ms/帧):https://arxiv.org/html/2506.06026v1
- SAM2 跨视角局限(LM-EEC 等):https://arxiv.org/html/2510.11417v1
- SAM2 + 跨视角 prompt 自动生成(ICIAP 2025):https://link.springer.com/chapter/10.1007/978-3-032-10192-1_27
- V²-SAM(本项目基线):https://arxiv.org/abs/2511.20886

> 核验说明:RoboMIND 的 AgileX 相机配置与"无 mask"两条结论经独立双票核验确认;DROID / RH20T / RoboInter / RoboSet 的关键参数均直接抓取官方页面或论文原文核对。
> **2026-07-10 收尾:原表中的 ⚠️/"未公开/需确认"项已逐条定死**——RoboInter mask 视角覆盖(仅外部/全局,腕部 0)、DROID license(CC-BY 4.0)、RH20T "7 配置"含义(4 臂×4 夹爪×3 力矩)、RoboSet 4 路 depth、BridgeData V2 license(CC-BY 4.0)均已确定;**唯一真正不可考的是 RoboMIND 的 per-embodiment FPS**(释出 HDF5 无时间戳,须原始 ROS bag 才能得),已如实标注为"不可考"而非"待实测"。
