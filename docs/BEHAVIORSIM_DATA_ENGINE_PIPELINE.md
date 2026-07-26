# Goal-Conditioned Simulator-Native Cross-View Data Engine

> Status: wording approved and locked on 2026-08-25. This document is the
> canonical BehaviorSim process description for the paper, dataset README, and
> pipeline figure. It describes the production path used for
> [Travor278/behaviorsim-crossview-seg](https://huggingface.co/datasets/Travor278/behaviorsim-crossview-seg),
> not every experimental branch retained in the repository.

## 1. 一句话概括

我们从 BEHAVIOR-1K 机器人操作 episode 出发，将任务级 BDDL 目标解析为场景中的精确物理实例，在每个采样时刻直接恢复同一份 simulator state，为目标物体合成受房间几何、物体尺度和运动轨迹约束的 ego/exo 视角，再通过 `seg_instance_id` 提取与精确 prim 对应的像素级 mask；系统先按照实测可见性生成候选 card 和时刻，人类只负责检查目标语义、跨视角有效性与时刻价值，而不描画 mask，最后构造可审计的双向跨视角 relation。

~~~text
BEHAVIOR-1K episode + BDDL goal
  → Goal-conditioned target instance discovery
  → State-restored synchronized replay
  → Target-centric adaptive camera synthesis
  → Simulator-native instance mask extraction
  → Visibility-aware candidate construction
  → Human-reviewed sparse moment selection
  → Bidirectional relation construction and audit
~~~

这条流程的关键不是“在仿真图像上运行一个自动分割模型”，而是把**任务语义、精确 simulator instance、多视角同步状态、目标驱动相机和人工语义筛选**连接成一条数据引擎。像素边界来自仿真器，人工决定哪些实例、视角和时刻具有可靠的跨视角监督价值。

## 2. 角色与职责边界

BehaviorSim 有意将任务语义、物理状态、相机生成、像素级监督和最终样本选择解耦。

| 参与者或模块 | 核心职责 | 不负责的内容 |
|---|---|---|
| BEHAVIOR-1K / BDDL task specification | 提供 activity、goal literal、scene scope、manipulandum 类型和 destination 关系 | 不直接提供最终的二维 relation，也不保证所有异常 episode 都能自动得到正确 target |
| Target resolver | 将 BDDL 中的目标类型解析为当前 scene scope 中的具体实例；保存 target name、prim identity 和备选 manipulanda | 不根据图像外观生成近似 box 或类别级 mask |
| OmniGibson / Isaac Sim replay | 在指定采样时刻恢复记录的 simulator state，并为不同相机渲染同一物理配置 | 不通过分别重新执行动作来近似同步 |
| Camera synthesis module | 构造受目标轨迹、房间边界、物体尺度和可见性约束的 exo 候选视角，并生成受前向视锥约束的 `head_gaze` | 不预先假设某个固定相机一定可用 |
| Simulator instance annotator | 输出 `seg_instance_id` 和 pixel-to-prim 映射，据此提取精确目标实例 mask | 不把同类别的多个 sibling instance 合并成一个语义区域 |
| Automatic candidate builder | 统计逐视角 mask 面积和可见率，定位 manipulation window，选择时间分散的候选时刻，并保留最佳 ego/exo 视角 | 不把自动排序结果直接宣布为最终 benchmark |
| Human reviewer | 检查目标是否正确、各视角是否对应同一实例、mask overlay 是否合理、渲染是否退化，并选择值得保留的稀疏时刻 | 不手工描边或修改 simulator-native mask |
| Deterministic exporter | 按人工 verdict 构造 ego→exo、exo→ego 和 exo↔exo relation，编码 COCO RLE，并执行统计与完整性审计 | 不把未审核、不可见或缺失的 relation 静默补齐 |

与 DROID companion dataset 不同，BehaviorSim 没有 Codex-guided SAM3、RoboInter fallback 或人工涂抹 mask。二者共享“以物理实例为中心、经人类验证、按可见性构造 relation”的思想，但像素监督来源不同。

## 3. 详细数据生成流程

### 3.1 Episode 输入与 task-conditioned card 单元

流程读取 BEHAVIOR-1K / OmniGibson 中记录的机器人操作 episode。每个 episode 包含一系列序列化 simulator states，以及对应 activity 的 BDDL problem definition。生产系统不是简单地为“一个 episode”导出一个样本，而是以 **episode × target physical instance** 为 card 单元。

当一个 BDDL goal 同时包含多个 manipulanda，例如要求机器人依次移动多个物体时，系统记录所有能够解析到 scene scope 的具体实例。完整的实例 ID buffer 只需渲染一次；随后可以针对不同 target prim 分别查询 mask，因此一个 episode 能在不增加额外 GPU 渲染成本的情况下生成多个 target-specific cards。

这一设计避免把“episode identity”和“object identity”混为一谈。两个 card 可以共享相同的物理状态和 RGB 帧，但只要目标 prim 不同，它们就是两个不同的跨视角实例对应问题。

### 3.2 Goal-conditioned target instance discovery：从任务语义到精确 prim

系统首先解析 activity 对应的 BDDL goal literal。对于 `inside`、`ontop`、`under`、`nextto`、`onfloor`、`filled`、`covered`、`overlaid`、`attached`、`draped`、`contains` 和 `saturated` 等操作关系，第一个对象参数通常是被操纵的物体，后续参数则是 container、support surface 或 destination。

例如：

~~~text
(inside ?pumpkin.n.01 ?cabinet.n.01)
~~~

系统将 `pumpkin.n.01` 视为 manipulandum 类型，将 `cabinet.n.01` 记录为它的 destination 类型，然后在当前 simulator scope 中解析具体的 pumpkin instance。这里选择的是**一个确定的 scene object / prim**，而不是 “pumpkin” 这一语义类别。

解析顺序为：

1. 如果生产计划显式提供 `PILOT_TARGET`，则优先匹配指定实例；
2. 否则依照 BDDL goal 中的 manipulandum 顺序，选择第一个存在具体 scene instance 的类型；
3. 如果 goal 解析失败或无法命中，则从非 floor、wall、ceiling、room 等 fixture 的 scope 对象中选择更可能是 manipulandum 的候选；
4. 最终 fallback 按 AABB 体积选择较小对象，而不是按名称排序，减少误把 refrigerator、microwave 或 cabinet 等 destination fixture 当成目标的风险。

系统保存 target scope key、实例名称、初始位置、prim path 映射，以及 goal 中解析出的全部 manipulanda。这个 target identity 会贯穿相机生成、mask 提取、候选 card、人工审核和关系导出，不能在不同视角间切换到同类别的另一个实例。

### 3.3 State-restored synchronized replay：无时间漂移的多视角回放

不同视角以独立 viewer-camera pass 渲染，但它们不分别重新执行机器人动作。对于需要渲染的采样时刻，系统直接调用 simulator state restoration，将 episode 中记录的序列化状态载入 OmniGibson，然后执行必要的 simulator step 和 settle renders。

在 fast-replay 路径中，系统只访问均匀采样的状态，而不顺序执行中间绝大多数动作：

~~~text
recorded episode state at raw step t
  → load_state(serialized_state[t])
  → simulator step
  → settle renders
  → position the selected camera
  → render RGB + seg_instance_id
~~~

这种方式有三个作用：

- 不同相机看到的是同一条记录中的同一物理配置，而不是多个 action replay 的近似结果；
- 避免长 episode 中因动力学重新执行造成的累积偏差和跨视角时间漂移；
- 只渲染约定数量的采样状态，降低长序列的 GPU 成本和 moving-camera segmentation pipeline 的累积崩溃风险。

不同 view pass 在重渲染、失败恢复或历史预算变化后可能具有不同的本地帧数。card builder 因此不直接复用某个 view 的局部数组下标，而是先把候选时刻表达为 reference timeline 上的归一化时间比例，再映射到每个 view 自己的本地时间轴。这样，同一 card row 表示相同的 episode progress，而不是碰巧相同的数组索引。

### 3.4 Target-centric adaptive camera synthesis：面向目标的自适应相机

#### Ego view

`head_gaze` 使用机器人头部传感器的真实安装位置和基础朝向。它不会在整个 episode 中机械地锁定当前 card target，而是从 BDDL 解析出的 manipulanda 中选择离机器人手部最近的活动对象，并用滞回规则避免在相邻物体之间频繁跳变。

当活动对象接近手部时，gaze point 会从物体本身适度向 BDDL destination 方向移动，用来近似操作过程中“视线先于手部到达目标位置”的行为。最终朝向受机器人头部真实前向的有限视锥约束；目标位于身后时，相机不会穿过机器人头部反向凝视。

因此 `head_gaze` 是一种 goal-conditioned active robot-head view，而不是人类眼动的真实复现。它提升操纵目标进入 ego 画面的机会，但仍保留机器人本体朝向和可实现观察范围的限制。

#### Exo views

最终生产路径采用 **over-generate, render, and select by measured visibility** 的策略。生产 runner 设置 `PILOT_EXO_RING=8`，为每个可接受 episode 尝试渲染最多八个 target-centric exo 候选视角，之后才选择最有用的两个。

候选相机构造包含以下约束：

- **Target-centred aim。** 使用 target AABB centre 而不是 mesh origin，避免物体 pivot 偏离真实几何中心；
- **Trajectory-aware segmentation。** 在 episode 内采样 target trajectory，并按照位置方差寻找运动切分点；exo 在每个 segment 内保持静态，只在 segment 边界重新瞄准该段目标质心；
- **Room-bound placement。** 从 BDDL scope 中的 floor object 获得房间 AABB，沿每个方位寻找仍位于房间边界内的最大可用半径，避免把相机放入墙体或室外；
- **Object-scale distance。** 根据 target AABB 对角尺寸缩放 near/far radius，使 ice cube、pen 等小物体获得更近视角，同时避免大物体被过度裁切；
- **Depression-angle height。** 根据目标距离和俯视角计算相机高度，并设置绝对高度上限，兼顾 drawer、sink 等低处目标和室内 ceiling；
- **View diversity。** 候选方位覆盖环绕目标的不同 azimuth，并交替使用 near/far distance，减少相邻机位共享同一遮挡的概率；
- **Measured selection。** 完成真实渲染后，以 target mask 的 median pixel count 为主、visible fraction 为辅排序，要求最低可见率且 median mask 非零，最终保留最佳两个 exo view。

这里的重点是：最终选择依据**真实 simulator mask 的实测可见性**，而不是一个可能与最终渲染不一致的廉价几何代理。

#### 最终生产路径与保留的实验分支

`pilot_viewercam.py` 中还保留一条 3 radii × 12 azimuths 的几何评分分支，它使用 frustum、AABB projected area、PhysX raycast 和 robot framing 预测候选视角质量。这条路径来自早期 geometric placement design，并对后续 room constraint、trajectory-aware aim 和相机尺度设计有重要影响。

但当前发布数据的 v5 runner 明确启用了 `PILOT_EXO_RING=8`。因此，论文和 dataset README 应将最终方法描述为“target-centric candidate over-generation followed by measured-visibility selection”，不能把 36-pose analytic ranking 写成已用于当前数据的最终选择器。

### 3.5 Simulator-native instance mask extraction：精确实例级监督

每个 view pass 同时输出 RGB 和 `seg_instance_id`。后者为每个像素提供一个实例 ID，伴随的 `id_map` 将整数 ID 映射到 simulator prim path。系统将 BDDL-resolved target instance name 与 prim-path label 精确匹配，得到目标 ID 集合：

~~~text
target physical instance
  → exact simulator name / prim path
  → seg_instance_id integer
  → pixels whose ID equals the target ID
  → binary target mask
~~~

之所以使用 `seg_instance_id` 而不是类别级 semantic segmentation，是因为同一个场景可能存在多个 pizza、plate、shoe 或 can。类别 mask 会把 sibling instances 合并，而 prim-path based instance mask 能维持物理实例身份。

如果 exact instance match 存在，系统不会再对其他同类别实例做模糊 substring 合并。仅在处理旧 semantic-format artifact 且不存在 exact match 时，才允许严格的 whole-label category fallback；当前生产渲染固定使用 `seg_instance_id`。

RGB 按 H.264 流式写入，instance-ID buffer 按 gzip-compressed HDF5 流式写入，避免把长 episode 的全部帧留在内存。输出尺寸以第一帧的实际 viewer shape 为准；当前发布快照保持原生 720×1280 mask，而不是缩放到参考 benchmark 的方形分辨率。

系统还使用 segmentation buffer 检测退化渲染。当下采样后的整帧只包含极少数 ID 时，该帧通常是未正确恢复的黑帧或无效 render。系统执行额外 settle renders 并重新读取 annotator；card sampling 也会再次排除仍然退化的时刻。

这一阶段没有 foundation-model prediction、box-to-mask approximation 或人工描边。最终 mask 边界就是目标 simulator instance 的可见像素集合。

### 3.6 Visibility-aware candidate construction：自动生成可审查候选

完成各视角渲染后，card builder 针对每个 target instance 统计整个采样时间轴上的逐视角 mask pixel count。系统不是把 episode 中均匀分布的随机帧直接交给人工，而是先定位真正发生操作且目标可见的时间窗口。

候选时刻的构造逻辑包括：

1. 以至少 200 个 target pixels 作为“具有实质可见性”的候选信号；
2. 统计每个时刻有多少 view 同时达到该阈值；
3. 结合多视角总 mask 面积和近距离 ego / manipulation view 的可见性进行排序；
4. 使用首次和末次可见时刻形成 manipulation window，并保留少量前后文；
5. 排除所有视角都不可见的时刻和任一关键 view 明显退化的时刻；
6. 在 manipulation window 内选择最多 8 个高分时刻，同时施加最小时间间隔，减少近重复帧；
7. 通过归一化时间比例把 reference time 映射到各 view 的本地帧索引；
8. 为候选时刻保存原生 RGB、binary mask、visible fraction、median target pixels 和 target metadata。

自动候选阶段的作用是提高人工审核密度：优先展示目标可见、跨视角信息充分且分布在操作过程不同阶段的帧。它不是最终 acceptance gate。

### 3.7 Human-reviewed sparse moment selection：人工复核稀疏时刻

review package 为每个 card 提供两级证据：

- 最多 8 个自动候选时刻的全分辨率 RGB，以及可以独立显示、隐藏和调节透明度的 target-mask overlay；
- 每个保留 view 的完整 scrub video，其中 simulator mask 以半透明颜色叠加，便于人工从自动候选之外寻找更有价值的时刻。

人工审核的内容包括：

- BDDL target label 与画面中的实际被操作物体是否一致；
- ego 和 exo 是否确实展示同一个物理实例；
- instance mask 是否对应正确 prim，而不是同类邻近物体；
- 目标面积是否足够支持可靠的跨视角判断；
- 是否存在严重遮挡、画面裁切、黑帧、渲染退化或时间错位；
- 该时刻是否包含有意义的 manipulation state，而不只是导航、静止或空场景；
- 两个 exo view 是否提供互补而不是近重复的观察信息。

review verdict 支持：

- `card = drop`：整个 card 作废；
- `keep_frames`：正选择模式，只导出人工明确选择的时刻；
- `drop_frames`：负选择模式，删除指定坏帧；
- `add_frames`：从完整 scrub video 中选择自动 sampler 未提供的时刻。

从 scrub video 新增的时刻必须先回到保留的 RGB video 和 segmentation HDF5 中 materialize，对每个 view 按归一化时间映射提取 JPEG 和精确 instance mask，之后才能进入导出。未审核 card 默认排除；“尚未看过”不等于“已经通过”。

人工在此阶段负责的是**目标语义、跨视角一致性、渲染质量和时刻价值**。人工不会重画 simulator-native mask，也不会用视觉模型生成一个替代 mask。

### 3.8 Bidirectional relation construction：按可见性构造有向关系

自动 view selector 优先使用可见率达到阈值的 `head_gaze` 作为 ego，并在所有成功渲染的 exo views 中按照 median target pixels、visible fraction 排序，最多保留两个 exo。最终关系由可用视角决定，而不是强制每个 card 具有相同数量的输出。

对于存在 ego 和 exo 的 card，系统分别导出：

~~~text
head_gaze → exo_i   as ego2exo
exo_i → head_gaze   as exo2ego
~~~

对于存在两个可用 exo 的 card，系统额外导出：

~~~text
exo_a → exo_b
exo_b → exo_a       as two directed exo2exo relations
~~~

对每个有向 relation，exporter 在人工选择的稀疏时刻中寻找第一个 source 和 target view 都具有非空 target mask 的时刻。source mask 写入 `prompt.first_frame_anns`，表示跨视角查询；同一时刻的 target-view mask 写入 `objects.0.segmentation`，表示需要预测的 ground truth。目标相机所有人工选择时刻按时间顺序记录在 relation 的 `video_path` 中。

binary mask 使用 compressed COCO RLE 编码，`size` 明确保留 `[H,W] = [720,1280]`。不存在双方共同可见时刻的 view pair 不会被导出，也不会通过复制另一时刻或另一实例的 mask 来凑齐 relation。

### 3.9 Auditable export and validation：可追溯导出

导出阶段通过确定性规则将 review decisions、bundle metadata、RGB 文件和 binary masks 转换为三个 relation JSON。审计至少覆盖：

- relation key 与 `video_id` 是否一致；
- source / target image path 是否真实存在；
- COCO RLE 能否解码且 shape 与原图一致；
- 保存的 mask area 是否等于 RLE 解码面积；
- source 与 target mask 是否来自同一个人工选择时刻；
- 未审核、整 card drop、空 mask 和不可用 view 是否已排除；
- 三个方向文件的数量之和是否等于总有向 relation 数；
- repository image 与 relation-referenced image 是否分别计数；
- 未被 JSON 引用的残留 JPEG 是否明确披露而不是误算为 benchmark 样本。

该设计使每个发布 record 都可以追溯到：

~~~text
BEHAVIOR activity and episode
  → BDDL goal and resolved target prim
  → restored simulator state
  → concrete camera pass
  → seg_instance_id buffer
  → exact target-ID mask
  → automatic visibility statistics
  → human card/frame verdict
  → directed relation record
~~~

## 4. 当前锁定快照的结果

截至 2026-08-25，Hugging Face private dataset revision
`ff0e77c9d37f43b9b10a8ef063e777bb34d5627a` 的数据文件统计为：

| 指标 | 数量 |
|---|---:|
| source episodes | 59 |
| cards（episode × target instance） | 132 |
| distinct target categories | 71 |
| human-selected card/frame moments | 391 |
| directed relations | 708 |
| `ego2exo.json` | 243 |
| `exo2ego.json` | 243 |
| `exo2exo.json` | 222 |
| repository JPEGs | 1,118 |
| relation-referenced JPEGs | 1,110 |
| disclosed unreferenced JPEGs | 8 |
| source/target RLE masks in relation records | 1,416 |
| native mask shape | 720×1280 |

mask area 的最小值为 43 pixels，中位数为 2,883.5 pixels，平均值为 10,324.6 pixels，最大值为 271,970 pixels。面积分布为：

| mask area | masks | share |
|---|---:|---:|
| < 1k px | 312 | 22.0% |
| 1k–5k px | 584 | 41.2% |
| 5k–20k px | 364 | 25.7% |
| ≥ 20k px | 156 | 11.0% |

每个 card 的人工选择时刻中位数为 3，范围为 1–11。由 relation counts 可以确定：111 个 cards 具有两个可用 exo views，因此各产生两个方向的 exo↔exo relations；其余 21 个 cards 具有一个可用 exo view，只贡献 ego↔exo relations。

source-camera usage 为：

~~~text
head_gaze 243
exo3      101
exo0       82
exo1       73
exo2       67
exo4       54
exo5       46
exo6       42
~~~

上述是当前发布快照的固定统计，不应与生产期间生成过但被人工 drop、因 view failure 未采用、或未进入 relation JSON 的中间 artifacts 混合计数。

## 5. Novelty 与常见仿真数据生成流程的差别

| 常见做法 | BehaviorSim data engine |
|---|---|
| 从任务名或类别标签生成 semantic mask | 从 BDDL goal 解析 manipulandum，再绑定精确 simulator prim，保留 sibling-instance identity |
| 为每个相机独立执行动作或近似同步视频 | 在相同采样时刻直接恢复记录的 simulator state，避免 action replay drift |
| 使用固定房间相机或围绕机器人盲目放置相机 | 围绕目标轨迹生成相机，约束房间边界、物体尺度、俯视角和运动 segment |
| 预先选少量机位并假设它们可用 | over-generate 多个 exo candidates，真实渲染后按 measured mask visibility 选二 |
| 用 semantic category、box、SAM 或人工多边形产生 mask | 用 `seg_instance_id` 到 exact prim 的映射产生原生分辨率实例 mask |
| 均匀抽帧或只保留自动最高分帧 | 自动定位 manipulation window，再由人类通过 overlay 和完整视频选择稀疏、有意义时刻 |
| 人工负责修边界 | 人工负责语义与样本有效性；像素边界由 simulator ground truth 固定 |
| 每个 episode 强制生成固定数量 pair | 按实际可见性构造 ego↔exo 和 exo↔exo relation，不制造缺失观察 |

核心 novelty 可以概括为六点：

1. **Task semantics to exact pixels。** 将 BDDL 中的任务目标逐级落到具体 scene instance 和逐像素 mask；
2. **State-restored multi-view synchronization。** 通过直接恢复记录状态，在多进程相机渲染中维持统一物理时刻；
3. **Target-centric adaptive view synthesis。** 相机由 target geometry、trajectory、room bounds 和 object scale 驱动，而不是固定在场景或机器人坐标附近；
4. **Over-generate and measure。** 先渲染多个几何多样的 exo 候选，再用真实 instance-mask 可见性选择，而不是盲信代理分数；
5. **Exact-boundary / semantic-review separation。** simulator 决定像素边界，人类决定语义正确性、视角价值和时刻价值；
6. **Visibility-aware directed supervision。** 只对双方共同可见的 source-target view 构造有向 relation，并保留 ego↔exo 与 exo↔exo 的方向差异。

## 6. 可直接用于论文的中文描述

我们构建了一套面向跨视角实例分割的 goal-conditioned simulator-native data engine。系统首先解析 BEHAVIOR-1K activity 的 BDDL goal literals，将操作谓词中的 manipulandum 类型绑定到 OmniGibson scene scope 中的精确物理实例，并记录其 destination 关系。随后，系统在多个独立相机 pass 中直接恢复 episode 的序列化 simulator state，而不是分别重新执行动作，从而使 ego 和 exo observation 对应同一记录物理状态，并避免长序列 action replay 引入的时间漂移。

围绕已解析的目标实例，我们合成一个受目标轨迹、房间几何和物体尺度约束的多视角观察集合。egocentric `head_gaze` 安装在机器人头部，根据手部邻近关系选择当前 manipulandum，并在有限前向视锥内向其 BDDL destination 提前注视。exocentric 路径围绕目标生成多个 azimuth-diverse 候选机位，按房间 floor AABB 限制位置，按目标尺寸调整观察半径，以俯视角确定高度，并根据目标运动轨迹分段重新瞄准。生产流程对最多八个 exo 候选进行真实渲染，再依据 simulator mask 的实测面积与可见率选择两个最有用视角。

每个视角同时渲染 RGB 与 `seg_instance_id`。pixel-to-prim 映射使目标 mask 能够绑定到 exact simulator instance，从而区分同类别的多个物理对象；退化黑帧通过实例 ID 多样性检测并重新渲染。系统随后在目标真实可见的 manipulation window 内自动生成时间分散的候选时刻。人工审核者查看全分辨率 mask overlay 和完整 scrub video，验证目标身份、跨视角一致性、渲染质量和操作语义，并选择最终稀疏时刻；人工不描画或修改 simulator-native mask。

最终，系统只在 source 和 target view 共同可见目标的人工选择时刻构造有向 ego→exo、exo→ego 和 exo→exo relations。source mask 作为 cross-view prompt，target-view mask 作为 ground truth，二者以原生 720×1280 分辨率的 compressed COCO RLE 保存。该流程把任务语义、精确实例身份、无漂移多视角同步、目标驱动相机生成、仿真器像素真值和人工语义筛选连接为一条可追溯的数据生成链。

## 7. Paper-ready English description

### Data Engine Pipeline

We build a goal-conditioned, simulator-native data engine for cross-view instance segmentation. For each BEHAVIOR-1K episode, the engine parses the BDDL goal literals and resolves each manipulandum type to a concrete object in the OmniGibson scene scope. This resolution preserves the identity of a physical simulator instance rather than collapsing multiple sibling objects into a semantic category. Destination objects extracted from the same goal literals are retained as task context.

To obtain temporally consistent observations, we render each camera pass by directly restoring the recorded serialized simulator state at the requested timestep. We do not independently re-simulate the action sequence for different cameras. State restoration therefore anchors every view to the same recorded physical configuration and avoids accumulated action-replay drift. When view passes contain different numbers of rendered frames because of recovery or sampling budgets, candidate timestamps are transferred through normalized episode time rather than reused as local array indices.

Camera synthesis is target-centric. The egocentric `head_gaze` camera is mounted at the robot head, attends to the BDDL manipulandum nearest either hand, gradually biases its gaze toward the corresponding task destination, and remains within a bounded cone around the robot's true forward direction. For exocentric observations, the production system over-generates up to eight azimuth-diverse camera candidates around the target. Their placement is constrained by the room floor AABB, their distance is scaled to the target AABB size, their elevation follows a depression-angle rule, and their aim is updated only at trajectory-derived segment boundaries. After rendering, the two most useful exocentric views are selected by measured target-mask size and visible-frame fraction. This measured selection is the production path used by the released data; the analytic frustum-and-raycast scorer retained in the codebase is an experimental alternative.

Each view produces RGB and simulator `seg_instance_id` output. The integer-to-prim map is matched to the resolved target instance, yielding native-resolution binary masks for the exact physical object. This prevents masks from merging multiple objects of the same category. Degenerate renders are detected from the instance-ID buffer and re-rendered or excluded. The system then identifies the manipulation window and proposes temporally separated frames with strong joint visibility. Human reviewers inspect full-resolution overlays and mask-tinted scrub videos, reject invalid cards or frames, and select sparse semantically meaningful moments. Reviewers curate target identity, cross-view consistency, render validity, and temporal usefulness; they do not draw or refine the simulator-native mask boundaries.

Finally, the exporter constructs all supported directed relations: ego-to-exo, exo-to-ego, and both directions between two usable exocentric views. A relation is emitted only when its source and target views contain non-empty masks for the same selected moment. The source-view mask forms the cross-view prompt and the target-view mask provides the ground truth, both encoded as compressed COCO RLE at the native 720×1280 resolution. The resulting records remain traceable from the BDDL goal and resolved prim through simulator state, camera pass, instance-ID mask, human decision, and exported relation.

### Core novelty

The central contribution is not simulator rendering alone, but the composition of task semantics, exact instance identity, state-restored multi-view synchronization, target-conditioned camera synthesis, measured visibility selection, simulator-native pixel supervision, and human semantic curation. In particular, the pipeline separates two forms of authority: the simulator defines exact visible boundaries, while the reviewer determines whether a target, view, and moment constitute useful cross-view supervision.

### Short paper version

We generate BehaviorSim using a goal-conditioned simulator-native data engine. BDDL goal literals are resolved to exact OmniGibson object prims, and synchronized ego/exo observations are rendered by restoring the same recorded simulator state for each camera pass. Target-centric exo candidates are constrained by room geometry, object scale, depression angle, and target trajectory, then selected using measured instance-mask visibility; an active robot-head view supplies the ego observation. Exact masks are extracted from `seg_instance_id` rather than predicted or drawn. Human reviewers validate target identity, cross-view consistency, render quality, and sparse manipulation moments without editing mask boundaries. We finally export only jointly visible, directed ego↔exo and exo↔exo relations as native-resolution COCO RLE.

## 8. Mermaid 流程图草稿

~~~mermaid
flowchart LR
    A[BEHAVIOR-1K episode<br/>recorded states + BDDL goal]
    B[Goal-conditioned target discovery<br/>manipulandum + destination]
    C[Exact target instance<br/>scene object / prim identity]
    D[State-restored replay<br/>same recorded physical state]
    E1[Active ego synthesis<br/>hand-aware head_gaze<br/>bounded forward cone]
    E2[Target-centric exo synthesis<br/>room bounds + object scale<br/>depression angle + trajectory segments]
    F[Render candidate views<br/>RGB + seg_instance_id]
    G[Exact instance mask<br/>pixel ID to target prim]
    H[Measured visibility statistics<br/>manipulation window + temporal spacing]
    I[Candidate card<br/>ego + best two exos<br/>full overlays + scrub videos]
    J{Human review<br/>target / view / moment validity}
    K[Drop card or frame]
    L[Keep or add sparse moments]
    M[Joint-visibility relation builder]
    N1[ego2exo.json]
    N2[exo2ego.json]
    N3[exo2exo.json]
    O[Audit<br/>paths + RLE + areas + counts]

    A --> B --> C --> D
    D --> E1
    D --> E2
    E1 --> F
    E2 --> F
    C --> G
    F --> G --> H --> I --> J
    J -->|reject| K
    J -->|approve / select| L --> M
    M --> N1
    M --> N2
    M --> N3
    N1 --> O
    N2 --> O
    N3 --> O
~~~

## 9. 流程图生成 Prompt

### 9.1 推荐英文 Prompt

Create a publication-quality scientific pipeline diagram titled “Goal-Conditioned Simulator-Native Cross-View Data Engine”. Use a clean 16:9 horizontal layout, white or very light warm-gray background, vector graphics, restrained robotics-conference visual style, crisp typography, and no decorative 3D effects.

Show seven numbered stages from left to right:

1. “Goal-Conditioned Target Instance Discovery”. Illustrate a BEHAVIOR-1K household manipulation episode beside a compact BDDL goal expression such as “inside pumpkin cabinet”. Draw an arrow from the manipulandum token to one highlighted physical object in the simulator scene. Label the result “exact scene object / prim”, and show the cabinet only as the destination, not as the target.

2. “State-Restored Synchronized Replay”. Show one serialized simulator-state timeline. At a selected timestep, branch the exact same physical state into multiple camera passes. Add the caption “restore recorded state; do not independently re-simulate actions”. Make it visually clear that all cameras observe one identical robot-object configuration.

3. “Target-Centric Adaptive Camera Synthesis”. Split the stage into two coordinated parts. On top, show a robot-head ego camera whose gaze follows the manipulandum nearest a hand and bends slightly toward the BDDL destination, while staying inside a forward-facing cone. On the bottom, show up to eight exo candidates distributed around the target with different azimuths and near/far radii. Draw room boundaries, a depression-angle arc, object-size-dependent camera distance, and two trajectory segments with piecewise-static re-aiming.

4. “Simulator-Native Instance Masks”. Show RGB and a color-coded `seg_instance_id` buffer. Connect the exact target prim to one integer ID and then to a tight binary mask of only that physical instance. Include a neighboring object of the same category with a different ID to emphasize instance separation. Add a small retry loop labeled “detect and re-render degenerate frames”.

5. “Measured Visibility and Candidate Construction”. Show visibility curves or mask-pixel bars for the ego and several exo views. Highlight the manipulation window, select temporally separated candidate moments, and retain the ego plus the two exo views with the strongest measured target visibility. Add a novelty badge reading “over-generate, render, then select by measured masks”.

6. “Human-Reviewed Sparse Moments”. Show a reviewer inspecting three synchronized views, full-resolution translucent mask overlays, and a scrub-video timeline. Provide controls or labels for “keep card”, “drop card”, “keep/drop frame”, and “add frame”. State explicitly: “human validates target, views, and moments; human does not draw mask boundaries”.

7. “Directed Relation Export and Audit”. Show the retained ego and two exo frames connected by directed arrows: ego→exo, exo→ego, exo A→exo B, and exo B→exo A. Show source mask as prompt and target mask as ground truth. End with three files named `ego2exo.json`, `exo2ego.json`, and `exo2exo.json`, compressed COCO RLE at 720×1280, followed by a green audit shield for image paths, RLE decoding, mask areas, and record counts.

Use consistent visual semantics:

- blue for task semantics and target identity;
- violet for state restoration and synchronization;
- teal for camera synthesis;
- green for simulator-native instance masks;
- amber for measured candidate selection;
- warm orange for human review;
- dark blue for relation export and audit;
- solid arrows for data flow;
- thin dashed arrows only for metadata or provenance.

Add a narrow footer with six novelty labels:

“BDDL semantics → exact prim” · “state-restored synchronization” · “target-centric views” · “over-generate and measure” · “exact masks, human semantic review” · “visibility-aware directed relations”.

The figure must clearly distinguish this simulated pipeline from a model-assisted annotation pipeline. Do not include Codex, SAM, SAM3, bounding-box prompting, official fallback masks, or human-painted polygons.

### 9.2 中文 Prompt

生成一张可直接用于机器人与计算机视觉论文的横向 16:9 矢量流程图，标题为“Goal-Conditioned Simulator-Native Cross-View Data Engine（目标条件化的仿真器原生跨视角数据引擎）”。使用白色或极浅暖灰背景，线条清晰、字体克制、信息密度高但不拥挤，不使用装饰性 3D 效果。

从左到右展示七个编号阶段：

1. “Goal-Conditioned Target Instance Discovery / 目标实例发现”。画出一个 BEHAVIOR-1K 家庭操作场景和简化 BDDL goal，例如 “inside pumpkin cabinet”。从 manipulandum token 指向场景中唯一高亮的具体 pumpkin 实例，标注 “exact scene object / prim”；cabinet 只作为 destination，不要高亮成目标。

2. “State-Restored Synchronized Replay / 状态恢复式同步回放”。画出一条序列化 simulator-state 时间轴，在某个采样时刻把同一物理状态分支给多个相机 pass。标注 “直接恢复记录状态，不为不同相机独立重放动作”，让多个视角中的机器人和目标姿态明显完全一致。

3. “Target-Centric Adaptive Camera Synthesis / 面向目标的自适应相机”。上半部分画机器人头部 ego camera：视线选择距离手部最近的 manipulandum，轻微提前指向 BDDL destination，同时限制在前向视锥内。下半部分围绕目标画最多八个 exo 候选机位，包含不同 azimuth、near/far radius、房间边界、俯视角、按物体尺寸缩放的相机距离，以及根据目标轨迹划分的两个 piecewise-static 瞄准 segment。

4. “Simulator-Native Instance Masks / 仿真器原生实例 mask”。并排展示 RGB、彩色 `seg_instance_id` 图和精确二值 mask。从 exact target prim 连接到一个整数 instance ID，再连接到只覆盖该物理实例的 mask。旁边放置同类别的另一个物体并赋予不同 ID，突出 sibling-instance separation。添加一个小型重试环：“检测并重渲染退化帧”。

5. “Measured Visibility and Candidate Construction / 实测可见性与候选构造”。画 ego 和多个 exo 的可见率曲线或 mask-pixel 柱状图，高亮真正发生操作的 manipulation window，选择时间上分散的候选时刻，并保留 ego 与实测 target visibility 最强的两个 exo。添加 novelty 标签：“over-generate, render, then select by measured masks”。

6. “Human-Reviewed Sparse Moments / 人工复核稀疏时刻”。画一名审核者同时查看三个同步视角、可调透明度的全分辨率 mask overlay 和可拖动的完整 scrub-video 时间轴。展示 “keep card”“drop card”“keep/drop frame”“add frame”。明确标注：“人工验证目标、视角与时刻；人工不描画 mask 边界”。

7. “Directed Relation Export and Audit / 有向关系导出与审计”。画保留的一个 ego 和两个 exo frame，并用有向箭头连接 ego→exo、exo→ego、exo A→exo B 和 exo B→exo A。source mask 标记为 prompt，target mask 标记为 ground truth。末端展示 `ego2exo.json`、`exo2ego.json`、`exo2exo.json`，原生 720×1280 compressed COCO RLE，以及一个绿色 audit shield，包含路径检查、RLE 解码、mask area 和 record count。

固定视觉语义：

- 蓝色：任务语义与目标身份；
- 紫色：状态恢复与多视角同步；
- 青色：相机合成；
- 绿色：仿真器原生实例 mask；
- 琥珀色：实测可见性与自动候选；
- 暖橙色：人工审核；
- 深蓝色：关系导出与审计；
- 实线箭头：主要数据流；
- 细虚线：metadata 或 provenance。

底部加入六个 novelty 短语：

“BDDL semantics → exact prim” · “state-restored synchronization” · “target-centric views” · “over-generate and measure” · “exact masks, human semantic review” · “visibility-aware directed relations”。

必须明确这是一条 simulator-native pipeline。不要出现 Codex、SAM、SAM3、box prompting、official fallback mask 或人工多边形描边。

### 9.3 Negative Prompt

~~~text
No photorealistic poster, no cinematic lighting, no dark background, no decorative
robots unrelated to the pipeline, no neural-network brain icon, no SAM or SAM3 logo,
no Codex or language-model box, no bounding-box-to-mask pipeline, no hand-drawn
polygon annotation, no semantic-category mask merging multiple instances, no
independent action replay per camera, no fixed blind room cameras, no claim that
humans refine pixel boundaries, no claim that analytic raycast ranking produced the
released dataset, no forced relation when a target is invisible, no square 704x704
mask, no illegible tiny labels, no excessive gradients, no 3D infographic style.
~~~

## 10. Implementation Evidence Map

| 流程环节 | 主要实现或记录 | 关键证据 |
|---|---|---|
| BDDL target 与 destination 解析 | `BehaviorPilot/remote/pilot_viewercam.py` | `MOVE_PREDS`、goal literal 解析、`target_key`、size-based fallback、`manipulanda` metadata |
| State-restored replay | `BehaviorPilot/remote/pilot_viewercam.py` | `FAST_REPLAY`、逐采样时刻 `og.sim.load_state`、settle renders、streamed capture |
| 最终 v5 相机配置 | `BehaviorPilot/remote/start_v5.sh` | `PILOT_EXO_RING=8`、`PILOT_NO_WRIST=1`、`PILOT_FAST_REPLAY=1` |
| Target-centric exo 与 active ego | `BehaviorPilot/remote/pilot_viewercam.py` | room AABB、object-size radii、depression angle、trajectory segments、`_gaze_point`、`_clamped_lookat` |
| 精确实例 mask | `BehaviorPilot/remote/pilot_viewercam.py`、`BehaviorPilot/remote/make_card_bundle.py` | `seg_instance_id`、pixel-to-prim map、exact instance matching、blank-frame detection |
| manipulation window 与候选时刻 | `BehaviorPilot/remote/make_card_bundle.py` | joint visibility score、200-pixel bar、time spacing、per-view normalized-time mapping |
| Review package | `BehaviorPilot/remote/build_review_host.py`、`BehaviorPilot/build_review.py`、`BehaviorPilot/review_page.html` | full-resolution overlays、mask-tinted scrub videos、frame/card verdict UI |
| 新增时刻 materialization | `BehaviorPilot/remote/extract_picked_frames.py` | `keep_frames ∪ add_frames`、per-view local index mapping、RGB/mask extraction |
| Reviewed relation export | `BehaviorPilot/apply_review.py` | unreviewed exclusion、positive/negative frame selection、ego↔exo 与 exo↔exo 双向 relation、COCO RLE |
| 早期几何 placement 设计 | `docs/superpowers/specs/2026-07-26-geometric-exo-placement-design.md` | 36-pose analytic scoring 的设计依据；作为实验分支记录，不冒充当前发布数据的最终选择路径 |
| 当前发布统计 | `.codex-tmp/project2-export-v1/behaviorsim-card-stats-20260811.json`、HF revision audit | 708 relations、132 cards、59 episodes、71 categories、1,416 masks |

## 11. Figure Caption

### 中文

**图 X：目标条件化的仿真器原生跨视角数据引擎。** 系统从 BEHAVIOR-1K 的 BDDL goal 中识别 manipulandum，并将其解析为 OmniGibson 场景中的精确物理实例。多个相机 pass 通过直接恢复同一记录 simulator state 获得无 action-replay 漂移的同步观察。目标驱动的 ego/exo 相机生成结合主动 head gaze、房间边界、物体尺度、俯视角和目标轨迹；多个 exo 候选完成真实渲染后，再按照 simulator mask 的实测可见性选出最佳视角。`seg_instance_id` 提供与 exact prim 对应的原生分辨率实例 mask。人工审核者验证目标身份、跨视角一致性、渲染质量和稀疏时刻，但不描画 mask 边界。最终系统只为共同可见的 source-target views 导出有向 ego↔exo 与 exo↔exo relations，并执行 RLE、路径、面积与数量审计。

### English

**Figure X: Goal-conditioned simulator-native cross-view data engine.** BDDL goal literals identify the manipulandum, which is resolved to an exact physical object prim in the OmniGibson scene. Multiple camera passes observe synchronized configurations by restoring the same recorded simulator state, avoiding action-replay drift. Target-centric ego and exo synthesis combines active head gaze, room-bound placement, object-scale distance, depression-angle elevation, and trajectory-aware re-aiming; exo candidates are rendered first and selected using measured simulator-mask visibility. Native `seg_instance_id` output provides exact instance masks. Human reviewers validate target identity, cross-view consistency, render quality, and sparse moments without drawing mask boundaries. The exporter then emits only jointly visible directed ego↔exo and exo↔exo relations and audits their paths, RLE masks, areas, and counts.
