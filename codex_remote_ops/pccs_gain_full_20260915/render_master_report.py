"""Rebuild the single user-requested report from verified experiment artifacts."""
from pathlib import Path
import json,hashlib
R=Path(__file__).parent;O=R.parent;report=R/'FINAL_REPORT.md'
backup=R/'FINAL_REPORT_pre_consolidation_20260916.md'
if report.exists() and not backup.exists():backup.write_bytes(report.read_bytes())
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
names={'baseline':'PCCS 基线','learned_gate':'学习判断器（0.03）','omama_margin005':'原几何 O-MaMa（0.05）','native_margin005':'保持纵横比 O-MaMa（0.05）','geometry_consensus':'两几何一致才替换'}
sources=[('Exo→Ego／作者权重',R/'runs/full/results.json'),('Exo→Exo／作者权重',O/'pccs_exoexo_transfer_20260916/runs/full/results.json'),('Exo→Exo／修复版权重',O/'pccs_corrected_exoexo_20260916/runs/full/results.json'),('Exo→Ego／修复版权重',O/'pccs_corrected_exoego_20260916/runs/full/results.json')]
parts=['# V2-SAM PCCS／O-MaMa 实验总报告','','> 唯一总报告；具体运行文件在各实验目录。更新到已取回并验证的机读结果，未完成项目保持待定。研究主目标为 **Exo→Exo 稳定增益**，Exo→Ego 用于验证源方向与比较迁移。',
'','## 1. 当前结论与完整结果','','- 指标与pipeline已按实验日记和`v2sam-pccs@0e3bc33`复核：新frame-level口径正确，零IoU保留，未发现历史DINO二次初始化或目标GT内容影响推理的问题；证据与边界见§5。',
'- 作者权重与用户修复版权重必须分列。前四组delta均减**同一轮、同权重、同候选bank**的PCCS基线；§9新增候选实验的主delta减当轮原O-MaMa几何一致性输出。',
'- 修复版权重Exo→Exo的预先指定主方案native-margin已获得正的95%序列bootstrap区间；几何一致方案点估计更高，但它是次要对照，未证明显著优于其他改进方法。',
'- “稳定”在这里指当前基准上主比较的统计证据，不是保证每个对象、每个随机种子或所有新场景都涨点。',
'','### 1.1 全量 frame-level 指标','','IoU／Dice／ContA 为百分数，LocE 为原始归一化距离（越低越好）。置信区间对应 **IoU 增益，单位百分点**。']
if sources[-1][1].exists():
 latest=load(sources[-1][1]);primary=latest['paired_comparison']['geometry_consensus'];ci=primary['paired_take_bootstrap_95ci_pp']
 parts.insert(parts.index('### 1.1 全量 frame-level 指标')-1,f"- 修复版权重Exo→Ego全量已完成：预先指定的几何一致性主方案IoU {latest['methods']['baseline']['frame']['IoU']*100:.4f}→{latest['methods']['geometry_consensus']['frame']['IoU']*100:.4f}，+{primary['delta_frame_iou_pp']:.4f}点，95%区间[{ci[0]:.4f}, {ci[1]:.4f}]。这轮canonical/native的数值非常接近，不应当成两份独立重复证据。")
up=O/'pccs_candidate_union_20260916/exo2exo_results.json'
if up.exists():
 u=load(up);ci=u['primary_95ci_pp'];parts.insert(parts.index('### 1.1 全量 frame-level 指标')-1,f"- 最新候选合并／质量适配探索已完成（§9）：Exo→Exo原一致性IoU {u['methods']['original_consensus']['frame'][0]*100:.4f}→{u['methods']['selected_union']['frame'][0]*100:.4f}，新增{u['primary_delta_vs_original_consensus_pp']:+.4f}点，95%区间[{ci[0]:.4f}, {ci[1]:.4f}]跨0；**未获得额外稳定增益**。小比例池化、可靠多点与rank-8 O-MaMa适配的负结果也完整记录。")
completed={}
for title,p in sources:
 parts+=['',f'#### {title}','']
 if not p.exists():
  parts+=['**待完成：46,515 对、109,253 对象、295 个拍摄序列。** 使用corrected Visual e19／Fusion e20，先32对四卡预检，再全量候选生成、五方案打分与汇总。主比较提前指定为几何一致性；门槛不变、不重新拟合。结果尚未回填，不能使用其他行的分数代替。'];continue
 a=load(p);assert a['coverage']=='exact';completed[title]=a
 parts += [f"范围：{a.get('pairs',46515):,} 对、{a.get('objects',109253):,} 对象、{a.get('takes',295)} 个拍摄序列。",'', '| 方法 | IoU | Dice | ContA | LocE ↓ | ΔIoU | 95% 序列区间 |','|---|---:|---:|---:|---:|---:|---|']
 for m,v in a['methods'].items():
  f=v['frame'];c=a['paired_comparison'].get(m);ci=c['paired_take_bootstrap_95ci_pp'] if c else None
  parts.append(f"| {names[m]} | {f['IoU']*100:.4f} | {f['Dice']*100:.4f} | {f['ContA']*100:.4f} | {f['LocE']:.6f} | {c['delta_frame_iou_pp'] if c else 0:+.4f} | {f'[{ci[0]:.4f}, {ci[1]:.4f}]' if ci else '—'} |")
 if title=='Exo→Exo／修复版权重':parts+=['','主比较为**保持纵横比 O-MaMa**；其他区间是未作多重比较校正的探索性结果。旧学习判断器在作者权重候选上训练，本轮只是冻结迁移对照；它的LocE并未改善。']
 if title=='Exo→Ego／修复版权重':parts+=['','主比较为**两几何一致才替换**，在本轮结果产生前已指定；其他方法为次要对照。']
 parts+=['',f"完整机读结果：[{p.parent.parent.parent.name}/results.json]({p.relative_to(O).as_posix() if p.is_relative_to(R) else '../'+p.relative_to(O).as_posix()})。"]
 # Correct links relative to the report directory, including its own experiment.
 parts[-1]='完整机读结果：[results.json]('+('runs/full/results.json' if p.is_relative_to(R) else '../'+p.relative_to(O).as_posix())+')。'
parts+=['','### 1.2 Object-level 与独立确认','','Exo→Exo当前每pair恰有一个对象，object和frame指标相同；Exo→Ego的对象数不均，两套聚合不能混写。','','| Exo→Ego 权重 | 方法 | Object IoU | Object Dice | Object ContA | Object LocE ↓ |','|---|---|---:|---:|---:|---:|']
for title in ('Exo→Ego／作者权重','Exo→Ego／修复版权重'):
 if title not in completed:continue
 for m,v in completed[title]['methods'].items():
  f=v['object'];parts.append(f"| {title.split('／')[1]} | {names[m]} | {f['IoU']*100:.4f} | {f['Dice']*100:.4f} | {f['ContA']*100:.4f} | {f['LocE']:.6f} |")
parts+=['','独立确认：作者权重下的512对／965对象／64序列holdout，PCCS **70.9472→74.9235**（学习判断器，+3.9762点，95%区间[2.5743,5.4232]）。这些序列排除了此前探索与第一批确认序列。后续全量包含历史已观察测试样本，因此全量是冻结方案的基准评测，不是整个测试集从未观察过的盲测。',
'','## 2. 数据、权重和统计口径','','### 2.1 任务方向与权重身份','','Exo→Exo的源和目标都是外部相机视角；没有专用权重时借用Exo→Ego权重，**任务标签仍是Exo→Exo**。Ego→Exo的新判断器尚未评测；该方向PCCS可能选组合／阈值mask，需要先适配当前三专家身份特征，不能只改启动方向。',
'','| 资产 | 来源／用途 | SHA256 或版本 |','|---|---|---|',
'| corrected Visual | 用户重训e19，926,619,220 bytes | `fd2e1b8f342c4468b45c962c61d4c914b9f6d71b6c6085014fb9a3ed2924b65f` |',
'| corrected Fusion | 用户重训e20，2,136,421,714 bytes | `8373b17180c198898fe9f9e39d09fe72991150ce3053b207b19e81986c60cab1` |',
'| 发布仓库 | `Travor278/V2-SAM`，两文件与平台逐字节一致 | `8b8310e9ee13023cb8711c2417e18cc7361dc3aa` |',
'| 原作者Visual／Fusion | 历史对照，来自`official_weights_wangzeze` | 分别以`55761ff4…`／`a4a09371…`开头，完整值在审计回执 |',
'| 冻结学习判断器 | 作者权重候选上训练的HGBoost | `f79543baa14e81ebc83fa3bb7458e0940e405afe31a556fb612281c6aa0d2678` |',
'| O-MaMa代码 | 公开Exo→Ego预训练匹配头；本轮冻结 | `0f187c65cb9d8f8df1d8b5f445edbb0485956d88` |',
'| DINOv2代码 | ViT-B/14-reg4，新增打分分支 | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |',
'','修复权重的远端文件位于`codex_remote_ops/hf_publish_exo2ego_20260903/tmp/`。原始SAM2/DINOv3继续用于V2-SAM候选生成，新增DINOv2/O-MaMa负责候选匹配，不应混称为同一个骨干。',
'','### 2.2 指标和可比性','','```text\nobject-level = 所有对象指标的等权平均\nframe-level  = mean_pair(mean_object_in_that_pair(metric))\n```',
'','零IoU对象保留。ContA是上游边界F-measure；LocE是最大外轮廓点坐标均值之间的距离除以图像对角线，并非mask面积质心距离。目标标注按上游nearest缩放为1024×1024，与预测对齐。置信区间使用10,000次按拍摄序列重采样，序列长度不等时保留frame的sum/count权重，不能改成每序列均值等权。',
'','“全量”的范围：Exo→Ego为46,515对／109,253对象／295序列；Exo→Exo为已构建基准的1,094对／20序列，不是Ego-Exo4D所有可能外部相机组合。前者图像对约为后者42倍、对象数约100倍，所以四卡运行约6小时和约11分钟并不矛盾。',
'','## 3. 方法原理与实际实现','','### 3.1 从候选生成到最终输出','','```mermaid\nflowchart TD\n  A[源图像与源物体mask + 目标图像] --> B[冻结V2-SAM产生visual / anchor / fusion]\n  B --> C[原PCCS得到baseline]\n  B --> D[冻结DINOv2与O-MaMa匹配]\n  C --> E[固定余弦门槛 / 学习收益门控 / 几何一致性]\n  D --> E\n  E --> F[保留baseline或输出一个已有候选]\n```',
'','Exo→Ego原PCCS为fusion-first：fusion通过面积／连通分量质量检查时优先采用，否则按源mask和循环对应信息回退到visual或anchor。所有方法先保留这个真实baseline，再按`baseline, visual, anchor, fusion`顺序去重相同二值mask；空替代候选不参与比较，空源提示保留baseline。新增选择模块不会修改mask边界，也不能在所有候选都错误时生成正确分割。',
'','候选只生成一次，按rank分别保存无损bit-packed缓存。上游RegionSampler仍使用随机抽点，所以跨GPU／跨分片重跑不保证逐位相同；同一次方法比较的mask内容则严格一致。',
'','### 3.2 O-MaMa怎样提取物体和上下文','','使用768维DINOv2 patch特征，双线性插值到输入约1/4网格（patch14网格约放大3.5倍，尺寸按源码取整）。mask最近邻对齐。',
'','```text\no(M) = mean{F(p): p在物体mask内}                       # 768维\nc(M) = mean{F(p): p在bbox每边扩张100px的矩形内}          # 768维\nd(M) = [o(M), c(M)]                                    # 1536维\n```',
'','100px位于预处理图像坐标中，越界裁剪。框内包含物体自身和背景，**不是减去前景后的环，也不是1.5／2倍面积缩放**。源物体和每个目标候选分别取框。当前不显式检测邻居物体，也没有实例关系图。bbox宽高、整数取整和边界裁剪跟随原代码。',
'','### 3.3 双向交叉注意力和匹配分数','','```text\na_q  = CrossAttention(o_q,  target整图tokens + P_t)\na_tk = CrossAttention(o_tk, source整图tokens + P_q)\nz_q  = normalize(MLP([a_q,  o_q,  c_q]))\nz_tk = normalize(MLP([a_tk, o_tk, c_tk]))\ns(k) = cosine(z_q, z_tk)\n```',
'','共享MLP把2304维投影为768维；注意力从另一张整图取信息，局部框描述符单独输入MLP。所有参数来自公开预训练头并冻结。当前不使用其可见性分支，也未复现原O-MaMa的整套候选生成和训练流程。',
'','四种用于判断器的分数：完整预训练匹配；将局部context和cross输入置零、仍用原MLP的物体主导匹配；DINOv2物体余弦；拼接`[object,context]`后的余弦。最后一项不是两个余弦的简单平均。置零只是推理干预，不是独立训练的object-only消融。',
'','### 3.4 几何与固定门槛','','- **canonical**：Exo→Ego源高532×宽952、目标700×700，对应38×68和50×50学习位置表；图像／mask按适配器固定nearest缩放，图像作ImageNet标准化。迁移到Exo→Exo时这会一同迁移ego目标的方形先验。',
'- **native**：保留每张图纵横比，最长边≤1024、向下取14倍数，对学习位置表作双线性插值；其余匹配参数不变。',
'- **固定门槛**：非空候选中取最高完整匹配分数，原始余弦比baseline高**超过0.05**才替换；baseline为空时选可用候选。0.05不是sigmoid概率差。',
'- **几何一致性**：canonical和native分别执行上述规则，只有二者选中同一候选才替换，否则保留PCCS。它不新增训练，但增加两路匹配计算。',
'','### 3.5 学习判断器：预测相对baseline的IoU增益','','其目的不是把语义相似度当成分割IoU，而是识别“更像物体、但边界可能更差”的替换风险。每个候选构造65项原始输入：',
'','| 组别 | 内容 | 数量 |','|---|---|---:|',
'| 专家身份 | 候选与baseline分别对visual/anchor/fusion one-hot | 6 |',
'| 匹配分数 | 四种分数各取候选值、baseline值、差值、相对其他候选最高分的margin | 16 |',
'| 全局信息 | 源mask面积比、连通数、pair物体数、DINO前向置信度、三组候选间IoU | 7 |',
'| 局部质量 | 12属性各取候选值、baseline值、差值 | 36 |',
'','12属性为：面积比、连通数、SAM预测IoU、SAM IoU margin、SAM object logit、循环距离、DINO反向置信度、前向点落入mask的比例，以及概率图的前景均值、`mean(2|p−0.5|)`、`p>0.55`与`p>0.45`的稳定IoU、`0.4<p<0.6`的不确定比例。SAM预测IoU和候选间IoU都不是目标真值IoU。',
'','缺失字段保留上游`-1`等约定；None／非有限值转NaN，用**拟合集**中位数填补，并增加缺失指示列。因此65是原始向量长度，不必等于填补器输出维数。',
'','```text\ny_ijk = IoU(candidate_ijk, GT_ij) − IoU(baseline_ij, GT_ij)\n训练：加权平方误差，权重∝1/(pair对象数 × 该对象可用替代候选数)\n推理：最大预测增益 > 0.03时替换，否则保留baseline\n```',
'','0.03是预测IoU增益的3个百分点，不保证每次真实增加3点；回归输出不是概率，不作0～1裁剪。它与余弦差0.05量纲不同。最终树为HistGradientBoosting，100次迭代、最多7叶、叶最少25样本、L2=10、seed42，其他参数沿用固定scikit-learn1.5.2默认值。',
'','## 4. 训练／校准／测试边界与探索记录','','判断器只在作者权重候选上拟合：48训练序列×8=384对，生成1352条候选回归样本；另16序列×8=128对／225对象用于校准。试验预先限定ridge alpha={1,10,100}与上述小树，门槛={0,0.01,0.03,0.05}，按校准frame IoU选择，相同值优先较少替换，无正收益则保留baseline。最终选小树0.03；校准数据不再并入训练。',
'','原专家看过官方训练集，因此训练内校准收益不能作为测试收益；官方val标注虽找到，图像当时缺失，没有将train校准冒充官方validation。随后512对独立确认排除了早期探索／确认序列。修复权重实验继续使用旧树是**冻结迁移对照**，没有暗中重训。',
'','| 早期探索 | 观察 | 含义 |','|---|---|---|',
'| 1.5／2倍背景环context | Exo→Exo原PCCS38.4343→均值context39.0209；Ego→Exo与Exo→Ego下降 | 简单扩背景未获跨方向统一收益 |',
'| O-MaMa32/64对探索 | 源方向出现较强信号，其他方向不一致 | 不能将小样本涨幅外推全量 |',
'| 首批512对独立确认 | canonical-margin约+1.11点但CI跨0；context/cross置零约−0.30点 | 不支持只保留物体分数即可解决问题 |',
'| 后续512对学习判断器确认 | +3.98点且CI为正 | 支持继续固定方案全量验证 |',
'','上表只是方法演进记录，不能将其中不同候选、不同权重、不同样本的分数直接相减。源码随机采样是冻结bank设计的重要原因。',
'','## 5. Pipeline复核结论','','依据用户实验日记§2.2／§16与固定上游实现，完成静态身份、全量独立复算和真实模型检查：',
'','| 检查 | 证据／结果 |','|---|---|',
'| 新frame/object口径 | 真实evaluator构造`[0,1]`与`[0.2]`两pair：frame0.35、object0.4，区别旧last-object／正IoU过滤0.6 |',
'| 零IoU与多卡覆盖 | 作者权重全量Exo→Ego32033、Exo→Exo455个baseline零IoU全部保留；逐pair对象数、身份集合和全量四指标独立复算通过，误差<3.8e-15 |',
'| DINO二次初始化 | 作者／修复两权重，每套forward及两个backward分支各368张量，MMEngine init前后未变且等于原DINO资产 |',
'| 权重实际内容 | Visual964、Fusion1335推理张量逐个匹配；修复权重仅额外忽略训练计数`contrast_schedule.step` |',
'| GT内容泄漏／输出路由 | 真实3对象与1对象pair，GT原值/零/一/随机；39/23个预测张量哈希及PCCS路由不变；Fusion输出来自Fusion分支 |',
'| 上游实现身份 | 11核心源文件可执行AST与0e3bc33一致，4个指标函数一致；文档/换行和跨Python空type_params单独处理，原始hash仍保留 |',
'| 仍存在的算法弱点 | soft-mask.nonzero随机区域采样、稀疏top1点均沿用上游；不伪称已修复 |',
'','当前推理不执行旧对比损失／优化器，不涉及训练contrast调度或mid-epoch游标。目标mask的**形状**仍用于输出分辨率；反事实测试证明所测样本的GT内容不影响预测，不是对所有可能输入作无bug保证。每轮另要求所有rank零跳过、exact覆盖、缓存mask指标核验<1e-5及冻结模型hash一致。审计任务已成功释放。',
'','完整回执：[AUDIT_REPORT.md](../pccs_pipeline_audit_20260916/AUDIT_REPORT.md)、[runtime_probe.json](../pccs_pipeline_audit_20260916/runtime_probe.json)、[read_only_audit.json](../pccs_pipeline_audit_20260916/read_only_audit.json)。',
'','## 6. 后续探索：优先改善候选质量，再适配匹配器','','| 优先级 | 方法 | 原因与验证方式 |','|---|---|---|',
'| 1 | 概率加权区域池化 | sigmoid soft mask通常处处非零，原nonzero采样近似包含全图。先在train/calibration固定样本比较概率加权、硬阈值与原采样，检查候选oracle是否提高；它改变训练分布，必要时只微调matcher／prompt投影，不能当作无代价bug修复 |',
'| 2 | 可靠多点对应 | 互为近邻、空间去重top-k和置信筛选，减小单个错误点对SAM的影响；固定坐标转换，k和阈值只在训练／校准选取 |',
'| 3 | 质量感知O-MaMa适配 | 在**修复权重候选**上重训小型IoU差排序／收益头，加入尺度和视角增强；主体冻结，独立校准后冻结测试 |',
'| 4 | 多尺度物体／背景可靠性 | 区分前景、框内邻域与跨图注意力，根据源目标尺度和候选质量学习可靠性，而非继续手工扫描测试倍率；与完整原头公平消融 |',
'| 5 | 更大独立Exo→Exo确认 | 当前只有20序列；增加未见外部相机组合／序列，固定单一主方案，并做组件归因。不能用重复测试替代独立证据 |',
 '','优先做**区域池化与可靠多点的训练／校准小实验**。当前O-MaMa门控只能选已有mask，改善候选本身更有机会突破上限。以上是有代码依据的研究假设，不承诺必然涨点；实际启动和结果状态见§8。',
'','## 7. 实现、运行与维护索引','','| 目录／文件 | 用途 |','|---|---|',
'| 本目录`score.py`、`gain_features.py`、`aligned_model.py` | 作者权重全量时的原始学习门控实现 |',
'| `../pccs_corrected_exoexo_20260916/` | 修复权重Exo→Exo；几何双路打分、完整日志和回执 |',
'| `../pccs_corrected_exoego_20260916/` | 修复权重Exo→Ego全量重跑，五方案同bank |',
'| `../pccs_gain_20260915/fit_gain_gate.py` | 判断器训练／校准与逐候选特征记录 |',
'| `../pccs_pipeline_audit_20260916/` | 日记对照审计、权重身份、初始化和反事实检查 |',
'| `../pccs_method/v2sam_pccs_gain.patch`（Git分支中） | 相对固定上游的可审阅候选生成补丁 |',
'','GitHub：[Travor278/EgoExoSeg，pccs-omama-gain-20260916](https://github.com/Travor278/EgoExoSeg/tree/pccs-omama-gain-20260916)。代码、原始日志和哈希记录同步该分支；大型权重、图像和mask bank不入Git。O-MaMa相关适配保留AGPL-3.0来源与许可；DINOv2遵循其原许可。',
'','运行沿用已审计的Python3.10.21／PyTorch2.3.1+cu121及任务专用依赖。四卡预检通过才跑全量；成功后核验平台状态和占用节点0。监督间隔按当前有效ETA的4/5动态调整，阶段切换或无可信ETA用5分钟。新结果只在覆盖和回执通过后更新§1；不得覆盖详细原理或改写历史基线。']
parts+=['','### 7.1 运行收尾记录','','| 任务 | 结束时间（北京时间） | 占用节点 |','|---|---|---:|']
for folder in ('pccs_gain_full_20260915','pccs_exoexo_transfer_20260916','pccs_corrected_exoexo_20260916','pccs_corrected_exoego_20260916'):
 p=O/folder/'resource_release.json'
 if not p.exists():continue
 rr=load(p);cells=rr.get('cells',[])
 if len(cells)>9:parts.append(f"| {cells[0]} | {cells[9]} | {cells[7]} |")
Q=O/'pccs_candidate_quality_20260916'
if (Q/'PLAN.md').exists():
 parts+=['','## 8. 新增候选质量 2×2 消融（2026-09-16）','','概率加权池化和可靠多点对应现已进入实际测试。四组为baseline、只改预测coarse-mask池化、只改可靠多点、两者结合；主体权重均冻结。源提示mask池化不改，加权分支保留原采样的RNG消耗；多点为严格双向最近邻、目标第一/第二匹配间隔≥0.01、双侧间隔≥2patch、最多3点，至少2点可靠时采用，否则退回原单点。',
 '','先用8对进行运行预检，再用128对／16训练序列筛查、另128对／16不重叠训练序列校准。上游专家已看过官方训练数据，故这不是独立测试收益。方案须同时改善筛查与校准候选oracle，且校准的固定几何一致性最终IoU为正增益；满足条件者按校准最终IoU选择，之后才比较预选方案与baseline的Exo→Exo结果。无合格方案则报告负结果，不在目标测试集调参。',
 '','分支隔离门要求：只改池化时Anchor不变；只改点时Visual不变；组合方案的对应分支与单项方案一致。不通过即停止。每个arm使用自己的新候选，并比较arm之间的同名最终选择方法；不能把某arm的新方法分数减去异次旧基线。']
 armnames={'baseline':'原始候选','weighted_pool':'概率加权池化','reliable_points':'可靠多点','both':'两者结合'}
 found=False
 for section_index,(phase,title) in enumerate([('screen','训练内筛查'),('calibration','训练内校准'),('exo2exo','预选方案 Exo→Exo 确认')],1):
  p=Q/(phase+'_summary.json')
  if not p.exists():continue
  a=load(p);found=True;parts+=['',f'### 8.{section_index} {title}','', '| 候选方案 | Oracle IoU | 原PCCS IoU | 几何一致性 IoU | 多点对象数 |','|---|---:|---:|---:|---:|']
  for arm,v in a['arms'].items():parts.append(f"| {armnames[arm]} | {v['frame']['oracle'][0]*100:.4f} | {v['frame']['pccs'][0]*100:.4f} | {v['frame']['consensus'][0]*100:.4f} | {v['multipoint_objects']} |")
 if not found:parts+=['','**当前结果待定，尚不宣称这两项改动产生收益。**']
 if (Q/'selection.json').exists():
  selection=load(Q/'selection.json');parts+=['','训练内选择结果：`'+selection['selected']+'`；规则按预注册计划执行。']
  if selection['selected']=='baseline':
   parts+=['','**本轮三个改动方案均未通过晋级条件，保留原始候选，未启动Exo→Exo目标测试。** 这是冻结权重下的直接替换实验负结果，不是任务运行失败，也不代表已证明所有概率池化或多点方法无效。',
    '','在筛查／校准两个阶段，池化、点及组合的候选oracle均下降；因此损失已经出现在候选生成，而不只是最终选择器没有选好。直接把原随机非零区域采样替换为稠密概率平均，同时改变了特征聚合分布；原训练网络可能已经适应旧分布，但该解释尚需描述符分析或轻量适配验证。',
    '','多点实际在筛查55/243对象、校准55/225对象启用，最多3点，其余回退单点。双向最近邻和间隔筛选并未在本轮转化成更好的候选；不能把特征匹配的“可靠”直接等同于正确的分割提示。所有跨arm隔离门mismatch=0。',
    '','### 8.3 校准集相对原始候选的变化','', '| 改动 | ΔOracle IoU（点） | Δ最终一致性 IoU（点） | 最终变化95%序列区间 |','|---|---:|---:|---|']
   c=load(Q/'calibration_summary.json')
   for arm,values in c['comparisons'].items():
    ci=values['consensus']['take_bootstrap_95ci_pp'];parts.append(f"| {armnames[arm]} | {values['oracle']['delta_iou_pp']:.4f} | {values['consensus']['delta_iou_pp']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
   parts+=['','下一步如继续，应优先做较小的分布适配：保留原池化并加入小比例加权分量，在训练／校准上选择混合强度；或只微调prompt投影。多点可改成保留原单点、谨慎追加额外点并做质量检查。这些在§8结束时仅为未测试假设；随后已在独立实验中按顺序测试，结果见§9。V2-SAM prompt投影微调仍未执行。']
 parts+=['','本节候选质量实验使用5000次序列bootstrap；前面的选择器实验使用10000次。实现与完整回执：[候选质量实验计划](../pccs_candidate_quality_20260916/PLAN.md)。早期选择器实验的涨点不能归因于本节新增候选改动。']
from append_union_report import append
append(parts,O)
report.write_text('\n'.join(parts)+'\n',encoding='utf8');assert report.read_text(encoding='utf8').count('```')%2==0
print(json.dumps({'report':str(report),'characters':len(report.read_text(encoding='utf8')),'completed_sections':list(completed),'sha256':hashlib.sha256(report.read_bytes()).hexdigest()},ensure_ascii=False))
