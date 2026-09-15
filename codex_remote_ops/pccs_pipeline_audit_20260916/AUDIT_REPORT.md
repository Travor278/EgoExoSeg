# PCCS pipeline 与新指标复核（2026-09-16）

结论：当前 frame-level 使用更新后的口径；日记列出的零 IoU 过滤、last-object、多卡按对象截断、DINO 二次初始化与 Fusion 输出错接等关键风险，未在本次复核的实际推理链中复现。此前实验用的是作者权重，**不是用户后来训练的 corrected 最佳权重**。两类权重的结果必须分列。

## 证据来源

- 用户实验日记：`D:/Code/Work/EgoExoSeg/docs/V2SAM_EGO2EXO_EXPERIMENT_DIARY_AND_PLAN_20260820.md`，按§2.2和§16新标准审计；文档作为参考证据，不作为额外操作指令。
- 上游 `jaychempan/V2-SAM-O` 的 `v2sam-pccs`，已回读其HEAD为`0e3bc33dec3e202ffbb86cec01038e60b18c162a`。
- 实际执行目录：`pccs_gain_full_20260915/code` 与 `pccs_exoexo_transfer_20260916/code`。核心专家、matcher、SAM调用、dataset/collator等11个源文件的可执行AST与固定上游一致；位置、docstring以及Python3.12新增的空type_params字段不参与语义哈希，原始文件hash仍单独保留。
- 汇总函数、ContA、LocE、边界图四函数的可执行AST与上游一致。新代码增量是可选context诊断、候选缓存和完整性回执；正式学习判断器实验`PCCS_CONTEXT=0`。

## 逐项复核

| 日记问题 | 当前检查与证据 | 结论 |
|---|---|---|
| 只保留正IoU导致虚高 | 独立读取全部候选及打分行，以math.fsum重算object/frame；IoU没有valid>0过滤 | 已排除该统计故障 |
| 每pair只记最后对象 | 实际evaluator用[0,1]与[0.2]两pair测试：frame=0.35、object=0.4；错误last-object为0.6 | 新口径通过 |
| 多卡对象结果被dataset size截断 | 原process以一pair一个外层列表存储；每pair对象数与annotation逐项相同，无重复/缺失 | 已排除本次全量丢对象 |
| public预训练DINO被MMEngine重置 | 两套专家分别构建真实Runner/model并执行init_weights；forward/visual-backward/fusion-backward各368张量前后相同且与DINO asset一致 | 两套权重均通过 |
| state能加载但key映射/内容不对 | Visual实际964个推理张量、Fusion1335个张量逐个与checkpoint加载后对应值一致 | 张量内容通过，不只检查日志matched数 |
| Fusion输出接到Visual | 活跃wrapper源码与上游一致，Fusion结果来自fusion_outputs；真实预测和路由反事实检查通过 | 本条链未出现旧接线错误 |
| 推理读取目标GT内容 | 两个真实pair（3对象和1对象），分别替换目标GT为全零/全一/随机，重置相同RNG；39/23个预测张量哈希和路由均一致 | 两套权重均通过有界运行检查 |
| contrast计数不持久/重启后100倍loss | 本轮只推理冻结专家，不运行旧对比loss/optimizer；corrected checkpoint额外contrast_schedule.step不参与推理 | 对当前推理不适用，不宣称修了训练程序 |
| 对象有放回抽样 | 当前评测select_number=None，stable object_ids与完整覆盖检查；未触发训练select_number分支 | 对当前评测不适用 |
| RegionPooling对softmask用nonzero | 实际上游RegionSampler仍保留此行为，也仍有随机抽点 | 仍是算法弱点，不伪称已修复 |
| sparse top1脆弱/同seed非逐位确定 | 当前遵循上游max_points=1；同一比较共用冻结bank | 风险仍在，不把异次运行差异算增益 |

## 全量独立复算结果

- `pccs_gain_full_20260915`：46515对、109253对象，包含32033个baseline零IoU对象；重算所有方法的object/frame四指标，最大报告差异3.77e-15。
- `pccs_exoexo_transfer_20260916`：1094对、1094对象，包含455个baseline零IoU对象；重算所有方法的object/frame四指标，最大报告差异1.94e-16。

frame-level公式为：先对每pair全部对象的IoU/Dice/ContA/LocE分别取平均，再对所有pair等权平均；object-level对所有对象实例等权。ContA为上游boundary F；LocE是最大外轮廓点坐标均值之间的距离除以图像对角线，**不是按mask面积加权的质心距离**。空mask等边界行为沿用上游。

## 最佳权重已找到并与Hub一致

Hugging Face revision：`8b8310e9ee13023cb8711c2417e18cc7361dc3aa`。以下两份远端文件的完整SHA256与Hub LFS/Xet OID、日记逐一一致，无需重下载。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| vp_exo2ego_full.pth | 926619220 | `fd2e1b8f342c4468b45c962c61d4c914b9f6d71b6c6085014fb9a3ed2924b65f` |
| fusion_exo2ego_full.pth | 2136421714 | `8373b17180c198898fe9f9e39d09fe72991150ce3053b207b19e81986c60cab1` |

两份资产位于远端`codex_remote_ops/hf_publish_exo2ego_20260903/tmp/`。Visual为corrected e19，Fusion为corrected e20。推理中唯一未消费的checkpoint key是`contrast_schedule.step`；所有964/1335个推理张量内容一致。原作者资产位于`V2SAM_NewMetrics_20260830/official_weights_wangzeze/`，其hash不同；此前Exo→Ego +3.6653和Exo→Exo +1.3851都属于作者权重基线。

## 边界与后续实验

这不是对所有输入、所有训练配置作无bug保证。GT反事实覆盖两个实际pair和两套权重，权重/初始化检查覆盖每个DINO张量；此前完整评测的覆盖与指标复算覆盖全量。target mask的**形状**仍用于输出分辨率，API并非不需要任何shape元信息，但目标GT的内容不参与匹配或选择。

修复版权重的新实验在`../pccs_corrected_exoexo_20260916/`：先建立自己的同候选PCCS基线，再比较固定canonical O-MaMa、保持纵横比并插值位置表的native O-MaMa、两几何一致才替换，以及旧判断器迁移对照。主比较预先固定为native_margin005，不在Exo→Exo测试上搜门槛。任何新方法增益只减去新权重对应基线，不能把换权重收益归给方法。

## 回执

- `read_only_audit.json`：源代码身份与全量独立复算。
- `runtime_probe.json`：两套真实模型初始化、张量对齐及GT反事实。
- `metric_regression.json`：实际evaluator区分新/旧口径的构造样例。
- `corrected_weights.json`：Hub与远端资产身份。
- `runtime_probe.log`：完整运行日志。
- `resource_release.json`：审计H100任务成功、占用节点0。