# PCCS dense-cycle/context batch — 2026-09-17

目标：在PCCS自身模块内实现接近外接O-MaMa组合的增益；不使用O-MaMa网络、DINOv2或其权重，不把借鉴思路直接写成原创性结论。当前状态：代码已实现，等待真实模型测试。所有原始候选、点提示、专家参数和原PCCS规则保持，新增模块作为有条件的循环置信度修正。

## 实现分析与改进依据

[O-MaMa官方实现](https://github.com/Maria-SanVil/O-MaMa)的mask/context encoder使用区域平均，邻域框包含前景；cross-attention从对侧整图取软信息，随后以预训练MLP组合物体/邻域/跨图表征，并用候选匹配及邻近难负样本训练。不能把它的收益简单解释为“背景环相似”。前一版native模块使用严格MNN和邻域落点判定，且在低支持时否决Fusion；它可能把跨视角不可比的邻域当成冲突，实际误伤了较好Fusion结果。

本轮借鉴三个有用动机：区域证据比单个点丰富；跨图对应应表达不确定性；邻近背景可帮助识别混淆。对应到我们自己的模块如下，代码没有复制其Attention_projector/Context_Attn/MLP实现或加载其参数：

| 路线 | PCCS内的具体实现 | 对照 |
|---|---|---|
| 稠密软循环 | 原DINOv3 patch余弦矩阵的双向softmax，累计源mask→目标候选及候选→源mask的质量，并计算两跳返回质量 | 原硬点循环；仅软循环 |
| 前景/邻域相对证据 | 在1.5/2倍bbox外环上计算前景对邻域混淆量、去背景后的区域余弦和软循环相对邻域泄漏量 | 同特征不含context；错位context |
| 循环置信度校准 | 以原点票数/距离/SAM质量，加上稠密循环及context特征，校准候选相对原PCCS的质量差 | base/cycle/context/full同容量消融 |

不是因当前Fusion的context偏低就强制退回VA。每次修正都要求另一个实际存在、非重复的候选通过原mask质量门，并且它的soft-cycle支持比原PCCS候选高至少0.005、绝对值>0.01。未满足条件或禁用修正时直接保留原PCCS。原PCCS的硬票数仍检查源mask外接框，原单点提示与专家pooling均不改。soft-cycle是双向mask传输质量的调和均值；另单独记录真正的两跳返回质量cycle_mass，不能把两者混写。

## 冻结实验范围

1. 8对smoke：Capture开/关完整预测签名必须相同；关闭policy或strength=0直接返回原选择；已有基线mask SHA与选择逐条复核。稠密循环另做正交特征恒等、错误对象、空mask数值测试。
2. TRAIN fit由16个take扩到既有48个take×8=384对，另16个不重叠take×8=128对校准。专家已见过官方TRAIN，校准也被多次探索过，不能当新盲测。旧cache覆盖的训练子集/校准全部验证候选逐位一致，新增fit样本记录新基线。
3. 8个无需训练的规则：两种软循环门槛；1.5/2倍前景-背景contrast；1.5/2倍残差区域余弦；1.5/2倍循环-context联合支持。具体固定配置见policies.py，缺乏有效context时回退。
4. 4组置信度输入base/cycle/context/full × ridge10/ridge100/小树 × 替换阈值0.03/0.05，共24个校准器配置。base组也共享同样的软循环准入条件；它是不含新增描述符的校准对照，不是完全没有稠密循环计算的系统。全部在48个fit take内做4-fold GroupKFold，产生逐对象OOF选择；每fold仅在其训练take拟合。预测目标为候选IoU减原PCCS IoU，权重1/(pair对象数×可用替代候选数)。最终模型只在fit数据重训，校准不进入优化。
5. 仅TRAIN OOF（固定规则为其TRAIN表现）与校准都正增益的配置可晋级，按校准IoU选择唯一主方案并冻结hash；无合格方案则保留baseline。目标测试只作冻结确认。训练方案同时冻结与主方案相同模型/门槛的4组输入消融；错位context保持原模型与准入条件，仅干预context特征，不参与选型。

## 接入与确认

`native_bridge.py`在原PCCS得到初始expert之后，调用本轮native循环置信度修正；默认无配置时原样返回。依赖本实验目录在PYTHONPATH，prepare.py把接入点写入隔离的PCCS代码。目标worker将本轮DINOv3证据送入原metric，通过真实PCCS路径再作一次决策；evaluate.py要求它与冻结policy离线回放逐对象一致。不是仅计算一个分数表后宣称完成集成。

若晋级：使用Exo→Exo1094对/20take，同一pair种子、corrected Visual e19/Fusion e20。三专家mask与已完成候选bank逐条比较；O-MaMa只读取该**相同候选**上的冻结历史输出作为参考，不出现在本轮推理特征、训练目标或决策输入中。报告原PCCS、主方案、输入消融、错位对照、O-MaMa参考及主方案相对O-MaMa的配对区间。允许报告接近它的增益，但不能承诺预定涨幅或在目标重选。

指标：逐pair先平均对象，再平均pair，0 IoU保留；10000次take聚类bootstrap。给出边界/定位指标与改选利弊，不只报IoU。全部脚本、配置和负结果保存在本目录；唯一总报告仍为../pccs_gain_full_20260915/FINAL_REPORT.md。四H100单任务，成功/失败保留1分钟，监督间隔0.8×当前阶段ETA，至少1分钟，无任意上限，初始化/切换阶段用5分钟。

## 历史缓存一致性诊断与恢复协议

R1在Exo→Exo一个knife对象处触发历史候选SHA检查而停止。独立单卡probe对同一输入/seed连续运行OFF、OFF、ON、ON、OFF：Visual/Anchor及原路由一致，第一次OFF的Fusion与历史相同，第二次OFF已相差25像素，之后ON/ON/OFF保持第二次结果；IoU/Dice/ContA都是0，LocE有小幅差异。故不能将此差异归因于新context；具体数值见failure_probe.json。这里只确认原推理重复运行有差异，不将底层原因猜测为已证实的“warmup bug”。

因此恢复时固定每个对象本轮实际生成的三专家候选，所有native方法共用；保留所有历史SHA/路由/指标差异及像素差数，绝不直接挪用不同候选的旧分数。历史一致条目沿用已核验参考，差异条目保存当前mask bank，在独立进程用原O-MaMa作外部参考重算，并以历史一致条目验证该参考实现。O-MaMa只存在于比较基线进程，不进入新方法特征、训练或决策。原始记录不覆盖，参考补充存为reference_updates.json，检查其mask SHA与当前记录完全相同；native决策前后mask SHA也必须相同。

R1及gate_study的参数选择均保持冻结，训练/校准不重跑。恢复追加尚未完成的目标记录，完成后分别报告两轮冻结主方案、同容量对照、错位context及同bank O-MaMa参考。gate_study是利用同一特征bank的决策回放；它通过native_bridge接口对真实缓存mask进行逐对象一致性核验，不声称重新运行了整套分割网络。
