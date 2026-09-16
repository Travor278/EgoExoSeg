import json
def append_dense(parts,root):
    r=root/'pccs_dense_context_20260917'
    if not (r/'PLAN.md').exists():return
    parts+=['','## 11. PCCS稠密循环与context多方案（2026-09-17）','','重新核对O-MaMa实现后，本轮借鉴区域证据、跨图软信息和邻近干扰消歧的动机，将其实现为PCCS内部的循环置信度修正；不加载O-MaMa网络、DINOv2或其参数。8个固定规则与24个小型校准器配置共享原三专家候选。完整定义：[本轮计划](../pccs_dense_context_20260917/PLAN.md)。', '', '### 11.1 机制与接入', '', '在原反向DINOv3特征上对patch余弦矩阵以τ=0.07做双向softmax。令u为源mask向目标候选传输的平均概率质量，v为候选向源mask传输的平均质量，soft-cycle=2uv/(u+v)；另记录源mask→候选→源mask的两跳返回质量。它提供硬循环票数为0时仍可比较的连续证据；两种统计不能混写。', '', '在1.5/2倍bbox外环分别计算：物体区域余弦；物体与对侧背景的交叉相似度；前景相对背景的差；减去0.5倍本侧背景原型后的区域余弦；对象循环质量相对邻域泄漏的差。对应关系仍来自自己的DINOv3匹配，不复制O-MaMa的可学习cross-attention/MLP结构。', '', '原PCCS先运行，只有另一候选通过原mask质量门、soft-cycle高于原选择至少0.005且绝对值>0.01时，才允许修正。无配置/strength=0直接返回原结果；缺少上下文不会单独否决Fusion。native_bridge.py接入原PCCS决策位置，目标阶段要求实际接口决策与冻结policy回放逐对象一致。', '', '### 11.2 训练、对照与选择', '', 'fit扩大到48个take/384对，另16个take/128对校准；原专家已见过TRAIN，当前校准也曾用于探索。固定规则不训练；小校准器仅拟合原PCCS点票数/距离/SAM质量及本轮证据对应的质量差，不微调专家或骨干。4-fold按take留出产生OOF预测，校准不进入拟合。base/cycle/context/full四种输入共享相同软循环准入；各用ridge10、ridge100、3叶小树和0.03/0.05门槛。', '', '仅OOF（固定规则为TRAIN表现）和校准都正的方案可晋级，按校准选唯一主方案并冻结。同容量输入消融及错位context一起冻结，目标不得重选。O-MaMa只读取相同旧候选bank上的冻结输出作参考，不作为本轮训练/推理输入。判定是否接近它的增益应看同一轮配对比较和区间。']
    p=r/'status.json';state=json.loads(p.read_text()) if p.exists() else {'state':'queued'};parts.append(f"\n当前已核验状态：{state.get('state')}，阶段{state.get('phase','awaiting startup')}。未完成时不宣称涨点。")
    p=r/'search.json'
    if p.exists():
        candidates=json.loads(p.read_text())['candidates'];best={}
        for x in candidates:
            group=x.get('group','hand')
            if group not in best or x['calibration']['frame'][0]>best[group]['calibration']['frame'][0]:best[group]=x
        parts+=['','| 家族中校准最高配置（仅摘要） | TRAIN OOF/固定规则ΔIoU | 校准ΔIoU |','|---|---:|---:|']
        for x in best.values():parts.append(f"| {x['name']} | {x['train_oof']['delta_pp']:+.4f} | {x['calibration']['delta_pp']:+.4f} |")
        parts.append('\n此表不等于晋级名单，完整搜索与负结果见search.json；必须另满足OOF为正。')
    p=r/'selection.json'
    if p.exists():
        s=json.loads(p.read_text());parts.append('\n冻结主方案：`'+s['selected']['name']+'`。')
    p=r/'exo2exo_results.json'
    if p.exists():
        a=json.loads(p.read_text());parts+=['','### 11.3 Exo→Exo冻结确认','','| 方法 | IoU | 相对原PCCSΔIoU | 95%take区间 |','|---|---:|---:|---|']
        for name,v in a['methods'].items():
            ci=v['ci95_pp'];parts.append(f"| {name} | {v['frame'][0]*100:.4f} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        v=a['primary_vs_omama'];ci=v['ci95_pp'];parts.append(f"\n主方案相对冻结O-MaMa参考：{v['delta_pp']:+.4f}点，95%区间[{ci[0]:.4f}, {ci[1]:.4f}]。主方案在PCCS实际接口中的选择与离线回放一致。不能把不含context校准器的增益归到context。")
        parts+=['','**原生方案在当前基准获得正增益，但没有追平O-MaMa的点估计。** 第一批预选主方案是cycle_ridge10_t0.03，使用原PCCS信息、软循环统计及区域物体余弦，不包含显式邻域描述符；因此其错位context对照应与主方案相同。显式context/full同容量对照没有优于该主方案，当前收益不能写成“局部邻域context已证明有效”。']
    if (r/'gate_target_results.json').exists() and (r/'lift_study/target_results.json').exists():
        g=json.loads((r/'gate_target_results.json').read_text());l=json.loads((r/'lift_study/target_results.json').read_text());parts+=['','### 11.4 缓存上的后续机制消融','','训练内发现：校准82个对象有更好的合格替代候选，但原始软循环硬门只覆盖7个。第二批据此比较原门、非零支持门和跨尺度正向context准入；第三批按随机对应的面积先验，对forward/backward/两跳循环概率作log-lift校正。两批都重放同一48-take划分，源OOF与校准选型后才读取本轮完整目标结果。', '', '| 冻结方案 | TRAIN OOF ΔIoU | 校准ΔIoU | Exo→Exo ΔIoU | 目标95%区间 |','|---|---:|---:|---:|---|']
        for label,source,target in [('相对context准入',json.loads((r/'gate_study/selection.json').read_text())['selected'],g),('面积先验校正',json.loads((r/'lift_study/selection.json').read_text())['selected'],l)]:
            v=target['methods']['primary'];ci=v['ci95_pp'];parts.append(f"| {label} | {source['train_oof']['delta_pp']:+.4f} | {source['calibration']['delta_pp']:+.4f} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        parts+=['','第二批主方案通过真实native_bridge入口对缓存mask逐对象核验，且标签反事实不改变选择；第三批是冻结policy的缓存回放，其R1控制逐对象复现真实接口结果，尚未单独部署第三批到新分割推理任务。两批均没有超过第一批的目标点估计；保留各自预选结果，不用目标集重新选超参数。第三批相对O-MaMa区间全部低于0，不能称为等效。', '', '本地重新生成GroupKFold时发现折分配与远端不同，首次本地复现停止于选型之前；随后直接使用远端保存的take列表，原R1各配置的OOF和校准差值精确复现。该诊断及第一次结果保留在gate_study中。']
    p=r/'failure_probe.json'
    if p.exists():
        parts+=['','### 11.5 推理重复性、共享候选与核验','','目标首次在knife对象处触发历史SHA检查并停止。单卡重复OFF/OFF/ON/ON/OFF确认：第二次未加新模块的Fusion本身与首次相差25像素，Visual/Anchor和原路由不变，IoU/Dice/ContA均为0，LocE略变；不能将差异归因于context，也没有证实底层成因。恢复采用固定当前候选bank、记录历史差异、必要时在独立参考进程重算O-MaMa的协议，避免混用候选。', '', '最终恢复的1094对全部与历史候选SHA及原PCCS输出一致，实际参考重算数为0。三批主方案的指标与10000次take bootstrap独立复算通过，原PCCS440个零IoU对象保留。原生方法没有使用O-MaMa网络或权重。完整证据见final_validation.json、failure_probe.json及各阶段回执。']
    p=r/'resource_release.json'
    if p.exists():
        c=json.loads(p.read_text(encoding='utf-8-sig')).get('cells',[])
        if len(c)>9:parts.append(f"\n本批资源已释放：{c[5]}，占用节点{c[7]}，结束时间{c[9]}（北京时间）。")
    if (root/'pccs_roi_context_20260917/PLAN.md').exists():parts+=['','### 11.6 下一步：原生局部视图循环','','目前已获得约2.3点原生增益，距离同bank O-MaMa约1.06点。下一步研究原DINOv3的物体局部视图与全图循环联合，并加入同分辨率前景-only crop对照，区分放大细节与context的作用。ROI几何模块和测试已准备，完整实验尚未启动；计划见[局部视图循环](../pccs_roi_context_20260917/PLAN.md)。', '', '几何说明修正：原BackwardCorrespondenceMatcher实际上保持图像纵横比，将高度归一到768、宽度取16倍数。早期native_context.py的“方形网格”注释不准确，但执行代码使用实际返回网格尺寸处理mask，没有硬编码方形。不能用“原图被拉成正方形”解释此前负结果。']
