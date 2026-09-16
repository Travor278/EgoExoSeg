import json
def append_native(parts,root):
    r=root/'pccs_native_context_20260916'
    if not (r/'native_context.py').exists():return
    parts+=['','### 10.1 已实现的原生PCCS context模块','','复用原Visual/Anchor和Fusion反向对应模块已提取的DINOv3特征，额外编码器前向次数为0。用源/预测mask外的1.75倍bbox邻域，检查全图双向最近邻是否落入对方物体邻域；匹配要求cosine≥0.4、top1−top2≥0.02，双向至少各1个、合计4个。可靠性r=min(1,N/8)×sqrt(平均有效覆盖率)，r≥0.2时原循环成功票乘以1−0.5r(1−support)；Fusion的可靠context支持<0.25则转入原循环复核。', '', '原函数`_count_points_in_mask`实际检查源mask外接矩形，首版保留这个精确定义及原距离平票规则。context权重在首版按对象/专家区域共享，并非逐点关系网络；错位对照将源邻域平移半幅图像且扣除源前景。没有O-MaMa匹配头、额外DINOv2或新参数训练。', '', '所有设置共享原三专家mask；8对smoke另做捕获开/关完整预测签名检查，各阶段验证λ=0精确回归及与旧baseline的候选SHA/PCCS路由一致。当前训练内样本此前已参与探索，不能称为独立验证。']
    p=r/'status.json';state=json.loads(p.read_text()) if p.exists() else {'state':'queued'}
    parts.append(f"\n最近已核验状态：{state.get('state')}，阶段{state.get('phase','awaiting startup')}。结果只在精确覆盖及一致性检查通过后写入下表。")
    names={'baseline':'原PCCS','zero':'λ=0','cycle':'仅循环加权','review':'仅Fusion复核','both':'两者结合','wrong_context':'错位context对照'}
    for phase in ('screen','calibration','exo2exo'):
        p=r/(phase+'_results.json')
        if not p.exists():continue
        a=json.loads(p.read_text());parts+=['',f'### 10.2 {phase}（{a["pairs"]}对/{a["objects"]}对象）','','| 设置 | Frame IoU | ΔIoU（点） | 95% take区间 | 改善/变差对象 |','|---|---:|---:|---|---|']
        for n,v in a['methods'].items():
            ci=v['ci95_pp'];parts.append(f"| {names[n]} | {v['frame'][0]*100:.4f} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {v['improved']}/{v['harmed']} |")
        parts.append(f"\n原Fusion接收{a['fusion_accepted_objects']}对象，其中context触发复核{a['fusion_recheck_objects']}；可靠context对象数{a['reliable_objects_by_expert']}。原候选逐位一致，λ=0路由一致。错位对照不得参加选型，其他组件的区间作为探索性对照解释。")
    p=r/'selection.json'
    if p.exists():
        s=json.loads(p.read_text());parts.append('\n训练内冻结选择：`'+s['selected']+'`。')
        if s['selected']=='baseline':parts.append('三个新增设置均未同时通过筛查与校准的正收益条件；保留原PCCS，未进入目标测试。')
    parts+=['','源码与冻结计划：[原生context实验](../pccs_native_context_20260916/PLAN.md)。']
