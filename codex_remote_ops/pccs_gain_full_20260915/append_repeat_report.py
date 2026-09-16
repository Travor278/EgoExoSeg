import json

def append_repeat(parts,root):
    r=root/'pccs_roi_seed_repeat_20260917';p=root/'pccs_roi_context_20260917'
    if not (r/'job_receipt.json').exists():return
    parts += ['', '## 14. 冻结模型的候选随机种子复验（2026-09-17）', '',
      '本轮仅把候选生成的SHA256种子前缀从candidate-quality-v1:改为candidate-quality-v2:。1.5倍原主方案、2倍关键次要对照、其他同配置对照及此前cycle模型全部冻结，不重新训练、改阈值或按结果再选随机种子。数据及每rank顺序、专家权重、DINOv3参数/精度、局部ROI模块与指标保持；新旧候选可以不同，但每轮所有方法必须共用自己的同一候选bank。', '',
      '1.5倍与2倍均通过完整PCCSMetric入口计算选择，并与清除GT指标字段的离线策略逐对象对照。启动前核对专家文件SHA和所有模型SHA；先8对新seed冒烟，再用旧bank已知样本复现O-MaMa参考分数，随后才跑Exo→Exo全量与既有Exo→Ego512对。参考网络在独立进程运行，只是外部对照；新候选对应的参考分数重新计算，只有候选SHA及原路由完全相同时才复用旧值。', '',
      '这是已有模型在重复观察基准上的随机性稳健性检查，不是全新盲测；两个seed不能当作新拍摄序列扩大样本量。2倍仍不能追认为上一轮的原主方案。完整预先声明见[复验协议](../pccs_roi_seed_repeat_20260917/PLAN.md)。']
    completed=False
    for phase in ('exo2exo','exo2ego'):
        path=r/(phase+'_results.json')
        if not path.exists():continue
        completed=True;new=json.loads(path.read_text());old=json.loads((p/(phase+'_results.json')).read_text())
        parts += ['',f'### {phase}：各seed分别报告','', '| 方法 | seed1 IoU | seed1 ΔPCCS | seed2 IoU | seed2 ΔPCCS | seed2 95%区间 |', '|---|---:|---:|---:|---:|---|']
        for name in ('baseline','frozen_cycle','primary','local20_matched','global_local20_matched','global_object20_matched','object20_matched','omama_reference'):
            if name not in new['methods']:continue
            a=old['methods'][name];b=new['methods'][name];ci=b['ci95_pp'];parts.append(f"| {name} | {a['frame'][0]*100:.4f} | {a['delta_pp']:+.4f} | {b['frame'][0]*100:.4f} | {b['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        parts += ['',f"新seed中有{new['seed1_bank_changed_objects']}个对象的候选/原路由/原指标与seed1不完全相同；这是预期检查项目，不删样本、不要求新seed复制旧mask。"]
    if not completed:parts += ['', '当前状态：代码与本地冻结/种子检查已完成，四卡任务已提交，尚无新seed的完整目标结果。']
    if all((r/'runs/reference_probe'/f'rank{k}/witness.json').exists() for k in range(4)):
        errors=[json.loads((r/'runs/reference_probe'/f'rank{k}/witness.json').read_text())['max_score_error'] for k in range(4)];parts += ['',f"新seed的8对pipeline冒烟已通过；四张卡分别复现旧bank见证样本的O-MaMa专家选择，最大分数绝对误差{max(errors):.8g}（预设容差1e-4）。完整benchmark比较仍须使用新候选对应的参考输出。"]
    if (r/'reference_updates.json').exists():
        a=json.loads((r/'reference_updates.json').read_text());parts += ['',f"外部参考重新计算{a['count']}个对象，复用完全一致bank的{a['reused_identical_bank']}个对象；每rank均用旧bank见证样本校对分数。"]
    if (r/'exo2exo_results.json').exists():parts += ['', 'Exo→Exo复验：预选1.5倍主方案在seed1/seed2分别提升2.7498/2.6975点；2倍次要对照分别提升3.3101/3.2772点。第二seed原PCCS39.0686，2倍42.3458，同bank O-MaMa42.5391。2倍相对此前cycle额外1.1718点，描述性区间[0.5506,1.7985]；与O-MaMa相差-0.1934点，区间[-1.6148,1.2403]，仍不是等效证明。', '', '1094对全部完成，Visual有1091个mask变化、Fusion1083个、Anchor0个；每对候选集合均有变化，所以O-MaMa参考1094个全部重算。独立核验包含新seed计算、同一数据键集合、三个实际metric路由（主方案/2倍/旧cycle）、frame聚合及参考专家由保存分数重放。原PCCS本轮434个零IoU对象保留。真实背景对前景-only的描述性差仍为正：local20差2.6951点，[1.6514,3.5674]；全图联合差2.1238点，[1.1205,3.0302]。灰背景分布变化的解释边界不变。']
    if (r/'exo2ego_results.json').exists():parts += ['', 'Exo→Ego复验（仍为512对/965对象）：第二seed原PCCS71.2670，原主方案72.4494（+1.1824点，[0.5496,1.8664]），2倍次要对照72.3418（+1.0748点，[0.2968,1.8486]）。两者在两个seed均取得相对原PCCS的正平均增益，置信区间下界均高于零。2倍相对旧cycle的第二seed额外变化为+0.2515点，区间[-0.3646,0.7746]，因此不能称其在Exo→Ego上对旧cycle的额外优势已稳定显著。965对象的bank均有变化，Visual962/Fusion958/Anchor0；本轮132个零IoU对象保留。', '', '方向差异必须保留：第二seed Exo→Ego的2倍真实背景比前景-only略低0.0238点，区间[-0.5772,0.3826]；全图联合的真实背景优势也跨零。总体ROI增益与context的额外作用是不同结论，当前真实背景贡献的较强证据限于Exo→Exo。', '', '**阶段结论：已经找到在原PCCS基础上、两个候选种子下均能复现的正向改进；2倍局部视图在主目标Exo→Exo上获得约3.3点增益，接近同bank O-MaMa的点估计。** 仍保留1.5倍原主方案/2倍次要对照的事前身份，不宣称统计等效或所有新数据必涨点，也不将两个seed合成新takes增加样本量。本轮有界复验到此结束，参数保持冻结，不再追加种子挑结果。', '', '如后续开展新的研究，应分别验证自然背景错配控制、独立数据或新模块的Exo→Ego全量；当前新ROI的Exo→Ego结论仅限这512对，不能用较早O-MaMa全量实验替代。']
    if (r/'resource_release.json').exists():
        a=json.loads((r/'resource_release.json').read_text(encoding='utf-8-sig'))['cells'];parts += ['',f"任务{a[5]}，占用节点{a[7]}，结束{a[9]}（北京时间）。"]
