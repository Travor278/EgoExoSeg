import json

def append_roi(parts,root):
    r=root/'pccs_roi_context_20260917'
    if not (r/'job_receipt.json').exists():return
    parts += ['', '## 13. 原生局部视图循环（2026-09-17）', '',
      '固定范围池化无法恢复已在全图patch中混合的细节，本轮检验额外物体局部视图是否改善原PCCS候选判断。原分割候选、原hard correspondence及回退不变；用现有DINOv3提取1.5/2倍物体bbox长边的方形裁图，形成局部双向mask传输、循环概率、面积先验log-lift与物体余弦证据，加入原PCCS挑战候选的置信判断。不导入O-MaMa编码器、投影头或权重。', '',
      '裁图移入图像边界，保留完整前景；小于16像素或长边无法纳入图像短边的对象不提供局部证据，保留原规则。最小裁图边长32；记录有效倍率，边界饱和的1.5/2倍裁图不能视为独立证据。前景-only对照使用完全相同的2倍crop/分辨率及前景RGB，将背景替换为[124,116,104]。这有分布变化，不能仅凭该干预宣称某个邻居物体的因果作用。', '',
      '源域预声明7个特征组×3种小判断器×2个阈值×2个准入门，共84组；同一准入门下所有组的候选集合相同，准入不读取ROI信息，避免把放宽门槛当context贡献。组别包括全图、局部1.5/2倍、前景-only及全图联合；两个门为原cycle dominance和原始soft-cycle>.001。仍用48个fit takes、16个cal takes及保存的四折，按正OOF/正校准后最高校准冻结，所有同模型/同门/同阈值的组构成目标对照。原global/dominance须精确复现此前源OOF/校准。小Ridge/HGB使用TRAIN标签，专家与DINOv3冻结，不称整个方法training-free。', '',
      '运行顺序：8对原pipeline smoke → 同一bank的独立缓存smoke → TRAIN384对 → 校准128对 → 冻结 → Exo→Exo1094对缓存评估 → Exo→Ego既有512对holdout。目标标签不进入策略。缓存smoke核对编码器全state SHA、TF32/autocast设置、批量/单图及逐项局部统计；避免用不同特征路径悄悄替换原生模块。Source保存私有mask缓存，Exo→Exo直接复用已经验证的候选，最终真实native_select接口须与离线选择逐对象一致，并通过清除GT指标字段的反事实检查。', '',
      '将报告额外编码视图数、局部处理时间、GPU峰值；当前同时提取全部消融视图，不能称其耗时为单个获胜方法的优化部署速度。详细设计与边界见[局部视图计划](../pccs_roi_context_20260917/PLAN.md)。']
    if not (r/'selection.json').exists():parts += ['', '当前状态：代码与几何/准入回退检查已完成，四卡任务已提交，尚无本轮源域选型或目标增益结果。']
    if all((r/'runs/cache_smoke'/f'rank{k}/receipt.json').exists() for k in range(4)):
        receipts=[json.loads((r/'runs/cache_smoke'/f'rank{k}/receipt.json').read_text()) for k in range(4)]
        assert all(x['source_encoder_state_parity'] and x['source_roi_feature_parity'] for x in receipts)
        parts += ['', 'GPU冒烟已通过：8对原pipeline Capture开关/候选不变/strength0回退，以及同8对独立缓存特征路径的编码器全state与局部统计一致性。实际使用FP32、TF32开启、autocast关闭；批量/单图特征对照也通过。尚未拟合时cache_smoke不执行新模型GT反事实，该检查在冻结后目标阶段执行。']
    else:
        s=json.loads((r/'selection.json').read_text())['selected'];parts += ['', f"冻结源域方案：`{s['name']}`。TRAIN take-OOF {s['train_oof']['delta_pp']:+.4f}点、校准{s['calibration']['delta_pp']:+.4f}点。预选主方案与同配置次要对照必须分别报告，不因目标结果更高改选主方案。"]
    for phase in ('exo2exo','exo2ego'):
        p=r/(phase+'_results.json')
        if not p.exists():continue
        data=json.loads(p.read_text());parts += ['',f'### {phase} 验证结果','', '| 方法 | IoU | Dice | ContA | LocE ↓ | 相对原PCCS变化（点） | 95%序列区间 |', '|---|---:|---:|---:|---:|---:|---|']
        for name,v in data['methods'].items():
            ci=v['ci95_pp'];f=v['frame'];parts.append(f"| {name} | {100*f[0]:.4f} | {100*f[1]:.4f} | {100*f[2]:.4f} | {f[3]:.6f} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        parts += ['', '| 配对机制对照 | 变化（点） | 95%区间 |', '|---|---:|---|']
        for name,v in data['paired_comparisons'].items():
            ci=v['ci95_pp'];parts.append(f"| {name} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        cost=data['cost'];parts += ['', f"全部消融视图的局部处理：每对象均值{cost['roi_seconds_per_object_mean']:.3f}秒，每对{cost['roi_seconds_per_pair_mean']:.3f}秒，平均每对象{cost['encoder_views_per_object_mean']:.2f}个视图。此计时排除模型初始化，且不是单主方案优化后端到端成本。"]
    if (r/'exo2exo_results.json').exists():
        parts += ['', 'Exo→Exo阶段结论：预选local15主方案41.7688，相对原PCCS+2.7498点，区间[0.8710,4.6268]；相对此前冻结cycle额外+0.4334点，但区间[-0.7644,1.4037]仍跨零。预设local20次要对照42.3292（+3.3101点，[1.5986,5.0157]），点估计接近O-MaMa42.3980，不能把它事后追认为主方案，也不能由接近的均值宣称统计等效。', '', '本轮context对照比固定区域池化更支持保留真实背景：同crop/同容量/同门的local20比前景-only高2.5371点，[1.5776,3.3932]；加入全图后差为2.1308点，[1.1692,3.0320]。这是保留自然背景优于均值背景的描述性干预结果，尚不能排除均值背景的分布偏移，也不能具体归因于邻居物体语义或跨视角关系。后续需自然背景错配等控制与冻结方法的重复验证。']
    if (r/'secondary_contrasts.json').exists():
        extra=json.loads((r/'secondary_contrasts.json').read_text())
        for phase,items in extra['phases'].items():
            parts += ['',f'次要对照的描述性配对差（{phase}，不作新选型）：']
            for name,v in items.items():
                ci=v['ci95_pp'];parts.append(f"- {name}: {v['delta_pp']:+.4f}点，[{ci[0]:.4f}, {ci[1]:.4f}]。")
