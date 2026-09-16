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
    else:
        s=json.loads((r/'selection.json').read_text())['selected'];parts += ['', f"冻结源域方案：`{s['name']}`。"]
    for phase in ('exo2exo','exo2ego'):
        p=r/(phase+'_results.json')
        if not p.exists():continue
        data=json.loads(p.read_text());parts += ['',f'### {phase} 验证结果','', '| 方法 | Frame IoU | 相对原PCCS变化（点） | 95%序列区间 |', '|---|---:|---:|---|']
        for name,v in data['methods'].items():
            ci=v['ci95_pp'];parts.append(f"| {name} | {100*v['frame'][0]:.4f} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        parts += ['', '| 配对机制对照 | 变化（点） | 95%区间 |', '|---|---:|---|']
        for name,v in data['paired_comparisons'].items():
            ci=v['ci95_pp'];parts.append(f"| {name} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        cost=data['cost'];parts += ['', f"全部消融视图的局部处理：每对象均值{cost['roi_seconds_per_object_mean']:.3f}秒，每对{cost['roi_seconds_per_pair_mean']:.3f}秒，平均每对象{cost['encoder_views_per_object_mean']:.2f}个视图。此计时排除模型初始化，且不是单主方案优化后端到端成本。"]
