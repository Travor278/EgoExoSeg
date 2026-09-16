import json

def append_roi(parts,root):
    r=root/'pccs_roi_context_20260917'
    if not (r/'job_receipt.json').exists():return
    parts += ['', '## 13. 原生局部视图循环（2026-09-17）', '',
      '固定范围池化无法恢复已在全图patch中混合的细节，本轮检验额外物体局部视图是否改善原PCCS候选判断。原分割候选、原hard correspondence及回退不变；用现有DINOv3提取1.5/2倍物体bbox长边的方形裁图，形成局部双向mask传输、循环概率、面积先验log-lift与物体余弦证据，加入原PCCS挑战候选的置信判断。不导入O-MaMa编码器、投影头或权重。', '',
      '裁图移入图像边界，保留完整前景；小于16像素或长边无法纳入图像短边的对象将局部统计置零并传入有效性标记，原PCCS测量仍可参与校准。实现核对说明：crop无效本身不强制回到原选择，原路由仅在无合格挑战者或预测改善未过阈值时回退；不能将实现描述成缺失ROI就绝不换候选。最小裁图边长32；记录有效倍率，边界饱和的1.5/2倍裁图不能视为独立证据。前景-only对照使用完全相同的2倍crop/分辨率及前景RGB，将背景替换为[124,116,104]。这有分布变化，不能仅凭该干预宣称某个邻居物体的因果作用。', '',
      '源域预声明7个特征组×3种小判断器×2个阈值×2个准入门，共84组；同一准入门下所有组的候选集合相同，准入不读取ROI信息，避免把放宽门槛当context贡献。组别包括全图、局部1.5/2倍、前景-only及全图联合；两个门为原cycle dominance和原始soft-cycle>.001。仍用48个fit takes、16个cal takes及保存的四折，按正OOF/正校准后最高校准冻结，所有同模型/同门/同阈值的组构成目标对照。原global/dominance须精确复现此前源OOF/校准。小Ridge/HGB使用TRAIN标签，专家与DINOv3冻结，不称整个方法training-free。', '',
      '运行顺序：8对原pipeline smoke → 同一bank的独立缓存smoke → TRAIN384对 → 校准128对 → 冻结 → Exo→Exo1094对缓存评估 → Exo→Ego既有512对holdout。目标标签不进入策略。缓存smoke核对编码器全state SHA、TF32/autocast设置、批量/单图及逐项局部统计；避免用不同特征路径悄悄替换原生模块。Source保存私有mask缓存，Exo→Exo直接复用已经验证的候选，最终真实native_select接口须与离线选择逐对象一致，并通过清除GT指标字段的反事实检查。', '',
      '将报告额外编码视图数、局部处理时间、GPU峰值；当前同时提取全部消融视图，不能称其耗时为单个获胜方法的优化部署速度。详细设计与边界见[局部视图计划](../pccs_roi_context_20260917/PLAN.md)。']
    parts += ['', '### 13.1 原PCCS内部的计算路径', '',
      '对每个对象，先让原V2-SAM/PCCS完成Visual、Anchor、Fusion候选及原路由。源提示mask和每个目标候选各定义局部裁图，随后才提取附加特征，因此ROI不会反馈修改候选。设a为目标候选在局部patch上的前景覆盖率，b为源提示的前景覆盖率，F/G为目标/源局部DINOv3末层逐token归一化特征；图像缩放至768方形，但原图PCCS分支仍保持原先的纵横比。', '',
      '```text',
      'S[i,j] = dot(F[i], G[j]) / 0.07',
      'A = row_softmax(S)              # target -> source',
      'B = row_softmax(transpose(S))   # source -> target',
      'forward  = (b.T @ B @ a) / sum(b)',
      'backward = (a.T @ A @ b) / sum(a)',
      'cycle    = (b.T @ B @ diag(a) @ A @ b) / sum(b)',
      'soft_cycle = 2*forward*backward / max(forward+backward, 1e-8)',
      'log_lift_forward  = log(forward)  - log(mean(a))',
      'log_lift_backward = log(backward) - log(mean(b))',
      'log_lift_cycle    = log(cycle) - log(mean(a)*mean(b))',
      '```', '',
      '各log输入下限1e-30，归一化分母保护1e-8；再加入前景池化余弦、有效性和实际源/目标裁图倍率。均匀随机对应的forward/backward/cycle分别为mean(a)、mean(b)、mean(a)×mean(b)，所以log-lift用于区分面积先验与真实匹配支持。这里不是显式识别邻居物体：周围像素通过DINOv3图像编码和全patch匹配竞争影响前景对应，模型没有邻居实例标签或语义关系图。', '',
      '每个挑战者的输入为自身特征、原PCCS所选候选的特征、两者差值及专家类型one-hot，保留原hard votes、回环距离、SAM IoU、面积和连通域信息。TRAIN监督量是挑战者IoU减原PCCS IoU，按每对中的对象数及有效挑战者数加权。预选Ridge100预测改善超过0.05时才替换；present门要求原source/候选有效、质量合格、mask不重复及原soft-cycle>0.001。校准器不读取测试GT、take名称或O-MaMa分数。因而改进的是原PCCS的局部循环证据和置信判断，而非用外部matcher替代PCCS。', '']
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
    if (r/'exo2ego_results.json').exists():parts += ['', 'Exo→Ego范围仍为既有512对/965对象/64 takes，不能与此前46515对全量的绝对分数混比。预选1.5倍方案相对原PCCS+1.1054点，[0.4194,1.8679]；2倍次要对照+1.1958点，[0.5652,1.9090]。2倍相对此前冻结cycle额外+0.3665点，描述性区间[0.0415,0.6838]。不过2倍真实背景相对前景-only仅+0.1312点，区间[-0.2064,0.4314]，context额外效应的证据主要来自Exo→Exo，尚不能推广为两方向都显著。']
    if (r/'missing_view_audit.json').exists():
        audit=json.loads((r/'missing_view_audit.json').read_text());parts += ['', '缺失ROI实现审计（保持已冻结算法不变，强制回退仅作事后敏感性诊断）：']
        for phase,items in audit['phases'].items():
            for name,v in items.items():parts.append(f"- {phase}/{name}：有{v['changed_to_missing_roi_objects']}个对象换到缺失局部视图的候选；如强制这些对象回原PCCS，总IoU变化为{-v['original_minus_forced_pp']:+.6f}点。主要增益不来自该边界路径，但文字应如实说明零填充与回退条件。")
    if (r/'final_validation.json').exists():parts += ['', '最终核验：1094对Exo→Exo、512对/965对象Exo→Ego覆盖完整，实际接口选择与冻结回放一致，原始候选/全图特征保持，frame聚合复算一致，保留440/128个零IoU对象。参数和模型来源见manifest、encoder_receipt及final_validation。']
    if (r/'resource_release.json').exists():
        c=json.loads((r/'resource_release.json').read_text(encoding='utf-8-sig'))['cells'];parts += [f"本轮任务{c[5]}，占用节点{c[7]}，于{c[9]}结束（北京时间），运行{c[10]}；gpu_after无残留计算进程。"]
    if (root/'pccs_roi_seed_repeat_20260917/PLAN.md').exists():parts += ['', '下一步先冻结已有1.5/2倍模型做一次新的候选随机种子复验，避免继续在已观察目标上选参数。新种子命名已预先声明，只改变候选生成随机性，所有比较共享同一新bank，并重新计算对应的外部参考；不能将种子当独立拍摄序列增加样本量。该复验尚未提交，见[冻结复验协议](../pccs_roi_seed_repeat_20260917/PLAN.md)。']
