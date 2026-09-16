import json

def append_fixed(parts,root):
    r=root/'pccs_fixed_context_20260917'
    if not (r/'PLAN.md').exists():return
    parts += ['', '## 12. 固定 100 像素上下文与 Exo→Ego 原生验证（2026-09-17）','',
      '用户提出固定100像素范围可能优于1.5/2倍bbox。核对O-MaMa官方DescriptorExtractor：bbox每边各扩100，边界裁剪，池化包含前景的整框；100定义在预处理后的图像坐标。此前倍率纯背景环不是这一实现。此次只借鉴区域构造，将证据加入原PCCS挑战候选判断，不导入O-MaMa网络或替换PCCS。', '',
      '设置50/100/150原生像素整框与纯背景环、1.5/2倍整框与纯背景环、O-MaMa canonical图像坐标的100像素整框与环，共12种。原生坐标指DINOv3输入高768/宽保持比例；canonical范围按source532×952、target700×700归一化映射。后者仅区域范围对齐，仍用原生DINOv3特征，并非复现O-MaMa的DINOv2四分之一尺寸插值池化。bbox采用像素外边界，相比其xmax/ymax最多差一个像素，已记录。', '',
      '区域权重为连续bbox对每个token格子的面积占比；背景环再扣除前景覆盖率。相同六项上下文证据加入原生软循环置信特征，保持原候选、对应点、cycle-dominance准入和回退。比较原cycle、历史ring15/ring20和12个区域组，Ridge10/100、阈值.03/.05共60个源域配置，固定四折take列表，按48个TRAIN takes的OOF及16个校准takes冻结。目标上报告匹配容量对照与错位邻域控制，不按目标结果选型。', '',
      'Exo→Ego先验证既有512对/965对象/64 takes holdout，包括上一轮冻结cycle_ridge10_t0.03及本轮源域预选方案；它与拟合/校准takes互斥，但历史被评估过，不能称全新盲测，也不是46515对全量。Exo→Exo继续1094对全量。实际PCCS接口选择与缓存回放逐对象核对；source原生特征、候选SHA和原路由对照上一轮，确保比较不是由候选变化造成。详细协议见[固定像素实验计划](../pccs_fixed_context_20260917/PLAN.md)。', '']
    if (r/'selection.json').exists():
        selection=json.loads((r/'selection.json').read_text(encoding='utf-8-sig'));s=selection['selected']
        parts += [f"源域选型已完成，主方案冻结为`{s['name']}`，OOF {s['train_oof']['delta_pp']:+.4f}点、校准{s['calibration']['delta_pp']:+.4f}点；两个源域区间均跨零，不能由此宣称稳定增益。原cycle各配置的OOF/校准精确复现上一轮，未按目标集改选。", '', '| 同正则/阈值的区域对照 | TRAIN OOF增益（点） | 校准增益（点） |', '|---|---:|---:|']
        for c in selection['matched_controls']:parts.append(f"| {c['group']} | {c['train_oof']['delta_pp']:+.4f} | {c['calibration']['delta_pp']:+.4f} |")
        parts += ['', '100像素整框与环均保留为预先声明对照，150像素主方案只由源域规则选中；后续不能把目标集上最好的某一行追认成预选主方案。', '']
    for phase,label in [('exo2ego','Exo→Ego 512对 holdout'),('exo2exo','Exo→Exo 全量')]:
        p=r/(phase+'_results.json')
        if not p.exists():continue
        data=json.loads(p.read_text(encoding='utf-8-sig'))
        parts += [f'### 12.{1 if phase=="exo2ego" else 2} {label}', '',f"源域预选：`{data['selected']['name']}`。所有下列对照均冻结后评测。", '', '| 方法 | frame IoU | 相对原PCCS增益（点） | take bootstrap 95%区间 |', '|---|---:|---:|---|']
        for name,v in data['methods'].items():
            ci=v['ci95_pp'];parts.append(f"| {name} | {v['frame'][0]*100:.4f} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        v=data['vs_frozen_cycle']['primary'];ci=v['ci95_pp'];parts += ['',f"本轮主方案相对上一轮冻结cycle额外变化：{v['delta_pp']:+.4f}点，区间[{ci[0]:.4f}, {ci[1]:.4f}]。区分总增益和context的额外贡献，不能将原cycle本已有收益算作新增context收益。"]
        comparisons=r/'context_comparisons.json'
        if comparisons.exists():
            c=json.loads(comparisons.read_text())['phases'].get(phase,{})
            parts += ['', '| 配对的额外贡献/几何比较 | IoU变化（点） | 95%区间 |', '|---|---:|---|']
            for a,b,label in [('px100_box_matched','cycle_matched','原生100px整框 − 同阈值cycle'),('om100_box_matched','cycle_matched','canonical100px整框 − 同阈值cycle'),('primary','cycle_matched','预选150px整框 − 同阈值cycle'),('primary','wrong_context_primary','预选方案 − 错位邻域'),('px100_box_matched','px100_ring_matched','100px整框 − 100px环'),('px100_box_matched','scale15_box_matched','100px整框 − 1.5倍整框'),('px100_box_matched','scale20_box_matched','100px整框 − 2倍整框'),('px100_ring_matched','scale20_ring_matched','100px环 − 2倍环')]:
                if a+' minus '+b in c:
                    v=c[a+' minus '+b];ci=v['ci95_pp'];parts.append(f"| {label} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
            parts += ['', '这些是描述性配对比较，不按目标数值重新选择模型。IoU点估计略高但区间跨零时，不能声称该区域定义已带来稳定额外收益。']
    if not (r/'exo2ego_results.json').exists():parts += ['当前状态：任务已提交，尚无本轮目标结果；100像素优于倍率环是待验证假设。']
    if (r/'resource_release_r1.json').exists():parts += ['', '运行记录：R1完成8对Capture开关/strength0回退检查、384对TRAIN及128对校准，原生特征/候选/路由一致性通过。Exo→Ego首次误用TRAIN-only图片根目录，四个分片均在第一条预测前发生空图像列表错误；该次没有目标指标。失败日志已保留，节点归零。R2仅修复测试图片根目录，采用此前全量成功配置并逐文件预检；不重拟合、不改选已冻结模型或阈值。']
    if (r/'resource_release_r2.json').exists():parts += ['', 'R2完成Exo→Ego后，在Exo→Exo同一knife对象的历史候选一致性检查处停止。这是上一轮已验证原始推理本身可能重复变动的对象；本轮断言前未保存差异mask，不能推断本次差异也是25像素。R3保留已完成记录，改用已记录的同当前bank协议：所有方法共享当前候选，必要时在独立参考进程重算有差异对象的冻结O-MaMa外部比较，并用未变对象检查参考分数复现。模型与阈值不变，不能将不同候选bank指标混用。']
    if (r/'reference_updates.json').exists():
        updates=json.loads((r/'reference_updates.json').read_text());parts += [f"最终外部参考重算对象数：{updates['count']}。"]
    if (r/'resource_release.json').exists():parts += ['资源释放证据已存于本实验resource_release.json。']
    parts += ['', 'ROI裁图重编码属于另一机制，暂缓，优先完成用户明确要求的固定像素区域池化。']
