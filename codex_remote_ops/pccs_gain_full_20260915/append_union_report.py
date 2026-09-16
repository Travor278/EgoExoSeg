"""Append reproducible status/results for the follow-up candidate-union run."""
import json

def append(parts,root):
    r=root/'pccs_candidate_union_20260916'
    if not (r/'PLAN.md').exists():return
    parts += ['', '## 9. 保留候选、小比例池化与 O-MaMa 适配（2026-09-16）', '',
      '本轮继续§8的负结果，按顺序测试：①保留原单点候选并加入可靠多点候选；②保留原池化并加入5%／10%／25%概率加权分量；③在修复权重的新候选上拟合质量增益排序器；④冻结O-MaMa主体，仅训练共享rank-8残差匹配投影。全部原候选和原几何一致性输出保留，作为候选集合及回退项。', '',
      '前三种排序器为ridge α=10/100与3叶小树，29项输入仅含双几何匹配分数、候选来源和预测mask几何；监督是候选相对原一致性输出的IoU差。第④项在O-MaMa完整物体／100px框邻域／跨图注意力表征之后加768→8→768残差投影，初始为恒等映射；用筛查集候选IoU软分布训练，主体及V2-SAM权重不变。③④需要少量训练，不能称为training-free；④也不是V2-SAM prompt投影微调。', '',
      '继续复用官方TRAIN内128对／243对象筛查、128对／225对象校准。只在筛查拟合，校准选模型；两者最终frame IoU都改善才晋级，按校准最终IoU选一个并冻结，再条件性验证Exo→Exo。训练内数据和目标benchmark此前已经用于探索，因此不把重复筛选当作盲测。新全量比较须使用同一逐pair随机种子的当轮原始基线，不直接减去§1的异次分数。', '',
      '### 9.1 多点候选合并：已完成的离线回放', '',
      '| 阶段 | 原一致性 IoU | 合并后 IoU | Δ最终（点） | Δ候选Oracle（点） |', '|---|---:|---:|---:|---:|']
    p=r/'stage1_results.json'
    if p.exists():
        for phase,v in json.loads(p.read_text())['phases'].items():
            f=v['frame_iou_percent'];parts.append(f"| {phase} | {f['baseline']:.4f} | {f['union']:.4f} | {v['delta_final_pp']:.4f} | {v['delta_oracle_pp']:.4f} |")
    parts += ['', '原候选保留保证该集合的oracle不降低，但不保证选择器最终涨点。本轮固定0.05阈值合并没有改善最终指标，新增可靠多点所提供的候选上限增量也很小。', '', '### 9.2 小比例池化候选合并', '']
    p=r/'stage2_results.json'
    if p.exists():
        a=json.loads(p.read_text());parts+=['| 增加候选 | 筛查Δ最终 | 校准Δ最终 | 校准ΔOracle |','|---|---:|---:|---:|']
        for name,v in a['phases']['calibration'].items():parts.append(f"| {name} | {a['phases']['screen'][name]['delta_final_pp']:.4f} | {v['delta_final_pp']:.4f} | {v['delta_oracle_pp']:.4f} |")
        parts+=['','以上都是固定余弦差0.05的描述性对照，单位为IoU百分点。5%接近不变，较大比例仍损害最终选择；全部合并在校准的候选上限增加0.5299点，但最终下降0.6013点。候选有所补充，如何选好仍待排序器／适配验证。']
    parts+=['','### 9.3 后续运行状态','']
    p=r/'status.json';state=json.loads(p.read_text()) if p.exists() else {'state':'queued'}
    jobs=sorted(r.glob('job_receipt*.json'));job=json.loads(jobs[-1].read_text()) if jobs else {}
    parts.append(f"当前任务 `{job.get('name','unknown')}`；最近已核验状态：`{state.get('state')}`，阶段：`{state.get('phase','awaiting startup')}`。单任务最多4张H100，成功／失败均保留1分钟。尚无完整结果时不宣称涨点。")
    if (r/'recovery_r2/precision_probe.json').exists():parts+=['','运行诊断：r1在O-MaMa embedding重放校验出现0.000308最大偏差，超过1e-4容差，停止于拟合前。r2单独切换cuDNN TF32未解决，故不能将原因归结于cuDNN。r3继续检查GEMM精度及原组／扩大候选组，保持原容差；历史失败证据归档，已完成候选缓存继续复用。r1／r2均已释放节点。']
    if (r/'job_receipt_r4.json').exists():parts+=['','r3确认GEMM TF32开启时原候选组重放误差为0，但扩大候选批次仍可引入>1e-4的数值偏差。r4修复为：共享冻结DINO图像特征，按每个原候选组的大小与顺序调用匹配头，再合并候选embedding；不放宽校验。诊断对象最大误差降至5.96e-8，原始分数保留，随后继续逐对象重放检查。r3也已释放节点。此修复不改变任何已生成mask或原始评价结果。']
    p=r/'selection.json'
    if p.exists():
        a=json.loads(p.read_text());parts.append('训练内冻结选择：`'+a['selected']['name']+'`。')
        selected=a['selected']
        if selected.get('family')!='baseline':parts.append(f"筛查相对原一致性输出{selected['screen']['delta_pp']:+.4f}点，校准{selected['calibration']['delta_pp']:+.4f}点；校准替换{selected['calibration']['changed']}个对象，其中改善{selected['calibration']['improved']}、变差{selected['calibration']['harmed']}。这是训练内选择结果，不能据此宣称目标集稳定涨点。")
    p=r/'model_search.json'
    if p.exists():
        search=json.loads(p.read_text())['candidates'];parts+=['','| 配置 | 筛查Δ最终 | 校准Δ最终 |','|---|---:|---:|']
        wanted=('ridge10_t0.05','ridge100_t0.01','tree_t0.05','omama_adapter_e5_m0.1','omama_adapter_e15_m0.1','omama_adapter_e40_m0.1')
        for v in search:
            if v['name'] in wanted:parts.append(f"| {v['name']} | {v['screen']['delta_pp']:+.4f} | {v['calibration']['delta_pp']:+.4f} |")
        parts+=['','完整配置均在[model_search.json](../pccs_candidate_union_20260916/model_search.json)记录，以上是紧凑摘录。O-MaMa残差投影共640个优化步，loss有限、零初始化恒等检查通过，但所有预设投影配置在校准集均下降；筛查上涨、校准下降呈现过拟合信号。因此本轮不将该投影送入目标测试。选中的ridge质量排序器也未经过目标集重新拟合。']
    p=r/'exo2exo_results.json'
    if p.exists():
        a=json.loads(p.read_text());parts += ['', '| Exo→Exo方法 | Frame IoU |', '|---|---:|']
        for name,v in a['methods'].items():parts.append(f"| {name} | {v['frame'][0]*100:.4f} |")
        ci=a['primary_95ci_pp'];parts.append(f"\n相对当轮原一致性输出变化{a['primary_delta_vs_original_consensus_pp']:.4f}点，take配对bootstrap 95%区间[{ci[0]:.4f}, {ci[1]:.4f}]；覆盖{a['pairs']}对／{a['objects']}对象。")
    parts += ['', '实现、冻结规则与完整运行证据见[本轮计划](../pccs_candidate_union_20260916/PLAN.md)。监督间隔按有效剩余ETA×4/5动态调整；初始化／切换阶段采用5分钟。']
