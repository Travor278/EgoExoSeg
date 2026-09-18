import json
def append_matched(parts,root):
    r=root/'pccs_omama_full_matched_20260918'
    if not (r/'PLAN.md').exists():return
    parts += ['', '## 17. 当前Exo→Ego全量同候选O-MaMa补测', '',
      '用户要求将O-MaMa与当前原生ROI放在同一候选bank上比较。两轮全量样本范围一致，历史与当前候选并未逐对象证明完全相同；不能仅因权重一样就当作相同候选。区域池化/稀疏对应的随机点采样、逐pair seed设置与GPU数值实现都可能改变专家输出，mask一致性以hash为准。当前ROI vs 原PCCS同轮增益已同候选核验，这个问题影响的是与历史外部参考的成对比较。', '',
      '当前全量只留存mask SHA、候选指标与选择，没有完整原始mask。因此新增严格重建：使用同一冻结代码/逐pair seed，三份专家mask SHA、基线路由与各候选指标须与full1完全一致。任一不一致保存drift并停止，不排除不一致样本或把新bank混作旧bank。64对预设pilot后再全量；O-MaMa另进程运行，历史bank见证先复现分数和选择；0.05阈值与canonical/native一致性规则冻结。完整流程见[PLAN.md](../pccs_omama_full_matched_20260918/PLAN.md)。', '']
    if (r/'job_receipt.json').exists():
        j=json.loads((r/'job_receipt.json').read_text());parts += [f"重建作业：`{j['job_id']}`，{j['submitted_at_local']}北京；既有ROI seed2继续，不修改其代码。",'']
    if (r/'resource_release_failed_pilot.json').exists():parts += ['首次64对/147对象的三路mask全部匹配当前seed1，四条历史O-MaMa见证分数和选择复现通过；之后evaluate.py误引用仅在本地存在的verify_results模块，汇总失败。失败任务已释放节点。修复为自包含的相同pair/take聚合，保留已完成的pilot记录及SHA，不重选样本，不改模型或候选。修复前后代码SHA由repair_receipt.json记录。','']
    if (r/'job_receipt_rebuild_r2.json').exists():
        j=json.loads((r/'job_receipt_rebuild_r2.json').read_text());parts += [f"修复续跑任务：`{j['job_id']}`，{j['submitted_at_local']}北京。",'']
    q=r/'full_results.json'
    if not q.exists():parts += ['同候选全量O-MaMa结果待完成；提交、pilot或重建完成均不等于最终对照完成。'];return
    d=json.loads(q.read_text());parts += ['| 方法 | IoU | ΔPCCS（点） | 95% take区间 |','|---|---:|---:|---|']
    for name,v in d['methods'].items():
        ci=v['ci95_pp'];parts.append(f"| {name} | {100*v['frame'][0]:.4f} | {v['delta_pp']:+.4f} | [{ci[0]:.4f},{ci[1]:.4f}] |")
    for name,v in d['paired_comparisons'].items():
        ci=v['ci95_pp'];parts += ['',f"{name}: {v['delta_pp']:+.4f}点，95%区间[{ci[0]:.4f},{ci[1]:.4f}]。"]
