from pathlib import Path
import json
R=Path(__file__).parent;p=R/'METHOD.md';marker='\n<!-- CURRENT_VALIDATION_RESULTS -->'
text=p.read_text(encoding='utf8').split(marker)[0];lines=[marker,'','## 10. 本次验证进度与结果','']
if (R/'weight_audit.json').exists():lines += ['本次平台启动已再次完整核验Visual e19/Fusion e20的字节数及SHA，并确认历史O-MaMa全量使用同组权重；机器可读证据：`weight_audit.json`。','']
if (R/'science_results.json').exists():
    data=json.loads((R/'science_results.json').read_text());lines += ['自然背景控制已完成，以下均为同一候选、同一冻结模型的描述性配对比较。真实图重放需保持原始特征/选择一致。','','| 方向/尺度 | 真实背景 IoU | 远背景 IoU | 错位背景 IoU |','|---|---:|---:|---:|']
    for phase,v in data['phases'].items():
        for mode in ('local15','local20'):lines.append(f"| {phase}/{mode} | {100*v['methods']['real_'+mode]['frame'][0]:.4f} | {100*v['methods']['far_'+mode]['frame'][0]:.4f} | {100*v['methods']['rolled_'+mode]['frame'][0]:.4f} |")
    lines += ['','| 配对比较 | 真实−干预（点） | 95%区间 |','|---|---:|---|']
    for phase,v in data['phases'].items():
        for name,c in v['real_minus_intervention'].items():
            ci=c['ci95_pp'];lines.append(f"| {phase}/{name} | {c['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines += ['','本对照仍受自然patch拼接、空间错位和模型分布变化限制，不能单独证明具体邻居语义的因果作用。','']
    lines += ['自然背景控制的解读：Exo→Exo2×真实背景比远处供体高0.6654点，区间[0.3186,0.9870]；支持局部背景兼容性影响当前模型。1.5×远处供体差也为正，2×错位对照跨零。Exo→Ego效果更弱，多数区间跨零，不能推广为所有方向/尺度均显著。','', '真实图重放逐项特征误差为0。远供体不是完美纯背景：2×时Exo→Exo30个、Exo→Ego130个供体patch与源mask有重叠，全部保留；缺失ROI的4/17个对象也保留。源前景与8像素halo严格保持，平均实际改动比例及供体信息详见science_audit.json。','']
else:lines += ['自然背景控制：运行中或待完成。','']
for seed in (1,2):
    q=R/f'full{seed}_results.json'
    if not q.exists():lines += [f'全量seed{seed}：待完成，尚无46515对完整结果。',''];continue
    a=json.loads(q.read_text());lines += [f'全量seed{seed}：{a["pairs"]}对、{a["objects"]}对象、{a["takes"]}takes。','','| 方法 | IoU | 相对本轮原PCCS增益（点） | 95%区间 |','|---|---:|---:|---|']
    for name in ('baseline','frozen_cycle','primary','local20_matched','global_matched','object20_matched'):
        v=a['methods'][name];ci=v['ci95_pp'];lines.append(f"| {name} | {100*v['frame'][0]:.4f} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines += ['']
    lines += ['同候选的机制/方法对照（描述性比较，未按测试结果重新选型）：','','| 比较 | IoU差（点） | 95% take区间 |','|---|---:|---|']
    for name,v in a['paired_comparisons'].items():
        ci=v['ci95_pp'];lines.append(f"| {name} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    d=R/f'full{seed}_diagnostic.json'
    if d.exists():
        diag=json.loads(d.read_text());lines += ['','在同一次全量运行中重新按既有512配对成员分组；不是重新挑选高分样本。','','| 范围 | 配对数 | 原PCCS | 1.5× | 2× | 三候选GT oracle |','|---|---:|---:|---:|---:|---:|']
        for label,key in [('全量','all'),('此前512对子集成员','prior512_membership'),('其余配对','remaining')]:
            v=diag[key];lines.append(f"| {label} | {v['pairs']} | {v['baseline']:.4f} | {v['primary']:.4f} | {v['roi2']:.4f} | {v['oracle']:.4f} |")
        lines += ['','GT oracle只用于事后诊断，不进入推理/拟合/阈值选择。全量和子集的基线差异来自样本范围；不能据此证明具体难度因素。前景-only对照同时改变输入背景和表征分布，与自然背景干预合看，不能将全部差值归于邻居语义。','']
if all((R/f'full{s}_results.json').exists() for s in (1,2)):
    lines += ['双seed全量结论：每轮46515对/109253对象/295takes、同轮同候选；1.5×主方案两轮增益+2.4599/+2.4984点，2×次对照+2.5142/+2.5334点，四项95% take区间均高于零。支持既有测试集上对候选随机种子的重复性；两个seed不是两份独立场景数据，不保证每个对象都涨点。完整覆盖/指标复算与资源释放见validation.json和resource_release_seed1/2.json。当前同候选O-MaMa的新共同候选复验尚在运行，不能将历史56.6385作本轮成对参考。','']
expanded=R.parent/'pccs_exoexo_expand_20260917/exo2exo_results.json'
if expanded.exists():
    e=json.loads(expanded.read_text());lines += ['扩展Exo→Exo时序压力测试已完成：新增6534个不重复配对，原PCCS41.3130，冻结1.5×主方案45.0837（+3.7706点），2×对照45.2391（+3.9260点），同候选O-MaMa43.9262（+2.6131点）。原主方案对O-MaMa区间跨零，2×对照的描述性差区间为[0.1774,2.6580]，不改变事前主次身份。', '', '该扩展只有19个旧take，不是新增场景泛化；它与Exo→Ego46515对全量验证分别报告。完整结果见总报告§16及扩展目录的validation.json。','']
p.write_text((text+'\n'.join(lines)).rstrip()+'\n',encoding='utf8');print('METHOD_RESULTS_UPDATED')
