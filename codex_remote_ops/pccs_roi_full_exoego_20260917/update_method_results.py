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
expanded=R.parent/'pccs_exoexo_expand_20260917/exo2exo_results.json'
if expanded.exists():
    e=json.loads(expanded.read_text());lines += ['扩展Exo→Exo时序压力测试已完成：新增6534个不重复配对，原PCCS41.3130，冻结1.5×主方案45.0837（+3.7706点），2×对照45.2391（+3.9260点），同候选O-MaMa43.9262（+2.6131点）。原主方案对O-MaMa区间跨零，2×对照的描述性差区间为[0.1774,2.6580]，不改变事前主次身份。', '', '该扩展只有19个旧take，不是新增场景泛化；它没有替代仍在运行的Exo→Ego46515对全量验证。完整结果见总报告§16及扩展目录的validation.json。','']
p.write_text((text+'\n'.join(lines)).rstrip()+'\n',encoding='utf8');print('METHOD_RESULTS_UPDATED')
