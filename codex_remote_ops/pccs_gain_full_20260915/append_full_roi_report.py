import json

def append_full_roi(parts,root):
    r=root/'pccs_roi_full_exoego_20260917'
    if not (r/'PLAN.md').exists():return
    parts += ['', '## 15. 当前原生ROI方案的Exo→Ego全量与自然背景控制', '',
      '用户要求补当前原生ROI的全量验证、独立方法说明及科学性消融。本轮使用已经确认的最佳Exo→Ego Visual e19 / Fusion e20，冻结已有1.5倍主方案、2倍关键对照及所有匹配特征控制。数据为46515对/109253对象/295takes，先seed1再seed2，各自完整覆盖并分别报告；此前512对子集结果不能替代全量。', '',
      '独立方法说明：[METHOD.md](../pccs_roi_full_exoego_20260917/METHOD.md)。解释原PCCS保留路径、局部ROI重编码、soft-cycle/面积校正、置信校准与回退、与O-MaMa的借鉴区别、固定像素方案与ROI方案的不同、权重来源及现有证据边界。它是原PCCS内部增强，不加载O-MaMa网络作为本方法，也不能称完全training-free或已证实新颖性。', '',
      '历史O-MaMa好权重核对：pccs_corrected_exoego_20260916的manifest/完整结果/执行预检记录确认使用相同专家SHA，完整46515对原PCCS53.8997、几何一致性O-MaMa56.6385（+2.7388点）。本次启动再次hash专家文件，并生成weight_audit.json。旧O-MaMa与当前全量候选运行不同，所以这是历史结果，不能冒充当前种子的同bank配对对照。', '',
      '新增自然背景干预使用既有seed1固定候选：保持源前景RGB、mask、crop与8原图像素边界带不变，目标图与目标mask全部不变，仅改变源crop的较远背景。比较真实重放、同源图远处天然背景供体、crop半边长错位背景，均使用冻结校准器，不重训干预模型。真实重放必须复现旧局部特征及选择。远背景供体按源mask重叠最低/距离最远确定，记录替换比例及供体重叠；无可编辑背景的样本保留。干预仍可能有拼接/分布偏移，错位背景可能携带移位物体纹理，不能称完美语义因果证明。', '']
    p=r/'science_results.json'
    if p.exists():
        data=json.loads(p.read_text());parts += ['### 15.1 冻结自然背景消融', '', '| 方向与方法 | 真实背景IoU | 远处背景IoU | 错位背景IoU |', '|---|---:|---:|---:|']
        for phase,v in data['phases'].items():
            for mode in ('local15','local20'):parts.append(f"| {phase}/{mode} | {100*v['methods']['real_'+mode]['frame'][0]:.4f} | {100*v['methods']['far_'+mode]['frame'][0]:.4f} | {100*v['methods']['rolled_'+mode]['frame'][0]:.4f} |")
        parts += ['', '| 配对比较 | 真实−干预（点） | 95% take区间 |', '|---|---:|---|']
        for phase,v in data['phases'].items():
            for name,c in v['real_minus_intervention'].items():
                ci=c['ci95_pp'];parts.append(f"| {phase}/{name} | {c['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        parts += ['', '上述为描述性机制对照，不能根据结果重选模型、删掉干预较弱样本或重新拟合阈值；新方法是否普遍获益与背景干预效应须分开。']
    for seed in (1,2):
        p=r/f'full{seed}_results.json'
        if not p.exists():parts += ['',f'全量seed{seed}：待完成。'];continue
        a=json.loads(p.read_text());parts += ['',f'### 全量 Exo→Ego seed{seed}（46515对）', '', '| 方法 | IoU | Dice | ContA | LocE ↓ | ΔPCCS（点） | 95%区间 |','|---|---:|---:|---:|---:|---:|---|']
        for name,v in a['methods'].items():
            f=v['frame'];ci=v['ci95_pp'];parts.append(f"| {name} | {100*f[0]:.4f} | {100*f[1]:.4f} | {100*f[2]:.4f} | {f[3]:.6f} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    parts += ['', '一次只运行一项四H100任务。第一任务先科学消融后full1；确认完成与节点释放后再启动full2。每次监督按当前阶段剩余ETA的4/5调整。完整协议见[PLAN.md](../pccs_roi_full_exoego_20260917/PLAN.md)。']
