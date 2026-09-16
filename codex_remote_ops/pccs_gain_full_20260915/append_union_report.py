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
    parts += ['', '原候选保留保证该集合的oracle不降低，但不保证选择器最终涨点。本轮固定0.05阈值合并没有改善最终指标，新增可靠多点所提供的候选上限增量也很小。', '', '### 9.2 后续运行状态', '']
    p=r/'status.json';state=json.loads(p.read_text()) if p.exists() else {'state':'queued'}
    parts.append(f"任务 `v2sam-candidate-union-adapt-20260916-r1`；最近已核验状态：`{state.get('state')}`，阶段：`{state.get('phase','awaiting startup')}`。提交于北京时间15:51，单任务最多4张H100，成功／失败均保留1分钟。尚无完整结果时不宣称涨点。")
    p=r/'selection.json'
    if p.exists():
        a=json.loads(p.read_text());parts.append('训练内冻结选择：`'+a['selected']['name']+'`。')
    p=r/'exo2exo_results.json'
    if p.exists():
        a=json.loads(p.read_text());parts += ['', '| Exo→Exo方法 | Frame IoU |', '|---|---:|']
        for name,v in a['methods'].items():parts.append(f"| {name} | {v['frame'][0]*100:.4f} |")
        ci=a['primary_95ci_pp'];parts.append(f"\n相对当轮原一致性输出变化{a['primary_delta_vs_original_consensus_pp']:.4f}点，take配对bootstrap 95%区间[{ci[0]:.4f}, {ci[1]:.4f}]；覆盖{a['pairs']}对／{a['objects']}对象。")
    parts += ['', '实现、冻结规则与完整运行证据见[本轮计划](../pccs_candidate_union_20260916/PLAN.md)。监督间隔按有效剩余ETA×4/5动态调整；初始化／切换阶段采用5分钟。']
