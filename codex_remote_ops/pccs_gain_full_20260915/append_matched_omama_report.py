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
    shared=root/'pccs_omama_shared_full_20260918'
    if (r/'resource_release_rebuild.json').exists():
        parts += ['严格重建随后在一个Fusion候选上遇到SHA不一致并按协议停止：take849aaee0-866d-4b85-9775-39360e27fa90/frame6120/obj0，Fusion IoU从0.76996195变成0.73738414，Visual/Anchor及基线路由相同。已保存791个匹配对象，不能据此前缀估计全量漂移率。失败任务节点已释放。底层非确定性原因未定位，不将其武断称为仅几个像素的舍入误差，也不放宽hash检查。','']
    if (shared/'PLAN.md').exists():
        parts += ['### 17.1 改为一次生成并持久化的共同候选全量复验','','新目录[pccs_omama_shared_full_20260918](../pccs_omama_shared_full_20260918/PLAN.md)：保持模型、权重、阈值与46515对数据不变，沿用完整原生ROI worker，仅增加同次三专家mask和source mask的存储；原PCCS/ROI1.5/ROI2/旧cycle由同次预测计算，O-MaMa另进程读取同一份不可变bank。因此所有方法同候选，但这是新生成的复验，不声称逐bit复现原full1。原full1的同轮ROI增益仍有效。','','本轮使用H100 CUDA12.8/183核组，仍为4 H10080GB、80CPU、900GB内存、同ngc-pytorch25.02-cuda12.8镜像与既有Python环境。此前将下拉列表可见部分误认为全部选项，错误声称原组不再提供；经搜索确认原H100 CUDA13.2/183核组（lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e）仍存在。用户明确同意已启动的12.8任务保持运行。记录资源组变化，不将跨运行数值差异混入成对增益。','']
        j=shared/'job_receipt.json'
        if j.exists():
            info=json.loads(j.read_text());parts += [f"新共同候选任务：`{info['job_id']}`，{info['submitted_at_local']}北京。",'']
        if (shared/'resource_release_generation.json').exists():parts += ['共同候选生成已于2026-09-19 09:27:35完成，4rank回执共46515对/109253对象，冻结编码器配置核验通过，节点0。mask已保存在远端，生成完成不等于O-MaMa全量打分完成。','']
        if (shared/'job_receipt_reference.json').exists():
            info=json.loads((shared/'job_receipt_reference.json').read_text());parts += [f"独立O-MaMa打分任务：`{info['job_id']}`，{info['submitted_at_local']}北京提交；读取已保存bank，不重新生成候选。",'']
        q=shared/'full1_results.json'
        if (shared/'PERFORMANCE.md').exists():parts += ['GPU利用率问题单列[PERFORMANCE.md](../pccs_omama_shared_full_20260918/PERFORMANCE.md)：7.9GiB峰值分配量不代表充分利用。41秒/9次实测四卡平均约46.3%/26.4%/17.1%/27.3%，四worker各接近满1个CPU核。用户随后要求先不再优化、恢复0.8×ETA监督，临时函数profiler已停止，主实验方法未改；不宣称具体函数瓶颈或优化提速已验证。','']
    if not q.exists():parts += ['同候选全量O-MaMa结果待完成；提交、pilot或重建完成均不等于最终对照完成。'];return
    d=json.loads(q.read_text());parts += ['| 方法 | IoU | ΔPCCS（点） | 95% take区间 |','|---|---:|---:|---|']
    for name,v in d['methods'].items():
        ci=v['ci95_pp'];parts.append(f"| {name} | {100*v['frame'][0]:.4f} | {v['delta_pp']:+.4f} | [{ci[0]:.4f},{ci[1]:.4f}] |")
    for name,v in d['paired_comparisons'].items():
        ci=v['ci95_pp'];parts += ['',f"{name}: {v['delta_pp']:+.4f}点，95%区间[{ci[0]:.4f},{ci[1]:.4f}]。"]
    if (shared/'full1_validation.json').exists():
        v=json.loads((shared/'full1_validation.json').read_text());drift=v['cross_run_diagnostic_only'];parts += ['', '**结论：已完成同候选公平比较。** 原生ROI在原PCCS上保持约2.46～2.51点增益；O-MaMa一致性参考约+2.71点，主1.5×/次2×的IoU点估计分别低0.2467/0.1926点，两项成对差区间均跨零。可以说点估计接近；不能据“无显著差异”宣称等效、非劣或优于O-MaMa，尤其未事前指定等效界限。', '', f"本地验证完整46515对/109253对象/295takes，4rank全部reference记录文件SHA、逐对象三mask SHA与选择对齐、历史分数/选择见证、逐帧均值和10000次take bootstrap均通过。相对旧seed1有{drift['objects_with_any_mask_drift']}个对象候选发生漂移、{drift['objects_with_baseline_route_drift']}个基线路由改变；这是跨运行诊断，不能确定底层原因，也不能代替本轮逐对象同bank核验。", '', '本轮报告全部canonical/native/consensus参考，不选最高值重新定义主对照；零IoU和不利样本完整保留，主方案身份仍为源域预选1.5×。两seed原生ROI与本轮共同候选使用同一已观察场景集，不能当三份新独立测试集。']
    if (shared/'resource_release_reference.json').exists():
        info=json.loads((shared/'resource_release_reference.json').read_text());parts += ['',f"参考打分任务已于{info['ended_at_beijing']}完成，节点0且GPU计算进程为空；生成和参考两阶段资源均已释放。完整可复算标量见full1_paired_compact.jsonl.gz、runs/reference_full1/rank*/records.jsonl.gz及full1_validation.json；verify_results.py兼容未压缩及压缩参考记录。"]
