import json
def append_expanded(parts,root):
    r=root/'pccs_exoexo_expand_20260917'
    if not (r/'temporal_expansion_manifest.json').exists():return
    d=json.loads((r/'temporal_expansion_manifest.json').read_text())
    parts += ['', '## 16. Exo→Exo样本规模核查与扩展验证', '',
      '**全文此前“Exo→Exo全量1094对”是指已构建人工筛选集的全量，不是官方Ego-Exo4D全部可用数据。** 来源为Jenk9/exo2exo-selected-EgoExo4D-pair-same-random@23b8f17b76aa，547个人工确认pseudo-source提示各配同帧/异帧各一条，共1094。20takes，79种label文字；按take与source_object_id精确组合为96条对象实例，不应将类别、实例、图像或pair混报。', '',
      '现有平台Exo→Ego全量有295takes，但逐条源路径审计每take只有一个exo相机被标注，不能直接拼出任意两个exo的可靠GT。现有另一exo源mask经过V2-SAM/SAM2生成及人工筛选；扩展到新takes需要更多可信源mask，已请求原候选/筛选目录。不能把生成mask当targetGT，也不能用方法输出选择“容易涨点”的新样本。', '',
      f"立即构建的扩展压力测试：新增{d['additional_pairs']}个配对（旧+新合计{d['combined_old_and_new_pairs']}），来自原集合内{d['takes']}takes、{d['take_scoped_object_instances']}条实例、{d['source_prompts']}个源提示、{d['target_images']}张目标图。按每个源提示的同物体/同目标相机可用GT时间等分选最多16个新帧，排除原有配对；源提示及4K目标mask完全保留。**无新增独立take、无新增mask；这是时序组合稳健性，不是场景泛化扩大。** 原1094仍独立保留，新6534单独评分，19take聚类bootstrap，不能把pair当独立场景。", '',
      '冻结1.5×历史主方案、2×关键次要对照和matched controls，使用好Exo2Ego权重。原PCCS候选只生成一次，所有方法共用；独立重算新候选的O-MaMa参考并校验旧bank见证分数。先8对smoke，后新增6534完整评估；不重训练或目标调参。新任务独立资源/目录，不修改进行中的Exo→Ego全量任务。', '',
      '另一条初步扩展尝试将540×960的现有Exo2Ego prompt标注充作更多目标GT，但发现实例匹配/分辨率一致性问题，只有722新配对通过预设核对，因此未把它作为本轮主评测。当前6534使用同原4K注释池，避免此口径变化；完整审计保留，不能降低核验门槛只为增加样本。', '',
      '更大独立场景覆盖仍待补更多take的可信源mask。来源及实现：[PLAN.md](../pccs_exoexo_expand_20260917/PLAN.md)、[后续调研](../pccs_method_research_20260917/RESEARCH.md)。']
    p=r/'exo2exo_results.json'
    if all((r/'runs/smoke'/f'rank{k}/receipt.json').exists() for k in range(4)):parts += ['', '扩展任务的8对原pipeline冒烟及四卡旧bank O-MaMa见证分数已通过；最终完成状态以下方完整结果及验证回执为准。']
    if not p.exists():parts += ['', '扩展6534结果：待完成，不能使用旧1094增益代替。']
    else:
        a=json.loads(p.read_text());parts += ['', '| 方法 | 新6534 frame IoU | Δ本轮PCCS（点） | 95%take区间 |','|---|---:|---:|---|']
        for name,v in a['methods'].items():
            ci=v['ci95_pp'];parts.append(f"| {name} | {100*v['frame'][0]:.4f} | {v['delta_pp']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
        parts += ['', '完整6,534对结果支持原生ROI在更多异步时序组合中继续获益：1.5倍预选主方案+3.7706点，[2.2539,5.5732]；2倍关键次要对照+3.9260点，[2.4658,5.5719]。此前cycle为+2.2914点，同当前候选重算的O-MaMa为+2.6131点。不能把这组绝对IoU与旧1,094直接相减来算方法增益，数据组成已变。', '',
          '相对O-MaMa：1.5倍+1.1575点，区间[-0.0201,2.5478]仍跨零；2倍+1.3129点，描述性区间[0.1774,2.6580]为正，但它仍是事前声明的次要对照，未作多重检验校正，不能改写成旧实验主方案或普遍优于O-MaMa。2倍相对1.5倍仅+0.1554点，区间[-0.0764,0.3936]跨零。', '',
          '原生增强相对旧cycle：1.5倍额外+1.4792点，[1.0345,1.9902]；2倍额外+1.6346点，[1.1364,2.2036]。2倍真实背景匹配组相对同配置前景-only组+2.5361点，[1.6863,3.3218]，但灰背景分布偏移的解释限制仍在。', '',
          '验证：6,534个新增pair全部覆盖，与旧源提示/目标图配对不重合，19takes；完整PCCSMetric的1.5/2/cycle选择与离线回放一致，保留2,606个零IoU对象。O-MaMa参考6,534条全部同bank计算，4卡历史见证校验通过，并从保存相似度分数独立重放其最终选择。原三候选、源/目标标注与数据构造hash核对通过；见validation.json及reference回执。', '',
          '**这仍只是19个旧take中的新组合，不是新增场景测试。** 新mask/场景扩充仍待更多可信源mask；不能用6,534pair把19个take伪装成6,534个独立样本。较好的配对结果不能消除这一局限。']
    p=r/'resource_release.json'
    if p.exists():
        cells=json.loads(p.read_text(encoding='utf-8-sig'))['cells']
        if str(cells[7])=='0':parts += ['',f"扩展任务已释放：{cells[5]}，占用节点0，结束{cells[9]}（北京时间）。"]
