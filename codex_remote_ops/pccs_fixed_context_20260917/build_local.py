"""Construct this isolated experiment from audited native-PCCS runners."""
from pathlib import Path
import json
R=Path(__file__).parent;D=R.parent/'pccs_dense_context_20260917'
assert not (R/'job_receipt.json').exists(), 'Historical bootstrap: never overwrite an already submitted experiment; edit its reviewed source files directly.'
for n in ('randomness.py','native_bridge.py','data_utils.py','prepare.py','worker.py','controller.py','fetch_artifacts.py','read_monitor.js','build_capsule.py'):
    (R/n).write_text((D/n).read_text(encoding='utf8').replace('pccs_dense_context_20260917','pccs_fixed_context_20260917'),encoding='utf8')
s=(D/'dense_context.py').read_text();needle='        assert all(np.isfinite(v) for v in base.values());out.append(base)'
assert s.count(needle)==1
s=s.replace(needle,"        from fixed_context import evidence\n        base.update(evidence(qf,tf,qa,ta,pm,sm,oq,ot,obj,soft_f,into_source,into_candidate))\n"+needle)
(R/'dense_context.py').write_text(s)
s=(D/'policies.py').read_text();s=s.replace("HAND=[", "HAND=[",1)
s += "\nfrom fixed_context import VARIANTS,KEYS\nGROUPS={'cycle':BASE+DENSE,'old_ring15':BASE+DENSE+tuple('ctx15_'+k for k in KEYS),'old_ring20':BASE+DENSE+tuple('ctx20_'+k for k in KEYS),**{v:BASE+DENSE+tuple(v+'_'+k for k in KEYS) for v in VARIANTS}}\nHAND=[]\n"
s=s.replace("k.replace('ctx','wrong',1) if wrong and k.startswith('ctx') else k", "(k.replace('ctx','wrong',1) if k.startswith('ctx') else 'wrong_'+k) if wrong and (k.startswith('ctx') or k.startswith(('px','scale','om100'))) else k")
(R/'policies.py').write_text(s)
s=(R/'prepare.py').read_text().replace("('dense_context.py','worker.py','randomness.py','native_bridge.py','policies.py'", "('fixed_context.py','dense_context.py','worker.py','randomness.py','native_bridge.py','policies.py'")
s=s.replace("    m['code_sha256']=", "    confirm=json.loads((G/'holdout512.json').read_text());assert len(confirm)==512 and sum(len(v['objects']) for v in confirm.values())==965\n    assert takes(confirm).isdisjoint(takes(train)|takes(cal))\n    add_phase(m,'exo2ego',confirm,cfg,Q,'smoke')\n    m['test_scope']='Previously evaluated fixed 512-pair/965-object/64-take Exo2Ego holdout; never used for this fitting; not a new blind test or full test.'\n    m['frozen_cycle']=json.loads((R.parent/'pccs_dense_context_20260917/selection.json').read_text())['selected']\n    m['code_sha256']=")
(R/'prepare.py').write_text(s)
s=(R/'worker.py').read_text()
start=s.index("        ref=Path(phase['reference_root'])");end=s.index("        ann=json.loads",start)
s=s[:start]+"        prior={}\n        if args.phase!='exo2ego':\n    "+s[start:end]+s[end:]
s=s.replace("                if args.phase=='exo2exo':\n                    pred[0]['dense_context']", "                if args.phase in ('exo2exo','exo2ego'):\n                    pred[0]['dense_context']")
needle="                    for row,decision in zip(batch_records,integrated):row['integrated_primary']=decision['best_expert']"
s=s.replace(needle,needle+"\n                    frozen_metric=mm.PCCSMetric(collect_device='cpu',enable_vis=False,routing_policy='fusion_first',diagnostic_output=None);frozen_metric.native_dense_config=plan['frozen_cycle'];frozen_metric.process(batch,pred)\n                    for row,decision in zip(batch_records,[r for pair in frozen_metric.results for r in pair]):row['integrated_frozen_cycle']=decision['best_expert']")
# Read all previous source scalar rows and check original features/candidates, not just old 128-pair screen.
s=s.replace("        ann=json.loads", "        old_dense={}\n        if args.phase in ('train','calibration'):\n            for oldfile in (R.parent/'pccs_dense_context_20260917/runs'/args.phase).glob('rank*/records.jsonl'):\n                for oldrow in map(json.loads,oldfile.read_text().splitlines()):old_dense[(oldrow['video_id'],oldrow['obj_id'])]=oldrow\n        ann=json.loads")
s=s.replace("                    batch_records.append(row)", "                    if old_dense:\n                        oldrow=old_dense[key];assert oldrow['mask_sha256']==masks and oldrow['baseline']==base\n                        for e in evidence:\n                            for k,v in oldrow['evidence'][e].items():assert np.isclose(evidence[e][k],v,atol=1e-6,rtol=1e-5),(key,e,k,evidence[e][k],v)\n                    batch_records.append(row)")
(R/'worker.py').write_text(s)
s=(D/'fit_and_select.py').read_text();a=s.index('    splits=list(GroupKFold');b=s.index('    for group in GROUPS:',a)
s=s[:a]+"    fold_receipts=json.loads((R.parent/'pccs_dense_context_20260917/folds.json').read_text());splits=[]\n    for fold in fold_receipts:\n        fit_ids=[i for i,r in enumerate(tr) if r['take_id'] in fold['fit_takes']];val_ids=[i for i,r in enumerate(tr) if r['take_id'] in fold['held_takes']];assert len(fit_ids)+len(val_ids)==len(tr);splits.append((fit_ids,val_ids))\n"+s[b:]
s=s.replace("('ridge10','ridge100','tree')", "('ridge10','ridge100')")
# The zero-context model is a control, not a new candidate discovery.
s=s.replace("    eligible=[r", "    old=json.loads((R.parent/'pccs_dense_context_20260917/search.json').read_text())['candidates']\n    for cfg in results:\n        if cfg['group']=='cycle':\n            ref=next(x for x in old if x['name']==cfg['name']);assert abs(ref['train_oof']['delta_pp']-cfg['train_oof']['delta_pp'])<1e-7;assert abs(ref['calibration']['delta_pp']-cfg['calibration']['delta_pp'])<1e-7\n    eligible=[r")
(R/'fit_and_select.py').write_text(s)
s=(R/'controller.py').read_text().replace("run('test_policies.py');run('test_dense.py');run('prepare.py')", "run('test_geometry.py');run('prepare.py')")
s=s.replace("    if choice['selected']['family']!='baseline':", "    assert choice['selected']['family']=='calibrated'\n    phase_run('exo2ego');read('exo2ego');run('evaluate_exoego.py')\n    if choice['selected']['family']!='baseline':")
(R/'controller.py').write_text(s)
s=(R/'build_capsule.py').read_text();a=s.index('names=');b=s.index('\npayload=',a)
s=s[:a]+"names=['fixed_context.py','dense_context.py','policies.py','native_bridge.py','test_geometry.py','randomness.py','prepare.py','worker.py','data_utils.py','fit_and_select.py','evaluate.py','evaluate_exoego.py','controller.py','PLAN.md']"+s[b:]
s=s.replace('pccs_dense_submit_20260917','pccs_fixed_submit_20260917')
(R/'build_capsule.py').write_text(s)
s=(R/'read_monitor.js').read_text().replace("'test_policies.py.log','test_dense.py.log'", "'test_geometry.py.log','evaluate_exoego.py.log'").replace("'calibration','exo2exo'", "'calibration','exo2ego','exo2exo'")
(R/'read_monitor.js').write_text(s)
print('FIXED_EXPERIMENT_BUILT')
