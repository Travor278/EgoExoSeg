from pathlib import Path
import hashlib,json
R=Path(__file__).parent
status=json.loads((R/'status.json').read_text());selection=json.loads((R/'selection.json').read_text());release=json.loads((R/'resource_release.json').read_text(encoding='utf-8-sig'))
assert status['state']=='complete' and selection['selected']=='baseline' and selection['eligible']==[]
assert release['cells'][5]=='已成功' and release['cells'][7]=='0'
summaries={p:json.loads((R/(p+'_summary.json')).read_text()) for p in ('smoke','screen','calibration')}
for phase,summary in summaries.items():
 assert summary['coverage']=='exact'
 assert all(v['mismatches']==0 for v in summary['isolation_checks'].values())
 assert len({(v['pairs'],v['objects']) for v in summary['arms'].values()})==1
assert all(v['pairs']==128 and v['objects']==243 for v in summaries['screen']['arms'].values())
assert all(v['pairs']==128 and v['objects']==225 for v in summaries['calibration']['arms'].values())
assert len((R/'gpu_after.txt').read_text().splitlines())==1
out={'state':'passed','outcome':'no_candidate_passed_preregistered_training_selection','target_exo2exo_run':False,'screen_pairs':128,'screen_objects':243,'calibration_pairs':128,'calibration_objects':225,'all_isolation_checks_passed':True,'multipoint_screen_objects':55,'multipoint_calibration_objects':55,'occupied_nodes':0,'summary_sha256':{p:hashlib.sha256((R/(p+'_summary.json')).read_bytes()).hexdigest() for p in summaries}}
(R/'final_validation.json').write_text(json.dumps(out,indent=2))
j=json.loads((R/'job_receipt.json').read_text());j.update(final_state='succeeded',finished_at=release['cells'][9],runtime=release['cells'][10],occupied_nodes_after_completion=0,selection=selection,target_test_run=False);(R/'job_receipt.json').write_text(json.dumps(j,indent=2))
print(json.dumps(out,indent=2))
