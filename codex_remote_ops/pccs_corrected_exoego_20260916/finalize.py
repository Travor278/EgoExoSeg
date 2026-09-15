from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).parent
a=json.loads((R/'runs/full/results.json').read_text(encoding='utf8'));plan=json.loads((R/'manifest.json').read_text(encoding='utf8'));release=json.loads((R/'resource_release.json').read_text(encoding='utf-8-sig'))
assert a['coverage']=='exact' and (a['pairs'],a['objects'],a['takes'])==(46515,109253,295)
assert set(a['methods'])=={'baseline','learned_gate','omama_margin005','native_margin005','geometry_consensus'}
assert plan['primary_method']=='geometry_consensus'
assert json.loads((R/'runs/full/candidate_receipts.json').read_text())=={str(i):[11629,0] for i in range(4)}
assert [r['objects'] for r in a['rank_receipts']]==[27314,27313,27313,27313]
assert all(r['cache_metric_max_error']<1e-5 and r['checkpoint_sha256']==a['frozen']['checkpoint_sha256'] for r in a['rank_receipts'])
assert release['cells'][5]=='已成功' and release['cells'][7]=='0'
assert len((R/'runs/full/gpu_after.txt').read_text().splitlines())==1
validation={'state':'passed','pairs':46515,'objects':109253,'takes':295,'primary_method':'geometry_consensus','primary_comparison':a['paired_comparison']['geometry_consensus'],'maximum_cache_iou_error':max(r['cache_metric_max_error'] for r in a['rank_receipts']),'occupied_nodes':0,'results_sha256':hashlib.sha256((R/'runs/full/results.json').read_bytes()).hexdigest()}
(R/'final_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
j=json.loads((R/'job_receipt.json').read_text());j.update(status='succeeded',finished_at=release['cells'][9],occupied_nodes_after_completion=0,primary_result=validation['primary_comparison']);(R/'job_receipt.json').write_text(json.dumps(j,indent=2),encoding='utf8')
subprocess.run([sys.executable,'-X','utf8',str(R.parent/'pccs_gain_full_20260915/render_master_report.py')],check=True)
print('CORRECTED_EXOEGO_VERIFIED_AND_MASTER_REPORT_UPDATED')
