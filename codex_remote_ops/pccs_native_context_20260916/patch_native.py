"""Small reviewed edits inside PCCS's existing fusion/cycle decisions."""
from pathlib import Path
import shutil
def patch(root):
    p=root/'projects/v2sam_pccs/evaluation/pccs_metric.py';s=p.read_text();original=s
    s='from ..native_context import recheck_fusion, weighted_count\n'+s
    old='                # 步骤1: 首先检查fusion的mask质量'
    assert s.count(old)==1
    s=s.replace(old,"                native_cfg = getattr(self, 'native_context_cfg', {})\n                native_info = data_samples[i].get('native_context', [{}] * num_obj)[j]\n"+old)
    old='                    and self.routing_policy == "fusion_first"';assert s.count(old)==1
    s=s.replace(old,old+'\n                    and not recheck_fusion(native_info, native_cfg)')
    for e in ('visual','anchor'):
        old=f'''                                in_mask_counts["{e}"] = self._count_points_in_mask(
                                    pts_{e}, raw_prompt_mask_j
                                )'''
        assert s.count(old)==1,e
        s=s.replace(old,f'''                                in_mask_counts["{e}"] = weighted_count(
                                    self, pts_{e}, raw_prompt_mask_j, native_info.get("{e}", {{}}), native_cfg
                                )''')
    assert s!=original;p.write_text(s)
    shutil.copy2(Path(__file__).parent/'native_context.py',root/'projects/v2sam_pccs/native_context.py')
