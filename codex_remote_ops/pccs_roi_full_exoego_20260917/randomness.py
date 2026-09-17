import hashlib,os
def stable_pair_seed(video_id):
    ns='candidate-quality-v'+os.environ['PCCS_EVAL_SEED']+':'
    return int.from_bytes(hashlib.sha256((ns+str(video_id)).encode()).digest()[:4],'little')%(2**31-1)
