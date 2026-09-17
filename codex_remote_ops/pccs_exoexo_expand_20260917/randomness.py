import hashlib
def stable_pair_seed(video_id):
    """Same per-pair seed as the already-audited candidate baseline."""
    return int.from_bytes(hashlib.sha256(('candidate-quality-v1:'+str(video_id)).encode()).digest()[:4],'little') % (2**31-1)
