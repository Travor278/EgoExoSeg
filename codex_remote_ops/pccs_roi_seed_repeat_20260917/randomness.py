import hashlib
def stable_pair_seed(video_id):
    """Predeclared repeat seed; no selection based on target outcomes."""
    return int.from_bytes(hashlib.sha256(('candidate-quality-v2:'+str(video_id)).encode()).digest()[:4],'little') % (2**31-1)
