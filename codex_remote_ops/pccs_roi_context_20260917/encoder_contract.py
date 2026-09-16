import hashlib

def fingerprint(matcher):
    import torch
    h=hashlib.sha256()
    for name,value in sorted(matcher.dinov3_model.state_dict().items()):
        a=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {'state_sha256':h.hexdigest(),'parameter_dtype':str(next(matcher.dinov3_model.parameters()).dtype),'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'autocast_enabled':torch.is_autocast_enabled(),'autocast_dtype':str(torch.get_autocast_gpu_dtype()),'mean':matcher.mean.cpu().tolist(),'std':matcher.std.cpu().tolist()}
