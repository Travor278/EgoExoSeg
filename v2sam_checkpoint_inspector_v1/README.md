# V2-SAM safe checkpoint inspector v1

This inspector statically interprets the audited protocol-2 pickle opcode
stream inside a PyTorch ZIP checkpoint. It does not import PyTorch, NumPy or
MMEngine, and it never calls `pickle.load`, `torch.load`, GLOBAL/REDUCE
callables, `__setstate__`, or `__getstate__`.

It is intentionally specific to the ten GLOBAL references found in
`fusion_ego2exo_full.pth`. Unknown globals, reductions, opcodes, storage
types, persistent IDs, and non-literal HistoryBuffer attribute lookups are
rejected.

## Inspect on Inspire Studio

```bash
INSPECTOR=/path/to/v2sam_checkpoint_inspector_v1/safe_checkpoint_inspector.py
CKPT=/inspire/hdd/project/luojianlan/zhubingwen-253108120125/v2sam_offline_assets/fusion_ego2exo_full.pth
REPORT=/inspire/ssd/project/luojianlan/zhubingwen-253108120125/v2sam-repro/fusion_ego2exo_tensor_inventory.json

TMP_REPORT="${REPORT}.incomplete.$$"
python -I -S "$INSPECTOR" --json "$CKPT" >"$TMP_REPORT" &&
mv -n "$TMP_REPORT" "$REPORT"
```

## Run the tests

```bash
V2SAM_TEST_CHECKPOINT="$CKPT" \
python -I test_safe_checkpoint_inspector.py
```

Expected reference structure:

- tensor count: 1335
- `grounding_encoder`: 900
- `sparse_correspondence`: 371
- `matcher`: 60
- `constr_prompt_fcs`: 4
- tensor-structure SHA256:
  `bb5ce1af90bd6beeea5c0c5f646e1653a41d5afaca1f04096bab354bd35a0a6a`
