# V2-SAM Resumable Dual Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify restart-safe four-GPU training for Fusion/NewMatcher on GPUs 0–3 and official B/Visual V2SAM on GPUs 4–7, while preventing either experiment from being mislabeled or resumed from an incomplete checkpoint.

**Architecture:** Create a portable deployment bundle in this repository, then use it to publish two immutable remote overlays. Both model implementations delegate the original 100-to-1 contrastive-loss switch to a persistent `ContrastScheduleState`; MMEngine saves complete epoch checkpoints with optimizer and scheduler state, and an external gate rejects any checkpoint or model lineage that does not match the run contract. Visual may proceed after its smoke test; the 24-epoch NewMatcher run additionally requires the existing official-checkpoint runtime mismatch to pass a contract gate, otherwise GPUs 0–3 run only the labeled two-epoch diagnostic.

**Tech Stack:** Python 3.10, PyTorch 2.3.1+cu121, MMEngine 0.11.0rc0, XTuner 0.1.23, `torchrun`, NCCL 2.20.5, Bash, four H100 GPUs per experiment.

**Spec:** `docs/superpowers/specs/2026-08-20-v2sam-resumable-dual-training-design.md`

## Global Constraints

- Fusion model is exactly `projects.v2sam.models.V2SAM_NEWMATCHER`; Visual model is exactly the Visual-source `projects.v2sam.models.V2SAM`.
- NewMatcher source model SHA256 before patching is `7f1c42d9089a834d4b9fbdee3f9809eca33d43d35186da704354c83120c3a1e4`.
- Visual source model SHA256 before patching is `f9550c6d81fafa43221c9af660c50fddfd2600397cf3928d676498c27941d3e8`.
- Training seed is `530358027`; author-aligned mode remains `deterministic=False`.
- Each experiment uses world size 4, per-device batch 16, accumulation 4, and nominal effective batch 256.
- Optimizer is AdamW at initial LR `4e-5`; Fusion horizon is 24 epochs and Visual horizon is 12 epochs.
- Ego2Exo FullTrain JSON SHA256 is `bb83cac92179b21833d2465ce29d5b0cf81668d8443d569279313c891472175a`.
- Training data closure report SHA256 is `ad6bc9dec706532e94b1b4bbd0b5a43579e6e6ce767e06479c780558803b140f` and requires 249,642 referenced images, zero missing images, and zero bad JPEGs.
- Checkpoints must contain top-level `state_dict`, `optimizer`, `param_schedulers`, `message_hub`, and `meta`.
- Existing inference overlays and old run directories are read-only inputs; never overwrite or reuse them.
- Do not launch 24-epoch NewMatcher while its official-checkpoint runtime contract is failed or unresolved.

---

### Task 1: Implement the Persistent Contrast Schedule State

**Files:**
- Create: `v2sam_resume_training/contrast_schedule_state.py`
- Create: `v2sam_resume_training/tests/test_contrast_schedule_state.py`
- Create: `v2sam_resume_training/__init__.py`

**Interfaces:**
- Produces: `ContrastScheduleState(threshold: int = 4000, high_scale: float = 100.0, low_scale: float = 1.0)`.
- Produces: `ContrastScheduleState.advance() -> tuple[torch.Tensor, torch.Tensor]`, returning scalar float tensors `(scale, step_for_logging)` while persisting `step` as a scalar `torch.long` buffer.
- Consumed by: both patched V2SAM model classes in Task 3.

- [ ] **Step 1: Write boundary and state-roundtrip tests**

```python
import torch

from v2sam_resume_training.contrast_schedule_state import (
    ContrastScheduleState,
)


def test_scale_switches_on_forward_4000():
    state = ContrastScheduleState()
    state.step.fill_(3998)

    scale_3999, logged_3999 = state.advance()
    scale_4000, logged_4000 = state.advance()

    assert scale_3999.item() == 100.0
    assert logged_3999.item() == 3999.0
    assert scale_4000.item() == 1.0
    assert logged_4000.item() == 4000.0


def test_state_dict_roundtrip_preserves_forward_count():
    source = ContrastScheduleState()
    source.step.fill_(4172)
    target = ContrastScheduleState()

    target.load_state_dict(source.state_dict(), strict=True)

    assert target.step.dtype == torch.long
    assert target.step.item() == 4172
    assert "step" in source.state_dict()


def test_advance_does_not_create_a_gradient_edge():
    state = ContrastScheduleState()
    scale, logged_step = state.advance()

    assert not scale.requires_grad
    assert not logged_step.requires_grad


def test_legacy_class_counter_is_not_checkpoint_state():
    class Legacy(torch.nn.Module):
        _constr_step = 0

    first = Legacy()
    first._constr_step += 4172
    second = Legacy()

    assert "_constr_step" not in first.state_dict()
    assert second._constr_step == 0
```

- [ ] **Step 2: Run the tests and verify the expected RED**

Run: `python -m pytest v2sam_resume_training/tests/test_contrast_schedule_state.py -q`

Expected: collection fails with `ModuleNotFoundError` for `contrast_schedule_state`.

- [ ] **Step 3: Implement the minimal persistent state module**

```python
from __future__ import annotations

import torch
from torch import nn


class ContrastScheduleState(nn.Module):
    def __init__(
        self,
        threshold: int = 4000,
        high_scale: float = 100.0,
        low_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if threshold < 1:
            raise ValueError("threshold must be positive")
        self.threshold = int(threshold)
        self.high_scale = float(high_scale)
        self.low_scale = float(low_scale)
        self.register_buffer(
            "step", torch.zeros((), dtype=torch.long), persistent=True
        )

    @torch.no_grad()
    def advance(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.step.add_(1)
        scale = torch.where(
            self.step < self.threshold,
            self.step.new_tensor(self.high_scale, dtype=torch.float32),
            self.step.new_tensor(self.low_scale, dtype=torch.float32),
        )
        return scale, self.step.detach().to(dtype=torch.float32)
```

- [ ] **Step 4: Run the focused and package test suites**

Run: `python -m pytest v2sam_resume_training/tests/test_contrast_schedule_state.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit the component**

```bash
git add v2sam_resume_training/__init__.py \
  v2sam_resume_training/contrast_schedule_state.py \
  v2sam_resume_training/tests/test_contrast_schedule_state.py
git commit -m "feat: persist V2-SAM contrast schedule state"
```

---

### Task 2: Implement the Checkpoint Resume Contract Gate

**Files:**
- Create: `v2sam_resume_training/checkpoint_gate.py`
- Create: `v2sam_resume_training/tests/test_checkpoint_gate.py`
- Create: `v2sam_resume_training/tests/fixtures/fusion_contract.json`

**Interfaces:**
- Consumes: a trusted, locally generated MMEngine checkpoint and an expected literal contract dictionary.
- Produces: `validate_checkpoint(path: Path, expected_contract: dict) -> dict`.
- CLI: `python -m v2sam_resume_training.checkpoint_gate CHECKPOINT --contract CONTRACT_JSON` returns 0 only on a complete match.

- [ ] **Step 1: Write tests for complete, optimizer-missing, counter-missing, and identity-mismatch checkpoints**

```python
import json
from pathlib import Path

import pytest
import torch

from v2sam_resume_training.checkpoint_gate import (
    CheckpointValidationError,
    validate_checkpoint,
)


CONTRACT = {
    "schema": "v2sam-resume-contract-v1",
    "variant": "fusion_newmatcher",
    "seed": 530358027,
    "world_size": 4,
    "per_device_batch_size": 16,
    "accumulative_counts": 4,
    "data_sha256": "b" * 64,
    "source_model_sha256": "c" * 64,
    "runtime_model_sha256": "d" * 64,
    "config_sha256": "e" * 64,
}


def write_checkpoint(path: Path) -> None:
    torch.save(
        {
            "meta": {
                "epoch": 1,
                "iter": 8,
                "seed": 530358027,
                "v2sam_resume_contract": CONTRACT,
            },
            "state_dict": {
                "contrast_schedule.step": torch.tensor(8, dtype=torch.long)
            },
            "optimizer": {
                "state": {
                    0: {
                        "step": torch.tensor(2.0),
                        "exp_avg": torch.zeros(1),
                        "exp_avg_sq": torch.zeros(1),
                    }
                },
                "param_groups": [{"lr": 4e-5}],
            },
            "param_schedulers": [{"last_step": 7}],
            "message_hub": {"runtime_info": {}},
        },
        path,
    )


def test_complete_checkpoint_is_accepted(tmp_path):
    path = tmp_path / "epoch_1.pth"
    write_checkpoint(path)
    report = validate_checkpoint(path, CONTRACT)
    assert report["status"] == "PASS"
    assert report["epoch"] == 1
    assert report["iter"] == 8
    assert report["contrast_step"] == 8


@pytest.mark.parametrize(
    ("removed", "message"),
    [("optimizer", "missing top-level key: optimizer"),
     ("param_schedulers", "missing top-level key: param_schedulers")],
)
def test_incomplete_training_state_is_rejected(tmp_path, removed, message):
    path = tmp_path / "bad.pth"
    write_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu")
    del checkpoint[removed]
    torch.save(checkpoint, path)
    with pytest.raises(CheckpointValidationError, match=message):
        validate_checkpoint(path, CONTRACT)


def test_missing_contrast_step_is_rejected(tmp_path):
    path = tmp_path / "bad.pth"
    write_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["state_dict"].clear()
    torch.save(checkpoint, path)
    with pytest.raises(CheckpointValidationError, match="contrast schedule"):
        validate_checkpoint(path, CONTRACT)


def test_empty_adam_state_is_rejected(tmp_path):
    path = tmp_path / "bad.pth"
    write_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["optimizer"]["state"].clear()
    torch.save(checkpoint, path)
    with pytest.raises(CheckpointValidationError, match="optimizer state"):
        validate_checkpoint(path, CONTRACT)


def test_counter_must_equal_runner_iteration(tmp_path):
    path = tmp_path / "bad.pth"
    write_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["state_dict"]["contrast_schedule.step"].fill_(7)
    torch.save(checkpoint, path)
    with pytest.raises(CheckpointValidationError, match="runner iteration"):
        validate_checkpoint(path, CONTRACT)


def test_wrong_variant_is_rejected(tmp_path):
    path = tmp_path / "bad.pth"
    write_checkpoint(path)
    expected = dict(CONTRACT, variant="visual_v2sam")
    with pytest.raises(CheckpointValidationError, match="variant"):
        validate_checkpoint(path, expected)
```

- [ ] **Step 2: Run the tests and verify the expected RED**

Run: `python -m pytest v2sam_resume_training/tests/test_checkpoint_gate.py -q`

Expected: import fails because `checkpoint_gate.py` does not exist.

- [ ] **Step 3: Implement strict structural and contract validation**

Implement `CheckpointValidationError` and `validate_checkpoint` with these exact rules:

```python
required = {"meta", "state_dict", "optimizer", "param_schedulers", "message_hub"}
missing = sorted(required - checkpoint.keys())
if missing:
    raise CheckpointValidationError(
        "missing top-level key: " + ", ".join(missing)
    )

state = checkpoint["state_dict"]
counter_keys = [key for key in state if key.endswith("contrast_schedule.step")]
if counter_keys != ["contrast_schedule.step"]:
    raise CheckpointValidationError("missing or ambiguous contrast schedule")

epoch = checkpoint["meta"].get("epoch")
iteration = checkpoint["meta"].get("iter")
if not isinstance(epoch, int) or epoch < 1:
    raise CheckpointValidationError("invalid checkpoint epoch")
if not isinstance(iteration, int) or iteration < 1:
    raise CheckpointValidationError("invalid checkpoint iteration")
if int(state[counter_keys[0]].item()) != iteration:
    raise CheckpointValidationError(
        "contrast schedule step does not match runner iteration"
    )

optimizer = checkpoint["optimizer"]
optimizer_states = optimizer.get("state") if isinstance(optimizer, dict) else None
if not optimizer_states or not all(
    {"step", "exp_avg", "exp_avg_sq"} <= set(item)
    for item in optimizer_states.values()
):
    raise CheckpointValidationError("missing AdamW optimizer state")
if not checkpoint["param_schedulers"]:
    raise CheckpointValidationError("missing parameter scheduler state")

actual_contract = checkpoint["meta"].get("v2sam_resume_contract")
for key, expected in expected_contract.items():
    actual = actual_contract.get(key) if isinstance(actual_contract, dict) else None
    if actual != expected:
        raise CheckpointValidationError(
            f"contract mismatch for {key}: expected {expected!r}, got {actual!r}"
        )
```

The CLI must state that `torch.load` is allowed only for trusted checkpoints created by these local training runs; it must not be reused for downloaded untrusted pickle files.

Create `v2sam_resume_training/tests/fixtures/fusion_contract.json` as the literal JSON equivalent of `CONTRACT` (all repeated SHA characters expanded to 64 characters). The CLI test below uses this file and a locally generated legacy-style fixture; it does not deserialize an untrusted downloaded checkpoint.

- [ ] **Step 4: Verify GREEN and exercise the CLI error code**

Run:

```bash
python -m pytest v2sam_resume_training/tests/test_checkpoint_gate.py -q
python - <<'PY'
from pathlib import Path
import torch

path = Path("build/legacy_checkpoint.pth")
path.parent.mkdir(exist_ok=True)
torch.save({"meta": {}, "state_dict": {}}, path)
print(path)
PY
python -m v2sam_resume_training.checkpoint_gate \
  build/legacy_checkpoint.pth \
  --contract v2sam_resume_training/tests/fixtures/fusion_contract.json
```

Expected: tests pass; the legacy-style fixture command exits nonzero because it lacks optimizer and the new persistent counter.

- [ ] **Step 5: Commit the gate**

```bash
git add v2sam_resume_training/checkpoint_gate.py \
  v2sam_resume_training/tests/test_checkpoint_gate.py \
  v2sam_resume_training/tests/fixtures/fusion_contract.json
git commit -m "feat: reject incomplete V2-SAM resume checkpoints"
```

---

### Task 3: Patch Both Model Variants Through an Idempotent Overlay Publisher

**Files:**
- Create: `v2sam_resume_training/overlay_publisher.py`
- Create: `v2sam_resume_training/tests/fixtures/v2sam_newmatcher_minimal.py`
- Create: `v2sam_resume_training/tests/fixtures/v2sam_visual_minimal.py`
- Create: `v2sam_resume_training/tests/test_overlay_publisher.py`
- Remote create: `projects/v2sam/models/contrast_schedule_state.py` in each new overlay.
- Remote modify: Fusion `projects/v2sam/models/v2sam_newmatcher.py` around the constructor and old `_constr_step` block.
- Remote modify: Visual `projects/v2sam/models/v2sam.py` around the constructor and old `_constr_step` block.

**Interfaces:**
- Produces: immutable `VariantSpec(name, model_relpath, class_name, expected_source_sha256, expects_sparse_correspondence)` records.
- Produces: `publish_overlay(source_root, target_root, spec, support_module) -> dict`.
- Production spec `fusion_newmatcher` requires source SHA `7f1c42...`, class `V2SAM_NEWMATCHER`, and sparse-correspondence parameters.
- Production spec `visual_v2sam` requires source SHA `f9550c...`, class `V2SAM`, and no sparse-correspondence parameters.

- [ ] **Step 1: Write fixture-based behavioral tests for publishing and patching**

The fixtures must contain the real constructor shape, a scalar `loss_contr`, and the exact original increment-before-threshold behavior. Tests execute the patched fixture with `torch` and assert:

```python
import hashlib
import importlib
from pathlib import Path
import sys

import pytest
import torch

from v2sam_resume_training.overlay_publisher import (
    PublishError,
    VariantSpec,
    publish_overlay,
)


SUPPORT_MODULE = (
    Path(__file__).resolve().parents[1] / "contrast_schedule_state.py"
)


def import_published(root, module_name):
    for name in list(sys.modules):
        if name == "projects" or name.startswith("projects."):
            del sys.modules[name]
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(root))


def publish_fixture(tmp_path, variant, target=None):
    source = tmp_path / f"source-{variant}"
    model_relpath = (
        "projects/v2sam/models/v2sam_newmatcher.py"
        if variant == "fusion_newmatcher"
        else "projects/v2sam/models/v2sam.py"
    )
    fixture = (
        Path(__file__).parent / "fixtures" /
        ("v2sam_newmatcher_minimal.py"
         if variant == "fusion_newmatcher"
         else "v2sam_visual_minimal.py")
    )
    model_path = source / model_relpath
    model_path.parent.mkdir(parents=True, exist_ok=True)
    for package_dir in (
        source / "projects",
        source / "projects/v2sam",
        source / "projects/v2sam/models",
    ):
        (package_dir / "__init__.py").touch()
    model_path.write_bytes(fixture.read_bytes())
    expected_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    spec = VariantSpec(
        name=variant,
        model_relpath=model_relpath,
        class_name=("V2SAM_NEWMATCHER"
                    if variant == "fusion_newmatcher" else "V2SAM"),
        expected_source_sha256=expected_sha,
        expects_sparse_correspondence=(variant == "fusion_newmatcher"),
    )
    return publish_overlay(
        source,
        target or (tmp_path / f"published-{variant}"),
        spec,
        SUPPORT_MODULE,
    )["target_root"]


def test_published_model_persists_counter_and_preserves_boundary(tmp_path):
    target = publish_fixture(tmp_path, "fusion_newmatcher")
    module = import_published(
        target, "projects.v2sam.models.v2sam_newmatcher"
    )
    model = module.V2SAM_NEWMATCHER()
    model.contrast_schedule.step.fill_(3998)
    first = model.forward_for_test(torch.tensor(2.0))
    second = model.forward_for_test(torch.tensor(2.0))
    assert first["loss_contr"].item() == 200.0
    assert second["loss_contr"].item() == 2.0
    assert second["constr_step"].item() == 4000.0


def test_existing_unexpected_target_is_never_overwritten(tmp_path):
    target = tmp_path / "published"
    target.mkdir()
    (target / "sentinel").write_text("user data")
    with pytest.raises(PublishError, match="unexpected existing target"):
        publish_fixture(tmp_path, "visual_v2sam", target=target)
    assert (target / "sentinel").read_text() == "user data"


def test_matching_published_target_is_idempotent(tmp_path):
    first = publish_fixture(tmp_path, "fusion_newmatcher")
    first_manifest = (first / ".v2sam-resume-overlay.json").read_bytes()
    second = publish_fixture(tmp_path, "fusion_newmatcher")
    assert second == first
    assert (second / ".v2sam-resume-overlay.json").read_bytes() == first_manifest


def test_diagnostics_do_not_change_parsed_total_loss(tmp_path):
    target = publish_fixture(tmp_path, "visual_v2sam")
    module = import_published(target, "projects.v2sam.models.v2sam")
    outputs = module.V2SAM().forward_for_test(torch.tensor(2.0))

    parsed_total = sum(
        value for name, value in outputs.items() if "loss" in name
    )
    assert set(outputs) >= {
        "loss_contr", "contr_raw", "contr_scale", "constr_step"
    }
    assert parsed_total.item() == outputs["loss_contr"].item()
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest v2sam_resume_training/tests/test_overlay_publisher.py -q`

Expected: import failure for `overlay_publisher`.

- [ ] **Step 3: Implement staged, SHA-gated, atomic publication**

The publisher must:

1. resolve the active model path inside the source root;
2. calculate SHA256 and require the exact pre-patch identity;
3. create a staging directory under the same remote `$BASE` filesystem;
4. copy with symlinks preserved;
5. copy `contrast_schedule_state.py` beside the active model;
6. insert `from .contrast_schedule_state import ContrastScheduleState`;
7. add `self.contrast_schedule = ContrastScheduleState()` in `__init__`;
8. replace only the unique class-attribute counter block with:

```python
loss_contr_raw = loss_contr
_contr_scale, _constr_step = self.contrast_schedule.advance()
loss_contr = loss_contr_raw * _contr_scale.to(
    device=loss_contr_raw.device,
    dtype=loss_contr_raw.dtype,
)
```

9. extend the loss dictionary with these non-loss diagnostic keys:

```python
"contr_raw": loss_contr_raw.detach(),
"contr_scale": _contr_scale.detach(),
"constr_step": _constr_step.detach(),
```

10. compile the patched Python file, import the state module, require exactly one replacement, write `.v2sam-resume-overlay.json` with source/patched hashes and variant identity, and atomically rename staging to target. A repeated call may return the existing target only when every manifest field and file hash still matches; otherwise it raises `PublishError` without modifying the target.

- [ ] **Step 4: Run all publisher and state tests**

Run: `python -m pytest v2sam_resume_training/tests/test_overlay_publisher.py v2sam_resume_training/tests/test_contrast_schedule_state.py -q`

Expected: all tests pass and both fixture variants have a `contrast_schedule.step` state key.

- [ ] **Step 5: Commit the publisher**

```bash
git add v2sam_resume_training/overlay_publisher.py \
  v2sam_resume_training/tests/fixtures \
  v2sam_resume_training/tests/test_overlay_publisher.py
git commit -m "feat: publish resumable Fusion and Visual overlays"
```

---

### Task 4: Embed the Immutable Run Contract in Every Checkpoint

**Files:**
- Create: `v2sam_resume_training/resume_contract_hook.py`
- Create: `v2sam_resume_training/tests/test_resume_contract_hook.py`
- Remote create: `projects/v2sam/hooks/resume_contract_hook.py` in both overlays.

**Interfaces:**
- Produces: registered MMEngine hook `ResumeContractHook(contract: dict)`.
- Writes: `checkpoint["meta"]["v2sam_resume_contract"]` in `before_save_checkpoint`.

- [ ] **Step 1: Write a real checkpoint-mutation test**

```python
from v2sam_resume_training.resume_contract_hook import ResumeContractHook


def test_hook_copies_contract_into_checkpoint_metadata():
    contract = {
        "schema": "v2sam-resume-contract-v1",
        "variant": "visual_v2sam",
        "seed": 530358027,
        "world_size": 4,
        "per_device_batch_size": 16,
        "accumulative_counts": 4,
        "data_sha256": "b" * 64,
        "source_model_sha256": "c" * 64,
        "runtime_model_sha256": "d" * 64,
        "config_sha256": "e" * 64,
    }
    checkpoint = {"meta": {"epoch": 1, "iter": 8}}

    ResumeContractHook(contract).before_save_checkpoint(None, checkpoint)

    assert checkpoint["meta"]["v2sam_resume_contract"] == contract
    contract["variant"] = "mutated"
    assert checkpoint["meta"]["v2sam_resume_contract"]["variant"] == "visual_v2sam"
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest v2sam_resume_training/tests/test_resume_contract_hook.py -q`

Expected: module import fails.

- [ ] **Step 3: Implement the hook with literal contract validation**

Require exactly these fields at construction: `schema`, `variant`, `seed`, `world_size`, `per_device_batch_size`, `accumulative_counts`, `data_sha256`, `source_model_sha256`, `runtime_model_sha256`, and `config_sha256`. Here `source_model_sha256` is the approved pre-patch lineage identity, while `runtime_model_sha256` is the actual patched model file used by the run. Reject missing or extra fields and deep-copy the dictionary before storing it.

- [ ] **Step 4: Verify the hook test and whole CPU suite**

Run: `python -m pytest v2sam_resume_training/tests -q`

Expected: all tests pass without CUDA initialization.

- [ ] **Step 5: Commit the hook**

```bash
git add v2sam_resume_training/resume_contract_hook.py \
  v2sam_resume_training/tests/test_resume_contract_hook.py
git commit -m "feat: embed V2-SAM training contract in checkpoints"
```

---

### Task 5: Generate Canonical Fusion and Visual Runtime Configs

**Files:**
- Create: `v2sam_resume_training/build_runtime_configs.py`
- Create: `v2sam_resume_training/tests/test_build_runtime_configs.py`
- Remote create: `$REPRO/configs/train_fusion_newmatcher_resumable.py`
- Remote create: `$REPRO/configs/train_visual_v2sam_resumable.py`

**Interfaces:**
- CLI inputs: variant, canonical source config, output path, asset/data paths, and contract fields.
- Produces: a compiled Python config that retains callable `template_map_fn` and exposes `v2sam_resume_contract`.

- [ ] **Step 1: Write config-generation tests**

Use minimal source configs with dict-shaped model/dataloader/hooks and assert these observable results after executing the generated config:

```python
assert generated.model["type"].__name__ == "V2SAM_NEWMATCHER"
assert generated.train_dataloader["batch_size"] == 16
assert generated.optim_wrapper["accumulative_counts"] == 4
assert generated.default_hooks["checkpoint"]["save_optimizer"] is True
assert generated.default_hooks["checkpoint"]["save_param_scheduler"] is True
assert generated.default_hooks["checkpoint"]["by_epoch"] is True
assert generated.default_hooks["checkpoint"]["interval"] == 1
assert callable(
    generated.train_dataloader["dataset"]["datasets"][0]
    ["template_map_fn"]["type"]
)
```

The Visual case must resolve `V2SAM`, use 12 epochs, and contain no DINO model arguments. The Fusion case must resolve `V2SAM_NEWMATCHER`, use 24 epochs, and contain the verified DINO repo/weight paths.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest v2sam_resume_training/tests/test_build_runtime_configs.py -q`

Expected: import failure for `build_runtime_configs`.

- [ ] **Step 3: Implement deterministic config generation**

Generate the config by copying the canonical source text and appending explicit overrides. The appended section must set:

```python
randomness = dict(seed=530358027, deterministic=False)
batch_size = 16
accumulative_counts = 4
train_dataloader["batch_size"] = 16
optim_wrapper["accumulative_counts"] = 4
optim_wrapper["optimizer"]["lr"] = 4e-5
default_hooks["checkpoint"].update(
    save_optimizer=True,
    save_param_scheduler=True,
    by_epoch=True,
    interval=1,
    max_keep_ckpts=-1,
)
default_hooks["logger"]["interval"] = 10
```

Set `train_cfg["max_epochs"]` and both scheduler endpoints from the variant horizon. Rebuild `template_map_fn` using the real `template_map_fn_factory` callable rather than a dumped string. Add `ResumeContractHook` through `custom_imports` and `custom_hooks`.

Compute the completed config SHA256 in the launcher and pass it as environment variable `V2SAM_CONFIG_SHA256`; compute the patched model file SHA256 and pass it as `V2SAM_RUNTIME_MODEL_SHA256`. The config holds the approved pre-patch `source_model_sha256` as a literal and reads the two runtime values into the checkpoint contract, avoiding a self-referential config hash while retaining both lineage identities.

- [ ] **Step 4: Run config tests and compile generated examples**

Run:

```bash
python -m pytest v2sam_resume_training/tests/test_build_runtime_configs.py -q
python -m py_compile v2sam_resume_training/build_runtime_configs.py
```

Expected: all tests pass and compilation exits 0.

- [ ] **Step 5: Commit the config builder**

```bash
git add v2sam_resume_training/build_runtime_configs.py \
  v2sam_resume_training/tests/test_build_runtime_configs.py
git commit -m "feat: generate canonical resumable V2-SAM configs"
```

---

### Task 6: Build and Verify the Remote Deployment Bundle

**Files:**
- Create: `v2sam_resume_training/remote_prepare.py`
- Create: `v2sam_resume_training/remote_preflight.py`
- Create: `v2sam_resume_training/bundle_builder.py`
- Create: `v2sam_resume_training/tests/test_remote_preflight.py`
- Create: `v2sam_resume_training/tests/test_bundle_builder.py`
- Create: `v2sam_resume_training/README.md`
- Create: `deliverables/v2sam-resumable-dual-training-20260820.tar.gz`
- Create: `deliverables/v2sam-resumable-dual-training-20260820.tar.gz.sha256`

**Interfaces:**
- `remote_prepare.py` consumes the two verified source roots and publishes the two new overlays.
- `remote_preflight.py` returns nonzero unless source, model type, dataset, callable config, checkpoint policy, environment, and GPU allocation all match the contract.
- `bundle_builder.py SOURCE_DIR ARCHIVE SIDECAR` creates a byte-reproducible tar.gz and its SHA256 sidecar on Windows or Linux.

- [ ] **Step 1: Write failing preflight tests for swapped model identities**

Construct two tiny overlay fixtures. The preflight must reject a Visual contract whose resolved class is `V2SAM_NEWMATCHER`, and reject a Fusion contract without `sparse_correspondence.*` state keys. It must accept only the mapping in the approved spec. Add a bundle test that builds the same source tree twice and requires identical archive bytes and sidecar content.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest v2sam_resume_training/tests/test_remote_preflight.py -q`

Expected: import failure for `remote_preflight`.

- [ ] **Step 3: Implement remote preparation and read-only preflight**

Use these exact remote roots:

```text
BASE=/inspire/ssd/project/luojianlan/zhubingwen-253108120125
REPRO=$BASE/v2sam-repro
FUSION_SOURCE=$BASE/v2sam-o-ego2exo-7b88c299-overlay-v1
VISUAL_SOURCE=$BASE/v2sam-visual-official-24ae5a-overlay-v2
FUSION_TARGET=$BASE/v2sam-newmatcher-resumable-7b88c299-overlay-v1
VISUAL_TARGET=$BASE/v2sam-visual-resumable-24ae5a-overlay-v1
```

The preflight must import configs from `/tmp` with explicit `PYTHONPATH`, print the resolved class/module/source, verify the two pre-patch SHAs, verify the FullTrain and closure report SHAs, and require GPUs 0–7 to have no compute processes.

The bundle builder must sort POSIX archive names, normalize uid/gid to 0, uname/gname to empty strings, modes to executable/non-executable canonical values, and every tar/gzip mtime to `2026-08-20T00:00:00Z`; it must reject symlinks that escape the source tree.

- [ ] **Step 4: Run the full local test suite and build a deterministic archive**

Run:

```bash
python -m pytest v2sam_resume_training/tests -q
python -m compileall -q v2sam_resume_training
python -m v2sam_resume_training.bundle_builder \
  v2sam_resume_training \
  deliverables/v2sam-resumable-dual-training-20260820.tar.gz \
  deliverables/v2sam-resumable-dual-training-20260820.tar.gz.sha256
```

Expected: tests and compile pass; the final archive SHA is stored in the sidecar `.sha256` file. `README.md` documents how to verify the sidecar but does not embed the archive hash, avoiding a circular rebuild.

- [ ] **Step 5: Commit the deployment bundle sources**

```bash
git add v2sam_resume_training \
  deliverables/v2sam-resumable-dual-training-20260820.tar.gz \
  deliverables/v2sam-resumable-dual-training-20260820.tar.gz.sha256
git commit -m "feat: package resumable dual V2-SAM training"
```

---

### Task 7: Run Real Four-GPU Save/Resume Smoke Tests

**Files:**
- Remote create: `$REPRO/smoke_inputs/ego2exo_train_first512.json`
- Remote create: two smoke configs under `$REPRO/configs/`
- Remote create: two immutable smoke run directories under `$REPRO/runs/`

**Interfaces:**
- Consumes: the published overlays/configs and 512 deterministic FullTrain records.
- Produces: one pre-resume checkpoint and one post-resume receipt for each model variant.

- [ ] **Step 1: Create the deterministic 512-record scope**

Read the verified FullTrain dict in insertion order and atomically write the first 512 entries. Record its SHA256 and require exactly 512 keys.

- [ ] **Step 2: Run Fusion/NewMatcher smoke on GPUs 0–3**

Use batch 16/GPU, accumulation 4, logger interval 1, one epoch, and master port 29710. With 512 records and four ranks this yields eight rank-local micro-iterations and two optimizer steps. Abort and capture diagnostics if no first `Iter(train)` appears within 10 minutes after initialization.

- [ ] **Step 3: Validate and explicitly resume Fusion smoke**

Run `checkpoint_gate.py` on `epoch_1.pth`, then resume with the canonical smoke config and explicit checkpoint path for epoch 2. Require the first resumed record to show epoch/iter continuity, LR continuity, `constr_step > 0`, and the same `contr_scale` branch.

- [ ] **Step 4: Repeat save/validate/resume on Visual GPUs 4–7**

Use master port 29711 and an independent work directory. Apply identical state checks; do not load the old epoch-12 Visual checkpoint.

- [ ] **Step 5: Write smoke receipts and verify GPU cleanup**

Each receipt contains config/source/data/checkpoint SHAs, pre/post iter, pre/post LR, pre/post contrast step, optimizer state entry count, exit codes, and `nvidia-smi` after termination. Both receipts must say `RESUME_SMOKE=PASS` before a formal launch is permitted.

---

### Task 8: Enforce the Visual Sanity and NewMatcher Runtime Contract Gates

**Files:**
- Remote create: `$REPRO/newmatcher_contract_gate.json`
- Read: completed public Fusion and NewMatcher official-checkpoint runs.

**Interfaces:**
- Produces an independent Visual sanity result plus one of `PASS`, `FAIL_DIAGNOSTIC_ONLY`, or `BLOCKED_MISSING_EVIDENCE` for NewMatcher.
- Consumed by the formal launcher in Task 9.

- [ ] **Step 1: Revalidate the immutable Visual and Fusion full-run evidence**

First run the official Visual checkpoint through the Visual overlay on a frozen small scope, require strict loading except for the newly introduced counter buffer, prove there are no `sparse_correspondence.*` parameters, and reproduce the established full-run scale near `0.3670/0.4280`. Record this as an independent `visual_sanity` field.

Then require the same official Fusion tensor identity and same 40,517-record test JSON for both Fusion runtime paths. Record the public `V2SAM` result near `0.4469/0.5097` and candidate NewMatcher result near `0.1727/0.2132`.

- [ ] **Step 2: Run a frozen small-scope check after the persistence patch**

Use the official Fusion checkpoint, identical sample IDs, seed, SAM2/DINO assets, and no training. The persistence patch must not affect prediction mode. Record model class/source, strict-load missing/unexpected keys excluding the new counter buffer, and per-sample result hashes.

- [ ] **Step 3: Classify the contract**

If the large metric gap remains, write:

```json
{
  "schema": "v2sam-runtime-gates-v1",
  "visual_sanity": {
    "status": "PASS",
    "allow_12_epoch": true
  },
  "newmatcher": {
    "status": "FAIL_DIAGNOSTIC_ONLY",
    "reason": "official checkpoint predictions diverge under candidate NewMatcher runtime",
    "allow_24_epoch": false,
    "allow_2_epoch_diagnostic": true
  }
}
```

Only set `allow_24_epoch=true` after the same frozen runtime produces the official-result scale and the source/config lineage is explained. Do not add a force flag to bypass this gate.

- [ ] **Step 4: Preserve the gate as an immutable receipt**

Write via temporary file plus atomic rename, then record SHA256. This gate controls only Fusion; it must not block the independent Visual launch.

---

### Task 9: Launch the Two Isolated Training Jobs

**Files:**
- Create: `v2sam_resume_training/launch_dual_training.sh`
- Create: `v2sam_resume_training/status_dual_training.sh`
- Create: `v2sam_resume_training/resume_one_training.sh`
- Remote create: immutable run directories and latest-run pointer files.

**Interfaces:**
- `launch_dual_training.sh` starts Visual full training only after its smoke PASS and `visual_sanity=PASS`.
- It starts Fusion for 24 epochs only when `newmatcher_contract_gate.json` permits it; otherwise it starts the explicitly labeled two-epoch diagnostic.
- `resume_one_training.sh VARIANT CHECKPOINT` runs the external checkpoint gate before any `torchrun` process.

- [ ] **Step 1: Write shell tests with stubbed `torchrun` and `nvidia-smi`**

Test that a failed Fusion contract never invokes a 24-epoch config, that a Visual sanity failure blocks only Visual, that Visual still launches when its independent smoke/sanity gates pass, and that a rejected checkpoint never invokes `torchrun`. Assert recorded argument arrays rather than grepping script source.

- [ ] **Step 2: Run shell tests and verify RED**

Run: `python -m pytest v2sam_resume_training/tests/test_launch_scripts.py -q`

Expected: failure because launch scripts do not exist.

- [ ] **Step 3: Implement safe launch and resume scripts**

Launch environment must include:

```bash
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
PYTHONHASHSEED=530358027
OMP_NUM_THREADS=4
NCCL_DEBUG=WARN
```

Use `CUDA_VISIBLE_DEVICES=0,1,2,3` and master port 29720 for Fusion; use `CUDA_VISIBLE_DEVICES=4,5,6,7` and master port 29721 for Visual. Export the approved pre-patch identity as `V2SAM_SOURCE_MODEL_SHA256`; compute and export `V2SAM_RUNTIME_MODEL_SHA256` and `V2SAM_CONFIG_SHA256` immediately before launch. The immutable manifest must contain both model hashes, `models/__init__.py`, config, FullTrain, closure report, SAM2, and (Fusion only) DINO hashes; resolved class/module/source; PyTorch/CUDA/NCCL/MMEngine/XTuner versions; seed; world size; batch; accumulation; and effective batch. Write PID/log/config/manifest paths before returning control to the terminal.

Resume requires an explicit checkpoint path, exact process identity checks, callable-config preflight, and `checkpoint_gate.py` PASS. It must never read the dumped config from the work directory.

- [ ] **Step 4: Verify scripts with stubs and ShellCheck where available**

Run:

```bash
python -m pytest v2sam_resume_training/tests/test_launch_scripts.py -q
bash -n v2sam_resume_training/launch_dual_training.sh
bash -n v2sam_resume_training/status_dual_training.sh
bash -n v2sam_resume_training/resume_one_training.sh
```

Expected: all tests pass and all scripts parse successfully.

- [ ] **Step 5: Start the jobs and inspect the first real iteration**

Start Visual on GPUs 4–7. On GPUs 0–3, start either the permitted 24-epoch Fusion run or the two-epoch `NEWMATCHER_CANDIDATE_DIAGNOSTIC` run. Require nonzero GPU memory/utilization, four live ranks per job, finite loss, expected LR, and monotonic contrast step.

- [ ] **Step 6: Commit launch tooling**

Rebuild the deterministic deployment archive and sidecar after adding the launch scripts so the final deliverable contains the complete package, then run the same archive verification command from Task 6.

```bash
git add v2sam_resume_training/launch_dual_training.sh \
  v2sam_resume_training/status_dual_training.sh \
  v2sam_resume_training/resume_one_training.sh \
  v2sam_resume_training/tests/test_launch_scripts.py \
  deliverables/v2sam-resumable-dual-training-20260820.tar.gz \
  deliverables/v2sam-resumable-dual-training-20260820.tar.gz.sha256
git commit -m "feat: launch and resume isolated V2-SAM training"
```

---

### Task 10: Verify the Full Story and Update the Experiment Diary

**Files:**
- Modify: `docs/V2SAM_EGO2EXO_EXPERIMENT_DIARY_AND_PLAN_20260820.md`
- Create: `docs/V2SAM_RESUMABLE_TRAINING_RUN_RECEIPT_20260820.md`

**Interfaces:**
- Consumes: test output, overlay manifests, smoke receipts, contract gate, launch logs, and first valid checkpoints.
- Produces: one human-readable evidence index without changing historical metrics.

- [ ] **Step 1: Run all CPU verification from a clean Python process**

Run:

```bash
python -m pytest v2sam_resume_training/tests -q
python -m compileall -q v2sam_resume_training
git diff --check -- v2sam_resume_training docs
```

Expected: zero failures and exit code 0 for every command.

- [ ] **Step 2: Verify remote evidence independently**

Recompute every config/source/data/checkpoint SHA, inspect checkpoint top-level keys, validate the persistent counter, count four ranks per active job, and confirm the GPU partition is 0–3 versus 4–7.

- [ ] **Step 3: Update labels and status in the diary**

Record Visual as `V2SAM Visual`, Fusion as either `V2SAM_NEWMATCHER strict` or `NewMatcher candidate diagnostic` according to the immutable gate. State explicitly that old Visual epoch 12 was not resumed.

- [ ] **Step 4: Commit documentation only after fresh verification**

```bash
git add docs/V2SAM_EGO2EXO_EXPERIMENT_DIARY_AND_PLAN_20260820.md \
  docs/V2SAM_RESUMABLE_TRAINING_RUN_RECEIPT_20260820.md
git commit -m "docs: record resumable V2-SAM training evidence"
```

- [ ] **Step 5: Report only evidence-backed status**

The handoff must include the exact active run paths, PIDs, latest iterations, LR, contrast steps, checkpoint gate outputs, and the next expected validation time. Do not claim strict Fusion reproduction if Task 8 remains `FAIL_DIAGNOSTIC_ONLY`.
