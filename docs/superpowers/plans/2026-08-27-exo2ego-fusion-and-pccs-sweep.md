# Exo2Ego Fusion and PCCS Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 恢复或确认 Exo2Ego Fusion 24-epoch 主线，并补测 Visual e7/e8/e13/e14 与固定 Fusion checkpoint 的全量 PCCS。

**Architecture:** 先从启智任务详情确认 Fusion 任务与最后 checkpoint，再从训练合同和持久化目录确认候选 checkpoint 实体。每个 PCCS 任务复用已成功的 e9 配置，只允许改变 `VISUAL_CKPT`、`TAG` 和任务名；first-8 门通过后运行 46,515 对象全量评测。

**Tech Stack:** 启智分布式训练、PyTorch、MMEngine、Playwright CLI、WRZ `seg_metric_fusionfirst_tripledecoder_newmetric.py`。

**Spec:** `docs/V2SAM_EGO2EXO_EXPERIMENT_DIARY_AND_PLAN_20260820.md`

## Global Constraints

- Exo2Ego 数据集固定为 46,515 个对象，不得切换到 Ego2Exo 40,517 对象集合。
- PCCS 的 Fusion checkpoint 在同一轮 sweep 中固定，不允许随 Visual epoch 改变。
- 只使用真实存在且稳定的 checkpoint；缺失 checkpoint 不创建必失败任务。
- 所有完成声明必须同时有 job 状态、exit code 和最终完整指标证据。

---

### Task 1: Audit Exo2Ego Fusion

**Files:**
- Modify after evidence: `docs/V2SAM_EGO2EXO_EXPERIMENT_DIARY_AND_PLAN_20260820.md`

- [ ] 在分布式任务列表搜索 `v2sam-exo2ego-fusion-official24`。
- [ ] 读取任务状态、最后日志、已保存 checkpoint 和严格续跑证据。
- [ ] 若任务停止且未到 epoch 24，从最后稳定 checkpoint 创建续跑任务；若仍运行，只记录 ETA。

### Task 2: Verify Visual Checkpoint Inventory

**Files:**
- Read: Exo2Ego Visual 训练任务配置与持久化 checkpoint 目录。

- [ ] 确认 `epoch_7.pth`、`epoch_8.pth`、`epoch_13.pth`、`epoch_14.pth` 是否存在。
- [ ] 对每个存在文件记录绝对路径；对缺失文件记录训练 `max_epochs` 证据。

### Task 3: Launch Valid PCCS Candidates

**Interfaces:**
- Consumes: 已验证的 Visual checkpoint 路径与固定 Fusion checkpoint 路径。
- Produces: 每个候选一个独立 4×H100 job ID。

- [ ] 复制已成功的 e9 PCCS 任务。
- [ ] 仅替换任务名、`VISUAL_CKPT` 和 `TAG`。
- [ ] 提交前复核 Exo2Ego 46,515、Fusion 固定点、WRZ evaluator 和 4×H100 资源合同。
- [ ] 提交任务并在任务详情验证实际执行命令。

### Task 4: Monitor and Publish Results

**Files:**
- Modify: `docs/V2SAM_EGO2EXO_EXPERIMENT_DIARY_AND_PLAN_20260820.md`

- [ ] 在 first-8 门、全量评测开始、最终 exit code 三个边界核验日志。
- [ ] 提取 IoU、Dice、Location、Shape、三专家选中数、三专家 IoU 和 Cycle 次数。
- [ ] 确认选中数之和为 46,515。
- [ ] 按 `(mean_IoU, mean_Dice)` 更新 Visual 候选排序，并保留 Fusion 尚未最终定点的说明。
