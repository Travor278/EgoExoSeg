# Label Studio LAN Direct Access Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Dell 上的 Label Studio 从单条 SSH 隧道切换为受源地址限制的局域网直连，降低图片并发加载的超时和队头阻塞。

**Architecture:** Label Studio 保持原 SQLite 数据库、媒体根和 8081 端口，只把监听地址从 `127.0.0.1` 攱为 `192.168.105.166`。Dell 的 INPUT 链在最前面允许 `192.168.105.67/32` 访问 TCP 8081，随后拒绝其他来源；直连验收通过后关闭本机 Plink 隧道。

**Tech Stack:** Label Studio 1.23、Linux iptables、OpenSSH、PowerShell/curl

## Global Constraints

- 不修改 Label Studio Project 2 的 task、prediction、annotation、draft 或 SQLite 内容。
- `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` 必须保持 `/home/dell/datasets/robointer_triview_1000`。
- Label Studio 数据目录必须保持 `/home/dell/datasets/robointer_triview_labelstudio/label-studio-data`。
- 仅允许 `192.168.105.67/32` 访问 Dell 的 TCP 8081；不得改变 SSH 22 或默认防火墙策略。
- 失败时恢复 `127.0.0.1:8081` 并保留本机 18082 隧道。
- 不创建 watchdog、计划任务或自启动脚本。

---

### Task 1: Preflight and failing baseline

**Files:**
- Read only: `/home/dell/datasets/robointer_triview_labelstudio/label-studio-data/label_studio.sqlite3`
- Read only: Dell process and firewall state

**Interfaces:**
- Consumes: Dell SSH endpoint `dell@192.168.105.166`
- Produces: exact current Label Studio PID, firewall baseline, Project 2 task count baseline

- [ ] **Step 1: Verify current process and data paths**

Run remote `ps`, `ss`, `free`, and `df`; require one Label Studio process using port 8081 and the expected data directory.

- [ ] **Step 2: Record the firewall baseline**

Run `sudo iptables -S INPUT` and save only the command output in the session transcript. Do not enable UFW or alter default policies.

- [ ] **Step 3: Verify the performance failure**

Run ten requests through `http://127.0.0.1:18082/projects/2/data?tab=1&task=1200` with a five-second timeout. Expected baseline: at least one timeout or multi-second response.

- [ ] **Step 4: Verify Project 2 baseline**

Use the existing Label Studio database/API environment to confirm Project 2 contains exactly 1200 tasks before mutation.

### Task 2: Restrict direct LAN access

**Files:**
- Modify runtime state only: Dell iptables INPUT chain

**Interfaces:**
- Consumes: source `192.168.105.67/32`, destination TCP 8081
- Produces: one source allow rule followed immediately by one deny-all rule for TCP 8081

- [ ] **Step 1: Add idempotent source allow rule**

Use `sudo iptables -C` before inserting an INPUT rule at position 1 that accepts TCP traffic from `192.168.105.67/32` to destination port 8081 and carries the comment `EgoExoSeg-LS-allow`.

- [ ] **Step 2: Add idempotent deny rule**

Use `sudo iptables -C` before inserting an INPUT rule immediately after the allow rule that rejects all other TCP traffic to destination port 8081 and carries the comment `EgoExoSeg-LS-deny`.

- [ ] **Step 3: Verify rule order without changing policy**

Run `sudo iptables -S INPUT`; require `EgoExoSeg-LS-allow` before `EgoExoSeg-LS-deny`, with no changes to policies or unrelated rules.

### Task 3: Rebind Label Studio with rollback

**Files:**
- Modify runtime process only: Label Studio PID
- Append: `/home/dell/datasets/robointer_triview_labelstudio/logs/labelstudio.log`
- Update: `/home/dell/datasets/robointer_triview_labelstudio/state/labelstudio.pid`

**Interfaces:**
- Consumes: existing SQLite data directory and local media root
- Produces: Label Studio listener at `192.168.105.166:8081`

- [ ] **Step 1: Stop only the verified Label Studio PID**

Send SIGTERM to the PID whose command line contains the expected data directory and `--port 8081`; wait for it to exit. Do not kill by broad process name.

- [ ] **Step 2: Start the LAN-bound process**

Start with `nohup` and these exact runtime values: local-files serving enabled, media root `/home/dell/datasets/robointer_triview_1000`, internal host `192.168.105.166`, port 8081, existing data directory, and legacy API token enabled.

- [ ] **Step 3: Verify remote health**

Poll `http://192.168.105.166:8081/projects/2/data?tab=1&task=1200` until it returns 200 or 302. Confirm the new PID and listener address.

- [ ] **Step 4: Roll back on any failure**

If Step 2 or Step 3 fails, stop only the new PID, restart the same command with internal host `127.0.0.1`, remove the two tagged iptables rules, verify the old 18082 route, and stop execution.

### Task 4: End-to-end verification and tunnel retirement

**Files:**
- Modify runtime process only: local Plink tunnel after direct verification

**Interfaces:**
- Consumes: `http://192.168.105.166:8081/projects/2/`
- Produces: verified direct browser endpoint with no SSH forwarding dependency

- [ ] **Step 1: Verify direct Windows access**

Run ten requests from Windows to `http://192.168.105.166:8081/projects/2/data?tab=1&task=1200`, each with a five-second timeout. Require ten responses with HTTP 200 or 302 and zero timeouts.

- [ ] **Step 2: Verify data identity**

Confirm Project 2 remains at exactly 1200 tasks and the database file path is unchanged.

- [ ] **Step 3: Stop only the verified 18082 Plink process**

After Steps 1–2 pass, stop the process whose command line contains both `127.0.0.1:18082:127.0.0.1:8081` and `dell@192.168.105.166`.

- [ ] **Step 4: Verify final state**

Require: direct endpoint healthy, no listener on local 18082, no watchdog task or script, Dell listener at `192.168.105.166:8081`, and ordered source-restriction rules present.
