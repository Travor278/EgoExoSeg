# SegMetric 评测覆盖率问题与修复

## 问题

`SegMetric.process()` 对**每个 object mask** 追加一条结果：

```python
for j in range(num_obj):
    ...
    self.results.append(dict(IoU=iou.item(), Dice=dice.item()))
```

但 `BaseMetric.evaluate(size)` 收到的 `size` 是 **image-pair 数量**（`len(dataset)`），
而 `mmengine/dist/dist.py` 的 `collect_results_cpu` 会执行：

```python
if world_size == 1:
    return result_part[:size]        # L37-38，单卡
...
ordered_results = ordered_results[:size]   # L84，多卡
```

Ego-Exo4D Relation Test 平均每个 pair 含 2.3~2.5 个 object，所以 object 总数
远多于 `size`，列表尾部被静默丢弃。

因为截断点落在 pair 边界附近，被保留的近似是**数据集的一段前缀**，而非每个 pair
里挑几个 object。也就是说：**聚合方式是对的（确实是 object level），但只统计了
测试集的前 40~44%。**

## 实测（fusion_*_full.pth + HF Ego-Exo4D-Relation-Test 全量）

| 方向 | 测试集 | 实际参与评测 | 被截断部分 IoU | 完整测试集 IoU |
|---|---|---|---|---|
| Ego2Exo | 40,517 pair / 100,223 mask | 前 16,213 pair (40.0%) → **44.65** | 37.24 | **40.24** |
| Exo2Ego | 46,515 pair / 109,253 mask | 前 20,412 pair (43.9%) → **49.04** | 44.41 | **46.38** |

未被评测的那部分数据系统性更难（Ego2Exo 差 7.4 点，Exo2Ego 差 4.6 点）。

### 与卡数无关

| 配置 | Exo2Ego IoU |
|---|---|
| 1 卡（本次） | 49.04 |
| 4 卡（另一位同学） | 49.05 |
| 8 卡（潘老师，`Iter(test) [5815/5815]`，46515÷8） | 49.03 |

`DefaultSampler` 不 shuffle 时 rank r 取 `[r::world_size]`，各 rank 结果交错重排后
截断，近似等于「每个 rank 各取前 43%」，整体仍是同一批数据。所以三种配置结果一致，
但都只覆盖了同一个前缀。

### 与推理后处理无关

`V2SAM.predict()` 中标注 `# debug` 的 `fill_holes_in_masks`，开关对比（各 2000 对）：

| 方向 | 开 | 关 |
|---|---|---|
| Ego2Exo | 44.04 | 43.83 |
| Exo2Ego | 48.42 | 48.42 |

影响 ≤0.21 点，可排除。

## 修复

核心思路：**让 `len(self.results)` 恰好等于 `size`，截断自然失效**——每个
image-pair 追加**一条**结果，条目内部装该 pair 的逐 object 数值列表。
不需要改动 mmengine 任何代码，也不改变 object level 的语义。

```python
def process(self, data_batch, data_samples):
    gt_masks_all = data_batch["data"].get("masks", None)
    for i in range(len(gt_masks_all)):
        gt_masks  = gt_masks_all[i]
        pred_masks = data_samples[i]["pred_masks"]

        ious, dices = [], []                      # ← 逐 object 收集
        for j in range(gt_masks.shape[0]):
            pred = pred_masks[j].float()
            gt   = gt_masks[j].float().to(pred.device)
            inter = (pred * gt).sum()
            union = (pred + gt - pred * gt).sum()
            ious.append((inter / (union + 1e-6)).item())
            dices.append(((2 * inter) / (pred.sum() + gt.sum() + 1e-6)).item())

        # 每个 image-pair 一条 -> len(results) == size -> [:size] 成为 no-op
        self.results.append(dict(ious=ious, dices=dices))

def compute_metrics(self, results):
    flat_iou = [v for r in results for v in r["ious"]]     # 展平回 object level
    mean_iou = sum(flat_iou) / len(flat_iou)
    ...
```

同样的模式可直接套用到 Location Score / Shape Accuracy：把每个指标的逐 object
值放进各自的 list，`compute_metrics` 里展平后求均值即可。

`seg_metric_full.py` 是可直接运行的完整参考实现，它同时输出三个数便于对照：

- `stock_IoU` —— 复现截断后的旧值（用于验证修复前后一致性，我们实测与原
  `SegMetric` 精确到小数点后四位一致）
- `perobject_IoU` —— 完整测试集的 object level 平均（**建议采用**）
- `perpair_IoU` —— 先 pair 内平均再对 pair 平均（供参考）

它还会打印 `n_pairs` / `n_objects` / 丢弃比例，可用来快速确认截断是否已消除：
修复生效时 `n_objects` 应显著大于 `n_pairs`，且 `stock` 与 `per-object` 出现差异。

## 待确认

论文 Table 1 的 Ego2Exo 44.5 / Exo2Ego 47.3 是在**完整测试集**上算的，还是也受此
截断影响？

- 若为完整测试集，则本次复现 40.24 / 46.38 仍有 −4.3 / −0.9 的缺口待解释；
- 若为截断后的值，则对应 44.65 / 49.04，Ego2Exo 吻合而 Exo2Ego 论文偏低 1.7 点。
