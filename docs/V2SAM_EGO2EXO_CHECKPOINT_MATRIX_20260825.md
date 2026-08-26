# V2-SAM Ego2Exo checkpoint 发布矩阵

> 更新时间：2026-08-25<br>
> 方向：Ego2Exo<br>
> 选择原则：以完整测试集的 IoU 主指标选择最佳 checkpoint；不同聚合口径不混算。

## 1. 已发布 checkpoint

| 发布名 | 实现与训练合同 | 最佳点 | 逐对象逐帧实例 IoU | 逐对象逐帧实例 Dice | 作者 frame-level / pair-macro IoU | 作者 frame-level / pair-macro Dice | 选择结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Strict NewMatcher Fresh24 | `V2SAM_NEWMATCHER`；从头训练；AdamW `4e-5`；Linear warmup `0→1.2 epoch`，Cosine `1.2→24` | epoch 16 | **0.405197334183** | **0.462906654638** | **0.450898164298** | **0.509557001338** | 24-epoch 轨迹最终最佳；epoch 18–24 均未刷新 |
| Visual resumable from-scratch, best | public Visual；从头训练；AdamW `4e-5`；Linear warmup `0→1.2 epoch`，Cosine `1.2→24`；完整可续训状态 | `iter_27600` | **0.3683** | **0.4305** | — | — | 本次指定发布的 public Visual checkpoint；后续 `iter_34400` 和 `iter_41376` 均回落 |

## 2. 与 V²-SAM 论文 Ego2Exo 结果对比

论文表 5 报告 Ego2Exo mIoU。2026-08-25 根据作者展示的 frame-level 结果与同一 raw 的独立复算，确认作者高点对应“先在每个 frame/pair 内平均对象，再对 frame/pair 等权”的聚合；因此本表用 `0.450898` 与论文 Fusion `0.445` 对齐。公开仓库 README 的测试命令仍调用本仓库 `tools/test.py`，没有证据表明论文主表通过 Ego-Exo4D EvalAI submission evaluator 生成，所以不把 challenge evaluator 与论文 evaluator 混称为同一实现。

| 本次 checkpoint | 对应论文组件 | 当前最接近官方聚合的 IoU | 论文 Ego2Exo IoU | 暂算差值 | IoU 百分点差 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Visual resumable from-scratch `iter_27600` | B / Visual Expert | **0.3683**（本地 public `SegMetric`） | 0.362 | **+0.0063** | **+0.63** | 与公开测试入口同类，但作者生成论文表的精确 evaluator commit 未披露，结论保留该限制 |
| Strict NewMatcher Fresh24 epoch 16 | C / Fusion Expert | **0.450898164298**（作者 frame-level / pair-macro） | 0.445 | **+0.005898164298** | **+0.5898** | 与作者 frame-level 口径对齐并略高；`0.405197334183` 是另一种 per-object 聚合，不与本列混算 |

注意：NewMatcher 的 `0.405197334183 / 0.462906654638` 是 100,223 个 object–frame 实例等权的 **per-object** 聚合；`0.450898164298 / 0.509557001338` 是同一 raw 结果先在每个 frame/pair 内平均对象、再对 40,517 个 frame/pair 等权的 **作者 frame-level** 聚合。前者比后者低是样本权重不同，不是模型结果矛盾。与作者/论文 frame-level 数值比较时使用后者，per-object 作为完整对象审计值并列保留。

## 3. 相对作者公开 checkpoint 的补充对照

| 模型 | 本次最佳 IoU / Dice | 作者公开权重实测 IoU / Dice | 差值（IoU / Dice） |
| --- | ---: | ---: | ---: |
| Visual | **0.3683 / 0.4305** | 0.367035 / 0.428027 | **+0.001265 / +0.002473** |
| Fusion / NewMatcher-compatible | **0.450898 / 0.509557**（作者 frame-level / pair-macro） | 0.446970 / 0.509795（历史 stock） | **+0.003928 / -0.000238** |

## 4. 文件身份

| 发布文件建议名 | 远端来源 | 大小（bytes） | SHA256 |
| --- | --- | ---: | --- |
| `fusion_ego2exo_full.pth` | `runs/train-newmatcher-v12-fresh24-fromscratch-seed530358027-20260822-v1/best_checkpoints/epoch_16_frame_iou_0.450898_perobject_iou_0.405197_strict_resume.pth` | 2,192,404,525 | `7a75826ffb7f065e7db9c20322567eb6058c9697e079180f77cda9ad4097d5b3` |
| `vp_ego2exo_full.pth` | `runs/train-visual-fromscratch-24epoch-resumable-seed530358027-20260821-v1/retained_eval_candidates/iter_27600.pth` | 982,288,288 | `bc94b285f5421359702b2c42ef2aec83ffee6787b301dea5821628ccdca8d0c4` |

两份文件均包含 optimizer、scheduler、message hub 和各自需要的持久状态，已通过严格续训审计。

## 5. Hugging Face 发布

- 私密模型仓库：[`Travor278/V2-SAM`](https://huggingface.co/Travor278/V2-SAM)
- 发布 revision：`07ee27c5b227a6e7a0fdaa2100d2b617773a50a6`
- 仓库结构与 `wangzeze/V2-SAM@50fd5a9a7e67d3fdaadab1cd0726b82896f89e02` 完全一致，共 8 个顶层文件。
- 服务端复制保留 `.gitattributes`、DINOv3、`json.tar.gz` 和两个 Exo2Ego checkpoint；只更新 `README.md`、`fusion_ego2exo_full.pth`、`vp_ego2exo_full.pth`。
- Windows 端独立回读已核验私密性、精确文件集合、README 指标说明，以及全部 6 个 LFS 对象的大小和 SHA256。

## 6. 论文参考

- V²-SAM Table 5：B / Visual Expert 为 `0.362`，C / Fusion Expert 为 `0.445`（Ego2Exo IoU）。A+B+C / PCCS 尚未在本实验中完成可审计测评，因此不进入 checkpoint 对比矩阵。这里的 frame-level 对齐来自作者结果展示与同一 raw 的独立聚合复算；公开 V²-SAM 测试入口使用本地 `SegMetric`，没有调用官方 EvalAI submission evaluator。
- 论文：https://arxiv.org/abs/2511.20886
