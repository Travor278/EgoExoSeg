# PCCS fixed-pixel context experiment — 2026-09-17

用户要求：在保留 PCCS 的前提下测试 O-MaMa 的 bbox 四边各扩 100 像素；同时将当前冻结原生 cycle 方法验证于 Exo→Ego。

## 区域定义与实现边界

核对官方 O-MaMa `descriptors/get_descriptors.py:38–84,109,126`：bbox 每边扩 100，裁剪边界，池化整框（包含前景），不是 bbox 倍率，也不是纯背景外环。它在处理后图像坐标中定义像素。本实验借鉴区域构造，保留原 V2-SAM/PCCS 的候选、DINOv3 特征、反向对应、原始路由与回退；不加载 O-MaMa 网络。

预声明 12 个区域：原生 DINOv3 预处理坐标（高 768、宽保持长宽比并取 16 倍数）的 50/100/150 像素整框和纯背景环；1.5/2 倍整框和纯背景环；O-MaMa canonical 坐标 source=532×952、target=700×700 中的 100 像素整框和环，映射到原生图像的归一化坐标。后者是区域范围对齐，不能声称复现其 DINOv2 四分之一分辨率池化或学习头。整框用 mask 原分辨率 bbox→连续 token 坐标→精确面积占比；环减去同格点前景覆盖率。bbox 使用像素外边界 xmax+1/ymax+1；与 O-MaMa xmax/ymax 有最多一个像素的离散约定差异，明确记录。

各区域产生相同六项证据：有效性、上下文余弦、前景/上下文交叉余弦、前景对比、前景减半上下文的残差余弦、前景软对应强于区域泄漏的差值。直接复用原 PCCS 最后层 DINOv3 特征，不裁图重编码。纯背景环与整框分开比较；错误邻域通过源区域平移半张图作控制。

## 选择、验证与可重复性

- 384 对/48 个官方 TRAIN takes 拟合；128 对/16 个不相交 TRAIN takes 校准。沿用已保存的四折 takes 划分，禁止重新按平台排序划分。专家已见过 TRAIN；这些不是独立官方 validation。
- 15 组同维度对照：cycle 原生主干、历史 ring15/ring20、上述 12 组。Ridge alpha10/100，阈值 .03/.05，共 60 个源域配置。挑战候选准入沿用 cycle-dominance，避免同时更改准入与上下文范围。
- 要求 TRAIN take-OOF 和校准均正增益，再按校准均值冻结主方案。匹配对照均用主方案相同正则与阈值；报告全组，禁止根据目标结果重新选方案。多个目标对照区间为描述性，非多重检验校正。
- 先完成 Exo→Ego 既有 512 对/965 对象/64 takes holdout。它排除拟合/校准 takes，但历史已评估过，不冒称新的盲测，也不是 46515 对全量。当前冻结 cycle_ridge10_t0.03 (SHA bf29e873ed7c5a6e6ebd157b2859b1f1a018564202bd6883e1964cae088cd06c) 与新主方案均接入实际 PCCS metric 路由，再与离线策略逐对象核对。
- 然后 Exo→Exo 全量 1094 对，同样权重和 per-pair seed。历史候选 hash、原始 PCCS 和 O-MaMa 比较值一致才直接复用参考；漂移则停止，不能比较不同候选。
- 所有源域原生 scalar evidence、候选 hash、路由对照上一轮完整记录；cycle 控制要求 source OOF/calibration 增益精确重现。smoke 比较 Capture 开/关输出、strength0 原样回退。模型选择不使用测试标签，标签仅用于最终指标。
- 指标保留零值，先对象均值再 pair 等权均值；95% take-cluster bootstrap 10000 次。纠正后的 Exo2Ego visual e19/fusion e20 权重沿用已审计配置。
- 监督间隔 = 当前阶段剩余 ETA × 4/5，四舍五入至分钟、最少 1 分钟；阶段转换/ETA 未知 5 分钟。完成后释放节点。

ROI 裁图重编码是另一机制，现暂缓，优先完成此次用户明确要求的固定像素区域池化。
