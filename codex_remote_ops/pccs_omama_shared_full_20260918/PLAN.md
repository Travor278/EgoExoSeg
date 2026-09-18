# Exo→Ego新生成固定候选上的公平全量复验

原严格重建在rank3/849aaee0-866d-4b85-9775-39360e27fa90_6120/obj0遇到Fusion mask SHA不一致：其IoU由0.76996195变为0.73738414，Visual/Anchor完全相同，基线仍Anchor。首个差异前保存791个匹配对象；不能把这一前缀估计为全量差异比例。原任务已按协议停止，未放宽hash门槛。

本轮新建目录，放弃“原full1精确重建”目标，完成用户真正需要的同候选公平对照。样本仍46515对/109253对象/295takes，专家权重与原生ROI1.5/2模型、O-MaMa适配器/权重/0.05阈值全部冻结，无拟合。

- 基于原完整ROI worker原样推理，唯一新增数据路径是在同次test_step后保存三专家与source mask packbits、hash及文件SHA；实际原PCCS、ROI1.5、ROI2、旧cycle均由同次预测计算。
- O-MaMa独立进程读取该不可变bank，逐对象验证mask/hash，先复现历史见证；保留canonical/native_interp/两几何一致性参考。所有方法面对同组候选，不能混入旧full1的mask指标或路由。
- 小规模smoke只验证capture开关、零强度回退、实际metric和重放一致、历史参考见证以及bank完整性。全量generation完成释放节点后再提交reference阶段，不提前报告全量效果。
- 本轮标为新生成共同候选复验；将与原full1比较候选漂移数量和基线变化，但不把两轮差异当方法收益。主要成对比较ROI1.5 vs O-MaMa一致性，ROI2为预设次要比较。
- frame指标先pair内对象平均再pair等权；零IoU完整保留，10000次take bootstrap。来源代码与权重SHA记录；原始mask留远端，公开标量、代码与日志。

固定seed仍出现Fusion差异的具体底层原因未定位，不武断称只是浮点误差；本方案通过同次生成并持久化候选消除比较混杂，不需要证明不同运行可逐bit复现。

## 资源记录与更正

运行在开发区H100 CUDA12.8/183核组，4H10080GB、80CPU、900GB RAM、512GB共享内存，镜像ngc-pytorch25.02-cuda12.8及既有runtime_env。此前只查看下拉框首屏后误称CUDA13.2/183核组不再提供；完整搜索确认lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e仍存在。用户明确接受已启动任务继续用12.8，无需为此重启。不能将资源组标签直接当作实际PyTorch编译CUDA版本，实际环境保存在运行日志中。
