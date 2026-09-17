"""Idempotently insert one real-case walkthrough into the existing method chapters."""
from pathlib import Path
import re,json
R=Path(__file__).parent;p=R/'METHOD.md';s=p.read_text(encoding='utf8')
s=re.sub(r'\n<!-- EXAMPLE:[^>]+ -->.*?<!-- /EXAMPLE -->\n','\n',s,flags=re.S)
def insert_before(anchor,key,body):
    global s
    assert s.count(anchor)==1,anchor
    s=s.replace(anchor,f'<!-- EXAMPLE:{key} -->\n{body.strip()}\n<!-- /EXAMPLE -->\n\n'+anchor)
insert_before('## 1. 目前用的究竟是哪种上下文','intro','''
## 贯穿全文的真实例子：同一只白碗，两个视角

![图1：白碗真实跨视角输入](figures/01_case.png)

**图1｜本文一直跟踪左图青色提示圈定的那只白碗。** 源图是 `gp01/3570.jpg`，目标图是 `gp04/3570.jpg`；均来自 Ego-Exo4D take `280ab94b-ebd0-4245-85a1-4cacd670588f` 的同步帧3570。这是 **Exo→Exo** 的已完成seed1样例，不是正在运行的Exo→Ego全量结果。目标图中有两只外观相似的碗，读下面每章时都可以回到这张图：输入只告诉模型“源图的这只碗”，要求它在另一视角找到对应区域。

这个样例为解释方法而在查看结果后挑选，是一个成功案例；不能代表平均效果。尤其是此前冻结的全图cycle在本例也选择Anchor，所以单凭这张图不能证明新增ROI优于旧cycle。全量/双seed与消融结论仍以统计表为准。图中mask和数值来自真实候选记录；展示时将1024×1024目标mask最近邻映射到960×540原图，原始hash已核对。颜色只帮助读图，不输入模型。

阅读顺序：图1看输入 → 图2看原候选 → 图3看crop → 图4看循环量 → 图5看选择 → 图6看科学性对照。机器可读样例依据见[example_evidence.json](figures/example_evidence.json)。
''')
insert_before('## 2. 保留的原 PCCS 流程','ch1','''
**用这只白碗理解“上下文”。** 图1中物体本身只有几十像素宽，周围还有台面、番茄、餐具及另一只碗。我们不会把它们自动标成一张“邻居关系图”。下面图3只是把含这只碗的真实图像局部放大再编码，使物体及周围像素共同参与特征形成；远处背景是否有用，要看图6的独立干预，不能靠肉眼指定某个番茄是涨点原因。
''')
insert_before('## 3. 局部视图怎么构造','ch2','''
![图2：同一目标图上的三个真实候选](figures/02_candidates.png)

**图2｜先看原PCCS已经拿到了什么。** 紫色Visual覆盖了桌上的碗，也出现手中另一只碗的碎片，共7个连通域，未通过质量门；绿色Anchor较完整地覆盖桌上的目标碗；红色Fusion偏向碗内的一小部分。原PCCS本例选择Fusion（`e0`）。后面的局部模块不生成第四个mask，也不把Fusion边界“修好”，它只判断是否应改用现成的Anchor。

图中的离线IoU分别为71.32%、96.74%、38.03%。这些是实验后用GT计算的评估数，不是模型在选择时能看的分数。
''')
insert_before('## 4. 局部循环证据的计算','ch3','''
![图3：白碗源提示和Anchor候选的真实ROI范围](figures/03_roi.png)

**图3｜方框是图像窗口，碗的mask仍是不规则轮廓。** 本例源提示bbox长边46像素，因此1.5×方形crop为69×69，2×为92×92；Anchor在目标原图上的bbox长边47像素，对应71×71及94×94。两幅crop都放到同样的768×768输入，产生48×48个DINOv3 token。图中1.5×下的碗占画面更大，2×则留出更多背景，这也是为何不能把倍率差异只解释成context多少：物体的编码分辨率也变了。

源2×crop的原图范围是 `[708,370,800,462]`，目标Anchor为 `[276,304,370,398]`（x0,y0,x1,y1，右下边界不含）。每个候选都按自己的mask取框；Fusion的2×框是84×84，Visual因包含离散区域为220×220。不是先用GT选一块统一“正确ROI”再去比较候选，也不是把bbox向外固定加100像素。
''')
insert_before('## 5. 如何与原PCCS一起决定最终候选','ch4','''
![图4：同一个白碗的局部循环计算示意](figures/04_cycle.png)

**图4｜把公式中的a、b落到这两张crop上。** 青色源碗在源patch网格上形成覆盖率向量b，绿色目标Anchor在目标网格上形成a；A、B是在两张完整crop的token之间算出来的softmax对应。`bᵀ B a`会累计“源前景token的对应是否落到目标候选”的支持，往返项进一步要求经过目标候选再回源前景。

本例Anchor的2×局部 `forward=0.9307`、`backward=0.8400`、`cycle=0.8073`。Fusion的backward反而更高（0.9505），但forward只有0.2898：一个偏小的区域可以很容易回到源物体，却未必覆盖源物体在目标中的完整对应。因此不能只比较某一个方向的数值。图4中的箭头是数学路径示意；本次没有保存完整token亲和矩阵，未绘制或捏造注意力热图。
''')
insert_before('## 6. 与 O-MaMa 的借鉴关系','ch5','''
![图5：白碗样例的真实证据与冻结校准器决定](figures/05_decision.png)

**图5｜本例到底怎样从Fusion换到Anchor。** 三个候选原hard votes都为1，不能靠这项区分。Visual质量不合格，被挡在挑战者准入门外；Anchor质量合格，原全图soft-cycle为0.3962，大于0.001，且与Fusion不是同一个mask，因此进入校准。

冻结的1.5×Ridge预测Anchor相对原Fusion的改善为0.2012；2×Ridge预测为0.0949，二者均超过0.05，所以都采用原先已有的Anchor mask。这里0.0949是**预测改善**，不是本例真实IoU差0.5871；不能把校准器分数当准确概率，也不是看到离线IoU96.74%后才选择Anchor。注意Visual在1.5×下即使有较高的离线预测改善，也因质量门不通过而不能挑战。
''')
insert_before('## 7. 最佳Exo→Ego权重与历史O-MaMa核对','ch6','''
**回到同一只白碗看借鉴边界。** O-MaMa给我们的启发是“源碗附近的视觉环境可能帮助区分目标图中相似的碗”。这里实际实现的图3→图4→图5路径，仍使用原PCCS的DINOv3、往返一致性和原候选回退：放大白碗所在局部，算局部区域循环支持，再决定是否替换Fusion。没有在这一步加载O-MaMa的DINOv2、cross-attention或投影头。

白碗例子说明的是数据怎样经过模块，不能凭这一例宣称方法独创、context因果有效或已追平O-MaMa。相关工作引用和整体对照仍然需要保留。
''')
insert_before('## 8. 已有结果与本次补充的科学性验证','ch7','''
**这个样例的模型来源也与上表对应。** 虽然图1是两个外部视角，本研究没有单独Exo→Exo训练权重，因此沿用这组修正后的Exo→Ego专家生成图2候选。图中好坏差异不能与另一组作者权重的结果混比。图1的Exo→Exo成功例子更不能替代本次46515对Exo→Ego全量验证。
''')
insert_before('## 9. 复现入口及当前状态','ch8','''
![图6：同一个白碗的真实与干预背景](figures/06_context.png)

**图6｜不动白碗，只改它旁边较远的像素。** 四图依次为真实源2×crop、同图远处供体背景、半幅错位背景、完全不变的目标crop。源前景和8原图像素halo保持原RGB；图上的青色透明叠加仅是显示标记，不参与模型。图是用科学消融的同一代码从原图和mask重建的，不是生成式合成样图。

**本例是重要的反面提醒：三个背景条件下，1.5×和2×都仍选Anchor。** 因此不能说“换背景后这只碗立刻匹配失败”，更不能把图6用作单例因果证明。它展示的是如何干预；真实背景平均优于某些干预的证据，来自前面列出的完整1094对/512对配对统计。旧全图cycle在这只碗上也选Anchor，故本例同样不能证明ROI对旧cycle的增益。样例没有替代负对照或全量表。
''')
insert_before('<!-- CURRENT_VALIDATION_RESULTS -->','ch9','''
### 按同一例子核对复现

`figures/example_evidence.json` 保存这只白碗的ID、真实候选hash、ROI坐标、原始局部统计、校准器预测和各干预选择；`render_walkthrough.py`生成六张解释图。原图及压缩mask留在私有`walkthrough/`目录，展示图按本次文档请求单独导出；未把未保存的注意力、关键点或GT轮廓画成真实输出。

案例图跟随文档放在`figures/`相对路径下，在本地与GitHub中都可显示。后续自动更新全量结果只更新下一节，保留本案例图文说明。
''')
p.write_text(s.rstrip()+'\n',encoding='utf8');assert s.count('<!-- EXAMPLE:')==10;print('METHOD_CASE_WALKTHROUGH_EMBEDDED')
