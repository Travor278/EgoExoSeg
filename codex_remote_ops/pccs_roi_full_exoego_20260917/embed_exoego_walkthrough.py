"""Preserve the Exo2Exo case, add a parallel Exo2Ego case before generated results."""
from pathlib import Path
import re,json
R=Path(__file__).parent;p=R/'METHOD.md';s=p.read_text(encoding='utf8')
if '<!-- COMMON_METHOD_START -->' in s:
    import subprocess,sys
    subprocess.run([sys.executable,str(R/'reorganize_method.py')],check=True)
    print('Preserved the two-direction document layout')
    raise SystemExit(0)
s=re.sub(r'\n<!-- EXOEGO_EXAMPLE:[^>]+ -->.*?<!-- /EXOEGO_EXAMPLE -->\n','\n',s,flags=re.S)
e=json.loads((R/'figures/exo2ego/example_evidence.json').read_text());assert e['video_id']=='02def476-33eb-4895-a448-a4398d44e16b_990' and e['baseline']=='anchor'
def before(anchor,key,body):
    global s
    assert s.count(anchor)==1
    s=s.replace(anchor,f'<!-- EXOEGO_EXAMPLE:{key} -->\n{body.strip()}\n<!-- /EXOEGO_EXAMPLE -->\n\n'+anchor)
before('## 1. 目前用的究竟是哪种上下文','intro','''
## 并列的 Exo→Ego 真实例子：从外部相机找到第一视角中的厨房毛巾

![图E1：外部相机源提示与第一视角目标图](figures/exo2ego/01_case.png)

**图E1｜从cam01看见的局部，转换到佩戴式相机aria01_214-1。** take为`02def476-33eb-4895-a448-a4398d44e16b`，frame990，obj0，标注类别名为`blue kitchen towel`。左图青色是不规则的源提示mask；右图是输入目标图，没有给模型目标GT。左图尺寸960×540，右图704×704：源物体很小，目标对应区域靠近右边界，视角、尺度与形状呈现差异。下排放大仅为阅读辅助，尚不是§3的模型ROI。

以下每章把该**毛巾案例（图E1～E6）**与原**白碗Exo→Exo案例（图1～6）**并列。图片、三候选mask和局部证据来自已保存的Exo→Ego seed1记录；哈希已与全量seed1及最终共同候选O-MaMa复验逐对象对齐，校准器预测也已重新计算。

这是查看结果后选取的教学成功案例，**不代表随机样本或整体效果**。重要限制一并保留：seed1得到修正，但seed2该对象没有修正；前景-only、全图匹配校准对照也能在seed1选对，背景干预未改变其最终选择。因此它适合解释数据流，不能单独证明局部context的独立因果贡献或方法对所有随机种子都成功。
''')
before('## 2. 保留的原 PCCS 流程','ch1','''
**对照Exo→Ego毛巾。** 图E1中的源提示邻近金属容器和灶台；第一视角还改变了这些像素的相对布局及可见性。模块不会输出“容器”“灶台”的类别或关系，只把候选周围的真实像素随ROI一起重新编码。不能因肉眼觉得邻近容器有帮助，就写成模型显式理解了这种关系。
''')
before('## 3. 局部视图怎么构造','ch2','''
![图E2：Exo→Ego同一目标图上的三份原候选](figures/exo2ego/02_candidates.png)

**图E2｜原PCCS在这一次选择了Anchor。** 三图都显示完整目标图：Visual覆盖右侧目标区域，离线IoU为87.24%；Anchor落在图像左上方，IoU为0；Fusion有少量重叠但碎片较多，IoU仅1.45%。原记录的选择原因是`cycle_distance`：三路原hard votes都是0，Visual与Anchor的归一化回环距离分别约0.4207和0.3664；原流程选择Anchor。Fusion质量门未通过，原PCCS转入回退选择。

后面的ROI增强不移动这些轮廓，也不重新分割。它只在**这一次已经生成的候选集合**里判断是否应从Anchor改为Visual。Anchor虽然没有通过新增挑战者质量门，仍可以作为原流程给出的基线`e0`；“原基线”与“允许的新挑战者”是两个概念。
''')
before('## 4. 局部循环证据的计算','ch3','''
![图E3：Exo→Ego源提示与Visual候选的两种ROI](figures/exo2ego/03_roi.png)

**图E3｜相同规则，不要求两视角的原始物体大小相同。** 源提示bbox长边45px，1.5×/2×得到68×68和90×90 crop；Visual候选在目标原图上bbox长边131px，得到197×197和262×262 crop。它们分别缩放到768×768，以相同48×48 token网格形成可比较的局部特征。

源2×范围为`[466,252,556,342]`，目标Visual为`[442,298,704,560]`，右下边界不含。目标crop碰到704px右边界时整体平移回图内，仍保留完整候选；不会为保持物体居中而截掉前景。不是把源框坐标原样搬到目标，也不是对两图统一向外扩100px。每个目标候选有自己的窗口：错误的Anchor 2×窗口为`[84,0,332,248]`，因此它看到的是另一块局部图像。
''')
before('## 5. 如何与原PCCS一起决定最终候选','ch4','''
![图E4：Exo→Ego毛巾的局部循环路径和真实统计](figures/exo2ego/04_cycle.png)

**图E4｜公式不因Exo→Ego而改变。** 青色源前景产生覆盖率向量b，紫色Visual前景产生a；匹配矩阵在两张完整crop的token之间计算。Visual的2× `forward=0.2307`、`backward=0.2453`、`cycle=0.0706`、`soft_cycle=0.2378`；错误Anchor分别约0.0755、0.0319、0.0025、0.0449，局部往返支持更弱。

这里的数值比白碗案例低，不代表它必然无效，也不是跨样本通用的置信概率。最终还要结合面积校正、原PCCS测量和固定校准器，不能设置一个看完此例才决定的“cycle大于多少就正确”阈值。箭头仍为数学路径示意，没有虚构实际关键点或注意力热图。
''')
before('## 6. 与 O-MaMa 的借鉴关系','ch5','''
![图E5：Exo→Ego毛巾的准入与真实校准选择](figures/exo2ego/05_decision.png)

**图E5｜明确地把`e0=Anchor`变成Visual。** Visual有4个连通域、通过质量门，原全图soft-cycle约0.003516，大于预设0.001；源提示有效，mask与基线不同，因此可以挑战。Fusion有12个连通域，未通过质量门，不能参与最终挑战；不能仅看其某项特征或模型预测值。

从同一冻结Ridge模型重算，Visual相对Anchor的预测IoU改善为**1.5×：0.8121，2×：0.7198**，均超过0.05，因此两种尺度都使用原Visual mask，离线IoU从0变为87.24%。0.8121/0.7198是校准器预测的改善，不是匹配概率，也不是从GT直接读取的真实差；将该行GT指标清空后，策略输出不变。
''')
before('## 7. 最佳Exo→Ego权重与历史O-MaMa核对','ch6','''
**在这个毛巾案例中，原生增强和O-MaMa确实是两套不同的选择机制。** 最终共同候选对照已确认本例三mask与图E2完全相同：原生1.5×/2×选择Visual，O-MaMa canonical、native_interp及一致性参考均选择Fusion。本例Visual IoU87.24%，Fusion为1.45%。这是一个对ROI有利的事后选例，不能用它声称总体优于O-MaMa；全量O-MaMa IoU点估计反而略高，差值区间见§10.1。
''')
before('## 8. 已有结果与本次补充的科学性验证','ch7','''
**毛巾样例的来源链。** 图E1～E6采用最佳Visual e19/Fusion e20生成的seed1候选；本地保存bank解包后的SHA与原512子集记录、全量seed1、共同候选复验一致。源码审计同时核对两份Ridge文件SHA。seed2是另一组候选，不能把它的mask或局部统计拼到图E2/图E4里解释seed1。
''')
before('## 9. 复现入口及当前状态','ch8','''
![图E6：Exo→Ego毛巾的固定前景背景干预](figures/exo2ego/06_context.png)

**图E6｜和白碗相同的科学性对照。** 对源2× crop保留前景及8原图像素halo，只干预保护区域以外的背景；目标候选不变。三张源图来自原实验干预代码重建，供体坐标和改动比例已与保存记录核对。真实、远处供体和半幅错位背景下，两种ROI尺度都仍选Visual；前景-only和同容量全图匹配校准对照也选Visual。故此例展示了干预怎么做，不能把成功完全归因于正确邻域背景。

| 同一对象的检查 | 原PCCS | 原生1.5× / 2× | 解释 |
|---|---|---|---|
| seed1及共同候选复验 | Anchor，IoU 0 | Visual，IoU 87.24% | 图E1～E6展示这一轮 |
| seed2全量 | Anchor，IoU 0 | 仍为Anchor，IoU 0 | 该轮已有Visual候选IoU88.30%，但选择器未修正 |
| seed1源背景三种干预 | 固定同一候选 | 都为Visual | 不能单例证明context的因果贡献 |

seed2失败的具体触发原因未在此案例说明中逐项定位，不凭候选IoU猜测准入或分数。这个反例说明：**总体两seed增益相近，不等于同一个样本在两seed都成功。** 不删除失败行，不将此例当盲测证据；整体效果仍由完整统计与成对区间支撑。
''')
before('<!-- CURRENT_VALIDATION_RESULTS -->','ch9','''
### Exo→Ego示例的核验与图文复现

毛巾的标量证据见[figures/exo2ego/example_evidence.json](figures/exo2ego/example_evidence.json)：包含原/新选择、三候选IoU和SHA、ROI坐标、真实局部统计、两尺度校准预测、背景干预及seed2/同候选O-MaMa对照。`audit_exoego_walkthrough.py`核验原始mask与冻结校准器，`render_exoego_walkthrough.py`生成六张科学示意图，`embed_exoego_walkthrough.py`将各图插入对应章节。

该图文补充只读取已有结果、在本地重放校准与绘图，未新增GPU实验或调整模型。原始图片、标注和mask保存在私有`walkthrough_exoego/`，仅按本次请求发布派生解释图和标量证据；图中可见的放大/颜色叠加不进入模型。更新§10的脚本保留两套案例块，原Exo→Exo白碗图文不被覆盖。
''')
p.write_text(s,encoding='utf8');print('EXOEGO_CHAPTERS_EMBEDDED')
