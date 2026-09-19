"""Keep two self-contained directional walkthroughs; preserve shared protocol and results."""
from pathlib import Path
import json,re,subprocess,sys
R=Path(__file__).parent;p=R/'METHOD.md';old=p.read_text(encoding='utf8')
cases={}
for tag in ('EXAMPLE','EXOEGO_EXAMPLE'):
    cases[tag]={k:v.strip() for k,v in re.findall(r'<!-- '+tag+r':([^>]+) -->\n(.*?)\n<!-- /'+tag+r' -->',old,re.S)}
    assert len(cases[tag])==10
marker='<!-- CURRENT_VALIDATION_RESULTS -->'
prefix=old.split(marker)[0]
if '<!-- COMMON_METHOD_START -->' in prefix:
    common=prefix.split('<!-- COMMON_METHOD_START -->\n',1)[1].split('\n<!-- COMMON_METHOD_END -->',1)[0]
else:
    common=prefix[prefix.index('## 1. 目前用的究竟是哪种上下文'):]
    for tag in cases:common=re.sub(r'\n<!-- '+tag+r':[^>]+ -->.*?<!-- /'+tag+r' -->\n','\n',common,flags=re.S)
    common=re.sub(r'^## (\d+)\. ',r'### A.\1 ',common,flags=re.M)
    common=re.sub(r'^### (\d+)\.(\d+) ',r'#### A.\1.\2 ',common,flags=re.M)
    common=re.sub(r'§10\.1', '附录B.1',common);common=common.replace('§10','附录B')
    common=re.sub(r'§([1-9])',r'附录A.\1',common)
    common=common.replace('完整数字见下节','完整数字见附录B').replace('见开头总览与附录B','见两节结果小结与附录B')
    common=re.sub(r'\n{3,}','\n\n',common).strip()
e1=json.loads((R/'figures/example_evidence.json').read_text());e2=json.loads((R/'figures/exo2ego/example_evidence.json').read_text())
parts=['# 在原PCCS上加入局部视图循环证据：Exo→Exo与Exo→Ego分节详解','',
'本文按两个方向分别完整讲解：**第一节以白碗为例说明Exo→Exo，第二节以厨房毛巾为例说明Exo→Ego**。每节从输入到最终选择独立展开，不必在两种案例之间来回切换。共用实现、权重与复算入口放在附录A，完整统计放在附录B。','',
'当前方法暂称 **PCCS + Local-view Cycle Evidence**：保留原Visual / Anchor / Fusion候选与PCCS回退，在原生DINOv3局部重编码上增加区域循环证据，再由源训练集拟合的小型校准器判断是否接受挑战者。1.5×是源域预选主方案，2×是预设次要对照；不加载O-MaMa网络作为本方法的一部分，也不能称完全免训练。','',
'截至2026-09-19，双seed Exo→Ego全量、同候选O-MaMa对照与Exo→Exo扩展时序验证均已完成。两个案例都在查看结果后选取，只解释机制，不代表随机样本、独立新场景或单例因果证据。','']
def add(s):parts.extend([s.strip(),''])
def case(tag,key,number):
    body=cases[tag][key]
    body=re.sub(r'^#{2,3} .+\n+','',body,count=1)
    body=body.replace('§10.1','附录B.1').replace('§10','附录B').replace('§3',f'§{number}.3')
    body=body.replace('读下面每章时都可以回到这张图','阅读本节时可以随时回到这张图')
    body=body.replace('以下每章把该**毛巾案例（图E1～E6）**与原**白碗Exo→Exo案例（图1～6）**并列。','本节完整跟踪**毛巾案例（图E1～E6）**；第一节已独立讲解**白碗Exo→Exo案例（图1～6）**。')
    body=body.replace('后续自动更新全量结果只更新下一节','后续自动更新全量结果只更新附录B')
    body=body.replace('前面列出的完整1094对/512对配对统计','附录B列出的完整1094对/512对配对统计')
    body=body.replace('将各图插入对应章节','生成该案例图文块，再由`reorganize_method.py`按方向整合成两节')
    body=body.replace('更新附录B的脚本保留两套案例块','更新统计的脚本保留两节案例块')
    add(f'<!-- {tag}:{key} -->\n{body}\n<!-- /{tag} -->')
def stats_table(e,field):
    ev=e[field];add('| 候选 | 2× forward | 2× backward | 2× cycle | 2× soft-cycle |\n|---|---:|---:|---:|---:|\n'+'\n'.join('| '+x.title()+' | '+' | '.join(f'{ev[x]["local20_"+k]:.4f}' for k in ('forward_mass','backward_mass','cycle_mass','soft_cycle'))+' |' for x in ('visual','anchor','fusion')))
def geometry_table(e,field):
    g=e[field];lines=['| 视图 | 1.5×方框 `[x0,y0,x1,y1]` | 2×方框 |','|---|---|---|']
    lines.append(f"| 源提示 | `{g['local15/visual']['source_box']}` | `{g['local20/visual']['source_box']}` |")
    for x in ('visual','anchor','fusion'):lines.append(f"| 目标{x.title()} | `{g['local15/'+x]['target_box']}` | `{g['local20/'+x]['target_box']}` |")
    add('\n'.join(lines));add('坐标均在各自原图上，右下边界不含；候选框由预测mask决定，不能使用目标GT框替代。')
formula='''```text
F：目标crop的归一化token；G：源crop的归一化token
a：目标候选在token网格的前景覆盖率；b：源提示前景覆盖率
A = row_softmax(F @ G.T / 0.07)       # 目标 → 源
B = row_softmax(G @ F.T / 0.07)       # 源 → 目标
forward  = b.T @ B @ a / sum(b)
backward = a.T @ A @ b / sum(a)
cycle    = b.T @ B @ diag(a) @ A @ b / sum(b)
soft_cycle = 2 * forward * backward / (forward + backward)
```

这里统计的是前景支持经过候选后回到源物体的程度。由于大mask容易承接更多对应质量，还输入`log(forward)-log(mean(a))`、`log(backward)-log(mean(b))`及`log(cycle)-log(mean(a)*mean(b))`，并加入前景加权特征余弦和有效倍率。背景参与整张crop的编码和对应竞争，但模型没有显式输出邻居物体类别。'''

add('## 第一节：Exo→Exo——白碗从一个外部视角到另一个外部视角')
add('### 1.1 输入与要解决的歧义');case('EXAMPLE','intro',1)
add('本例的源与目标都是外部相机，但位置不同；目标图同时出现相似碗，单凭外观容易混淆。模型只收到源图、源mask和目标图，目标GT只在预测结束后用于评估。由于没有单独Exo→Exo专家权重，这个方向使用用户确认的Exo→Ego Visual e19/Fusion e20迁移验证。')
add('### 1.2 原PCCS提供哪些候选，为什么需要再判断');case('EXAMPLE','ch2',1)
add('| 候选 | 离线IoU | 连通域数 | 新挑战者质量门 | 本例角色 |\n|---|---:|---:|---|---|\n| Visual | 71.32% | 7 | 不通过 | 不可挑战 |\n| Anchor | 96.74% | 1 | 通过 | 可挑战Fusion |\n| Fusion | 38.03% | 1 | 通过 | 原PCCS基线e0 |')
add('“质量合格”只说明面积/碎片等规则可接受，不等于语义身份正确或IoU最高。因此Fusion可以被原流程选中，而新增证据仍有机会支持Anchor。最终输出始终是三份原mask之一，不是混合轮廓或生成第四份mask。')
add('### 1.3 含背景的方形ROI怎样构造');case('EXAMPLE','ch1',1);case('EXAMPLE','ch3',1);geometry_table(e1,'roi_geometry')
add('一般规则为`side=min(图像宽,图像高,max(32,ceil(scale×bbox长边)))`；越界整体平移。方形crop缩放到768×768不会再改变该crop自身宽高比，mask仍保持不规则形状。1.5×放大物体更多，2×带入背景更多，因此倍率消融同时改变context范围与物体有效分辨率，不能只归因于背景。')
add('### 1.4 从两张crop计算双向支持与软循环');add(formula);case('EXAMPLE','ch4',1);stats_table(e1,'local_evidence')
add('Fusion的backward=0.9505看似很高，但forward=0.2898：偏小的候选可以匹配回源物体，却只承接少量源前景支持。Anchor在两方向都较高。Visual也有较高软循环，但质量门不合格，说明“循环量最大”和“最终可选”不是同一步。面积校正和其他原PCCS测量仍由校准器综合使用。')
add('### 1.5 冻结校准器怎样从Fusion改为Anchor');case('EXAMPLE','ch5',1)
add('| 挑战者 | 1.5×预测改善 | 2×预测改善 | 准入 | 实际处理 |\n|---|---:|---:|---|---|\n| Visual | 0.1831 | 0.0442 | 质量门不通过 | 两尺度均排除 |\n| Anchor | 0.2012 | 0.0949 | 通过 | 均超过0.05，采用Anchor |')
add('输入并非只有循环分数：包括挑战者、原Fusion各自的原PCCS测量与局部统计、二者差值及专家one-hot。Ridge100在TRAIN拟合并冻结。真实IoU改善约58.71个百分点只用于事后展示，不能与模型预测的0.2012/0.0949混为一谈；不满足准入或没有预测改善超过0.05的候选时保留e0。')
add('### 1.6 与O-MaMa的联系及本例的模型来源');case('EXAMPLE','ch6',1);case('EXAMPLE','ch7',1)
add('### 1.7 固定前景的背景干预能说明什么');case('EXAMPLE','ch8',1)
add('本例的前景-only与同容量全图匹配对照仍选Fusion，而局部有背景ROI选Anchor；但旧全图cycle也能选Anchor，且三种自然背景干预均不改变选择。这些事实要一起保留：本例支持解释“如何利用附加证据改选”，不足以证明只有正确背景语义才是成功原因。')
add('### 1.8 从单例回到Exo→Exo整体证据');add('原1094对是20个take人工筛选源mask构建集的全量，不是整个Ego-Exo4D。2×两seed增益为+3.3101/+3.2772点；新增6534个不重复时间配对来自19个已有takes，1.5×/2×分别+3.7706/+3.9260点，同候选O-MaMa为+2.6131点。扩展主1.5×对O-MaMa区间跨零；2×的描述性差区间高于零，但不能据此重选主方案，也不能把新配对称为新场景。');case('EXAMPLE','ch9',1)

add('## 第二节：Exo→Ego——厨房毛巾从外部视角到第一视角')
add('### 2.1 输入、目标位置与视角差异');case('EXOEGO_EXAMPLE','intro',2)
add('本例与白碗使用相同模型与附加证据流程，但不要求对应物体呈现相似几何：外部图提示在容器边缘附近很小，佩戴式相机目标在画面右边缘，存在不同投影和较暗的成像。图像看起来困难只是案例描述，不是对整个Exo→Ego方向增益差异的因果解释。')
add('### 2.2 原PCCS为何选错，以及三候选各覆盖哪里');case('EXOEGO_EXAMPLE','ch2',2)
add('| 候选 | 离线IoU | 连通域数 | 原全图soft-cycle | 本轮状态 |\n|---|---:|---:|---:|---|\n| Visual | 87.24% | 4 | 0.003516 | 质量合格，可挑战 |\n| Anchor | 0.00% | 5 | 0.000062 | 原PCCS基线e0 |\n| Fusion | 1.45% | 12 | 0.005974 | 质量门不通过 |')
add('Fusion全图soft-cycle比Visual高，但它仍不可挑战，因为准入同时检查质量和不同mask等条件。Anchor是原PCCS已经给出的基线，不能因为新增门不合格就从记录里删除；增强模块在原基线之上寻找合法挑战者。')
add('### 2.3 不同物体尺度与边界下的ROI');case('EXOEGO_EXAMPLE','ch1',2);case('EXOEGO_EXAMPLE','ch3',2);geometry_table(e2,'geometry')
add('源45px与目标131px的bbox长边分别决定窗口，随后都变成768×768输入。目标窗口右边界到704，并不意味着直接用GT裁图：Visual和Fusion框相近来自各自预测范围，Anchor的错误区域则在图像另一侧。小候选碎片也会影响bbox和可见context，这是实际方法的一个敏感性来源。')
add('### 2.4 相同循环公式在Exo→Ego上如何解释');add(formula);case('EXOEGO_EXAMPLE','ch4',2);stats_table(e2,'evidence')
add('这里所有循环量的绝对值都不能直接与白碗案例比较来判定哪个方向更好；它们是给定图像、候选面积与特征分布下的统计量。当前选择器没有采用人工挑选的单项最大值，也不把0.2378当成23.78%的真实匹配概率。')
add('### 2.5 从Anchor改为Visual的具体决策');case('EXOEGO_EXAMPLE','ch5',2)
add('| 挑战者 | 1.5×预测改善 | 2×预测改善 | 准入 | 实际处理 |\n|---|---:|---:|---|---|\n| Visual | 0.8121 | 0.7198 | 通过 | 均超过0.05，采用Visual |\n| Fusion | 0.3696 | 0.2880 | 质量门不通过 | 诊断分数再高也不参与选择 |')
add('Fusion这一行的预测值仅是对冻结校准器的离线诊断，实际准入先排除它。Visual的原全图soft-cycle超过固定0.001，质量合格且不同于基线，因此成为合法挑战者。结果从Anchor的IoU0到Visual的87.24%，改变的是选用哪一份原mask，候选生成网络、分割边界和原seed不被这一步修改。')
add('### 2.6 同候选O-MaMa对照与记录对齐');case('EXOEGO_EXAMPLE','ch6',2);case('EXOEGO_EXAMPLE','ch7',2)
add('### 2.7 背景干预、seed2反例与解释边界');case('EXOEGO_EXAMPLE','ch8',2)
add('这一案例的价值不只是展示从0到87.24%的成功，还展示三个边界：有较好候选不保证选择器选中；改变背景不一定改变最终决策；同一对象在另一seed可能失效。因此论文中应并列给出逐例与整体统计，不能用这只毛巾代替全量或独立测试。')
add('### 2.8 从单例回到Exo→Ego全量结果');add('46,515对/109,253对象/295takes的两seed中，1.5×主方案分别+2.4599/+2.4984点，2×次对照+2.5142/+2.5334点，四项增益95%区间均高于零。另一共同候选全量对照为原PCCS53.9195、1.5×56.3794、2×56.4334、O-MaMa56.6261；ROI主/次与O-MaMa分别相差−0.2467/−0.1926点，区间均跨零。能说点估计接近，不构成等效、非劣或优越性证明。原512子集基线约71，不能与全量53.9混为一组。');case('EXOEGO_EXAMPLE','ch9',2)
add('## 附录A：两节共用的实现、来源与复现协议')
add('本附录集中保留公式、训练/校准划分、权重指纹、原PCCS集成位置与总体科学性约束；两个案例没有分别训练两套方法。')
add('<!-- COMMON_METHOD_START -->\n'+common+'\n<!-- COMMON_METHOD_END -->')
p.write_text('\n'.join(parts).rstrip()+'\n\n'+marker+'\n',encoding='utf8')
subprocess.run([sys.executable,str(R/'update_method_results.py')],check=True)
print('METHOD_REORGANIZED_TWO_DIRECTIONS')
