"""Scientific figures from one verified real case; no synthesized model maps.

Inputs stay in private walkthrough/. Only derived explanatory figures, this
script and scalar example evidence are intended for the method document.
"""
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).parent
sys.path.insert(0,str(R.parent/'method_figure_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,ConnectionPatch
from matplotlib.font_manager import FontProperties
from PIL import Image
sys.path.insert(0,str(R))
from roi_views import crop_view
from science_views import intervene

F=R/'figures';F.mkdir(exist_ok=True);W=R/'walkthrough'
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False,'figure.facecolor':'#f7f9fc','savefig.facecolor':'#f7f9fc'})
row=json.loads((W/'record.json').read_text());dec=json.loads((W/'decisions.json').read_text());scores=json.loads((W/'scores.json').read_text());science=json.loads((W/'science.json').read_text())
source=Image.open(W/'source.jpg').convert('RGB');target=Image.open(W/'target.jpg').convert('RGB')
z=np.load(W/'bank.npz')
def unpack(e):return np.unpackbits(z[e])[:int(np.prod(z[e+'_shape']))].reshape(z[e+'_shape']).astype(np.uint8)
sm=unpack('query');masks={e:unpack(e) for e in ('visual','anchor','fusion')}
assert {e:hashlib.sha256(v.tobytes()).hexdigest() for e,v in masks.items()}==row['mask_sha256']
for mode,scale in (('local15',1.5),('local20',2.)):
    for e,mask in masks.items():
        assert list(crop_view(source,sm,scale)['box'])==row['roi_metadata']['geometry'][mode+'/'+e]['source_box']
        assert list(crop_view(target,mask,scale)['box'])==row['roi_metadata']['geometry'][mode+'/'+e]['target_box']
colors={'source':'#00c7e8','visual':'#b677ff','anchor':'#16b879','fusion':'#ef6355','15':'#ffb000','20':'#00c7e8'}
def mask_image(mask,im):return np.asarray(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST))>0
def panel(ax,im,title,mask=None,color='#00c7e8',zoom=None):
    ax.imshow(im);ax.set_title(title,fontproperties=font,pad=9,fontsize=12,color='#18243a');ax.axis('off')
    if mask is not None:
        m=mask_image(mask,im);overlay=np.zeros((*m.shape,4));from matplotlib.colors import to_rgb
        overlay[m,:3]=to_rgb(color);overlay[m,3]=.32;ax.imshow(overlay);ax.contour(m,levels=[.5],colors=[color],linewidths=1)
    if zoom:ax.set_xlim(zoom[0],zoom[2]);ax.set_ylim(zoom[3],zoom[1])
def box(ax,b,color,label):
    x,y,x1,y1=b;ax.add_patch(Rectangle((x,y),x1-x,y1-y,fill=False,edgecolor=color,lw=2));ax.text(x,y-4,label,color=color,fontsize=10,fontproperties=font,weight='bold',bbox=dict(facecolor='#152030',alpha=.8,edgecolor='none',pad=2))
def save(fig,name,caption):
    fig.text(.025,.025,caption,fontproperties=font,fontsize=10,color='#42516b');fig.savefig(F/name,dpi=155,bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.45));fig.suptitle('同一个真实例子：从 gp01 的白碗，找到 gp04 中的同一只碗',fontproperties=font,fontsize=17,y=1.01)
panel(axs[0],source,'源视角 gp01｜青色：输入的提示 mask',sm,colors['source']);panel(axs[1],target,'目标视角 gp04｜输入时没有目标 GT')
axs[0].annotate('本文跟踪这只碗',xy=(756,416),xytext=(470,495),arrowprops=dict(color=colors['source'],arrowstyle='->',lw=2),color='white',fontproperties=font,bbox=dict(facecolor='#18243a',edgecolor='none'))
save(fig,'01_case.png','Ego-Exo4D，frame 3570，同步 Exo→Exo 样例；为解释流程选取的成功案例，不代表总体增益。')

fig,axs=plt.subplots(1,3,figsize=(12,4.5));fig.suptitle('原 V2-SAM 已经给出的三个候选：ROI 模块不修改任何一个 mask',fontproperties=font,fontsize=16,y=1.01)
for ax,e in zip(axs,('visual','anchor','fusion')):
    panel(ax,target,f'{e.title()}｜离线 IoU {row["metrics"][e][0]*100:.2f}%',masks[e],colors[e],(235,245,475,425))
    note={'visual':'质量门不通过：7 个连通域','anchor':'1.5× / 2× 最终均选择此候选','fusion':'原 PCCS 的选择 e0'}[e]
    ax.text(.5,-.10,note,transform=ax.transAxes,ha='center',fontproperties=font,fontsize=11,color=colors[e])
save(fig,'02_candidates.png','三图使用相同目标图、相同观察窗口；彩色轮廓是真实预测。IoU只作离线评估，不进入选择。')

fig,axs=plt.subplots(2,3,figsize=(12,8));fig.suptitle('局部 ROI：先按 bbox 长边构造方形，再放大编码；不是圆形 mask',fontproperties=font,fontsize=16,y=1.01)
for i,(im,m,label) in enumerate(((source,sm,'源提示'),(target,masks['anchor'],'Anchor 候选'))):
    v15=crop_view(im,m,1.5);v20=crop_view(im,m,2.)
    y,x=np.where(mask_image(m,im));bbox=(int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1))
    zoom=(bbox[0]-90,bbox[1]-80,bbox[2]+90,bbox[3]+80)
    panel(axs[i,0],im,label+'：原图坐标中的范围',m,colors['source'] if i==0 else colors['anchor'],zoom)
    box(axs[i,0],bbox,'#eeeeee','bbox');box(axs[i,0],v20['box'],colors['20'],'2×');box(axs[i,0],v15['box'],colors['15'],'1.5×')
    for j,v in enumerate((v15,v20),1):
        panel(axs[i,j],v['image'],f'{label} {1.5 if j==1 else 2}×｜{v["image"].width}×{v["image"].height} px',v['mask'],colors['source'] if i==0 else colors['anchor'])
        axs[i,j].text(.5,-.08,'→ 768×768 输入 → 48×48 token',transform=axs[i,j].transAxes,ha='center',fontproperties=font,fontsize=10)
fig.subplots_adjust(hspace=.3)
save(fig,'03_roi.png','源 bbox 长边46px：1.5×=69px，2×=92px；Anchor bbox 长边47px：实际crop为71px / 94px。')

fig=plt.figure(figsize=(12,5.1));grid=fig.add_gridspec(1,3,width_ratios=[1,1.5,1]);a=fig.add_subplot(grid[0]);mid=fig.add_subplot(grid[1]);b=fig.add_subplot(grid[2])
sv=crop_view(source,sm,2.);tv=crop_view(target,masks['anchor'],2.)
panel(a,sv['image'],'源 crop：前景覆盖率 b',sv['mask'],colors['source']);panel(b,tv['image'],'目标 Anchor crop：前景覆盖率 a',tv['mask'],colors['anchor']);mid.axis('off')
mid.text(.5,.86,'同一个原生 DINOv3',ha='center',fontproperties=font,fontsize=16,color='#15283f')
mid.text(.5,.65,'源前景  → B →  目标候选\n目标候选 → A →  源前景\n源前景 → 目标候选 → 源前景',ha='center',va='center',fontproperties=font,fontsize=13,linespacing=1.9)
mid.text(.5,.22,'这个样例的真实统计（Anchor）\nforward = 0.9307\nbackward = 0.8400\ncycle = 0.8073',ha='center',va='center',fontproperties=font,fontsize=12,bbox=dict(boxstyle='round,pad=.6',facecolor='#e5f4ef',edgecolor='none'))
fig.suptitle('循环一致性：对应质量是否经过目标候选，再回到源物体？',fontproperties=font,fontsize=17)
save(fig,'04_cycle.png','箭头仅示意计算方向，不是实测关键点连线；图中未绘制未保存的相似度矩阵或注意力热图。')

fig,axs=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.45,1]});labels=['forward','backward','cycle','soft_cycle'];x=np.arange(4);width=.24
for j,e in enumerate(('visual','anchor','fusion')):
    v=[row['evidence'][e]['local20_'+k] for k in ('forward_mass','backward_mass','cycle_mass','soft_cycle')];axs[0].bar(x+(j-1)*width,v,width,color=colors[e],label=e.title())
axs[0].set_xticks(x,labels);axs[0].set_ylim(0,1.13);axs[0].legend(ncol=3,loc='upper left');axs[0].spines[['top','right']].set_visible(False);axs[0].set_title('三候选的2×局部证据（真实记录）',fontproperties=font)
axs[1].axis('off');axs[1].text(.02,.9,'原 PCCS：Fusion',fontproperties=font,fontsize=15,color=colors['fusion'])
axs[1].text(.02,.68,'Visual：质量门不通过，不能挑战\nAnchor：质量合格，原 soft-cycle > 0.001',fontproperties=font,fontsize=11,linespacing=1.8)
axs[1].text(.02,.41,'冻结 Ridge100 预测 Anchor 相对 Fusion\n1.5×：ΔIoU = 0.2012 > 0.05\n2×：   ΔIoU = 0.0949 > 0.05',fontproperties=font,fontsize=12,linespacing=1.8)
axs[1].text(.02,.12,'最终：采用现有 Anchor mask',fontproperties=font,fontsize=15,color=colors['anchor'])
fig.suptitle('原 PCCS 先给 e0，局部证据再判断是否接受挑战者',fontproperties=font,fontsize=17)
save(fig,'05_decision.png','预测改善是冻结校准器的输出，不是实际GT差值；选择不是简单取某一个循环统计的最大值。')

fig,axs=plt.subplots(1,4,figsize=(12,4.1));fig.suptitle('同一个白碗：固定前景和近边界，只干预源背景',fontproperties=font,fontsize=16,y=1.02)
for ax,arm,title in zip(axs[:3],('real','far','rolled'),('真实背景','同图远处背景供体','半幅错位背景')):
    v,meta=intervene(sv,source,sm,arm);panel(ax,v['image'],title,sv['mask'],colors['source'])
    saved=science['metadata']['source_interventions']['local20/'+arm]
    for k in ('editable_fraction','changed_pixel_fraction','donor_foreground_fraction'):
        if k in meta:assert abs(meta[k]-saved[k])<1e-12,(arm,k)
    if 'donor_box' in meta:assert list(meta['donor_box'])==saved['donor_box']
panel(axs[3],tv['image'],'目标 crop 完全不变',tv['mask'],colors['anchor'])
save(fig,'06_context.png','前景及8原图像素halo保持；这些是按真实实验代码重建的干预图。本例各干预仍选Anchor，不能拿它证明干预有增益。')

evidence={'video_id':row['video_id'],'source_path':json.loads((W/'provenance.json').read_text())['source'],'target_path':json.loads((W/'provenance.json').read_text())['target'],'selection_rationale':'Illustrative success case with similar bowls, chosen after inspecting outcomes; not representative or independent validation. Earlier frozen full-image cycle also selects anchor.','baseline':row['baseline'],'decisions':dec['choices'],'metrics':row['metrics'],'calibrator_predictions':scores,'local_evidence':{e:row['evidence'][e] for e in masks},'source_mask_shape':list(sm.shape),'candidate_mask_shapes':{e:list(v.shape) for e,v in masks.items()},'mask_sha256':row['mask_sha256'],'roi_geometry':row['roi_metadata']['geometry'],'science_decisions':science['choices'],'figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in F.glob('0*.png')},'notes':['Figures are scientific plots over observed images/masks, not generated images.','Masks nearest-resized for display; original 1024x1024 candidate hashes verified.','No measured affinity/attention maps available; cycle arrows illustrative.','Raw source images and packed masks remain local/private and are not published.']}
(F/'example_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8');print('WALKTHROUGH_FIGURES_CREATED',len(evidence['figures']))
