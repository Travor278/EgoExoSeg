"""Real Exo2Ego image/mask scientific panels; never synthesized attention or geometry."""
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).parent;W=R/'walkthrough_exoego';F=R/'figures/exo2ego';F.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R.parent/'method_figure_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgb
from PIL import Image
from roi_views import crop_view
from science_views import intervene
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False,'figure.facecolor':'#f7f9fc','savefig.facecolor':'#f7f9fc'})
r=json.loads((W/'record.json').read_text());s=json.loads((W/'scores.json').read_text());sc=json.loads((W/'science.json').read_text());full=json.loads((W/'full_crosscheck.json').read_text());prov=json.loads((W/'provenance.json').read_text())
source=Image.open(W/'source.jpg').convert('RGB');target=Image.open(W/'target.jpg').convert('RGB')
z=np.load(W/'bank.npz')
def unpack(e):return np.unpackbits(z[e])[:int(np.prod(z[e+'_shape']))].reshape(z[e+'_shape']).astype(np.uint8)
sm=unpack('query');masks={e:unpack(e) for e in ('visual','anchor','fusion')};colors={'source':'#00b9ce','visual':'#9161dc','anchor':'#10a875','fusion':'#e55b4d','15':'#edab00','20':'#00b9ce'}
assert {e:hashlib.sha256(m.tobytes()).hexdigest() for e,m in masks.items()}==r['mask_sha256']
def mask_image(m,im):return np.asarray(Image.fromarray(m).resize(im.size,Image.Resampling.NEAREST))>0
def bbox(m,im):
    y,x=np.where(mask_image(m,im));return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
def zoom(b,im,pad=40):return [max(0,b[0]-pad),max(0,b[1]-pad),min(im.width,b[2]+pad),min(im.height,b[3]+pad)]
def panel(ax,im,title,m=None,color='#00b9ce',limits=None):
    ax.imshow(im);ax.set_title(title,pad=10,fontsize=12,color='#18243a');ax.axis('off')
    if m is not None:
        q=mask_image(m,im);overlay=np.zeros((*q.shape,4));overlay[q,:3]=to_rgb(color);overlay[q,3]=.30;ax.imshow(overlay);ax.contour(q,levels=[.5],colors=[color],linewidths=1.2)
    if limits:ax.set_xlim(limits[0],limits[2]);ax.set_ylim(limits[3],limits[1])
def box(ax,b,color,label):
    x,y,x1,y1=b;ax.add_patch(Rectangle((x,y),x1-x,y1-y,fill=False,edgecolor=color,lw=2));ax.text(x,y,label,color=color,fontsize=10,va='bottom',bbox=dict(facecolor='#142031',alpha=.8,edgecolor='none',pad=2))
def save(fig,n,caption):
    fig.text(.025,.018,caption,fontsize=10,color='#42516b');fig.savefig(F/n,dpi=155,bbox_inches='tight');plt.close(fig)
geom={}
for mode,scale in [('local15',1.5),('local20',2.)]:
    q=crop_view(source,sm,scale)
    for e,m in masks.items():
        t=crop_view(target,m,scale);g=r['roi_metadata']['geometry'][mode+'/'+e]
        assert list(q['box'])==g['source_box'] and list(t['box'])==g['target_box'];geom[mode+'/'+e]=g
sb=bbox(sm,source);tb=bbox(masks['visual'],target)
fig,axs=plt.subplots(2,2,figsize=(12,9),gridspec_kw={'height_ratios':[1.35,1]});fig.suptitle('Exo→Ego：外部视角中的小毛巾，到了第一视角的画面边缘',fontsize=17,y=1.01)
panel(axs[0,0],source,'源 cam01｜青色为输入提示 mask',sm);panel(axs[0,1],target,'目标 aria01_214-1｜输入不含目标 GT')
box(axs[0,0],zoom(sb,source,15),colors['source'],'源提示位置')
panel(axs[1,0],source,'源图局部放大｜只作阅读辅助',sm,limits=zoom(sb,source,65))
panel(axs[1,1],target,'目标右侧局部｜按 Visual 候选范围放大',limits=zoom(tb,target,40))
fig.subplots_adjust(hspace=.3)
save(fig,'01_case.png','真实 frame 990，目标类别来自标注 blue kitchen towel；下排仅为显示放大，不是额外模型输入。')

fig,axs=plt.subplots(1,3,figsize=(13,5.4));fig.suptitle('三个原始候选，位置与覆盖均不同；ROI 只改变最终选择',fontsize=17)
for ax,e in zip(axs,masks):
    panel(ax,target,f'{e.title()}｜离线 IoU {100*r["metrics"][e][0]:.2f}%',masks[e],colors[e])
    b=bbox(masks[e],target);bb=zoom(b,target,7);ax.add_patch(Rectangle((bb[0],bb[1]),bb[2]-bb[0],bb[3]-bb[1],fill=False,edgecolor=colors[e],lw=2))
    ax.text(bb[0],min(target.height-30,bb[3]+10),'预测范围',color=colors[e],va='top',fontsize=10,bbox=dict(facecolor='#142031',alpha=.8,edgecolor='none',pad=2))
    notes={'visual':'ROI 1.5× / 2× 采用此候选','anchor':'原 PCCS 与旧 cycle 选择','fusion':'有局部重叠，但 IoU 很低'}
    ax.text(.5,-.055,notes[e],transform=ax.transAxes,ha='center',fontsize=11,color=colors[e])
save(fig,'02_candidates.png','三图为同一目标全图，轮廓和矩形由真实预测生成。离线IoU不进入选择器。')

fig,axs=plt.subplots(2,3,figsize=(12,8));fig.suptitle('同一 ROI 规则面对更强的尺度差与目标图边界',fontsize=17,y=1.01)
for i,(im,m,label,e) in enumerate([(source,sm,'源提示','source'),(target,masks['visual'],'Visual 候选','visual')]):
    b=bbox(m,im);v15=crop_view(im,m,1.5);v20=crop_view(im,m,2.)
    panel(axs[i,0],im,label+'：原图坐标',m,colors[e],zoom(v20['box'],im,25));box(axs[i,0],v20['box'],colors['20'],'2×');box(axs[i,0],v15['box'],colors['15'],'1.5×')
    for j,(v,scale) in enumerate([(v15,1.5),(v20,2.)],1):
        panel(axs[i,j],v['image'],f'{label} {scale}×｜{v["image"].width}×{v["image"].height}px',v['mask'],colors[e]);axs[i,j].text(.5,-.09,'→ 768×768 → 48×48 token',transform=axs[i,j].transAxes,ha='center',fontsize=10)
fig.subplots_adjust(hspace=.3)
save(fig,'03_roi.png','源 bbox 长边45px，crop为68/90px；Visual长边131px，crop为197/262px。右边界饱和时整体平移保留前景。')

fig,axs=plt.subplots(1,3,figsize=(13,5.3),gridspec_kw={'width_ratios':[1,1.5,1]});v=crop_view(source,sm,2.);t=crop_view(target,masks['visual'],2.)
panel(axs[0],v['image'],'源 crop：前景覆盖率 b',v['mask']);panel(axs[2],t['image'],'目标 Visual crop：前景覆盖率 a',t['mask'],colors['visual']);axs[1].axis('off')
ev=r['evidence']['visual'];axs[1].text(.5,.83,'原生 DINOv3 重新编码',ha='center',fontsize=16,color='#18243a');axs[1].text(.5,.59,'源前景 → B → 目标候选\n目标候选 → A → 源前景\n源 → 目标候选 → 源',ha='center',va='center',fontsize=13,linespacing=1.8)
axs[1].text(.5,.24,'Visual 的真实 2× 统计\n'+ '\n'.join(f'{label} = {ev["local20_"+k]:.4f}' for label,k in [('forward','forward_mass'),('backward','backward_mass'),('cycle','cycle_mass'),('soft-cycle','soft_cycle')]),ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round,pad=.6',facecolor='#ece5f6',edgecolor='none'))
fig.suptitle('相同循环公式，处理从外部到第一视角的对应',fontsize=17)
save(fig,'04_cycle.png','箭头是公式路径示意，不是测量的关键点连线；没有生成或伪造注意力热图。')

fig,axs=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1.1,1.2]});x=np.arange(4)
for j,e in enumerate(masks):axs[0].bar(x+(j-1)*.25,[r['evidence'][e]['local20_'+k] for k in ('forward_mass','backward_mass','cycle_mass','soft_cycle')],.25,label=e.title(),color=colors[e])
axs[0].set_xticks(x,['forward','backward','cycle','soft-cycle']);axs[0].set_ylim(0,1);axs[0].legend(ncol=3,fontsize=9);axs[0].spines[['top','right']].set_visible(False);axs[0].set_title('三候选局部证据｜不是直接取最大值')
axs[1].axis('off');axs[1].text(.02,.93,'原 PCCS：Anchor → 原生 ROI：Visual',fontsize=16,color='#18243a')
axs[1].text(.02,.73,f"Visual：质量门通过；原全图 soft-cycle={ev['soft_cycle']:.5f}\nFusion：未通过挑战者质量门",fontsize=11,linespacing=1.9)
axs[1].text(.02,.43,'Visual 相对 Anchor 的冻结预测改善\n'+f"1.5×：{s['local15']['candidates']['visual']['predicted_gain']:.4f} > 0.05\n2×：   {s['local20']['candidates']['visual']['predicted_gain']:.4f} > 0.05",fontsize=13,linespacing=1.8)
axs[1].text(.02,.10,'两个尺度都选择现有 Visual mask\nGT 只用于事后计算 IoU 87.24%',fontsize=12,color=colors['visual'],linespacing=1.8)
fig.suptitle('附加证据纠正一次原 PCCS 选择，但不产生新 mask',fontsize=17)
save(fig,'05_decision.png','Fusion有某些较高的局部统计也不代表可选；准入、原始选择和校准器共同决定结果。')

fig,axs=plt.subplots(1,4,figsize=(13,4.7));fig.suptitle('同一个 Exo→Ego 案例的背景控制：所有干预仍选 Visual',fontsize=16,y=1.01)
for ax,arm,title in zip(axs[:3],('real','far','rolled'),('真实源背景','同图远处背景','半幅错位背景')):
    view,meta=intervene(v,source,sm,arm);panel(ax,view['image'],title,view['mask']);old=sc['metadata']['source_interventions']['local20/'+arm]
    for k in ('editable_fraction','changed_pixel_fraction','donor_foreground_fraction'):
        if k in meta:assert abs(meta[k]-old[k])<1e-12
    if 'donor_box' in meta:assert list(meta['donor_box'])==old['donor_box']
panel(axs[3],t['image'],'目标 crop 不变',t['mask'],colors['visual'])
save(fig,'06_context.png','保护源前景及8px halo；此例不能单独证明context因果。另一次seed2未修正，详见正文。')

out={'video_id':r['video_id'],'obj_id':r['obj_id'],'label':prov['object_metadata']['crop_caption'],'source_path':prov['source'],'target_path':prov['target'],'source_image_hw':[source.height,source.width],'target_image_hw':[target.height,target.width],'source_bbox':sb,'visual_bbox':tb,'baseline':r['baseline'],'metrics':r['metrics'],'mask_sha256':r['mask_sha256'],'calibrator_predictions':s,'evidence':r['evidence'],'geometry':geom,'science_choices':sc['choices'],'full_crosscheck':full,'selection_note':'Chosen after inspecting seed1 successes for illustration, not a representative or blind sample. seed2 fails to correct; all background controls retain Visual.','figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in F.glob('0*.png')}}
(F/'example_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print('EXOEGO_FIGURES_CREATED',len(out['figures']))
