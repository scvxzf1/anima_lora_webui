"""CPU-only static Anima adapter analysis. Writes only beside this script."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).resolve().parent / 'mpl-cache')
import csv,json,math,re,struct,time,collections
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
OUT=Path(__file__).resolve().parent
SRC=OUT.parent
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
START=time.time()
FIELDS=['file','module','canonical','block','scope','group','component','type','shape','rank','alpha','alpha_source','scale','fro','rms','energy','energy_share','dense_params','adapter_params','status']
FILEFIELDS=['file','size_bytes','type','family','family_evidence','tensor_count','module_count','reconstructed','unsupported','adapter_params','scalar_alpha_count','rank_min','rank_max','alpha_min','alpha_max','scale_min','scale_max','energy','global_fro','delta_rms','dense_params','nonfinite_tensors','unused_tensors','seconds']
METAKEYS=['ss_base_model_version','modelspec.architecture','ss_network_module','ss_network_dim','ss_network_alpha','ss_steps','ss_epoch','ss_num_epochs','ss_learning_rate','ss_mixed_precision','modelspec.resolution','modelspec.date','ss_optimizer','ss_lr_scheduler','training_info']

def header(p):
    with p.open('rb') as h:return json.loads(h.read(struct.unpack('<Q',h.read(8))[0]))

def canonical(s):
    s=re.sub(r'^(diffusion_model\.|lora_unet_)','',s)
    return s.replace('.','_')

def classify(p):
    p=canonical(p)
    m=re.match(r'blocks_(\d+)_(.*)',p)
    if m:block=int(m[1]);tail=m[2];scope='dit'
    else:block='';tail=p;scope='llm_adapter' if 'llm_adapter' in p else 'nonblock'
    if 'adaln' in tail or 'modulation' in tail:group='AdaLN/modulation';component=tail
    elif 'self_attn' in tail:group='self-attn';component=re.sub(r'.*self_attn_','',tail).replace('output_proj','out').replace('o_proj','out').replace('_proj','')
    elif 'cross_attn' in tail:group='cross-attn';component=re.sub(r'.*cross_attn_','',tail).replace('output_proj','out').replace('o_proj','out').replace('_proj','')
    elif 'mlp' in tail:group='MLP';component=tail
    else:group='other';component=tail
    return p,block,scope,group,component

def energy(B,A):
    # float32 Gram matrices; float64 final reduction. Avoid dense B @ A.
    return float(((B.T@B).double()*(A@A.T).double().T).sum().item())

def write_csv(path,fields,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def safe_meta(m):
    out={k:m[k] for k in METAKEYS if k in m}
    try:
        args=json.loads(m.get('ss_network_args','{}'));out['network_args_safe']={k:v for k,v in args.items() if k in ['algo','factor','conv_dim','conv_alpha','module_dropout','dropout','rank_dropout','use_dora','rs_lora','use_rslora']}
    except Exception:pass
    return out

allrows=[];files=[];details=[]
with (OUT/'progress.log').open('w',encoding='utf-8') as progress:
 for path in sorted(SRC.glob('*.safetensors')):
    t0=time.time();h=header(path);meta=h.pop('__metadata__',{})
    groups=collections.defaultdict(dict)
    suffixes=['.lora_down.weight','.lora_up.weight','.lora_A.weight','.lora_B.weight','.alpha','.inv_scale','.lokr_w1','.lokr_w2','.lokr_w2_a','.lokr_w2_b']
    unknown=[]
    for k in h:
        suffix=next((s for s in suffixes if k.endswith(s)),None)
        if suffix:groups[k[:-len(suffix)]][suffix]=k
        else:unknown.append(k)
    rows=[];used=set();nonfinite=[];issues=[];types=set();alphas=[];ranks=[];scales=[]
    with safe_open(path,framework='pt',device='cpu') as sf:
        # Every tensor is inspected for finite values, including unsupported extras.
        for k in h:
            v=sf.get_tensor(k)
            if not torch.isfinite(v).all().item():nonfinite.append(k)
        for p,keys in groups.items():
            canon,block,scope,group,comp=classify(p)
            row=dict(file=path.name,module=p,canonical=canon,block=block,scope=scope,group=group,component=comp,type='',shape='',rank='',alpha='',alpha_source='',scale='',fro='',rms='',energy='',energy_share='',dense_params=0,adapter_params=sum(math.prod(h[k]['shape']) for s,k in keys.items() if s!='.alpha'),status='')
            try:
                if any(k in nonfinite for k in keys.values()):raise ValueError('nonfinite tensor')
                if '.lokr_w1' in keys:
                    typ='LoKr';w1=sf.get_tensor(keys['.lokr_w1']).float()
                    if '.lokr_w2_a' not in keys or '.lokr_w2_b' not in keys:raise ValueError('unsupported LoKr factor layout')
                    B=sf.get_tensor(keys['.lokr_w2_a']).float();A=sf.get_tensor(keys['.lokr_w2_b']).float();rank=A.shape[0]
                    if w1.ndim!=2 or A.ndim!=2 or B.ndim!=2 or B.shape[1]!=rank:raise ValueError('incompatible LoKr dimensions')
                    shape=(w1.shape[0]*B.shape[0],w1.shape[1]*A.shape[1]);e=energy(B,A)*float(w1.double().square().sum())
                else:
                    typ='LoRA';ak='.lora_down.weight' if '.lora_down.weight' in keys else '.lora_A.weight';bk='.lora_up.weight' if '.lora_up.weight' in keys else '.lora_B.weight'
                    if ak not in keys or bk not in keys:raise ValueError('missing LoRA A/B pair')
                    A=sf.get_tensor(keys[ak]).float();B=sf.get_tensor(keys[bk]).float()
                    if A.ndim!=2 or B.ndim!=2 or B.shape[1]!=A.shape[0]:raise ValueError('unsupported dimensions')
                    rank=A.shape[0];shape=(B.shape[0],A.shape[1])
                    if '.inv_scale' in keys:
                        inv=sf.get_tensor(keys['.inv_scale']).float()
                        if inv.ndim!=1 or inv.numel()!=A.shape[1]:raise ValueError('incompatible inv_scale')
                        A=A*inv.unsqueeze(0)
                    e=energy(B,A)
                if '.alpha' in keys:alpha=float(sf.get_tensor(keys['.alpha']).item());source='tensor'
                elif 'ss_network_alpha' in meta:alpha=float(meta['ss_network_alpha']);source='metadata ss_network_alpha'
                else:alpha=float(rank);source='fallback rank (assumed)'
                if alpha==0:alpha=float(rank);source+='; zero treated as rank'
                scale=alpha/rank;e*=scale**2
                if not math.isfinite(e) or e<-1e-8:raise ValueError('invalid Gram energy')
                e=max(e,0);n=math.prod(shape)
                row.update(type=typ,shape='x'.join(map(str,shape)),rank=rank,alpha=alpha,alpha_source=source,scale=scale,fro=math.sqrt(e),rms=math.sqrt(e/n),energy=e,dense_params=n,status='ok')
                used.update(keys.values());ranks.append(rank);alphas.append(alpha);scales.append(scale);types.add(typ)
            except Exception as exc:
                row.update(type='unsupported',status=str(exc));issues.append({'module':p,'reason':str(exc)})
            rows.append(row)
    total=sum(r['energy'] for r in rows if r['status']=='ok');dense=sum(r['dense_params'] for r in rows if r['status']=='ok')
    for r in rows:
        if r['status']=='ok':r['energy_share']=r['energy']/total if total else 0
    unused=sorted(set(h)-used)
    evidence='metadata' if meta.get('ss_base_model_version')=='anima' or 'anima' in meta.get('modelspec.architecture','').lower() else 'inferred: 28 blocks, 2048 hidden, Anima AdaLN/cross-attn naming; metadata absent'
    f=dict(file=path.name,size_bytes=path.stat().st_size,type='+'.join(sorted(types)),family='Anima',family_evidence=evidence,tensor_count=len(h),module_count=len(groups),reconstructed=sum(r['status']=='ok' for r in rows),unsupported=len(issues),adapter_params=sum(math.prod(v['shape']) for k,v in h.items() if not k.endswith('.alpha')),scalar_alpha_count=sum(k.endswith('.alpha') for k in h),rank_min=min(ranks) if ranks else '',rank_max=max(ranks) if ranks else '',alpha_min=min(alphas) if alphas else '',alpha_max=max(alphas) if alphas else '',scale_min=min(scales) if scales else '',scale_max=max(scales) if scales else '',energy=total,global_fro=math.sqrt(total),delta_rms=math.sqrt(total/dense) if dense else 0,dense_params=dense,nonfinite_tensors=len(nonfinite),unused_tensors=len(unused),seconds=time.time()-t0)
    d=dict(summary=f,metadata=safe_meta(meta),rank_distribution=dict(collections.Counter(ranks)),alpha_distribution=dict(collections.Counter(alphas)),scale_distribution=dict(collections.Counter(scales)),nonfinite_keys=nonfinite,unused_keys=unused,issues=issues)
    details.append(d);files.append(f);allrows.extend(rows)
    # Save progress after every file; no need to wait for final aggregation.
    write_csv(OUT/'per_file.csv',FILEFIELDS,files);write_csv(OUT/'per_layer.csv',FIELDS,allrows)
    (OUT/'partial.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    msg=f'{path.name}: {f["reconstructed"]}/{len(groups)} modules, energy={total:.7g}, Fro={f["global_fro"]:.7g}, nonfinite={len(nonfinite)}, unused={len(unused)}, {f["seconds"]:.1f}s'
    print(msg,flush=True);progress.write(msg+'\n');progress.flush()

GROUPS=['self-attn','cross-attn','MLP','AdaLN/modulation','other']
# All files comparable as static deltas. Keep conventional LoRA subset and main-block/common-target comparisons to expose coverage.
subsets={'all':[f['file'] for f in files],'plain_lora':[f['file'] for f in files if f['type']=='LoRA'],'plain_lora_main_blocks':[f['file'] for f in files if f['type']=='LoRA']}
aggregate={}

def statistics(v):
    a=np.array(v,dtype=float)
    return {'mean':float(np.mean(a)),'median':float(np.median(a)),'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75)),'min':float(a.min()),'max':float(a.max())}

for name,names in subsets.items():
    maps={};bmaps={};cmaps={};topcounts=collections.Counter()
    for filename in names:
        rows=[r for r in allrows if r['file']==filename and r['status']=='ok' and (name!='plain_lora_main_blocks' or r['scope']=='dit')]
        et=sum(r['energy'] for r in rows);gm=collections.defaultdict(float);bm=collections.defaultdict(float);cm=collections.defaultdict(float)
        for r in rows:
            gm[r['group']]+=r['energy']/et;cm[r['group']+'/'+r['component']]+=r['energy']/et
            if r['scope']=='dit':bm[str(r['block'])]+=r['energy']/et
        maps[filename]=dict(gm);bmaps[filename]=dict(bm);cmaps[filename]=dict(cm)
        topcounts.update(k for k,v in sorted(bm.items(),key=lambda x:x[1],reverse=True)[:5])
    comps=sorted(set().union(*(c.keys() for c in cmaps.values())))
    aggregate[name]={'n':len(names),'files':names,'group_stats':{g:statistics([m.get(g,0) for m in maps.values()]) for g in GROUPS},'block_stats':{str(b):statistics([m.get(str(b),0) for m in bmaps.values()]) for b in range(28)},'component_stats':{c:statistics([m.get(c,0) for m in cmaps.values()]) for c in comps},'per_file_group_share':maps,'per_file_block_share':bmaps,'per_file_component_share':cmaps,'block_top5_counts':dict(topcounts)}
# Common canonical main-block targets across ordinary LoRA files, normalized only within these targets.
plain=subsets['plain_lora'];keysets=[set(r['canonical'] for r in allrows if r['file']==f and r['scope']=='dit' and r['status']=='ok') for f in plain]
common=set.intersection(*keysets);commonrows=[r for r in allrows if r['file'] in plain and r['canonical'] in common]
common_groups={};common_blocks={}
for f in plain:
    rr=[r for r in commonrows if r['file']==f];et=sum(r['energy'] for r in rr);g=collections.defaultdict(float);b=collections.defaultdict(float)
    for r in rr:g[r['group']]+=r['energy']/et;b[str(r['block'])]+=r['energy']/et
    common_groups[f]=g;common_blocks[f]=b
aggregate['plain_lora_common_targets']={'n':len(plain),'common_target_count':len(common),'group_stats':{g:statistics([m.get(g,0) for m in common_groups.values()]) for g in GROUPS},'block_stats':{str(b):statistics([m.get(str(b),0) for m in common_blocks.values()]) for b in range(28)},'common_targets':sorted(common)}
# Identical 280 targets across all 10 files, including LoKr.
allnames=[f['file'] for f in files]
allcommon=set.intersection(*(set(r['canonical'] for r in allrows if r['file']==f and r['scope']=='dit' and r['status']=='ok') for f in allnames))
cg={};cb={};ct=collections.Counter()
for f in allnames:
    rr=[r for r in allrows if r['file']==f and r['canonical'] in allcommon];et=sum(r['energy'] for r in rr);gm=collections.defaultdict(float);bm=collections.defaultdict(float)
    for r in rr:gm[r['group']]+=r['energy']/et;bm[str(r['block'])]+=r['energy']/et
    cg[f]=dict(gm);cb[f]=dict(bm);ct.update(b for b,e in sorted(bm.items(),key=lambda x:x[1],reverse=True)[:5])
aggregate['all_common_targets']={'n':len(allnames),'common_target_count':len(allcommon),'group_stats':{g:statistics([m.get(g,0) for m in cg.values()]) for g in GROUPS},'block_stats':{str(b):statistics([m.get(str(b),0) for m in cb.values()]) for b in range(28)},'block_top5_counts':dict(ct),'per_file_group_share':cg,'per_file_block_share':cb,'common_targets':sorted(allcommon)}
result={'method':'scale=alpha/rank, LoRA Gram; LoKr ||W1||F² * Gram(W2A,W2B), multiplier=1','thread_limit':4,'seconds':time.time()-START,'files':details,'aggregate':aggregate,'limitations':['Static weight updates only; no base weight relative norm, activation, quantization sensitivity or image quality measured.','A/B no tensor alpha uses metadata; exporter conventions can differ.','Turbo metadata absent: tensor representation is ordinary LoRA, provenance/distillation goal not verified.','Missing targets are absence of adapter updates, not evidence of low intrinsic sensitivity.','Rank is configured/factor rank, not numerical effective rank.']}
(OUT/'aggregate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for label,cat,keys,png in [('Module energy share (%)','per_file_group_share',GROUPS,'module_heatmap.png'),('28 block energy share (%)','per_file_block_share',[str(i) for i in range(28)],'block_heatmap.png')]:
    a=aggregate['all'];mat=np.array([[a[cat][f].get(k,0)*100 for k in keys] for f in a['files']]);fig,ax=plt.subplots(figsize=(14 if len(keys)>10 else 10,6));im=ax.imshow(mat,aspect='auto',cmap='viridis');ax.set_yticks(range(len(files)),labels=[f'{i+1}: '+Path(f).stem[:30] for i,f in enumerate(a['files'])]);ax.set_xticks(range(len(keys)),labels=keys,rotation=25 if len(keys)<10 else 0);ax.set_title(label);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/png,dpi=180);plt.close(fig)

lines=['# Anima 画风权重静态分析','',f'10 个文件；{sum(f["size_bytes"] for f in files)/1e9:.3f} GB；CPU 最多4线程；耗时 {time.time()-START:.1f}s。','', '分析的是 multiplier=1 时的静态等价 ΔW。它不代表量化敏感性、输出质量、画风强弱或某层的因果重要性。未加载基座/DiT、未使用 GPU、未做全量 SVD。','', '## 计算与保存格式','', '普通 LoRA：ΔW=(α/r)BA；如有 inv_scale 则先 A←A·inv_scale。范数用 trace((BᵀB)(AAᵀ))，以 FP32 计算 Gram、FP64 求和。LoKr：ΔW=(α/r)kron(W1,W2A·W2B)，能量为 (α/r)²||W1||²·||W2A·W2B||²，r 从 W2B.shape[0] 得到，不用 W1 维度代替。','', '读取 alpha tensor 优先；没有 tensor 的 Anima_in_real 使用 metadata ss_network_alpha=32、实际 rank=64，scale=0.5。所有这些绝对值依赖保存者的 alpha 约定；其文件内能量占比不受统一 scale 影响。配置 rank 不等于数值有效秩。','', '参考实时源码：networks/lora_modules/base.py:78–81；lora.py:116–131（inv_scale）；networks/plugins/lokr/module.py:179–185、418–438；networks/attn_fuse.py:52–55。归一化下划线/dot 命名，q/k/v 与 out 细分；若有 fused qkv/kv 应按其输出维度拆分。本批次没有 fused projection tensor；所有保存pair为二维Linear更新。','', '## 每文件摘要','', '|文件|类型|模块|rank|alpha|scale|参数量|ΔW能量|global Fro|delta RMS|','|---|---|---:|---|---|---|---:|---:|---:|---:|']
for f in files:lines.append(f'|{f["file"]}|{f["type"]}|{f["reconstructed"]}|{f["rank_min"]}–{f["rank_max"]}|{f["alpha_min"]:g}–{f["alpha_max"]:g}|{f["scale_min"]:g}–{f["scale_max"]:g}|{f["adapter_params"]:,}|{f["energy"]:.6g}|{f["global_fro"]:.6g}|{f["delta_rms"]:.6g}|')
lines+=['', '参数量不含 alpha 标量；模块数按保存的 split target 计数；dense 参数量是所有可重建 ΔW 矩阵的元素数。完整大小、tensor 数、alpha/scale 分布、训练步数和元信息见 per_file.csv / aggregate.json，逐层见 per_layer.csv。','', '## 等权归一化统计','', '先按每文件总能量归一化，再对文件等权平均/中位数/四分位数；不将不同 checkpoint/scale 的原始能量混加。全部10文件和普通LoRA9文件分别列出；Turbo 按实际张量可重建成普通 LoRA，未因文件名排除。另提供普通LoRA的主28 blocks及交集目标统计，以隔离附加模块/覆盖偏差。','']
for name,a in aggregate.items():
    lines += [f'### {name}（n={a["n"]}）','', '|模块组|均值%|中位数%|Q25–Q75%|','|---|---:|---:|---:|']
    for g,s in sorted(a['group_stats'].items(),key=lambda x:x[1]['mean'],reverse=True):lines.append(f'|{g}|{s["mean"]*100:.2f}|{s["median"]*100:.2f}|{s["q25"]*100:.2f}–{s["q75"]*100:.2f}|')
    top=sorted(a['block_stats'].items(),key=lambda x:x[1]['mean'],reverse=True)[:5]
    lines+=['', '平均能量占比前5 block（0起算）：'+ '；'.join(f'{b}: {s["mean"]*100:.2f}%（中位 {s["median"]*100:.2f}%）' for b,s in top),'']
    if 'block_top5_counts' in a:lines.append('上述 block 进入每文件 top5 次数：'+ '；'.join(f'{b}: {a["block_top5_counts"].get(b,0)}/{a["n"]}' for b,s in top));lines.append('')
lines+=['## 覆盖、异常与限制','', '- 所有文件均覆盖主 DiT blocks 0–27，但 target 不同：448=每block16 targets；280=每block10个 attention/MLP targets，无AdaLN；364=每block13 targets，未覆盖AdaLN第一层；sky02v6 为514模块，包含额外 llm_adapter/embedders/final layer；Turbo为454模块，有6个主blocks外targets。','- sky02v6 为分解 LoKr，数学上可计算同维度 ΔW，但结构/覆盖不同，单独标明；rank/alpha可跨模块变化。','- Turbo-ANIMA-v4 的metadata为空，仅能根据键与shape推断Anima；实际是普通A/B+alpha保存格式，没有发现动态router/magnitude向量等额外张量。是否蒸馏/加速用途不能靠这些静态权重确认。','- lenovo 实际 tensor rank=16、alpha=16，与metadata一致；其文件体积接近rank32文件源于FP32保存，不能据文件大小推断rank。','- Anima_in_real 文件名 epoch20 与 metadata ss_epoch=1 / training_info epoch2 不一致；保留元信息，不推断文件名为实际训练epoch。',f'- 非finite张量 {sum(f["nonfinite_tensors"] for f in files)}；未使用张量 {sum(f["unused_tensors"] for f in files)}；不支持模块 {sum(f["unsupported"] for f in files)}。逐键异常列表完整保存在 aggregate.json。','- block热图以各文件完整能量为分母；没有保存adapter的target不等于低敏感度。附加模块会稀释主block占比。共同target交集统计另提供，此处普通LoRA交集 '+str(len(common))+' targets。','- 更新能量还受矩阵大小影响；per_layer.csv给出 RMS，可按每元素强度补充观察。没有基座权重范数，因此没有“相对基座变化率”。','', '![模块占比](module_heatmap.png)','', '![28blocks占比](block_heatmap.png)','', '## 复现与验证','', '`项目根目录 .venv/bin/python "'+str(OUT/'analysis.py')+'"`。脚本不会写入仓库或原始权重，仅写脚本所在分析目录。现有分析产物会被重生成；新分析应复制脚本到新的日期时间目录运行。','', '本次逐文件完整有限值检查、配对/shape检查、逐文件汇总/归一化。数学验证与文件守恒检查见 validation.json。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
# Independent tiny direct delta checks validate formulas without dense production deltas.
g=torch.Generator().manual_seed(7);A=torch.randn(4,11,generator=g);B=torch.randn(13,4,generator=g);W=torch.randn(2,3,generator=g)
lo=energy(B,A);direct=float((B.double()@A.double()).square().sum());kr=lo*float(W.double().square().sum());kd=float(torch.kron(W.double(),B.double()@A.double()).square().sum())
validation={'gram_relative_error':abs(lo-direct)/direct,'lokr_relative_error':abs(kr-kd)/kd,'all_files':len(files),'all_modules':len(allrows),'reconstructed':sum(f['reconstructed'] for f in files),'unsupported':sum(f['unsupported'] for f in files),'nonfinite':sum(f['nonfinite_tensors'] for f in files),'unused':sum(f['unused_tensors'] for f in files),'group_share_max_error':max(abs(sum(x.values())-1) for x in aggregate['all']['per_file_group_share'].values()),'per_file_energy_max_error':max(abs(sum(r['energy'] for r in allrows if r['file']==f['file'] and r['status']=='ok')-f['energy']) for f in files),'seconds':time.time()-START}
assert validation['gram_relative_error']<1e-6 and validation['lokr_relative_error']<1e-6
assert validation['group_share_max_error']<1e-10
(OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print('VALIDATION',json.dumps(validation),flush=True)
