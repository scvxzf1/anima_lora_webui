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
STYLE=Path('/home/scv/nvme0n1p1/权重分析/anima/画风/analysis-20261001-204544')
with (STYLE/'per_layer.csv').open(encoding='utf-8-sig') as f:style=list(csv.DictReader(f))
for r in style:
    if r['status']=='ok':r['energy']=float(r['energy']);r['dense_params']=int(r['dense_params'])
style_source=json.loads((STYLE/'aggregate.json').read_text())
def statistics(v):
    a=np.array(v,dtype=float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),iqr=float(np.quantile(a,.75)-np.quantile(a,.25)),min=float(a.min()),max=float(a.max()))
def intersection(rows,names):
    return set.intersection(*(set(r['canonical'] for r in rows if r['file']==f and r['scope']=='dit' and r['status']=='ok') for f in names))
def summarize(rows,names,targets=None,main=False):
    gs={};bs={};ms={};concentration={};tc=collections.Counter();mc=collections.Counter()
    for f in names:
        rr=[r for r in rows if r['file']==f and r['status']=='ok' and (not main or r['scope']=='dit') and (targets is None or r['canonical'] in targets)]
        et=sum(r['energy'] for r in rr);g=collections.defaultdict(float);b=collections.defaultdict(float);m={}
        for r in rr:
            share=r['energy']/et;g[r['group']]+=share;m[r['canonical']]=share
            if r['scope']=='dit':b[str(r['block'])]+=share
        gs[f]=dict(g);bs[f]=dict(b);ms[f]=m
        top=sorted(b,key=b.get,reverse=True)[:5];tc.update(top);mc.update(sorted(m,key=m.get,reverse=True)[:10])
        main_total=sum(b.values());normalized=[v/main_total for v in b.values()] if main_total else []
        concentration[f]={'top5_block_share':sum(b[k] for k in top),'effective_block_count':1/sum(v*v for v in normalized) if normalized else 0,'denominator_energy':et,'main_block_share':main_total,'target_count':len(rr)}
    return {'n':len(names),'files':names,'common_targets':sorted(targets) if targets is not None else None,'common_target_count':len(targets) if targets is not None else None,'group_stats':{g:statistics([x.get(g,0) for x in gs.values()]) for g in GROUPS},'block_stats':{str(i):statistics([x.get(str(i),0) for x in bs.values()]) for i in range(28)},'per_file_group_share':gs,'per_file_block_share':bs,'per_file_module_share':ms,'block_top5_counts':dict(tc),'module_top10_counts':dict(mc),'concentration':concentration}
names=[f['file'] for f in files];snames=[d['summary']['file'] for d in style_source['files']]
pc=intersection(allrows,names);joint=pc & intersection(style,snames)
aggregate={'character_all':summarize(allrows,names),'character_common':summarize(allrows,names,pc),'character_joint':summarize(allrows,names,joint),'style_joint':summarize(style,snames,joint)}
plain=[d['summary']['file'] for d in style_source['files'] if d['summary']['type']=='LoRA']
aggregate['style_plain_joint']=summarize(style,plain,joint)
comparison={'denominator':'intersection of canonical main DiT targets across all character and style files; normalize each file first, then equal weight','target_count':len(joint),'targets':sorted(joint),'group_difference_pp':{g:100*(aggregate['character_joint']['group_stats'][g]['mean']-aggregate['style_joint']['group_stats'][g]['mean']) for g in GROUPS},'block_difference_pp':{b:100*(aggregate['character_joint']['block_stats'][b]['mean']-aggregate['style_joint']['block_stats'][b]['mean']) for b in map(str,range(28))}}
# Preserve the available provenance hashes without inferring shared training conditions.
for d in details:
    meta=header(SRC/d['summary']['file']).get('__metadata__',{})
    d['metadata'].update({k:v for k,v in meta.items() if 'hash' in k.lower() or k in ['ss_sd_model_name','modelspec.base_model','ss_training_started_at','ss_training_finished_at']})
    d['header_flags']=[k for k in header(SRC/d['summary']['file']) if any(s in k.lower() for s in ['dora','magnitude','router','inv_scale','qkv','kv_proj'])]
result={'method':'multiplier=1; effective delta alpha/rank*BA; FP32 Gram with FP64 reduction; main DiT blocks anchored by blocks_<number>','files':details,'aggregate':aggregate,'comparison':comparison,'style_source':str(STYLE),'thread_limit':4,'seconds':time.time()-START}
(OUT/'aggregate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for cat,keys,png in [('per_file_group_share',GROUPS,'module_heatmap.png'),('per_file_block_share',list(map(str,range(28))),'block_heatmap.png')]:
    a=aggregate['character_common'];mat=np.array([[a[cat][f].get(k,0)*100 for k in keys] for f in names]);fig,ax=plt.subplots(figsize=(14 if len(keys)>10 else 9,5));im=ax.imshow(mat,aspect='auto',cmap='viridis');ax.set_yticks(range(len(names)),labels=[f'{i+1}: '+Path(f).stem[:33] for i,f in enumerate(names)]);ax.set_xticks(range(len(keys)),labels=keys,rotation=20 if len(keys)<10 else 0);ax.set_title('Character common targets: energy share (%)');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/png,dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
for name,label in [('character_joint','Character n=5'),('style_joint','Style n=10'),('style_plain_joint','Style plain LoRA n=9')]:
    axes[0].plot(range(28),[aggregate[name]['block_stats'][str(i)]['mean']*100 for i in range(28)],marker='.',label=label)
axes[0].legend();axes[0].set_ylabel('Mean share (%)');axes[0].set_title(f'Identical {len(joint)} main DiT targets, equal weight per file');axes[1].bar(range(28),[comparison['block_difference_pp'][str(i)] for i in range(28)]);axes[1].axhline(0,color='black',lw=.7);axes[1].set_ylabel('Character - style (pp)');axes[1].set_xlabel('Block (zero based)');fig.tight_layout();fig.savefig(OUT/'comparison_blocks.png',dpi=180);plt.close(fig)
bmat=np.array([[aggregate['character_common']['per_file_block_share'][f].get(str(i),0) for i in range(28)] for f in names]);norm=bmat/np.linalg.norm(bmat,axis=1,keepdims=True);cosine=norm@norm.T
fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(cosine,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(len(names)),labels=range(1,len(names)+1));ax.set_yticks(range(len(names)),labels=range(1,len(names)+1));ax.set_title('Character block-share cosine (file index)');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/'block_cosine.png',dpi=180);plt.close(fig)
# Real saved module dense reference; use minimum-size reconstructed LoRA module.
rr=min((r for r in allrows if r['status']=='ok' and r['type']=='LoRA'),key=lambda r:r['dense_params'])
with safe_open(SRC/rr['file'],framework='pt',device='cpu') as sf:
    A=sf.get_tensor(rr['module']+'.lora_down.weight').double();B=sf.get_tensor(rr['module']+'.lora_up.weight').double();dense_energy=float((B@A).square().sum())*rr['scale']**2
real_error=abs(rr['energy']-dense_energy)/dense_energy
checks={'real_module_dense':{'file':rr['file'],'module':rr['module'],'shape':rr['shape'],'gram_energy':rr['energy'],'dense_float64_energy':dense_energy,'relative_error':real_error},'all_files':len(files),'style_files':len(snames),'all_modules':len(allrows),'reconstructed':sum(f['reconstructed'] for f in files),'unsupported':sum(f['unsupported'] for f in files),'nonfinite':sum(f['nonfinite_tensors'] for f in files),'unused':sum(f['unused_tensors'] for f in files),'character_common_targets':len(pc),'joint_targets':len(joint),'joint_share_max_error':max(abs(sum(v.values())-1) for name in ['character_joint','style_joint'] for v in aggregate[name]['per_file_block_share'].values()),'cosine_matrix':cosine.tolist(),'seconds':time.time()-START}
assert real_error<1e-5 and checks['joint_share_max_error']<1e-10
(OUT/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# Anima 人物权重展开静态分析与画风对照','',f'人物 {len(files)} 文件，{sum(f["size_bytes"] for f in files)/1e6:.1f} MB；对照画风 {len(snames)} 文件。CPU 最多4线程，未加载基座、未使用GPU、未训练、未做全量SVD。','', '静态等价更新 ΔW=(alpha/rank)BA，multiplier=1；Fro能量用小Gram矩阵计算，FP32矩阵乘、FP64归约；LoKr沿用已验收的 Kronecker 范数公式。alpha优先tensor，其次metadata，具体来源逐模块记录。绝对范数不是质量、人物还原度或风格强弱排名；没有基座范数、激活、图片验证，不能作因果敏感性结论。','', '主DiT严格锚定 blocks_数字，llm_adapter等不混入block；dot/underscore命名统一canonical，本批人物未发现fused qkv/kv、DoRA/magnitude/router/inv_scale张量。所有张量完整检查finite；保存配对、shape、不支持模块和未消费键列表。配置rank不是数值有效秩。','', '## 每文件规模与静态更新','', '|文件|类型|主DiT/总模块|rank|alpha|scale|Fro|RMS|参数量（不含alpha）|','|---|---|---:|---|---|---|---:|---:|---:|']
for f in files:
    dit=sum(r['file']==f['file'] and r['scope']=='dit' and r['status']=='ok' for r in allrows)
    lines.append(f'|{f["file"]}|{f["type"]}|{dit}/{f["reconstructed"]}|{f["rank_min"]}–{f["rank_max"]}|{f["alpha_min"]:g}–{f["alpha_max"]:g}|{f["scale_min"]:g}–{f["scale_max"]:g}|{f["global_fro"]:.6g}|{f["delta_rms"]:.6g}|{f["adapter_params"]:,}|')
for i,d in enumerate(details):
    f=d['summary'];a=aggregate['character_all'];g=a['per_file_group_share'][f['file']];b=a['per_file_block_share'][f['file']];m=a['per_file_module_share'][f['file']]
    lines+=['',f'### {i+1}. {f["file"]}','',f'- 模型族依据：{f["family_evidence"]}。tensor数 {f["tensor_count"]}，重建 {f["reconstructed"]}/{f["module_count"]}；alpha来源：'+', '.join(sorted(set(r['alpha_source'] for r in allrows if r['file']==f['file']))),'- rank分布：'+json.dumps(d['rank_distribution'],ensure_ascii=False)+'；alpha分布：'+json.dumps(d['alpha_distribution'],ensure_ascii=False),'- 模块组（该文件全部可重建更新为分母）：'+'；'.join(f'{k} {v*100:.2f}%' for k,v in sorted(g.items(),key=lambda x:x[1],reverse=True)),'- top5 blocks：'+'；'.join(f'{k} {v*100:.2f}%' for k,v in sorted(b.items(),key=lambda x:x[1],reverse=True)[:5]),'- top10模块：'+'；'.join(f'{k} {v*100:.2f}%' for k,v in sorted(m.items(),key=lambda x:x[1],reverse=True)[:10]),'- 可用元信息：`'+json.dumps(d['metadata'],ensure_ascii=False)+'`']
def section(title,a):
    lines.extend(['',title,'',f'n={a["n"]}；共同主DiT目标 {a["common_target_count"]} 个。先在共同目标内按每文件能量归一化，再等权计算，不混加原始能量。','', '|模块组|mean%|median%|Q25–Q75%|IQR百分点|','|---|---:|---:|---:|---:|'])
    for g,s in sorted(a['group_stats'].items(),key=lambda x:x[1]['mean'],reverse=True):lines.append(f'|{g}|{s["mean"]*100:.3f}|{s["median"]*100:.3f}|{s["q25"]*100:.3f}–{s["q75"]*100:.3f}|{s["iqr"]*100:.3f}|')
    lines.extend(['','|block|mean%|median%|Q25–Q75%|进入文件top5次数|','|---|---:|---:|---:|---:|'])
    for b,s in sorted(a['block_stats'].items(),key=lambda x:x[1]['mean'],reverse=True):lines.append(f'|{b}|{s["mean"]*100:.3f}|{s["median"]*100:.3f}|{s["q25"]*100:.3f}–{s["q75"]*100:.3f}|{a["block_top5_counts"].get(b,0)}/{a["n"]}|')
section('## 人物内部共同目标分布与一致性',aggregate['character_common'])
lines+=['','|文件|top5 block能量占比%|effective block count|','|---|---:|---:|']
for f,c in aggregate['character_common']['concentration'].items():lines.append(f'|{f}|{c["top5_block_share"]*100:.3f}|{c["effective_block_count"]:.3f}|')
lines+=['','effective block count=1/sum(block_share²)，衡量能量集中度，不是权重有效秩；均匀28层为28。top5以每文件各自热点选取。','', '共同目标top10模块热点出现次数（每文件各取10，模块很多时次数小不等于不存在一致性）：'+ '；'.join(f'{k}: {v}/{len(names)}' for k,v in sorted(aggregate['character_common']['module_top10_counts'].items(),key=lambda x:x[1],reverse=True)[:20])]
section('## 人物与画风真正共同目标：人物',aggregate['character_joint'])
section('## 人物与画风真正共同目标：画风全部',aggregate['style_joint'])
lines+=['','## 两组差异（人物减画风，百分点）','',f'严格使用全部 {len(names)+len(snames)} 文件主DiT共同canonical交集 {len(joint)} targets。不同覆盖分母不可直接比较；这里每个文件都在同一交集重归一化。','', '|模块组|人物mean%|画风mean%|差值百分点|普通LoRA画风mean%（n=9）|','|---|---:|---:|---:|---:|']
for g in GROUPS:lines.append(f'|{g}|{aggregate["character_joint"]["group_stats"][g]["mean"]*100:.3f}|{aggregate["style_joint"]["group_stats"][g]["mean"]*100:.3f}|{comparison["group_difference_pp"][g]:+.3f}|{aggregate["style_plain_joint"]["group_stats"][g]["mean"]*100:.3f}|')
lines+=['','|block|人物mean%|画风mean%|差值百分点|','|---|---:|---:|---:|']
for b in sorted(comparison['block_difference_pp'],key=lambda x:abs(comparison['block_difference_pp'][x]),reverse=True):lines.append(f'|{b}|{aggregate["character_joint"]["block_stats"][b]["mean"]*100:.3f}|{aggregate["style_joint"]["block_stats"][b]["mean"]*100:.3f}|{comparison["block_difference_pp"][b]:+.3f}|')
lines+=['','## 样本差异、例外与可比性边界','',f'- 人物完整目标与共同目标分别为：'+ '；'.join(f'{f["file"]} {f["reconstructed"]}' for f in files)+f'；人物交集 {len(pc)}，两组交集 {len(joint)}。额外覆盖仅在逐文件完整统计出现；人物共同统计与对照统计排除它们。',f'- 非finite {checks["nonfinite"]}，unsupported {checks["unsupported"]}，未消费张量 {checks["unused"]}；完整键名、原因见aggregate.json。','- 本批人物例外：Sakura覆盖448个主DiT目标（多168个AdaLN目标），其余4份各280个；共同目标分析排除了Sakura额外AdaLN。remielle为rank16、alpha8、scale0.5且metadata为空，其共同目标top5 blocks占29.97%、effective block count23.75，比分散的roxy（21.13%、27.61）更集中；这些只描述本批静态能量。','- sshs_model_hash/sshs_legacy_hash是adapter自身hash，不作为基座证据；只有ss_sd_model_hash/ss_sd_model_sha256/modelspec.base_model等可帮助核对基座，缺失时无法确认同源。','- 人物5份与画风10份为本地选择样本，不代表全部人物/画风LoRA。作者、训练步数/学习率、rank、保存约定、基座hash和数据均可能不同；metadata缺失不能视为相同条件，不能把分布差异归因为题材。','- 画风含LoKr；它的附加llm_adapter不混入共同主DiT。普通LoRA画风9份对照另列，仍使用同一15文件交集。','- Fro受矩阵尺寸与scale影响；RMS是每元素强度补充。缺失target是未保存更新，不代表该层低敏感。余弦比较仅描述block占比分布相似，不能证明人物语义相似。','', '![人物模块组共同目标热图](module_heatmap.png)','', '![人物block共同目标热图](block_heatmap.png)','', '![相同交集两组block均值及差值](comparison_blocks.png)','', '![人物blockshare余弦矩阵](block_cosine.png)','', '## 复现与验证','', '`.venv/bin/python "'+str(OUT/'analysis.py')+'"`（项目根目录执行）。写入仅限本analysis目录，原始权重、画风报告和仓库只读。','', f'实际权重模块dense FP64对照：{rr["file"]} / {rr["module"]}，shape={rr["shape"]}，相对误差 {real_error:.3g}；共同block share守恒最大误差 {checks["joint_share_max_error"]:.3g}。验证数据见validation.json；完整逐模块矩阵参数量、范数与alpha来源见per_layer.csv，完整文件汇总见per_file.csv，统计分位数与共同targets见aggregate.json。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('VALIDATION',json.dumps(checks,ensure_ascii=False),flush=True)
print('COMPARISON',json.dumps(comparison['group_difference_pp']),flush=True)
