#!/usr/bin/env python3
"""Presentation-only redraw of verified v4 aggregates. No model or raw-data access."""
from pathlib import Path
import csv, json, hashlib, math
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]; IN=ROOT/'inputs'
(ROOT/'qa').mkdir(exist_ok=True)
C={'reg':'#236B8E','act':'#A95528','ref':'#5A6269','ink':'#22272B','line':'#8A949B','fill':'#F3F5F6','uncertain':'#E9EEF1'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'axes.edgecolor':'#67727A','axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'text.color':C['ink'],'axes.labelcolor':C['ink'],'xtick.color':C['ink'],'ytick.color':C['ink'],'grid.color':'#E5E8EB','grid.linewidth':.55,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})

def rows(n):
 with (IN/n).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def box(ax,x,y,w,h,s,fc='white',fs=9,weight='normal'):
 ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=C['line'],lw=.7))
 ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=fs,linespacing=1.28,weight=weight)
def arrow(ax,a,b):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.8,color=C['line'],shrinkA=1.8,shrinkB=2))
def panel(ax,label,title):
 ax.text(0,1.02,label,transform=ax.transAxes,weight='bold',fontsize=10,ha='left',va='bottom')
 ax.annotate(title,xy=(0,1.02),xycoords='axes fraction',xytext=(14,0),textcoords='offset points',fontsize=10,ha='left',va='bottom')
def diagram(rect,fig):
 ax=fig.add_axes(rect);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');return ax
def save(fig,name):
 for ext in ['pdf','svg','png']:
  kwargs={'dpi':300} if ext=='png' else {}
  fig.savefig(ROOT/(name+'.'+ext),**kwargs)
 fig.savefig(ROOT/'qa'/(name+'_150dpi.png'),dpi=150)
 dims[name]={'width_in':fig.get_figwidth(),'height_in':fig.get_figheight(),'png_dpi':300,'font_family':'DejaVu Sans'}
 plt.close(fig)

flow=rows('cohort_flow.csv'); metrics=rows('metric_comparison.csv'); rel=rows('reliability_bins.csv'); curves=rows('roc_pr_curve_points.csv'); mods=rows('module_metrics.csv'); b=rows('primary_brier_bounds.csv')[0]
F={r['stage']:int(r['records']) for r in flow}; M={r['predictor']:r for r in metrics};dims={};curve_checks=[]
assert 10809-890==9919 and 9919-1596==8323 and 8323-1494==6829 and 6829==6771+58
assert 8845-1169==7676 and 6141+1535==7676 and 174+44==218
assert len(mods)==6 and sum(int(r['records']) for r in mods)==6771 and sum(int(r['positives']) for r in mods)==186

# 1: sequential denominators and retained uncertainty.
fig=plt.figure(figsize=(6.4,4.8));a=diagram([.035,.055,.43,.88],fig);d=diagram([.53,.055,.435,.88],fig)
panel(a,'A','2013J source');panel(d,'B','2014J target')
box(a,.06,.81,.88,.15,'Six shared modules\n8,845 registration records',C['fill'])
arrow(a,(.5,.81),(.5,.60));a.text(.5,.705,'Exclude / quarantine 1,169',ha='center',fontsize=9,bbox={'facecolor':'white','edgecolor':'none','pad':1})
box(a,.06,.43,.88,.17,'Known eligible records: 7,676\n218 events; 7,458 horizon negatives',C['fill'])
arrow(a,(.5,.43),(.5,.27));box(a,.06,.10,.88,.17,'Source fit: 6,141 (174 events)\nCalibration: 1,535 (44 events)')
a.text(.06,.005,'Record = student × module × presentation',fontsize=8.5,va='bottom')
box(d,.05,.855,.9,.105,'All modules: 10,809 records',C['fill'])
arrow(d,(.5,.855),(.5,.71));d.text(.5,.782,'Remove source overlap: 890',ha='center',fontsize=8.7,bbox={'facecolor':'white','edgecolor':'none','pad':1})
box(d,.05,.62,.9,.09,'After overlap removal: 9,919')
arrow(d,(.5,.62),(.5,.47));d.text(.5,.545,'Remove remaining CCC: 1,596',ha='center',fontsize=8.7,bbox={'facecolor':'white','edgecolor':'none','pad':1})
box(d,.05,.38,.9,.09,'Six shared modules: 8,323')
arrow(d,(.5,.38),(.5,.225));d.text(.5,.305,'Ineligible / quarantined: 1,494',ha='center',fontsize=8.7,bbox={'facecolor':'white','edgecolor':'none','pad':1})
box(d,.05,.08,.9,.145,'Retained analysis records: 6,829\nKnown: 6,771 (186 events)\nUncertain: 58 (retained for bounds)',C['uncertain'],8.8)
d.text(.05,.005,'Overlap uses the full 2013J source cohort.',fontsize=8.5,va='bottom')
save(fig,'figure_01_cohort_flow')

# 2: timeline and realised information flow.
fig=plt.figure(figsize=(6.4,4.7));ax=fig.add_axes([.075,.70,.88,.22]);panel(ax,'A','Landmark and observation windows')
ax.set(xlim=(-1,58),ylim=(0,1),yticks=[],xlabel='Course day');ax.spines['left'].set_visible(False)
for lo,hi in [(0,6),(7,13),(14,20),(21,27),(28,28)]:
 ax.axvspan(lo-.5,hi+.5,color='#DEE8EE',ec='white',lw=.7)
 if lo!=28:ax.text((lo+hi)/2,.61,f'{lo}–{hi}',ha='center',fontsize=8.5)
ax.axvspan(28.5,56.5,color='#EEE5DF');ax.axvspan(56.5,57.5,color='#E6E9EB');ax.axvline(28.5,c=C['ink'],lw=1);ax.axvline(56.5,c=C['ref'],lw=.8,ls='--')
ax.text(28,.61,'28',ha='center',fontsize=7.8)
ax.text(14,.17,'VLE features: days 0–28 inclusive',ha='center',fontsize=8.5)
ax.text(42,.58,'Recorded withdrawal',ha='center',fontsize=9);ax.text(42,.25,'days 29–56',ha='center',fontsize=9)
ax.text(28.5,.94,'Landmark',ha='center',va='top',fontsize=8.5,bbox={'facecolor':'white','edgecolor':'none','pad':1})
ax.set_xticks([0,7,14,21,28.5,42,56.5],['0','7','14','21','28/29','42','56/57'])
fig.text(.075,.595,'Eligibility at landmark: registered and not withdrawn.',fontsize=8.5)
a=diagram([.035,.05,.93,.48],fig);panel(a,'B','Source-only learning; frozen target scoring')
xs=[.005,.265,.525,.785];w=.21
box(a,xs[0],.53,w,.34,'2013J source fit\n6,141 records\n174 events',C['fill'],8.7)
box(a,xs[1],.53,w,.34,'Two LR predictors\nRegistration\nRegistration + VLE',fs=8.6)
box(a,xs[2],.53,w,.34,'Source calibration\n1,535 records\n44 events\nSigmoid, separately',fs=8.4)
box(a,xs[3],.53,w,.34,'Freeze and score\n2014J target\nNo target tuning',C['fill'],8.6)
for j in range(3):arrow(a,(xs[j]+w,.70),(xs[j+1],.70))
box(a,.005,.08,.285,.22,'Constant reference\nSource-fit prevalence\n174 / 6,141 = 0.028334',C['fill'],8.4)
arrow(a,(xs[0]+w/2,.53),(.14,.30))
box(a,.37,.08,.30,.22,'Known labels: 6,771\n186 events\nDescriptive diagnostics',fs=8.6)
box(a,.73,.08,.265,.22,'58 uncertain records\nPaired bounds with\nknown-label records',C['uncertain'],8.6)
arrow(a,(.89,.53),(.52,.30));arrow(a,(.89,.53),(.86,.30));arrow(a,(.29,.19),(.37,.19))

save(fig,'figure_02_temporal_design')

# 3: allowed missing-time states and exact primary bounds, with no invented allocations.
fig=plt.figure(figsize=(6.4,4.6));a=diagram([.04,.58,.92,.34],fig);panel(a,'A','Allowed states for each of 58 missing-date records')
box(a,.005,.32,.26,.38,'Withdrawal recorded;\nwithdrawal day missing',C['uncertain'],9)
for y,txt in [(.70,'By day 28: ineligible → exclude'),(.38,'Days 29–56: eligible event → y = 1'),(.06,'After day 56: eligible non-event → y = 0')]:
 box(a,.40,y,.595,.23,txt,fs=9);arrow(a,(.265,.51),(.40,y+.115))
a.text(.005,-.045,'Feasible alternatives, not observed counts; every cross-record combination is admissible.',fontsize=8.5)
ax=fig.add_axes([.10,.20,.85,.28]);panel(ax,'B','Exact paired Brier identification bounds')
lo=float(b['lower_decimal']);hi=float(b['upper_decimal']);ax.axvline(0,c=C['ref'],ls='--',lw=.85)
ax.plot([lo,hi],[.2,.2],c=C['act'],lw=3.3);ax.scatter([lo,hi],[.2,.2],c=C['act'],s=28,zorder=3)
for v,ms,ha in [(lo,b['lower_attaining_m'],'left'),(hi,b['upper_attaining_m'],'right')]:
 ax.annotate(f'{v:.9f}',(v,.2),xytext=(0,14),textcoords='offset points',ha=ha,fontsize=9)
 ax.annotate(f'Eligible uncertain m = {ms}',(v,.2),xytext=(0,-13),textcoords='offset points',ha=ha,fontsize=8.5)
known_difference=float(M['registration_logistic']['brier_calibrated_or_reference'])-float(M['registration_plus_vle_logistic']['brier_calibrated_or_reference'])
assert abs(known_difference-.000434277263142975)<1e-15
ax.scatter([known_difference],[-.43],marker='D',c=C['ref'],s=24,zorder=3)
ax.annotate(f'Known-label contrast: {known_difference:.10f}',(known_difference,-.43),xytext=(9,0),textcoords='offset points',va='center',fontsize=8.5)
ax.set(ylim=(-.68,.82),xlim=(-.0001,.00168),yticks=[],xlabel='Brier(registration, calibrated) − Brier(activity, calibrated)')
ax.spines['left'].set_visible(False);ax.set_xticks([0,.0004,.0008,.0012,.0016]);ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.grid(axis='x')
fig.text(.10,.065,'Finite-cohort identification bounds — not a confidence interval.',fontsize=9,weight='bold')
fig.text(.10,.025,'Positive values favour activity. Known labels n = 6,771; uncertain K = 58.',fontsize=8.5)
save(fig,'figure_03_missing_time_bounds')

# 4: native sequence ROC / PR; overlapping raw/calibrated rank curves represented once.
fig,axs=plt.subplots(1,2,figsize=(6.4,3.75));fig.subplots_adjust(left=.105,right=.98,bottom=.33,top=.91,wspace=.40)
for ax,l,t in zip(axs,['A','B'],['ROC','Precision–recall']):panel(ax,l,t);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks([0,.25,.5,.75,1]);ax.grid(lw=.45)
for model,key,name in [('registration_logistic','reg','Registration'),('registration_plus_vle_logistic','act','Registration + VLE')]:
 for c,ax,xk,yk,metric,metriclabel in [('ROC',axs[0],'x_fpr','y_tpr','auroc','AUROC'),('PR',axs[1],'x_recall','y_precision','average_precision','AP')]:
  rr=[r for r in curves if r['curve']==c and r['predictor']==model and r['probability']=='calibrated'];xy=[(float(r[xk]),float(r[yk])) for r in rr]
  raw=[(float(r[xk]),float(r[yk])) for r in curves if r['curve']==c and r['predictor']==model and r['probability']=='raw'];assert raw==xy
  if c=='ROC':v=sum((q[0]-p[0])*(q[1]+p[1])/2 for p,q in zip(xy,xy[1:]))
  else:v=sum((p[0]-q[0])*p[1] for p,q in zip(xy,xy[1:]));assert all(p[0]>=q[0] for p,q in zip(xy,xy[1:]))
  assert abs(v-float(M[model][metric]))<1e-12
  curve_checks.append({'curve':c,'model':model,'points':len(xy),'metric':v,'raw_equals_calibrated':True,'native_order':True})
  ax.plot([p[0] for p in xy],[p[1] for p in xy],c=C[key],lw=1.1,ls='-' if key=='act' else '--',label=f'{name}: {metriclabel} {float(M[model][metric]):.3f}')
axs[0].plot([0,1],[0,1],c=C['ref'],ls=':',lw=.8);axs[0].set(xlabel='False-positive rate',ylabel='True-positive rate')
prev=186/6771;axs[1].axhline(prev,c=C['ref'],ls=':',lw=.8);axs[1].set(xlabel='Recall',ylabel='Precision')
for ax in axs:ax.legend(loc='upper left',bbox_to_anchor=(-.02,-.24),frameon=False,handlelength=2,fontsize=8.4)
fig.text(.105,.055,'Known labels: n = 6,771; 186 events. Descriptive; no confidence bands.',fontsize=8.5)
fig.text(.105,.018,'Dotted references: chance ROC diagonal; target prevalence (0.02747) in PR.',fontsize=8.5)
save(fig,'figure_04_discrimination')

# 5: probability state encoded by marker/line, model colour consistent across panels.
fig,axs=plt.subplots(1,2,figsize=(6.4,4.3));fig.subplots_adjust(left=.105,right=.98,bottom=.45,top=.88,wspace=.40)
for ax,model,key,title,l in zip(axs,['registration_logistic','registration_plus_vle_logistic'],['reg','act'],['Registration','Registration + VLE'],['A','B']):
 panel(ax,l,title)
 for state,marker,ls,fill in [('raw','s','--','white'),('calibrated','o','-',C[key])]:
  rr=[r for r in rel if r['predictor']==model and r['probability']==state and int(r['count'])];rr.sort(key=lambda r:int(r['bin']))
  ax.plot([float(r['mean_probability']) for r in rr],[float(r['observed_prevalence']) for r in rr],c=C[key],marker=marker,mfc=fill,mec=C[key],ms=4,lw=1,ls=ls)
 ref=next(r for r in rel if r['predictor']=='source_prevalence_constant' and int(r['count']))
 ax.scatter(float(ref['mean_probability']),float(ref['observed_prevalence']),c=C['ref'],marker='D',s=17,zorder=5)
 ax.plot([0,.3],[0,.3],c=C['ref'],ls=':',lw=.8);ax.set(xlim=(0,.3),ylim=(-.005,.3),xlabel='Bin mean predicted probability',ylabel='Observed event proportion');ax.set_xticks([0,.1,.2,.3]);ax.set_yticks([0,.1,.2,.3]);ax.grid(lw=.45)
 rr={state:{int(r['bin']):int(r['count']) for r in rel if r['predictor']==model and r['probability']==state and int(r['count'])} for state in ['raw','calibrated']}
 vals=[[f'[{j/10:.1f}, {(j+1)/10:.1f})',f"{rr['raw'].get(j,0):,}",f"{rr['calibrated'].get(j,0):,}"] for j in [0,1,2]]
 table=ax.table(cellText=vals,colLabels=['Bin','Raw n','Calibrated n'],loc='bottom',cellLoc='center',bbox=[-.03,-.73,1.06,.38])
 table.auto_set_font_size(False);table.set_fontsize(8.2)
 for (r,c),cell in table.get_celld().items():cell.set_linewidth(0);cell.set_facecolor('white');cell.PAD=.02
 for c in range(3):table[(0,c)].get_text().set_weight('bold')
handles=[Line2D([0],[0],c=C['ref'],ls='--',marker='s',mfc='white',ms=4,label='Raw'),Line2D([0],[0],c=C['ref'],ls='-',marker='o',ms=4,label='Calibrated'),Line2D([0],[0],c=C['ref'],ls='None',marker='D',ms=4,label='Source-prevalence reference')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,frameon=False,fontsize=8.4,columnspacing=1.1,handlelength=2)
fig.text(.105,.055,'Displayed probability range: 0–0.30. All higher bins are empty.',fontsize=8.5)
fig.text(.105,.018,'n = 6,771; 186 events. Ten fixed bins. Descriptive; no confidence bands.',fontsize=8.5)
save(fig,'figure_05_reliability')

# S1: dot plot, not ranked; all comparators including GGG counterexample retained.
fig,ax=plt.subplots(figsize=(6.4,3.7));fig.subplots_adjust(left=.31,right=.96,top=.82,bottom=.26)
y=np.arange(len(mods))
for off,col,key,marker,label in [(-.16,'brier_constant','ref','D','Source constant'),(0,'brier_registration_calibrated','reg','o','Registration'),(.16,'brier_activity_calibrated','act','^','Registration + VLE')]:
 ax.scatter([float(r[col]) for r in mods],y+off,c=C[key],marker=marker,s=23,label=label)
ax.set_yticks(y,[f"{r['module']}  n={int(r['records']):,}; events={r['positives']}" for r in mods]);ax.invert_yaxis();ax.set_xlim(.017,.037);ax.set_xticks([.02,.025,.03,.035]);ax.set_xlabel('Brier score (lower is better)');ax.grid(axis='x');ax.spines['left'].set_visible(False)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.51,.99),frameon=False,ncol=3,fontsize=8.8,columnspacing=1)
fig.text(.06,.085,'Modules retain source order. Complete cases only; 6–62 events per module.',fontsize=9)
fig.text(.06,.04,'Descriptive point estimates; no uncertainty intervals or module-superiority inference.',fontsize=9)
save(fig,'figure_S1_module_brier')

# S2: target feature medians/IQR and known event-day histogram, explicitly different populations.
f=rows('feature_summary.csv');e=rows('event_day_counts.csv');assert sum(int(r['known_recorded_withdrawals']) for r in e)==186
fig=plt.figure(figsize=(6.4,4.45));gs=fig.add_gridspec(2,3,left=.09,right=.97,bottom=.20,top=.89,hspace=.95,height_ratios=[.75,1.2])
for i,(r,title) in enumerate(zip(f,['Total clicks','Active days','Recency (days)'])):
 ax=fig.add_subplot(gs[0,i]);med=float(r['median']);q1=float(r['q1']);q3=float(r['q3']);ax.errorbar(med,0,xerr=[[med-q1],[q3-med]],fmt='o',c=C['reg'],ms=4,lw=1.2,capsize=3);ax.set_yticks([]);ax.set_ylim(-1,1);ax.set_xlim(left=0);ax.set_title(title,fontsize=9);ax.spines['left'].set_visible(False);ax.set_xlabel('Observed value',fontsize=8.5);ax.text(.5,.95,f'{med:g} [{q1:g}, {q3:g}]',transform=ax.transAxes,ha='center',fontsize=8.5);ax.grid(axis='x')
ax=fig.add_subplot(gs[1,:]);ax.bar([int(r['day']) for r in e],[int(r['known_recorded_withdrawals']) for r in e],color=C['act'],width=.8);ax.set(xlabel='Recorded withdrawal day',ylabel='Known event count',xlim=(28,57));ax.set_xticks([29,35,42,49,56]);ax.set_yticks([0,5,10,15,20]);ax.grid(axis='y');ax.set_axisbelow(True)
fig.text(.09,.965,'A  Feature descriptions: all 6,829 target records; dot = median, whisker = Q1–Q3',fontsize=9)
fig.text(.09,.54,'B  Recorded event days: 186 known positives only',fontsize=9)
fig.text(.09,.06,'Post-hoc descriptions. Feature summaries include 235 zero-activity records.',fontsize=8.5)
fig.text(.09,.025,'Missing-date records (n = 58) are absent from event bars but retained in the bounds.',fontsize=8.5)
save(fig,'figure_S2_feature_event_descriptions')

# Reproducibility: hash copied aggregate inputs and exported figure files; all data unchanged.
expected_input_hashes=json.loads((ROOT/'source_input_hashes.json').read_text())
inputs=[]
for p in sorted(IN.glob('*.csv')):
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected_input_hashes[p.name]
 inputs.append({'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
manifest={'source_aggregate_archive_sha256':'4dd2f8e209ce7a92817aa1729971085bc61c8004ce3549fd7f4adb48f0301482','source_partition_events':{'fit_records':6141,'fit_events':174,'calibration_records':1535,'calibration_events':44,'provenance':'Locked source split; source reference 174/6141 and source total 218 events reconcile.'},'scope':'Presentation-only redraw; no new analyses, fits, predictions, recalibration, inferential bands or raw-data access.','inputs':inputs,'figures':dims,'numerical_preservation':{'all_12_retained_input_tables_byte_identical':True,'curve_checks':curve_checks,'exact_bound_fraction_matches_decimal':all(abs(float(Fraction(b[k+'_fraction']))-float(b[k+'_decimal']))<1e-15 for k in ['lower','upper']),'counts_reconciled':True,'known_label_difference':known_difference},'figure_inputs':{'figure_01_cohort_flow':['cohort_flow.csv','label_availability_summary.csv','source partition facts'],'figure_02_temporal_design':['cohort_flow.csv','reliability_bins.csv','source partition facts'],'figure_03_missing_time_bounds':['primary_brier_bounds.csv','metric_comparison.csv','declared Cartesian three-state feasible set'],'figure_04_discrimination':['roc_pr_curve_points.csv','metric_comparison.csv'],'figure_05_reliability':['reliability_bins.csv'],'figure_S1_module_brier':['module_metrics.csv'],'figure_S2_feature_event_descriptions':['feature_summary.csv','zero_activity_summary.csv','event_day_counts.csv']},'outputs':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(ROOT.glob('figure_*.*')) if p.suffix in ['.pdf','.svg','.png']}}
(ROOT/'inputs_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'PASS','figures':dims,'numerical_preservation':manifest['numerical_preservation']},indent=2))
