from pathlib import Path
import json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
root=Path(sys.argv[1]); out=root/'report/figures';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':12,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
fig,ax=plt.subplots(figsize=(7,2.8))
colors=['#9dc5df','#f5ce93','#dfabb3','#ddd']
labels=['Correct','Wrong answer','Execution error','Unscorable']
for i,vals in enumerate([[36,44,19,1],[40,49,10,1]]):
    left=0
    for val,c,l in zip(vals,colors,labels):
        ax.barh(i,val,left=left,color=c,label=l if i==0 else None,height=.48)
        if val>2:ax.text(left+val/2,i,str(val),ha='center',va='center',color='black',fontweight='bold')
        left+=val
ax.set_yticks([0,1],['Untuned 7B','QLoRA 7B']);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Questions out of 100');ax.legend(loc='upper center',bbox_to_anchor=(.45,1.3),ncol=4,frameon=False,fontsize=10);fig.tight_layout();fig.savefig(out/'outcomes.png',bbox_inches='tight');plt.close(fig)
state=json.loads((root/'results/qlora7b/trainer_state.json').read_text());hist=[x for x in state['log_history'] if 'loss' in x];val=json.loads((root/'results/qlora7b/validation_results.json').read_text())
fig,ax=plt.subplots(figsize=(7,2.8));ax.plot([x['step'] for x in hist],[x['loss'] for x in hist],color='#2475ad',lw=1.7,label='Logged training loss');ax.scatter([state['global_step']],[val['eval_loss']],color='#d88b25',s=48,marker='D',label='Final validation loss',zorder=5);ax.set_xlabel('Optimizer step');ax.set_ylabel('Cross entropy loss');ax.legend(frameon=False,fontsize=11);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(out/'training_loss.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,1.75));ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis('off')
boxes=[(0.1,'SQL drafts\ndisagree'),(2.65,'Check the\ndiffering clause'),(5.2,'Repair using\ndatabase evidence'),(7.75,'Verify or flag\nuncertainty')]
for x,t in boxes:
    ax.add_patch(FancyBboxPatch((x,.55),2.1,.9,boxstyle='round,pad=0.08',facecolor='white',edgecolor='#555',lw=1))
    ax.text(x+1.05,1,t,ha='center',va='center',fontsize=11)
for x in [2.25,4.8,7.35]:ax.annotate('',xy=(x+.35,1),xytext=(x,1),arrowprops={'arrowstyle':'->','color':'#2475ad','lw':1.5})
fig.tight_layout();fig.savefig(out/'proposed_repair.png',bbox_inches='tight');plt.close(fig)
