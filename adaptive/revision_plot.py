#!/usr/bin/env python3
import json
from pathlib import Path


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparison=json.loads((out/'comparison.json').read_text())['methods']
    robustness=out/'analytical_robustness.json'
    if robustness.exists():
        extra=json.loads(robustness.read_text())
        if 'groups' in extra:comparison['integral full range*']=extra
    models={m['name']:m for m in json.loads((out/'models/report.json').read_text())}
    figure,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    names=['conventional','analytical','gravity','integral','shared_matched','per_matched','shared_full','per_full']
    if 'integral full range*' in comparison:names.insert(4,'integral full range*')
    for i,name in enumerate(names):
        item=comparison[name]['groups']['primary']['joint_rmse_rad'];mean=item['mean'];lo,hi=item['ci95']
        axes[0,0].errorbar(mean*1000,i,xerr=[[max(0,mean-lo)*1000],[max(0,hi-mean)*1000]],fmt='o',capsize=3,color='#175e86')
    axes[0,0].set_yticks(range(len(names)),names);axes[0,0].invert_yaxis();axes[0,0].set_xlabel('Joint RMSE (mrad); family bootstrap 95% CI')
    axes[0,0].set_title('Held-out closed-loop comparisons: 40 families')
    fit_counts=[];means=[];lows=[];highs=[]
    for count in [8,16,32,64]:
        name=f'curve_{count}' if count<64 else 'shared_full'
        record=models[name];item=comparison[name]['groups']['primary']['joint_rmse_rad']
        fit_counts.append(len(record['fit_families']));means.append(item['mean']*1000)
        lows.append(max(0,item['mean']-item['ci95'][0])*1000);highs.append(max(0,item['ci95'][1]-item['mean'])*1000)
    axes[0,1].errorbar(fit_counts,means,yerr=[lows,highs],fmt='o-',capsize=3,color='#8f3b73')
    axes[0,1].set_xticks(fit_counts);axes[0,1].set_xlabel('Families contributing gradient updates (7/14/28/56)')
    axes[0,1].set_ylabel('Joint RMSE (mrad)');axes[0,1].set_title('Shared model learning curve; fixed final cohort')
    for lead in [0,20]:
        directory=out/f'convergence-expanded-lead-{lead}'
        if not (directory/'convergence.json').exists():directory=out/f'convergence-lead-{lead}'
        values=json.loads((directory/'convergence.json').read_text())
        teachers=json.loads((directory/'teacher_manifest.json').read_text())
        baseline=sum(m['baseline_rmse'] for m in teachers)/len(teachers)
        axes[1,0].plot(range(9),[baseline*1000]+[row['mean_proposal_rmse']*1000 for row in values],'o-',label=f'{lead*2} ms lead')
    axes[1,0].set_xlabel('ILC proposal iteration');axes[1,0].set_ylabel('Mean proposed-run joint RMSE (mrad)')
    axes[1,0].set_title('Training-only convergence; all proposals retained');axes[1,0].legend()
    for i,name in enumerate(['shared_matched','per_3_matched','per_4_matched','shared_full','per_3_full','per_4_full']):
        item=models[name]
        axes[1,1].scatter(i,item['validation_mse'],marker='o',color='#175e86',label='MLP' if i==0 else None)
        axes[1,1].scatter(i,item['zero_validation_mse'],marker='x',color='#c46922',label='zero' if i==0 else None)
        axes[1,1].scatter(i,item['copy_previous_validation_mse'],marker='^',color='#4f8436',label='copy previous' if i==0 else None)
    axes[1,1].set_xticks(range(6),['shared\nmatched','3-joint\nmatched','4-joint\nmatched','shared\nfull','3-joint\nfull','4-joint\nfull'])
    axes[1,1].set_ylabel('Validation MSE normalized by joint correction bound')
    axes[1,1].set_title('Offline prediction error; not a closed-loop success test');axes[1,1].legend()
    for axis in axes.flat:axis.grid(alpha=.2)
    figure.suptitle('* Full-range PI is a separate exploratory robustness analysis; see protocol',fontsize=10)
    target=out/'comparison.png';figure.savefig(target,dpi=170);plt.close(figure)
    return target

if __name__=='__main__':
    import sys
    print(plot(Path(sys.argv[1])))
