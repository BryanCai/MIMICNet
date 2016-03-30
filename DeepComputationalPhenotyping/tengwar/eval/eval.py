import numpy as np
import sys

from pandas import DataFrame
from sklearn.metrics import accuracy_score, roc_auc_score

# 
def compute_performance_metrics(Y, Yd, trix, teix, ylist, yclist=None):
    labels = ['All']
    sys.stdout.write('Computing performance across all...')
    sys.stdout.flush()
    Mtr = [ roc_auc_score(Y[trix], Yd[trix], average='macro') ]
    mtr = [ roc_auc_score(Y[trix], Yd[trix], average='micro') ]
    Mte = [ roc_auc_score(Y[teix], Yd[teix], average='macro') ]
    mte = [ roc_auc_score(Y[teix], Yd[teix], average='micro') ]
    F   = [ Y.mean() ]
    sys.stdout.write('DONE!\n')
    
    cat_ix = np.zeros(ylist.shape,dtype=bool)
    if yclist is not None:
        ylist = np.hstack([ylist,yclist])
        cat_ix = np.hstack([cat_ix, np.ones(yclist.shape, dtype=bool)])
    
    if np.any(cat_ix):
        sys.stdout.write('Computing performance for labels, categories...')
        sys.stdout.flush()
        labels.extend(['Labels', 'Categories' ])
        Mtr.extend([ roc_auc_score(Y[trix][:,~cat_ix], Yd[trix][:,~cat_ix], average='macro'),
                     roc_auc_score(Y[trix][:,cat_ix], Yd[trix][:,cat_ix], average='macro') ])
        mtr.extend([ roc_auc_score(Y[trix][:,~cat_ix], Yd[trix][:,~cat_ix], average='micro'),
                     roc_auc_score(Y[trix][:,cat_ix], Yd[trix][:,cat_ix], average='micro') ])
        Mte.extend([ roc_auc_score(Y[teix][:,~cat_ix], Yd[teix][:,~cat_ix], average='macro'),
                     roc_auc_score(Y[teix][:,cat_ix], Yd[teix][:,cat_ix], average='macro') ])
        mte.extend([ roc_auc_score(Y[teix][:,~cat_ix], Yd[teix][:,~cat_ix], average='micro'),
                     roc_auc_score(Y[teix][:,cat_ix], Yd[teix][:,cat_ix], average='micro') ])
        F.extend([ Y[:,~cat_ix].mean(), Y[:,cat_ix].mean() ])
        sys.stdout.write('DONE!\n')
    
    
    labels.extend([ 'Label {0:4d}'.format(l) for l in ylist[~cat_ix] ])
    if np.any(cat_ix):
        labels.extend([ 'Category {0:2d}'.format(l) for l in ylist[cat_ix] ])
    
    sys.stdout.write('Computing individual performances...')
    sys.stdout.flush()
    
    Mtr.extend([None] * Y.shape[1])
    mtr.extend(roc_auc_score(Y[trix], Yd[trix], average=None))
    Mte.extend([None] * Y.shape[1])
    mte.extend(roc_auc_score(Y[teix], Yd[teix], average=None))
    F.extend(Y.mean(axis=0))
    sys.stdout.write('DONE!\n')
    
    df = DataFrame(data={ 'MacroAucTraining': Mtr,
                          'MacroAucTest': Mte,
                          'MicroAucTraining': mtr,
                          'MicroAucTest': mte,
                          'Frequency': F}, index=labels)
    return df

def do_per_frame_performance(model, X, Y, trix, teix, ydlist, yclist=None):
    Yd = model.decision_function(X)
    return compute_performance_metrics(Y, Yd, trix, teix, ydlist, yclist)

# evaluate classification accuracy per esisode
# combining results from each frame that belongs to this episode
def do_per_episode_combined_performance(model, Ep, EpE, X, YE, trixE, teixE, ydlist, yclist=None, cb='mean'):
    Yd = model.decision_function(X)
    if cb == 'median':
        YdE = np.vstack([ np.median(Yd[Ep==e],axis=0) for e in EpE ])
    elif cb == 'max':
        YdE = np.vstack([ Yd[Ep==e].max(axis=0) for e in EpE ])
    else: # by default we use mean
        YdE = np.vstack([ Yd[Ep==e].mean(axis=0) for e in EpE ])
    
    return compute_performance_metrics(YE, YdE, trixE, teixE, ydlist, yclist)

def make_firstN_indeces(Ep, N=3):
    Ix = []
    EpE = np.unique(Ep)
    for i,e in enumerate(EpE):
        ix = list(np.where(Ep==e)[0][0:N])
        if len(ix) < N:
            ix.extend([ix[-1]] * (N-len(ix)))
        Ix.extend(ix)
    Ix = np.array(Ix)
            
    return Ix



