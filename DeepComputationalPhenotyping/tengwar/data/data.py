from __future__ import division

import cPickle
import numpy as np
import scipy
import sys

from itertools import chain
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit
from scipy.stats import norm


def make_theano_shared(X, y=None, borrow=True):
    import theano
    import theano.tensor as TT

    Xsh = theano.shared(np.asarray(X, dtype=theano.config.floatX),
                        borrow=borrow)
    if y is not None:
        ysh = TT.cast(theano.shared(np.asarray(y, dtype=theano.config.floatX),
                                    borrow=borrow), 'int32')
    else:
        ysh = None

    return Xsh, ysh

def get_first_measurement(x, t=None, ranges=None):
    xfirst = np.zeros((x.shape[1],))
    xfirst[:] = np.nan
    for vi,xv in enumerate(x.T):
        if np.any(~np.isnan(xv)):
            xfirst[vi] = xv[~np.isnan(xv)][0]
    if ranges is not None:
        xfirst[np.isnan(xfirst)] = ranges[np.isnan(xfirst),2]
    return xfirst

def get_last_measurement(x, t=None, ranges=None):
    xlast = np.zeros((x.shape[1],))
    xlast[:] = np.nan
    for vi,xv in enumerate(x.T):
        if np.any(~np.isnan(xv)):
            xlast[vi] = xv[~np.isnan(xv)][-1]
    if ranges is not None:
        xlast[np.isnan(xlast)] = ranges[np.isnan(xlast),2]
    return xlast

def get_trend(x, t=None, ranges=None):
    if t is None:
        t = np.array(x.shape[0])
    t = t-t.min()
    w = np.zeros((2,x.shape[1]))
    w[:,:] = np.nan
    for vi in np.arange(x.shape[1]):
        tv = t[~np.isnan(x[:,vi])]
        xv = x[~np.isnan(x[:,vi]),vi]
        if xv.shape[0] > 0:
            wv,_,_,_ = np.linalg.lstsq(np.vstack((tv,np.ones(tv.shape))).T,xv)
            w[:,vi] = wv
            #if xv.shape[0] == 1:
            #    print 'WV!', vi, tv, xv, wv
    if ranges is not None:
        w[0,np.isnan(w)] = 0
        w[1,np.isnan(w)] = ranges[np.isnan(w),2]
    return w

def get_basic_classification_features(x, t=None, ranges=None):
    xfeat = np.vstack( ( get_first_measurement(x, ranges=ranges),
                         get_last_measurement(x, ranges=ranges),
                         np.nanmean(x, axis=0),
                         np.nanstd(x, axis=0),
                         np.nanmin(x, axis=0),
                         np.nanmedian(x, axis=0),
                         np.nanmax(x, axis=0),
                         get_trend(x, t=t, ranges=ranges)[0,:] ) )
    if ranges is not None:
        xfeat[np.isnan(xfeat),[2,4,5,6]] = np.tile(ranges[np.isnan(xfeat),2], (4,1)).T
        xfeat[np.isnan(xfeat),[3]] = 0
    return xfeat

def get_raw(x, t=None, ranges=None):
    return x

def get_resampling_features(x, t=None, ranges=None):
    return np.vstack( (np.nanmean(x, axis=0),) )

def extract_features_with_sliding_window(x, fn, timestamps=None, max_length=None, width=1, stride=1,
                                         normal=None, impute=False):
    if timestamps is None:
        timestamps = np.arange(x.shape[0])
    
    if max_length is None or max_length <= 0:
        max_length = timestamps.max()+1
    
    if impute:
        xprev = get_first_measurement(x)
        if normal is not None:
            xprev[np.isnan(xprev)] = normal[np.isnan(xprev)]
    else:
        xprev = np.zeros((x.shape[1],))
        xprev[:] = np.nan
    xfeat = []
    xdummy = np.zeros((1,x.shape[1]))
    xdummy[:] = np.nan
    
    t = 0
    while t <= max_length-width:
        ix = (timestamps>=t)&(timestamps<t+width)
        if np.any(ix):
            xt = x[ix,:]
            tt = timestamps[ix]
        else:
            xt = xdummy.copy()
            tt = np.array([t])
        xlst = get_last_measurement(xt)
        
        missix = np.all(np.isnan(xt), axis=0)                
        if impute:
            xt[0,missix] = xprev[missix]
            #missix = np.all(np.isnan(xt), axis=0)
            #if ranges is not None:
            #    xt[0,missix] = normal[missix,2]

        xprev[~missix] = xlst[~missix]                
        xf = fn(xt, t=tt)
        
        xfeat.append(xf)
        t += stride

    return np.dstack(xfeat)

def generate_splits_for_episodes(Y, num_splits=10, minpos=0):
    se = []

    sys.stdout.write('Generating {0} training ({1}), valid/test ({2}) splits'.format(num_splits,
                                                                                     (num_splits-2)/num_splits,
                                                                                     1/num_splits))
    while (len(se) < num_splits):
        for not_teix,teix in ShuffleSplit(Y.shape[0], n_iter=2*num_splits, test_size=1/num_splits):
            sys.stdout.write('_')
            sys.stdout.flush()
            assert(np.all(np.sort(np.hstack((not_teix,teix))) == np.arange(Y.shape[0])))
            if np.all(Y[not_teix,:].sum(axis=0)>minpos) and np.all(Y[teix,:].sum(axis=0)>minpos):
                for trix,vix in ShuffleSplit(Y[not_teix].shape[0], n_iter=2*num_splits, test_size=1/num_splits):
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    assert(np.all(np.sort(np.hstack((trix,vix))) == np.arange(Y[not_teix].shape[0])))
                    if np.all(Y[not_teix,:][trix,:].sum(axis=0)>minpos) and np.all(Y[not_teix,:][vix,:].sum(axis=0)>minpos):
                        se.append((np.sort(not_teix[trix]), np.sort(not_teix[vix]), np.sort(teix)))
                        sys.stdout.write(str(len(se)))
                        sys.stdout.flush()
                        break

    return se[0:num_splits]
