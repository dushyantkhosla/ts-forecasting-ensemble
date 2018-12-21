import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

from ..obtain import *
from ..scrub import *
from ..explore import *
from ..model import *

dict_ISO_to_pmiMarket = {v:k for k, v in dict_pmiMarket_to_ISO.items()}
dict_ypred = {}

def fit_SCurve(mkt, till, conf_ints, bool_return, bool_plot, kwargs):
    '''
    Fit an Logistic Curve to the data using `scipy.optimize.curve_fit`
    The function's domain is (-5, 5) split over (beginning of series, timestamp provided via `till`)
    '''
    global dict_ypred
    
    dict_pmiMarket_to_ISO = kwargs['dict_pmiMarket_to_ISO']
    df_som = kwargs['df_som']
    func_logit = kwargs['func_logit']
    
    mkt = dict_pmiMarket_to_ISO[mkt]
    
    y = (df_som
         .loc[:, mkt]
         .copy()
         .dropna())
    
    till = pd.to_datetime(str(till))
    IDX = pd.date_range(start=y.index.min(), 
                        end=till, 
                        freq='W')
    
    Xs = np.linspace(-5, 5, len(IDX))
    
    # Finding parameters of the logistic function via least-squares
    (a, b, c), _ = \
    curve_fit(func_logit, 
              xdata=Xs[:len(y)], 
              ydata=y,
              p0=[0, 1, 1],
              bounds=((-1., y.max(), .1), 
                      (1., 1., 2.))
             )

    y_pr = pd.Series(func_logit(x=Xs, xo=a, L=b, k=c), index=IDX)
    
    dict_ypred[till.year] = y_pr
    if mkt not in dict_ypred.keys():
        dict_ypred[mkt] = y
    
    # Quality of fit
    rmse = np.sqrt(mean_squared_error(y, y_pr.iloc[:len(y)]))
    
    if bool_return:
        return {
            'ypred': y_pr,
            'rmse': rmse,
            'year': till.year,
            'L': b,
            'k': c,
            'x0': a
        }
       
    if bool_plot:
        # Plot actual, predicted, and optionally, confidence-intervals
        ax = y.plot(style='k--', 
                    linewidth=4,
                    figsize=(15, 6), 
                    ylim=(0, 1),
                    label='Actual SoM')
    
        y_pr.plot(ax=ax, 
                  label='Fitted SoM')
        
        ax.hlines(y=y_pr.max(), 
                  xmin=IDX.min(), 
                  xmax=IDX.max(),  
                  color='darkgray')
        ax.set_title(f'---- {dict_ISO_to_pmiMarket.get(mkt)} ---- \n RMSE = {round(rmse, 3)} \nProjected SoM for {till.strftime("%Y-%b")} is {int(100 * y_pr.max())}%')
        ax.set(ylabel='Actual Share of Market')
        ax.legend()
        ax.set_yticklabels([f'{int(100 * x)}%' for x in ax.get_yticks().tolist()])
        
        if conf_ints:
            Series(func_logit(x=Xs, xo=a-.3, L=b - (.15 * b) , k=c), index=IDX).plot(ax=ax, alpha=.35, label='Pessimistic (15% lower)', color='darkred', style='-.')
            Series(func_logit(x=Xs, xo=a-.3, L=b - (.25 * b) , k=c), index=IDX).plot(ax=ax, alpha=.35, label='Highly Pessimistic (25% lower)', color='red', style='-.')
            Series(func_logit(x=Xs, xo=a, L=b + (.10 * b) , k=c), index=IDX).plot(ax=ax, alpha=.35, label='Optimistic (10% higher)', color='darkgreen', style='-.')

        format_plot(ax)    
    
        if till.year >= 2025:
            df_ = pd.DataFrame(dict_ypred)
            ax = df_.drop(mkt, axis=1).plot(figsize=(15, 4), title='All Predictions', alpha=0.7)
            df_[mkt].plot(ax=ax, style='k--', linewidth=4)
            ax.set(ylabel='Actual Share of Market')
            ax.legend()
            ax.set_yticklabels([f'{int(100 * x)}%' for x in ax.get_yticks().tolist()])
            format_plot(ax)
            dict_ypred = {}
            
