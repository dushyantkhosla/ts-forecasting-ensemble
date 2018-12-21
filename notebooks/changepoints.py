import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from ..obtain import *
from ..scrub import *
from ..explore import *
from ..model import *

def regress(y_, intercept):
    '''
    Fit a linear regression and return the r-squared, rmse and predictions
    '''
    y = y_.copy().values
    X = np.arange(len(y)).reshape(-1, 1)

    lr = LinearRegression(fit_intercept=intercept)
    lr.fit(X, y)
    
    r_sq = lr.score(X, y)
    y_pr = lr.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pr))
    
    result = {
        'r_squared': round(r_sq, 3),
        'y_predicted': pd.Series(y_pr, index=y_.index, name='predicted'),
        'RMSE': round(rmse, 3),
        'slope': round(lr.coef_[0], 3)
    }
    return result


def get_rmse_ypr(y_, pt, intercept):
    '''
    Splits the input series at the provided point
    Fits regress() on each part 
    
    Returns the combined RMSE, concatenated predictions
    '''
    reg_one, reg_two = regress(y_.loc[:pt].copy(), intercept=intercept), regress(y_.loc[pt:].copy(), intercept=intercept)

    y_pred = pd.concat([
        reg_one.get('y_predicted'),
        reg_two.get('y_predicted')
    ])

    result = {
        'RMSE': reg_one.get('RMSE') + reg_two.get('RMSE'),
        'y_pr': y_pred,
        'slope_1': reg_one.get('slope'),
        'slope_2': reg_two.get('slope')
    }
    return result


def get_minima(srs_, splits_, limit_time=True, select_last_n_weeks=32):
    '''
    Split the Series into given number of parts
    Ignore the 1st and last
    Split the remaining into a search space
    Treat each point in this space as a change-point and find the total RMSE
    Return the change-point that minimizes overall RMSE
    '''
    try:        
        if limit_time:
            #search_space = (srs_
            #                .iloc[-select_last_n_weeks:]
            #                .index
            #                .tolist())
            srs_ = (srs_.iloc[-select_last_n_weeks:])

        split_points = np.linspace(start=0, stop=len(srs_), num=splits_).astype(int)[1:-1]
        search_space = (srs_
                        .iloc[np.arange(split_points[0], 
                                        split_points[-1], 
                                        int(split_points[-1]/split_points[0]))]
                        .index
                        .tolist())

        pt_minima = pd.Series({pt:get_rmse_ypr(srs_, pt, intercept=True).get('RMSE') \
                               for pt in search_space}).idxmin()
        return pt_minima
    except Exception as e:
        print(f"Failed to find minima! {e}")
        
        
def change_points(MKT_, LAG, bool_plot, bool_return, kwargs):
    '''
    '''
    try:
        MKT = dict_pmiMarket_to_ISO.get(MKT_)

        srs_som = (kwargs['df_som']
                   .loc[:, MKT]
                   .copy()
                   .where(lambda i: i > 0)
                   .dropna()
                   .ewm(alpha=0.3)
                   .mean())

        pt_minima_som = get_minima(srs_=srs_som, splits_=5)
        slope_som_1, slope_som_2 = [get_rmse_ypr(y_=srs_som, pt=pt_minima_som, intercept=True).get(i) for i in ['slope_1', 'slope_2']]

        srs_gtr = (kwargs['df_gtr']
                   .loc[:, MKT]
                   .copy()
                   #.ewm(alpha=0.3)
                   #.mean()
                   .rolling(window=kwargs['filter_window']).median()
                   .shift(LAG)
                   .loc[srs_som.index[0]:pt_minima_som]
                   .dropna())

        pt_minima_gtr = get_minima(srs_=srs_gtr, splits_=5)
        slope_gtr_1, slope_gtr_2 = [get_rmse_ypr(y_=srs_gtr, pt=pt_minima_gtr, intercept=True).get(i) for i in ['slope_1', 'slope_2']]

        days_between = pt_minima_som - pt_minima_gtr


        if bool_plot:
            _, axs = plt.subplots(nrows=2, figsize=(16, 18), sharex=True)
            ax = axs[0]
            (get_rmse_ypr(srs_som, pt_minima_som, intercept=True)
                  .get('y_pr')
                  .plot(label='Fitted', 
                        style='-.', 
                        ax=ax)
                 )
            srs_som.plot(ax=ax, label='Actual SoM', linewidth=4)
            ax.vlines(x=pt_minima_som, ymin=srs_som.min(), ymax=srs_som.max(), linestyle='--', label='Change point')
            ax.set(title=f"""
            =====  {MKT_}  =====
            Change in SoM starts in {pt_minima_som.strftime('%Y-%b')}
            {days_between.days} days between SoM and Gtr changepoints""", 
                   ylabel='Share of Market [%]', 
                   xlabel='Date')
            ax.legend() 
            format_plot(ax)

        if bool_plot:
            ax = axs[1]
            ax.set_xlim([srs_som.index.min(), srs_som.index.max()])
            (get_rmse_ypr(srs_gtr, pt_minima_gtr, intercept=True)
                  .get('y_pr')
                  .plot(label='Fitted', 
                        style='-.', 
                        ax=ax,
                        color='peru')
                 )
            srs_gtr.plot(ax=ax, label='Google Trend Score', linewidth=4, color='seagreen')
            ax.vlines(x=pt_minima_gtr, ymin=srs_gtr.min(), ymax=srs_gtr.max(), linestyle='--', label='Change point', color='indigo')
            ax.set(ylabel='Google Trends [0-100]', 
                   xlabel='Date', 
                   title=f"""Ratio SoM Slopes: {round(slope_som_2/slope_som_1, 3)}\nRatio Gtr Slopes: {round(slope_gtr_2/slope_gtr_1, 3)}""")
            ax.legend() 
            format_plot(ax)

        if bool_return:
            return {
                'market': MKT_,
                'lag': LAG,
                'change_pt_SOM': pt_minima_som,
                'change_pt_GTR': pt_minima_gtr,
                'slope_som_1': slope_som_1,
                'slope_som_2': slope_som_2,
                'slope_gtr_1': slope_gtr_1,
                'slope_gtr_2': slope_gtr_2
            }
    except:
        print(f"No data to show for {MKT}")
        
