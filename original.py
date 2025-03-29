import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from evaluation import plot_key_figures, calc_key_figures
import pickle

from risk_algos import rolling_portfolio_vol


    
def trend_model_vectorized_risk_adjusted(ret, vol, trend_window=240, risk_window=140):
    
    # calculate signal
    sig = np.sign(ret.rolling(window=trend_window).sum())
    # scale with inverse vol to get pos
    pos = sig/vol
    # calculate model risk
    # adjust with model risk
    # model_risk = (pos.shift(2)*ret).dropna(how='all').sum(axis=1).rolling(risk_window, min_periods=20).std()
    model_returns = (pos.shift(2) * ret).dropna(how='all').sum(axis=1)

    short_term_risk = (
        model_returns
        .ewm(span=risk_window//2, min_periods=10)  # Shorter window for recent volatility
        .std()
        .clip(lower=1e-8)
    )
    
    long_term_risk = (
        model_returns
        .ewm(span=risk_window*2, min_periods=20)  # Longer window for structural volatility
        .std()
        .clip(lower=1e-8)
    )
    
    # Combine both risk measures with more weight on short-term risk
    original_model_risk = 0.4 * short_term_risk + 0.6 * long_term_risk
    pos_adjusted = pos.div(original_model_risk, axis=0)

    return pos_adjusted



def forward_looking_bias(ret, vol, vol_short): 
    pos = trend_model_vectorized_risk_adjusted(ret, vol)
    pos_short = trend_model_vectorized_risk_adjusted(ret.iloc[:-200], vol_short)
    temp = (pos-pos_short).abs().sum()
    if temp.sum() > 0:
        print('Forward looking bias detected!')
    else:
        print('No forward looking bias detected!')
    


def main(): 
    #global parameters

    prices = pd.read_csv('close_prices_insample.csv',index_col='AsOfDate',parse_dates=True)
    ret = prices.diff()

    #volatility metrics
    with open('vols.pkl', 'rb') as f:
        loaded_vols = pickle.load(f)

    with open('vols_short.pkl', 'rb') as f:
        loaded_vols_short = pickle.load(f)


    # model risk 

    # original_vol = np.sqrt((ret**2).rolling(window=vol_window, min_periods=vol_window//2).sum())


    pos = trend_model_vectorized_risk_adjusted(ret, loaded_vols)
    plot_key_figures(pos, prices)
    figures = (calc_key_figures(pos, prices))
    print(figures)
    forward_looking_bias(ret, loaded_vols, loaded_vols_short)


    
if __name__ == "__main__":
    main()

