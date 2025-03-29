import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from evaluation import plot_key_figures, calc_key_figures

    
def trend_model_vectorized_risk_adjusted(ret, trend_window=50, vol_window=100, risk_window=100):
    # calculate vol
    vol = np.sqrt((ret**2).rolling(window=vol_window, min_periods=vol_window//2).sum())
    # calculate signal
    sig = np.sign(ret.rolling(window=trend_window).sum())
    # scale with inverse vol to get pos
    pos = sig/vol
    # calculate model risk
    model_risk = (pos.shift(2)*ret).dropna(how='all').sum(axis=1).rolling(risk_window, min_periods=20).std()
    # adjust with model risk
    pos_adjusted = pos.div(model_risk, axis=0)
    return pos_adjusted

def forward_looking_bias(ret): 
    pos = trend_model_vectorized_risk_adjusted(ret)
    pos_short = trend_model_vectorized_risk_adjusted(ret.iloc[:-20])
    temp = (pos-pos_short).abs().sum()
    if temp.sum() > 0:
        print('Forward looking bias detected!')
    
def main(): 

    prices = pd.read_csv('example_prices.csv', index_col='dates', parse_dates=True)
    ret = prices.diff()
    vol = np.sqrt((ret**2).rolling(window=100, min_periods=10).mean()).shift(1)
    norm_ret = ret/vol
    pos = trend_model_vectorized_risk_adjusted(ret)
    plot_key_figures(pos, prices)
    figures = (calc_key_figures(pos, prices))
    print(figures)
    forward_looking_bias(ret)


    
if __name__ == "__main__":
    main()

