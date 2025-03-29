import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from evaluation import plot_key_figures, calc_key_figures
import pickle

from macros import generate_cpi_signals, generate_gdp_signals


    
def trend_model_vectorized_risk_adjusted(ret, cpi_growth, gdp_growth, trend_window=243, risk_window=20, vol_window=187):
    
    # calculate signal
    sig = np.sign(ret.rolling(window=trend_window).sum())

    # add cpi and gdp signals
    cpi_signals = generate_cpi_signals(cpi_growth)
    gdp_signals = generate_gdp_signals(gdp_growth)

    # add cpi signals to bonds columns
    for col in cpi_signals.columns:
        bonds_col = f"{col}_bonds"
        if bonds_col in sig.columns:
            sig[bonds_col] = sig[bonds_col] + cpi_signals[col]

    # add cpi signals to fx columns
    for col in cpi_signals.columns:
        # Find all columns in sig that contain both the country name and '_fx'
        fx_cols = [c for c in sig.columns if '_fx' in c and (
            f"{col}_fx" in c or  # matches NAME_..._fx
            f"_{col}_fx" in c    # matches ..._NAME_fx
        )]
        for fx_col in fx_cols:
            sig[fx_col] = sig[fx_col] + cpi_signals[col]

    for col in gdp_signals.columns:
        stocks_col = f"{col}_stocks"
        if stocks_col in sig.columns:
            sig[stocks_col] = sig[stocks_col] + gdp_signals[col]
    

    # scale with inverse vol to get pos
    original_vol = np.sqrt((ret**2).rolling(window=vol_window, min_periods=vol_window//2).sum())
    pos = sig/original_vol
    # calculate model risk
    # original below
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



def forward_looking_bias(ret): 
    pos = trend_model_vectorized_risk_adjusted(ret)
    pos_short = trend_model_vectorized_risk_adjusted(ret.iloc[:-200])
    temp = (pos-pos_short).abs().sum()
    if temp.sum() > 0:
        print('Forward looking bias detected!')
    else:
        print('No forward looking bias detected!')
    
def generate_cpi_signals(cpi_growth, lookback_period=30, threshold=0.00):
    """
    Generate trading signals based on CPI growth rates
    - Bearish signal when CPI growth is above threshold (inflation concerns)
    - Bullish signal when CPI growth is below threshold (inflation moderating)
    """
    # Initialize signals DataFrame with same structure as input
    signals = pd.DataFrame(0, index=cpi_growth.index, columns=cpi_growth.columns)
    
    # Calculate rolling average and generate signals for each column
    for column in cpi_growth.columns:
        rolling_cpi = cpi_growth[column].rolling(window=lookback_period).mean()
        signals.loc[rolling_cpi > threshold, column] = -1  # Bearish signal
        signals.loc[rolling_cpi < -threshold, column] = 1  # Bullish signal
    
    return signals

#  

def generate_gdp_signals(gdp_growth, lookback_period=30, threshold=0.00):
    """
    Generate trading signals based on GDP growth rates
    - Bullish signal when GDP growth is above threshold (economic expansion)
    - Bearish signal when GDP growth is below negative threshold (economic contraction)
    """
    # Initialize signals DataFrame with same structure as input
    signals = pd.DataFrame(0, index=gdp_growth.index, columns=gdp_growth.columns)
    
    # Calculate rolling average and generate signals for each column
    for column in gdp_growth.columns:
        rolling_gdp = gdp_growth[column].rolling(window=lookback_period).mean()
        signals.loc[rolling_gdp > threshold, column] = 1     # Bullish signal
        signals.loc[rolling_gdp < -threshold, column] = -1   # Bearish signal
    
    return signals

def main(): 
    #global parameters

    prices = pd.read_csv('close_prices_insample.csv',index_col='AsOfDate',parse_dates=True)
    ret = prices.diff()

    # Import data with multi-level columns
    macro_data = pd.read_csv('macro_data_insample.csv', header=[0, 1], index_col=0)

    # Convert index to datetime
    macro_data.index = pd.to_datetime(macro_data.index)

    # Example of how to access specific data:
    # Get all GDP data
    gdp_data = macro_data['gdp']
    cpi_data = macro_data['cpi']

    gdp_growth = gdp_data.pct_change()

    cpi_growth = cpi_data.pct_change()

    pos = trend_model_vectorized_risk_adjusted(ret, cpi_growth, gdp_growth)
    pos_short = trend_model_vectorized_risk_adjusted(ret.iloc[:-200], cpi_growth.iloc[:-200], gdp_growth.iloc[:-200])
    plot_key_figures(pos, prices)
    figures = (calc_key_figures(pos, prices))
    print(figures)
    

    temp = (pos-pos_short).abs().sum()
    if temp.sum() > 0:
        print('Forward looking bias detected!')
    else:
        print('No forward looking bias detected!')


    
if __name__ == "__main__":
    main()

