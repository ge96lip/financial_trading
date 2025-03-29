# garch_volatility.py

import numpy as np
import pandas as pd
from arch import arch_model
import pickle

def forecast_volatility(lookback, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate GARCH(1,1) volatility forecasts for each asset.
    Returns a DataFrame with forecasted volatilities (sigma), not variances.
    Returns predicted volatilities for day t using data from day t-lookback to t-1
    """
    vol_forecasts = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)

    for t in range(lookback + 1, len(returns_df)):
        print("in loop")
        for asset in returns_df.columns:
            hist_returns = returns_df[asset].iloc[t - lookback:t]
            try:
                model = arch_model(hist_returns, vol='GARCH', p=1, q=1, rescale=False)
                res = model.fit(disp='off', options={'maxiter': 1000}, method='BFGS')
                forecast = res.forecast(horizon=1)
                vol = np.sqrt(forecast.variance.values[-1, 0])
            except:
                vol = hist_returns.std()
            vol_forecasts.loc[returns_df.index[t], asset] = vol            
    return vol_forecasts.astype(float)

def forecast_volatility_enhanced(lookback: int, 
                               returns_df: pd.DataFrame,
                               prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced GARCH model using Garman-Klass volatility as external regressor.
    Handles multi-level DataFrame with OHLC prices per asset.
    
    Parameters:
    -----------
    lookback : int
        Number of periods to use for volatility calculation
    returns_df : pd.DataFrame
        DataFrame of asset returns
    prices_df : pd.DataFrame
        Multi-level DataFrame with 'Open', 'High', 'Low', 'Close' columns per asset
    """
    vol_forecasts = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)

    for t in range(lookback + 1, len(returns_df)):
        for asset in returns_df.columns:
            hist_returns = returns_df[asset].iloc[t - lookback:t]
            
            # Extract OHLC data for current asset
            o = prices_df[asset]['Open'].iloc[t - lookback:t]
            h = prices_df[asset]['High'].iloc[t - lookback:t]
            l = prices_df[asset]['Low'].iloc[t - lookback:t]
            c = prices_df[asset]['Close'].iloc[t - lookback:t]
            
            # Calculate GK volatility as external regressor
            gk_vol = garman_klass_volatility(
                h.to_frame(),
                l.to_frame(),
                o.to_frame(),
                c.to_frame()
            )
            
            try:
                # Use GK volatility as external regressor
                model = arch_model(hist_returns, vol='GARCH', p=1, q=1,
                                 x=gk_vol, rescale=False)
                res = model.fit(disp='off', options={'maxiter': 1000}, method='BFGS')
                forecast = res.forecast(horizon=1)
                vol = np.sqrt(forecast.variance.values[-1, 0])
            except:
                vol = gk_vol.iloc[-1, 0]  # Fallback to GK
            
            vol_forecasts.loc[returns_df.index[t], asset] = vol            
    return vol_forecasts.astype(float)

def garman_klass_volatility(high: pd.DataFrame, low: pd.DataFrame, 
                           open_: pd.DataFrame, close: pd.DataFrame, 
                           scale: float = 252) -> pd.DataFrame:
    """
    Implements the Garman-Klass volatility estimator using OHLC prices.
    This estimator is 7.4 times more efficient than close-to-close volatility.
    
    Parameters:
    -----------
    high : pd.DataFrame
        High prices
    low : pd.DataFrame
        Low prices
    open_ : pd.DataFrame
        Opening prices
    close : pd.DataFrame
        Closing prices
    scale : float
        Annualization factor (252 for daily data)
    
    Returns:
    --------
    pd.DataFrame
        Estimated volatilities
    """
    # Calculate log differences
    log_hl = (high / low).apply(np.log)
    log_co = (close / open_).apply(np.log)
    
    # Garman-Klass estimator
    vol = np.sqrt(
        scale * (
            0.5 * log_hl**2 -  # High-Low component
            (2 * np.log(2) - 1) * log_co**2  # Close-Open component
        )
    )
    
    return vol

# Fix the data loading
prices = pd.read_csv('prices_insample.csv', index_col=0, skiprows=1, header=[0,1])

# Create proper multi-index columns
prices.columns = prices.columns.str.split(',', expand=True)
prices.index = pd.to_datetime(prices.index)

# Calculate returns using Close prices
ret = prices.xs('Close', axis=1, level=1).pct_change()

# WARNING: THESE TAKE A LONG TIME TO RUN
vols = forecast_volatility(lookback=200, returns_df=ret)
vols_short = forecast_volatility(lookback=200, returns_df=ret.iloc[:-200])

vols_enhanced = forecast_volatility_enhanced(
    lookback=200,
    returns_df=ret,
    prices_df=prices  # Your multi-level DataFrame with OHLC data
)

vols_enhanced_short = forecast_volatility_enhanced(
    lookback=200,
    returns_df=ret.iloc[:-200],
    prices_df=prices.iloc[:-200]
)

# ORIGINAL VOLS - NOT ENHANCED
# with open('vols.pkl', 'wb') as f:
#     pickle.dump(vols, f)

# with open('vols_short.pkl', 'wb') as f:
#     pickle.dump(vols_short, f)

# with open('vols.pkl', 'rb') as f:
#     loaded_vols = pickle.load(f)

# with open('vols_short.pkl', 'rb') as f:
#     loaded_vols_short = pickle.load(f)





# ENHANCED VOLS
with open('enhanced_vols.pkl', 'wb') as f:
    pickle.dump(vols_enhanced, f)

with open('enhanced_vols_short.pkl', 'wb') as f:
    pickle.dump(vols_enhanced_short, f)

with open('enhanced_vols.pkl', 'rb') as f:
    enhanced_loaded_vols = pickle.load(f)

with open('enhanced_vols_short.pkl', 'rb') as f:
    enhanced_loaded_vols_short = pickle.load(f)
























# add loaded_vols to csv
# loaded_vols.to_csv('garch_vols.csv')


trend_window=50
sig = np.sign(ret.rolling(window=trend_window).sum())
sig_short = np.sign(ret.iloc[:-200].rolling(window=trend_window).sum())


pos = sig/loaded_vols
pos_short = sig_short/loaded_vols_short

temp = (pos-pos_short).abs().sum()
if temp.sum() > 0:
    print('Forward looking bias detected!')
else:
    print('No forward looking bias detected!')