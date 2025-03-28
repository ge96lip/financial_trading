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

    RETURNS SHOULD BE SHIFTED BY 2 DAYS AHEAD!!
    """
    vol_forecasts = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)

    for t in range(lookback + 1, len(returns_df)):
        print("in loop")
        for asset in returns_df.columns:
            hist_returns = returns_df[asset].iloc[t - lookback:t]
            try:
                model = arch_model(hist_returns, vol='GARCH', p=1, q=1, rescale=False)
                res = model.fit(disp='off')
                forecast = res.forecast(horizon=1)
                vol = np.sqrt(forecast.variance.values[-1, 0])
            except:
                vol = hist_returns.std()
            vol_forecasts.loc[returns_df.index[t], asset] = vol            
    return vol_forecasts.astype(float)


prices = pd.read_csv('example_prices.csv',index_col='dates',parse_dates=True)
ret = prices.diff()

# WARNING: THESE TAKE A LONG TIME TO RUN
# vols = forecast_volatility(lookback=200, returns_df=ret)
# vols_short = forecast_volatility(lookback=200, returns_df=ret.iloc[:-200])


# I HAVE SAVED VOLS (NOT SHORT) SO WE DONT HAVE TO RUN THE FORECASTING EVERY TIME
# Save the variable to a file
# with open('vols.pkl', 'wb') as f:
#     pickle.dump(vols, f)

# Later, load the variable from the file
with open('vols.pkl', 'rb') as f:
    loaded_vols = pickle.load(f)


trend_window=50
sig = np.sign(ret.rolling(window=trend_window).sum())
sig_short = np.sign(ret.iloc[:-200].rolling(window=trend_window).sum())


pos = sig/loaded_vols
# pos_short = sig_short/vols_short

# temp = (pos-pos_short).abs().sum()
# if temp.sum() > 0:
#     print('Forward looking bias detected!')
# else:
#     print('No forward looking bias detected!')


# OBS IT SAID NO FORWARD LOOKING BIAS




# if given highs and lows

def parkinson_volatility(high: pd.DataFrame, low: pd.DataFrame, scale=252) -> pd.DataFrame:
    """
    Parkinson volatility estimator using high-low range
    More efficient than close-to-close estimator
    """
    log_hl = np.log(high/low)
    vol = np.sqrt(scale/(4 * np.log(2)) * log_hl**2)
    return vol