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
    Fast volatility forecasting using only Garman-Klass estimator.
    Uses OHLC prices for more efficient volatility estimation.
    """
    vol_forecasts = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    
    for asset in returns_df.columns:
        # Extract OHLC data for current asset
        o = prices_df[asset]['Open']
        h = prices_df[asset]['High']
        l = prices_df[asset]['Low']
        c = prices_df[asset]['Close']
        
        # Calculate rolling Garman-Klass volatility
        gk_vol = garman_klass_volatility(
            h.to_frame(),
            l.to_frame(),
            o.to_frame(),
            c.to_frame()
        )
        
        # Convert to series and shift by 1 to avoid lookahead bias
        vol_forecasts[asset] = gk_vol.iloc[:, 0].shift(1)
    
    return vol_forecasts.astype(float)

def garman_klass_volatility(high: pd.DataFrame, low: pd.DataFrame, 
                           open_: pd.DataFrame, close: pd.DataFrame, 
                           scale: float = 252) -> pd.DataFrame:
    """
    Implements the Garman-Klass volatility estimator using OHLC prices.
    This estimator is 7.4 times more efficient than close-to-close volatility.
    """
    # Debug prints
    print("High shape:", high.shape)
    print("High head:", high.head())
    
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

# Fix the data loading - using the first row as headers
prices = pd.read_csv('prices_insample.csv', header=[0,1], skiprows=[2])  # Skip the 'AsOfDate' row
prices.index = pd.to_datetime(prices.iloc[:, 0])
prices = prices.iloc[:, 1:]  # Remove the date column since it's now the index

# Clean up the column structure
# First level should be asset names, second level should be OHLC
assets = [col[0] for col in prices.columns if 'Unnamed' not in col[0]]
assets = list(dict.fromkeys(assets))  # Remove duplicates

# Restructure the DataFrame
price_data = {}
for asset in assets:
    price_data[asset] = {
        'Open': prices[asset]['Open'],
        'High': prices[asset]['High'],
        'Low': prices[asset]['Low'],
        'Close': prices[asset]['Close']
    }

print("price_data", price_data)

# Convert to multi-level DataFrame
prices = pd.concat({k: pd.DataFrame(v) for k, v in price_data.items()}, axis=1)

# Calculate returns using Close prices
ret = prices.xs('Close', axis=1, level=1).pct_change()

# Debug: Print a sample of the data to verify structure
# print("\nSample of returns:")
# print(ret.head())
# print("\nSample of prices:")
# print(prices.head())

# WARNING: THESE TAKE A LONG TIME TO RUN
# vols = forecast_volatility(lookback=200, returns_df=ret)
# vols_short = forecast_volatility(lookback=200, returns_df=ret.iloc[:-200])

vols_enhanced = forecast_volatility_enhanced(
    lookback=200,
    returns_df=ret,
    prices_df=prices  # Your multi-level DataFrame with OHLC data
)

print("VOLS ENHANCED", vols_enhanced)

# vols_enhanced_short = forecast_volatility_enhanced(
#     lookback=200,
#     returns_df=ret.iloc[:-200],
#     prices_df=prices.iloc[:-200]
# )
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

# with open('enhanced_vols_short.pkl', 'wb') as f:
#     pickle.dump(vols_enhanced_short, f)

with open('enhanced_vols.pkl', 'rb') as f:
    enhanced_loaded_vols = pickle.load(f)

print(enhanced_loaded_vols)

with open('enhanced_vols_short.pkl', 'rb') as f:
    enhanced_loaded_vols_short = pickle.load(f)
























# add loaded_vols to csv
# loaded_vols.to_csv('garch_vols.csv')


# trend_window=50
# sig = np.sign(ret.rolling(window=trend_window).sum())
# sig_short = np.sign(ret.iloc[:-200].rolling(window=trend_window).sum())


# pos = sig/loaded_vols
# pos_short = sig_short/loaded_vols_short

# temp = (pos-pos_short).abs().sum()
# if temp.sum() > 0:
#     print('Forward looking bias detected!')
# else:
#     print('No forward looking bias detected!')