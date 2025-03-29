

#implement Markowitz Mean-Variance Model
# IDEA: do not output asset weights (we let the ML conduct the selection), but use this method to calculate the risk of a portfolio
# so this method is meerly emblematic of Makowitz method

import pandas as pd
import numpy as np
import pickle

# Use rolling cov matrix for more accurate risk
def rolling_portfolio_vol(positions, returns, risk_window=100):
    # Multiply shifted positions with returns to get PnL
    pnl = (positions.shift(2) * returns).dropna(how='all')
    covs = returns.rolling(window=risk_window).cov(pairwise=True)
    
    port_vol = []
    for t in pnl.index:
        if t not in covs.index:  # avoid missing cov
            port_vol.append(np.nan)
            continue
        pos_t = positions.loc[t].values.reshape(-1, 1)
        cov_t = covs.loc[t].values
        vol_t = np.sqrt(pos_t.T @ cov_t @ pos_t).item()
        port_vol.append(vol_t)
    
    return pd.Series(port_vol, index=pnl.index)


prices = pd.read_csv('close_prices_insample.csv',index_col='AsOfDate',parse_dates=True)
ret = prices.diff()
trend_window=50
vol_window=100
risk_window=100

# calculate risk adjusted positions
loaded_vols = pickle.load(open('vols.pkl', 'rb'))
# calculate signal
sig = np.sign(ret.rolling(window=trend_window).sum())
# scale with inverse vol to get pos
pos = sig/loaded_vols
model_risk = rolling_portfolio_vol(pos, ret, risk_window)


# save model_risk in pickle
with open('model_risk.pkl', 'wb') as f:
    pickle.dump(model_risk, f)


# load model_risk from pickle
with open('model_risk.pkl', 'rb') as f:
    model_risk = pickle.load(f)

pos_adjusted = pos.div(model_risk, axis=0)



# calculate risk adjusted positions SHORT VERSION to check for forward looking bias
loaded_vols_short = pickle.load(open('vols_short.pkl', 'rb'))
# calculate signal
sig_short = np.sign(ret.iloc[:-200].rolling(window=trend_window).sum())
# scale with inverse vol to get pos
pos_short = sig_short/loaded_vols_short
model_risk_short = rolling_portfolio_vol(pos_short, ret.iloc[:-200], risk_window)
pos_adjusted_short = pos_short.div(model_risk_short, axis=0)


# save model_risk_short in pickle
with open('model_risk_short.pkl', 'wb') as f:
    pickle.dump(model_risk_short, f)

with open('model_risk_short.pkl', 'rb') as f:
    model_risk_short = pickle.load(f)





temp = (pos_adjusted-pos_adjusted_short).abs().sum()
if temp.sum() > 0:
    print('Forward looking bias detected!')
else:
    print('No forward looking bias detected!')


# this can be used as a little smarter way to calculate position 
# expected_ret = (pos.shift(2) * ret).rolling(risk_window).mean().sum(axis=1)
# sharpe_like = expected_ret / model_risk
# pos_adjusted = pos.multiply(sharpe_like, axis=0)


