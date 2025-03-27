import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from evaluation import plot_key_figures, calc_key_figures
"""
XGBoost Model
threshold — Adjusted Sharpe: 0.376, Holding: 23.28
smooth — Adjusted Sharpe: 0.210, Holding: 10.35
normalize — Adjusted Sharpe: 0.376, Holding: 23.28
threshold_then_normalize — Adjusted Sharpe: 0.376, Holding: 23.28
smooth_then_normalize — Adjusted Sharpe: 0.209, Holding: 9.61
Best strategy: threshold with Adjusted Sharpe: 0.376
No strategy Adjusted Sharpe: 0.917
No strategy is better than the best strategy (threshold).
{'sharpe': np.float64(0.9090724899312874), 'sharpe_1d_lag': np.float64(1.0460940446116729), 
'av_holding': np.float64(23.71928926034637), 'position_bias': np.float64(0.19901017381619684), 
'skewness': np.float64(-0.012498194097319637), 'kurtosis': np.float64(2.705701340727947), 
'adjusted_sharpe': np.float64(0.916563441144595)}
"""
# --- Feature Engineering ---
def compute_volatility(prices, window=100):
    returns = prices.pct_change()
    vol = returns.rolling(window=window).std().add_suffix(f'_vol_{window}d')
    return vol

def compute_lagged_returns(prices, lags=[1, 5, 10]):
    return pd.concat([(prices / prices.shift(lag) - 1).add_suffix(f'_ret_{lag}d') for lag in lags], axis=1)

def compute_momentum(prices, window=20):
    max_price = prices.rolling(window).max()
    min_price = prices.rolling(window).min()
    momentum = ((prices - min_price) / (max_price - min_price + 1e-9)).add_suffix(f'_mom_{window}d')
    return momentum

def compute_rolling_corr(prices, window=50):
    returns = prices.pct_change()
    corr_features = []
    for col1 in prices.columns:
        for col2 in prices.columns:
            if col1 != col2:
                corr = returns[col1].rolling(window).corr(returns[col2])
                corr_features.append(corr.rename(f'{col1}_{col2}_corr_{window}d'))
    return pd.concat(corr_features, axis=1)

# --- Combine Features ---
def build_features(prices):
    return pd.concat([
        compute_lagged_returns(prices),
        compute_volatility(prices),
        compute_momentum(prices),
        compute_rolling_corr(prices)
    ], axis=1)

# --- ML Model Fitting and Position Generation ---
def ml_trading_model(prices, model_type='ridge'):
    ret = prices.pct_change()
    features = build_features(prices)

    # Target: next day returns (shifted -1)
    execution_lag = 2
    trend_window = 20  # Try values like 10, 20, 30
    y = ret.shift(-execution_lag).rolling(window=trend_window).sum()
    #y = ret.shift(-execution_lag)
    # y = ret.shift(-1)  # y_t = return at t+1
    X = features.shift(1)  # Feature values must be available at time t-1 (to avoid lookahead bias)

    pos = pd.DataFrame(index=ret.index, columns=ret.columns)

    for col in prices.columns:
        valid_idx = y[col].dropna().index.intersection(X.dropna().index)
        if len(valid_idx) < 250:  # Require enough samples
            continue

        X_ = X.loc[valid_idx]
        y_ = y[col].loc[valid_idx]

        # Time-based train-test split (80/20)
        split = int(0.8 * len(X_))
        X_train, X_test = X_.iloc[:split], X_.iloc[split:]
        y_train, y_test = y_.iloc[:split], y_.iloc[split:]

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Select model
        if model_type == 'ridge':
            model = Ridge()
        elif model_type == 'gbr':
            model = GradientBoostingRegressor()
        elif model_type == 'xgb':
            model = XGBRegressor()
        else:
            raise ValueError("Invalid model type")

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        #preds = pd.Series(preds, index=X_test.index).ewm(span=10).mean()
        # volatility scalling
        vol = np.sqrt((ret[col]**2).rolling(window=100, min_periods=10).mean()).shift(1)
        pos.loc[X_test.index, col] = preds / vol.loc[X_test.index]
        # Convert predictions into positions: sign(prediction)
        # pos.loc[X_test.index, col] = np.sign(preds)
        
        
    pos_check = pos.shift(-execution_lag)
    future_returns = prices.pct_change().shift(-execution_lag)

    # If any signal correlates with unseen returns before the 2-day lag, warn
    overlap = pos_check.index.intersection(future_returns.index)
    correlation_check = (pos_check.loc[overlap] * future_returns.loc[overlap]).sum().sum()

    if correlation_check != correlation_check:  # check for NaNs
        print(" Warning: NaNs in forward-looking bias check")
    elif correlation_check > 1e-8:
        print("Forward-looking bias detected via returns overlap!")
    else:
        print(" No forward-looking bias detected via returns overlap.")
    return pos.astype(float)


def apply_threshold(preds: pd.Series, threshold: float = 0.01):
    return np.where(np.abs(preds) > threshold, np.sign(preds), 0)

def smooth_positions(preds: pd.Series, window: int = 20):
    rolling_std = preds.rolling(window=window, min_periods=5).std().replace(0, np.nan)
    return (preds / rolling_std).clip(-1, 1).fillna(0)
def normalize_daily_positions(pos: pd.DataFrame):
    daily_risk = np.sqrt((pos**2).sum(axis=1))
    target_risk = 1.0
    scaling_factor = target_risk / daily_risk
    return pos.mul(scaling_factor, axis=0).clip(-1, 1).fillna(0)

def test_position_strategies(prices, pos_raw, strategy_name):
    from evaluation import calc_key_figures

    if strategy_name == "threshold":
        pos = pos_raw.copy()
        for col in pos.columns:
            pos[col] = apply_threshold(pos[col], threshold=0.01)

    elif strategy_name == "smooth":
        pos = pos_raw.copy()
        for col in pos.columns:
            pos[col] = smooth_positions(pos[col], window=20)

    elif strategy_name == "normalize":
        pos = normalize_daily_positions(pos_raw)

    elif strategy_name == "threshold_then_normalize":
        pos = pos_raw.copy()
        for col in pos.columns:
            pos[col] = apply_threshold(pos[col], threshold=0.01)
        pos = normalize_daily_positions(pos)

    elif strategy_name == "smooth_then_normalize":
        pos = pos_raw.copy()
        for col in pos.columns:
            pos[col] = smooth_positions(pos[col], window=20)
        pos = normalize_daily_positions(pos)

    else:
        raise ValueError("Unknown strategy")

    results = calc_key_figures(pos, prices)
    print(f"{strategy_name} — Adjusted Sharpe: {results['adjusted_sharpe']:.3f}, Holding: {results['av_holding']:.2f}")
    return results

# --- Main Evaluation Pipeline ---
def forward_looking_bias(pos, pos_short): 
    common_idx = pos.index.intersection(pos_short.index)
    temp = (pos.loc[common_idx] - pos_short.loc[common_idx]).abs().sum()
    if temp.sum() > 0:
        print('Forward looking bias detected!')
            
def evaluate_ml_strategy(prices_final, model_type='ridge', costs=-0.02, rolling_window=260):
    pos = ml_trading_model(prices_final, model_type=model_type)
    pos_short = ml_trading_model(prices_final.iloc[:-20], model_type=model_type)
    forward_looking_bias(pos, pos_short)
    strategies = ["threshold", "smooth", "normalize", "threshold_then_normalize", "smooth_then_normalize"]
    results = []
    for strat in strategies:
        results.append((strat, test_position_strategies(prices_final, pos, strat)))

    best_strategy = max(results, key=lambda x: x[1]['adjusted_sharpe'])
    print(f"Best strategy: {best_strategy[0]} with Adjusted Sharpe: {best_strategy[1]['adjusted_sharpe']:.3f}")

    # Calculate key figures for no strategy (raw positions)
    no_strategy_results = calc_key_figures(pos, prices_final, costs=costs, rolling_window=rolling_window)
    
    print(f"No strategy Adjusted Sharpe: {no_strategy_results['adjusted_sharpe']:.3f}")

    # Compare best strategy with no strategy
    if best_strategy[1]['adjusted_sharpe'] > no_strategy_results['adjusted_sharpe']:
        print(f"The best strategy ({best_strategy[0]}) is better than no strategy.")
        best_sharpe = best_strategy[1]['adjusted_sharpe']
        print(best_strategy[1])
    else:
        print(f"No strategy is better than the best strategy ({best_strategy[0]}).")
        best_sharpe = no_strategy_results['adjusted_sharpe']
        print(no_strategy_results)

    plot_key_figures(pos, prices_final, costs=costs, rolling_window=rolling_window)
    
    return best_sharpe

def main(): 
    prices = pd.read_csv('example_prices.csv',index_col='dates',parse_dates=True)
    print("Ridge Model")
    ridge_results = evaluate_ml_strategy(prices, model_type='ridge')
    print("Gradient Boosting Model")
    # gbr_results = evaluate_ml_strategy(prices, model_type='gbr')
    print("XGBoost Model")
    xgb_results = evaluate_ml_strategy(prices, model_type='xgb')
    
if __name__ == "__main__":
    main()
