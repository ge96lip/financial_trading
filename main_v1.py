import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from evaluation import plot_key_figures, calc_key_figures
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import sys


def get_features_and_target(ret_series, trend_window=50, vol_window=100):
    """
    For a single asset (a pandas Series of returns), compute features and target.
    Features (all lagged to avoid lookahead bias):
      - rolling_sum: rolling sum over trend_window (lagged by one day)
      - lag1 and lag2: one-day and two-day lagged returns
      - vol: volatility as sqrt(rolling sum of squared returns over vol_window)
    Target:
      - The sign of the current (non-lagged) rolling sum.
    """
    df = pd.DataFrame(index=ret_series.index)
    df['rolling_sum'] = ret_series.rolling(window=trend_window).sum().shift(1)
    df['lag1'] = ret_series.shift(1)
    df['lag2'] = ret_series.shift(2)
    df['vol'] = np.sqrt((ret_series**2).rolling(window=vol_window, min_periods=vol_window//2).sum()).shift(1)
    
    # Target: sign of the current rolling sum (non-lagged)
    target = ret_series.rolling(window=trend_window).sum()
    target = np.sign(target)
    
    # Drop NaNs created by rolling and shifting
    df = df.dropna()
    target = target.loc[df.index]
    return df, target

def walk_forward_prediction(ret_series, model, start_train=200, trend_window=50, vol_window=100, retrain_frequency=5):
    """
    Walk-forward prediction for one asset.
    Train the model on past data and predict for the current time step.
    To speed up the process, retrain only every 'retrain_frequency' steps.
    """
    features, target = get_features_and_target(ret_series, trend_window, vol_window)
    predictions = pd.Series(index=features.index, dtype=float)
    
    current_model = None  # holds the model trained at the last retraining point
    for i in range(start_train, len(features)):
        # Retrain only every 'retrain_frequency' steps or at the very first prediction
        if current_model is None or (i - start_train) % retrain_frequency == 0:
            train_X = features.iloc[:i]
            train_y = target.iloc[:i]
            model.fit(train_X, train_y)
            current_model = model
        test_X = features.iloc[[i]]
        pred = current_model.predict(test_X)[0]
        predictions.iloc[i] = pred
        
    return predictions

def generate_model_positions(ret, model_constructor, model_name, trend_window=50, vol_window=100,
                             risk_window=100, start_train=200, smoothing_window=5, retrain_frequency=5):
    """
    For each asset (column in the returns DataFrame), use a walk-forward approach with the specified model.
    After obtaining predictions, smooth the signal with a short moving average before taking its sign.
    Finally, risk-adjust the positions using the same method as in the original code.
    
    The parameters are tuned to mimic the original model’s behavior (e.g. high average holding period).
    """
    pos = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    
    for asset in ret.columns:
        print(f"Generating positions for {asset} using {model_name}")
        # Create a new instance for each asset
        model = model_constructor()
        predictions = walk_forward_prediction(ret[asset], model,
                                              start_train=start_train,
                                              trend_window=trend_window,
                                              vol_window=vol_window,
                                              retrain_frequency=retrain_frequency)
        # Smooth the predictions to reduce frequent flipping
        predictions_smoothed = predictions.rolling(window=smoothing_window, min_periods=1).mean()
        # Final signal is the sign of the smoothed predictions
        pos[asset] = np.sign(predictions_smoothed)
    
    # Risk adjust positions:
    model_risk = (pos.shift(2) * ret).dropna(how='all').sum(axis=1).rolling(risk_window, min_periods=20).std()
    pos_adjusted = pos.div(model_risk, axis=0)
    return pos_adjusted

def forward_looking_bias_model(ret, pos_func, **kwargs):
    """
    Checks for forward-looking bias by comparing positions on the full return dataset
    versus a truncated version (dropping the last 20 rows).
    """
    pos_full = pos_func(ret, **kwargs)
    pos_short = pos_func(ret.iloc[:-20], **kwargs)
    print(pos_full.shape)
    print(pos_short.shape)
    # Compare on common indices:
    common_idx = pos_short.index
    diff = (pos_full-pos_short).abs().sum()
    #diff = (pos_full.loc[common_idx] - pos_short).abs().sum().sum()
    if diff.sum() > 0:
        print(f'Forward looking bias detected in {kwargs.get("model_name", "model")}')
    else:
        print(f'No forward looking bias detected in {kwargs.get("model_name", "model")}')

def main(name):
    # Load prices (CSV with 'dates' index)
    prices = pd.read_csv('example_prices.csv', index_col='dates', parse_dates=True)
    ret = prices.diff()
    if name == "XGB": 
        # ---------------------------
        # XGBoost Model Evaluation
        # ---------------------------
        print("Evaluating XGBoost Model")
        # Use a lambda to create an XGBRegressor with reduced n_estimators and quiet mode.
        pos_xgb = generate_model_positions(ret, 
                                        model_constructor=lambda: XGBRegressor(n_estimators=15, max_depth=3, verbosity=0, n_jobs=1),
                                        model_name='XGB',
                                        trend_window=50, vol_window=100,
                                        risk_window=100, start_train=200, smoothing_window=5,
                                        retrain_frequency=5)
        # n_estimators=10, max_depth=3, verbosity=0, n_jobs=1: {'sharpe': np.float64(0.6668094194616517), 'sharpe_1d_lag': np.float64(0.614868825626963), 'av_holding': np.float64(31.735723853247656), 'position_bias': np.float64(-0.014938770859661351), 'skewness': np.float64(0.07796390102892065), 'kurtosis': np.float64(3.6427993549698052), 'adjusted_sharpe': np.float64(0.6646461032897002)}
        # estimators:15: {'sharpe': np.float64(0.6846537621706345), 'sharpe_1d_lag': np.float64(0.645968540441132), 'av_holding': np.float64(31.580620424230826), 'position_bias': np.float64(-0.013994759880333916), 'skewness': np.float64(0.08952193855895944), 'kurtosis': np.float64(3.6316091192585014), 'adjusted_sharpe': np.float64(0.6832016932594196)}
        # max_depth = 5: {'sharpe': np.float64(0.6436507049366413), 'sharpe_1d_lag': np.float64(0.5815819380125943), 'av_holding': np.float64(31.14161562829988), 'position_bias': np.float64(-0.010930426234156437), 'skewness': np.float64(0.09658071873623306), 'kurtosis': np.float64(3.614493831421452), 'adjusted_sharpe': np.float64(0.6434919554104483)}
        # verbosity=1: {'sharpe': np.float64(0.6668094194616517), 'sharpe_1d_lag': np.float64(0.614868825626963), 'av_holding': np.float64(31.735723853247656), 'position_bias': np.float64(-0.014938770859661351), 'skewness': np.float64(0.07796390102892065), 'kurtosis': np.float64(3.6427993549698052), 'adjusted_sharpe': np.float64(0.6646461032897002)}
        # estimators = 20: {'sharpe': np.float64(0.6826177301260691), 'sharpe_1d_lag': np.float64(0.6027561366132076), 'av_holding': np.float64(31.50518975949255), 'position_bias': np.float64(-0.014150037277940135), 'skewness': np.float64(0.08242805550006745), 'kurtosis': np.float64(3.6402481601555925), 'adjusted_sharpe': np.float64(0.6805338380254925)}
        plot_key_figures(pos_xgb, prices)
        print("pos_xgb tail: ", pos_xgb.tail())
        figures_xgb = calc_key_figures(pos_xgb, prices)
        print("XGBoost Model Key Figures:")
        print(figures_xgb)
        
        forward_looking_bias_model(ret, generate_model_positions,
                                model_name='XGB', model_constructor=lambda: XGBRegressor(n_estimators=10, max_depth=3, verbosity=0, n_jobs=1),
                                trend_window=50, vol_window=100,
                                risk_window=100, start_train=200, smoothing_window=5,
                                retrain_frequency=5)
    elif name == "Ridge":
        # ---------------------------
        # Ridge Model Evaluation
        # ---------------------------
        print("\nEvaluating Ridge Model")
        pos_ridge = generate_model_positions(ret, 
                                            model_constructor=lambda: Ridge(alpha=1.0),
                                            model_name='Ridge',
                                            trend_window=50, vol_window=100,
                                            risk_window=100, start_train=200, smoothing_window=5,
                                            retrain_frequency=5)
        
        plot_key_figures(pos_ridge, prices)
        figures_ridge = calc_key_figures(pos_ridge, prices)
        print("Ridge Model Key Figures:")
        print(figures_ridge)
        
        forward_looking_bias_model(ret, generate_model_positions,
                                model_name='Ridge', model_constructor=lambda: Ridge(alpha=1.0),
                                trend_window=50, vol_window=100,
                                risk_window=100, start_train=200, smoothing_window=5,
                                retrain_frequency=5)
    else: 
        print("Invalid model name")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        main(model_name)
    else:
        print("Please provide a model name as a command-line argument (e.g., 'XGB' or 'Ridge').")