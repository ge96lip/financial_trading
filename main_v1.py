import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from evaluation import plot_key_figures, calc_key_figures
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import sys
from utils import forward_looking_bias_model
from model_utils import generate_model_positions
from data_analysis import load_data


def main(name, data_path = 'example_prices.csv'):
    # Load prices (CSV with 'dates' index)
    prices = load_data(data_path) #pd.read_csv('example_prices.csv', index_col='dates', parse_dates=True)
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
    if len(sys.argv) > 2:
        model_name = sys.argv[1]
        data_path = sys.argv[2]
        main(model_name, data_path)
    else:
        print("Please provide a model name and data_path as a command-line argument (e.g., 'XGB' or 'Ridge', 'example_prices.csv').")