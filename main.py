import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from evaluation import plot_key_figures, calc_key_figures
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import sys
from utils import forward_looking_bias_model, test_hindsight_bias_using_target
from model_utils import generate_model_positions
from data_analysis import prepare_data


def main(name, data_path = 'example_prices.csv'):
    # Load prices (CSV with 'dates' index)
    prices = prepare_data(data_path) #pd.read_csv('example_prices.csv', index_col='dates', parse_dates=True)
    ret = prices.diff()
    
    if name == "XGB": 
        # ---------------------------
        # XGBoost Model Evaluation
        # ---------------------------
        print("Evaluating XGBoost Model")
        # Use a lambda to create an XGBRegressor with reduced n_estimators and quiet mode.
        pos_xgb, model = generate_model_positions(ret, 
                                        model_constructor=lambda: XGBRegressor(n_estimators=15, max_depth=3, verbosity=0, n_jobs=1),
                                        model_name='XGB',
                                        trend_window=50, vol_window=100,
                                        risk_window=100, start_train=200, smoothing_window=5,
                                        retrain_frequency=5)
        
        plot_key_figures(pos_xgb, prices)
        figures_xgb = calc_key_figures(pos_xgb, prices)
        print("XGBoost Model Key Figures:")
        print(figures_xgb)
        
        forward_looking_bias_model(ret, generate_model_positions,
                                model_name='XGB', model_constructor=lambda: XGBRegressor(n_estimators=10, max_depth=3, verbosity=0, n_jobs=1),
                                trend_window=50, vol_window=100,
                                risk_window=100, start_train=200, smoothing_window=5,
                                retrain_frequency=5)
        
        # Running the test method
        # Create synthetic returns data (simulate 100 days of returns for one asset)
        """dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        ret = pd.DataFrame({"asset": np.linspace(0.1, 1, 100)}, index=dates)
        
        test_hindsight_bias_using_target(ret, pretrained_model = model)"""
        
    elif name == "Ridge":
        # ---------------------------
        # Ridge Model Evaluation
        # ---------------------------
        print("\nEvaluating Ridge Model")
        pos_ridge, model = generate_model_positions(ret, 
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
    elif name == "RF":
        print("Evaluating Random Forest")
        model_constructor = lambda: RandomForestRegressor(
                n_estimators=50,    # try 50, 100, 200 in hyperparameter tuning
                max_depth=5,        # try None, 5, 10, 20
                min_samples_split=5, # try 2, 5, 10
                min_samples_leaf=2   # try 1, 2, 4
            )
        pos_rf, model = generate_model_positions(
            ret, 
            model_constructor=model_constructor,
            model_name='RandomForest',
            trend_window=50, vol_window=100,
            risk_window=100, start_train=200, smoothing_window=5,
            retrain_frequency=5
        )
        plot_key_figures(pos_rf, prices)
        figures_rf = calc_key_figures(pos_rf, prices)
        print("Random Forest Key Figures:")
        print(figures_rf)
        forward_looking_bias_model(ret, generate_model_positions,
                                model_name='RF', model_constructor=model_constructor,
                                trend_window=50, vol_window=100,
                                risk_window=100, start_train=200, smoothing_window=5,
                                retrain_frequency=5)
        
    else: 
        print("Invalid model name")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        data_path = sys.argv[2] if len(sys.argv) > 2 else 'example_prices.csv'
        main(model_name, data_path)
    else:
        print("Please provide a model name and data_path as a command-line argument (e.g., 'XGB' or 'Ridge', 'example_prices.csv').")