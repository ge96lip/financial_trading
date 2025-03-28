from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from evaluation import plot_key_figures, calc_key_figures
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import sys
from utils import forward_looking_bias_model, test_hindsight_bias_using_target
from model_utils import generate_model_positions
from data_analysis import load_data
#from catboost import CatBoostRegressor



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
        
    elif name == "SVM":
        print("Evaluating Support Vector Machine (SVR)")
        pos_svm, model = generate_model_positions(
            ret,
            model_constructor=lambda: SVR(
                kernel='rbf',    # or 'linear' depending on experiments
                C=1,             # try values such as 0.1, 1, 10, 100
                gamma='scale'    # or numeric values like 0.001, 0.01, 0.1, 1
            ),
            model_name='SVM',
            trend_window=50, vol_window=100,
            risk_window=100, start_train=200, smoothing_window=5,
            retrain_frequency=5
        )
        plot_key_figures(pos_svm, prices)
        figures_svm = calc_key_figures(pos_svm, prices)
        print("SVM Key Figures:")
        print(figures_svm)
        forward_looking_bias_model(ret, generate_model_positions,
                                model_name='SVM', model_constructor=model_constructor,
                                trend_window=50, vol_window=100,
                                risk_window=100, start_train=200, smoothing_window=5,
                                retrain_frequency=5)
    elif name == "MLP":
        print("Evaluating Multi-Layer Perceptron (MLP)")
        pos_mlp, model = generate_model_positions(
            ret,
            
            model_constructor = lambda: MLPRegressor(
                hidden_layer_sizes=(100,),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate_init=0.0005,
                max_iter=2000,
                early_stopping=True,
                n_iter_no_change=10,
                verbose=False # can make it True to see validation loss but it sucks because there are so many iterations
            ), 
            model_name='MLP',
            trend_window=50, vol_window=100,
            risk_window=100, start_train=200, smoothing_window=5,
            retrain_frequency=5
        )
        plot_key_figures(pos_mlp, prices)
        figures_mlp = calc_key_figures(pos_mlp, prices)
        print("MLP Key Figures:")
        print(figures_mlp)
        """forward_looking_bias_model(ret, generate_model_positions,
                                model_name='MLP', model_constructor=model_constructor,
                                trend_window=50, vol_window=100,
                                risk_window=100, start_train=200, smoothing_window=5,
                                retrain_frequency=5)"""
    elif name == "Logistic":
        print("Evaluating Logistic Regression")
        pos_logistic, model = generate_model_positions(
            ret,
            model_constructor=lambda: LogisticRegression(
                penalty='l2',          # try 'l1' (with solver 'liblinear') as an alternative
                C=1.0,                 # try 0.01, 0.1, 1, 10, 100
                solver='lbfgs',        # for multinomial logistic regression
                max_iter=200
            ),
            model_name='Logistic',
            trend_window=50, vol_window=100,
            risk_window=100, start_train=200, smoothing_window=5,
            retrain_frequency=5
        )
        plot_key_figures(pos_logistic, prices)
        figures_logistic = calc_key_figures(pos_logistic, prices)
        print("Logistic Regression Key Figures:")
        print(figures_logistic)
        """forward_looking_bias_model(ret, generate_model_positions,
                                model_name='Logistic', model_constructor=model_constructor,
                                trend_window=50, vol_window=100,
                                risk_window=100, start_train=200, smoothing_window=5,
                                retrain_frequency=5)"""
        
    elif name == "LightGBM":
        print("Evaluating LightGBM")
        pos_lgbm, model = generate_model_positions(
            ret,
            model_constructor=lambda: LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                min_child_samples=5,  # default is 20
                min_data_in_leaf=5,
                max_depth=3,
                num_leaves=7
            ),
            model_name='LightGBM',
            trend_window=50, vol_window=100,
            risk_window=100, start_train=200, smoothing_window=5,
            retrain_frequency=5
        )
        plot_key_figures(pos_lgbm, prices)
        figures_lgbm = calc_key_figures(pos_lgbm, prices)
        print("LightGBM Key Figures:")
        print(figures_lgbm)
        """forward_looking_bias_model(ret, generate_model_positions,
                                model_name='Logistic', model_constructor=model_constructor,
                                trend_window=50, vol_window=100,
                                risk_window=100, start_train=200, smoothing_window=5,
                                retrain_frequency=5)"""
        
    else: 
        print("Invalid model name")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        data_path = sys.argv[2] if len(sys.argv) > 2 else 'example_prices.csv'
        main(model_name, data_path)
    else:
        print("Please provide a model name and data_path as a command-line argument (e.g., 'XGB' or 'Ridge', 'example_prices.csv').")