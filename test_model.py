from itertools import product
import sys

from sklearn.linear_model import Ridge
from model_utils import generate_model_positions
from evaluation import calc_key_figures
from xgboost import XGBRegressor
from data_analysis import load_data


def test_hyperparameters(model_name, ret, prices):
    
    def evaluate_model(model_constructor, model_name):
        pos = generate_model_positions(
            ret,
            model_constructor=model_constructor,
            model_name=model_name,
            trend_window=50,
            vol_window=100,
            risk_window=100,
            start_train=200,
            smoothing_window=5,
            retrain_frequency=5
        )
        figures = calc_key_figures(pos, prices)
        print(f"Hyperparameters: {model_constructor()}")
        print("Key Figures:", figures)
        print("-" * 60)
        return figures['adjusted_sharpe'], model_constructor()

    results = []

    if model_name == "XGB":
        param_grid = {
            'n_estimators': [10, 15, 20],
            'max_depth': [3, 5],
            'verbosity': [0],
            'n_jobs': [1]
        }

        for values in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), values))
            constructor = lambda p=params: XGBRegressor(**p)
            score, used_params = evaluate_model(constructor, model_name)
            results.append((score, used_params))

    elif model_name == "Ridge":
        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'fit_intercept': [True, False]
        }

        for values in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), values))
            constructor = lambda p=params: Ridge(**p)
            score, used_params = evaluate_model(constructor, model_name)
            results.append((score, used_params))

    else:
        print(f"Model '{model_name}' not supported.")
        return

    results.sort(reverse=True, key=lambda x: x[0])
    print("\nTop configurations:")
    for score, model_params in results[:5]:
        print(f"Score: {score:.4f} | Params: {model_params}")
        
def main(name, data_path): 
    prices = load_data(data_path)
    ret = prices.diff()
    test_hyperparameters(model_name, ret, prices)
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        data_path = sys.argv[2]
        main(model_name, data_path)
    else:
        print("Please provide a model name as a command-line argument (e.g., 'XGB' or 'Ridge').")