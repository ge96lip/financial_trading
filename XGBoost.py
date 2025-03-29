import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from evaluation import plot_key_figures, calc_key_figures
from data_analysis import load_data


###########################
# 1. Data Loading Functions
###########################

def clean_dublicated_indices(df):
    duplicates = df.index.duplicated().sum()
    print(f"Duplicate dates found: {duplicates}")
    return df[~df.index.duplicated()]

def load_close_prices(path='close_prices_insample.csv'):
    print("Loading closing prices from:", path)
    df = pd.read_csv(path, index_col='AsOfDate', parse_dates=True)
    df = clean_dublicated_indices(df)
    df.sort_index(inplace=True)
    return df

def load_full_prices(path='prices_insample.csv'):
    """
    Loads prices_insample.csv, which has a two-row header.
    """
    print("Loading multi-header full price data from:", path)
    df = pd.read_csv(path, header=[0,1], index_col=0, parse_dates=True)
    df.columns = [f"{upper}_{lower}" for upper, lower in df.columns]
    df.sort_index(inplace=True)
    return df

def load_macro_data(path='macro_data_insample.csv'):
    """
    Loads macro_data_insample.csv, which has a two-row header:
      Row 0: 'gdp' or 'cpi'
      Row 1: planet names ('titan', 'io', 'callisto', 'europa', 'ganymede')
    The first column holds the dates (no explicit 'AsOfDate' label).
    We parse them as the DataFrame's index.
    
    The final columns become: 'gdp_titan', 'gdp_io', 'gdp_callisto', 'gdp_europa', 'gdp_ganymede',
                             'cpi_titan', 'cpi_io', 'cpi_callisto', 'cpi_europa', 'cpi_ganymede'
    """
    print("Loading macro data from:", path)
    # header=[0,1] -> first two rows are column headers (multi-level)
    # index_col=0  -> first column is the date index
    # parse_dates=True -> parse that first column as dates
    df = pd.read_csv(path, header=[0,1], index_col=0, parse_dates=True)
    
    # Flatten the two-level columns into single-level: ('gdp','titan') => 'gdp_titan'
    df.columns = [f"{upper}_{lower}" for (upper, lower) in df.columns]
    
    # Sort the index to ensure chronological order
    df.sort_index(inplace=True)
    
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {path}.")
    print("Columns:", df.columns.tolist())
    return df

###########################
# 2. Merging Data
###########################

def merge_datasets(full_df, macro_df):
    """
    Merges the multi-header price data with macro data on date index.
    """
    merged = full_df.join(macro_df, how='left')
    merged.fillna(method='ffill', inplace=True)  # forward-fill macro data
    return merged

###########################
# 3. Feature Construction Function
###########################

def get_enhanced_features(merged_df, asset, trend_window=50, vol_window=100, corr_window=20):
    """
    Build a rich set of features for a given asset using merged data.
    The features include:
      - Returns and basic technical features (from close prices)
      - Extra features from full OHLC data (e.g. intraday range)
      - Rolling correlation with other assets
      - Macro variables
    The target is defined as the sign of the rolling sum of returns.
    """
    df_features = pd.DataFrame(index=merged_df.index)
    # 1. Returns from the closing prices (the asset column in close_df)
    close_col = f"{asset}_Close"
    macro_cols = [col for col in merged_df.columns if col.startswith("gdp_") or col.startswith("cpi_")]

    # Make sure it exists
    if close_col not in merged_df.columns:
        raise ValueError(f"Column '{close_col}' not found in merged_df. Available columns:\n{merged_df.columns.tolist()}")
    
    df_features['return'] = merged_df[close_col].pct_change()
    
    # 2. Basic technical features (lagged to avoid lookahead bias)
    df_features['rolling_sum'] = df_features['return'].rolling(window=trend_window).sum().shift(1)
    df_features['lag1'] = df_features['return'].shift(1)
    df_features['lag2'] = df_features['return'].shift(2)
    df_features['lag3'] = df_features['return'].shift(3)
    df_features['vol'] = np.sqrt((df_features['return']**2).rolling(window=vol_window, min_periods=vol_window//2).sum()).shift(1)
    
    # 3. Additional technical features
    df_features['rolling_mean'] = df_features['return'].rolling(window=trend_window).mean().shift(1)
    df_features['rolling_std'] = df_features['return'].rolling(window=trend_window).std().shift(1)
    df_features['momentum'] = (df_features['return'].shift(1) / df_features['rolling_mean']) - 1
    df_features['ema'] = df_features['return'].ewm(span=trend_window, adjust=False).mean().shift(1)
    df_features['ema12'] = df_features['return'].ewm(span=12, adjust=False).mean().shift(1)
    df_features['ema26'] = df_features['return'].ewm(span=26, adjust=False).mean().shift(1)
    df_features['macd'] = df_features['ema12'] - df_features['ema26']
    df_features['rs_vol'] = df_features['rolling_sum'] / df_features['vol']
    
    # 4. Extra feature from full price data: intraday range (High - Low)
    high_col = f"{asset}_High"
    low_col  = f"{asset}_Low"
    if high_col in merged_df.columns and low_col in merged_df.columns:
        df_features['intraday_range'] = merged_df[high_col] - merged_df[low_col]
    
    # 5. Correlation features: rolling correlation of the asset’s returns with those of other assets.
    # We assume that the close prices (for all assets) are exactly the columns of close_df.
    all_close_cols = [c for c in merged_df.columns if c.endswith('_Close')]
    returns_df = merged_df[all_close_cols].pct_change()
    asset_returns = returns_df[close_col]
    other_assets = [col for col in all_close_cols if col != close_col]
    if other_assets:
        corr_list = []
        for other in other_assets:
            corr_series = asset_returns.rolling(window=corr_window).corr(returns_df[other])
            corr_list.append(corr_series)
        df_features['avg_corr'] = pd.concat(corr_list, axis=1).mean(axis=1)
    else:
        df_features['avg_corr'] = np.nan
    
    # 6. Include macro variables as additional predictors.
    for col in macro_cols:
        df_features[col] = merged_df[col]
    
    # 7. Define target: sign of the non-lagged rolling sum of returns
    target = df_features['return'].rolling(window=trend_window).sum()
    target = np.sign(target)
    
    # Drop rows with NaN values (from rolling calculations or merging)
    df_features = df_features.dropna()
    target = target.loc[df_features.index]
    
    return df_features, target

###########################
# 4. Walk-Forward Prediction
###########################

def walk_forward_prediction(features, target, model, start_train=200, retrain_frequency=5):
    predictions = pd.Series(index=features.index, dtype=float)
    current_model = None
    for i in range(start_train, len(features)):
        if current_model is None or (i - start_train) % retrain_frequency == 0:
            train_X = features.iloc[:i]
            train_y = target.iloc[:i]
            model.fit(train_X, train_y)
            current_model = model
        test_X = features.iloc[[i]]
        predictions.iloc[i] = current_model.predict(test_X)[0]
    return predictions, current_model

###########################
# 5. Full Pipeline: Loop Over Assets and Generate Positions
###########################

def model_constructor():
    # Example: use XGBoost with a small number of trees; adjust parameters as needed.
    return XGBRegressor(n_estimators=15, max_depth=3, verbosity=0, n_jobs=1)

def full_pipeline(start_train=200, retrain_frequency=5):
    # Load each data source
    close_df = load_close_prices()
    full_df  = load_full_prices()
    macro_df = load_macro_data()
    
    # Merge the three datasets into one master DataFrame
    merged_df = merge_datasets(full_df, macro_df)
    print("merged_df shape:", merged_df.shape)
    print("merged tail: ", merged_df.tail())
    # Create an empty DataFrame to hold position signals; columns as in close_df (the traded assets)
    pos = pd.DataFrame(index=merged_df.index, columns=close_df.columns, dtype=float)
    models_trained = {}  # Optional: store last trained model per asset
    
    # Loop over each asset (using close_df's columns)
    for asset in close_df.columns:
        print(f"Generating positions for {asset} using model {model_constructor.__name__}")
        # Build features and target for this asset
        features, target = get_enhanced_features(merged_df, asset)
        
        # Run walk-forward prediction using the enhanced features and target
        predictions, trained_model = walk_forward_prediction(features, target, model=model_constructor(), 
                                                              start_train=start_train, retrain_frequency=retrain_frequency)
        # Smooth predictions over a 5-day rolling window and then take the sign as the final signal
        pos[asset] = np.sign(predictions.rolling(window=5, min_periods=1).mean())
        models_trained[asset] = trained_model
    
    return pos, models_trained

def main(): 
    # Run the full pipeline:
    prices = load_data("close_prices_insample.csv") #pd.read_csv('example_prices.csv', index_col='dates', parse_dates=True)
    #close_df = load_close_prices()
    #full_df  = load_full_prices()
    #macro_df = load_macro_data()
    ret = prices.diff()
    
    pos, trained_models = full_pipeline()
    
    model_risk = (pos.shift(2) * ret).dropna(how='all').sum(axis=1).rolling(100, min_periods=20).std()
    pos_adjusted = pos.div(model_risk, axis=0)

    # positions is the final DataFrame of signals (one column per asset)
    print("Final Positions:")
    print(pos.head())
    figures_xgb = calc_key_figures(pos_adjusted, prices)
    print("XGBoost Model Key Figures:")
    print(figures_xgb)
    forward_looking_bias_model(ret, generate_model_positions,
                                model_name='XGB', model_constructor=lambda: XGBRegressor(n_estimators=10, max_depth=3, verbosity=0, n_jobs=1),
                                trend_window=50, vol_window=100,
                                risk_window=100, start_train=200, smoothing_window=5,
                                retrain_frequency=5)

if __name__ == "__main__":
    main()
