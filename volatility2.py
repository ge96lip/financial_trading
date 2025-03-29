import pandas as pd
import numpy as np

def calculate_garman_klass_volatility(df, window=30):
    """
    Calculate daily Garman-Klass volatility estimator for each asset in the dataframe.
    Assumes a multi-level column structure: (Asset) on level 0, (Open/High/Low/Close) on level 1.
    """
    # Copy to avoid altering the original
    df_copy = df.copy()
    
    # Ensure index is DateTime
    df_copy.index = pd.to_datetime(df_copy.index)
    
    # Create DataFrame to store results with same index
    vol_df = pd.DataFrame(index=df_copy.index)
    
    # Get unique asset names (first level of column MultiIndex)
    # Some CSVs might have an empty '' level if there's a blank column. Let's filter them out.
    asset_names = [name for name in df_copy.columns.levels[0] if name != '']
    print("Detected asset names:", asset_names)
    
    # Calculate GK volatility for each asset
    for asset in asset_names:
        try:
            # Extract sub-DataFrame for just this asset
            asset_data = df_copy[asset]
            
            # Safely get each OHLC column as numeric
            open_prices = pd.to_numeric(asset_data['Open'], errors='coerce')
            high_prices = pd.to_numeric(asset_data['High'], errors='coerce')
            low_prices  = pd.to_numeric(asset_data['Low'],  errors='coerce')
            close_prices= pd.to_numeric(asset_data['Close'],errors='coerce')
            
            # Garman-Klass daily volatility:
            # daily_var = 0.5 * (ln(H/L))^2 - (2ln(2) - 1) * (ln(C/O))^2
            # daily_vol = sqrt(daily_var)
            log_hl = np.log(high_prices / low_prices)
            log_co = np.log(close_prices / open_prices)
            
            daily_var = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
            daily_vol = np.sqrt(np.clip(daily_var, a_min=0, a_max=None))  # clip negatives to 0 for safety
            
            vol_df[asset] = daily_vol
            
        except Exception as e:
            print(f"Error processing {asset}: {e}")
            continue
    
    return vol_df


if __name__ == "__main__":
    # --- KEY PART: read_csv with skipping rows + multi-level headers ---
    # Skip the very first line and use the next TWO lines as the multi-level header
    # The 'index_col=0' is 'AsOfDate'.
    df = pd.read_csv(
        "prices_insample.csv",
        skiprows=[0],           # skip line 1 (the purely comma-separated line)
        header=[0, 1],          # lines 2 and 3 become the multi-level column
        index_col=0
    )
    
    # Now calculate daily Garman-Klass volatility
    volatilities = calculate_garman_klass_volatility(df)
    
    # Print results
    print("\nDaily Volatilities:")
    print(volatilities.head())
    print("\nSummary statistics:")
    print(volatilities.describe())
