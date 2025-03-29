
import pandas as pd
import matplotlib.pyplot as plt

def clean_dublicated_indices(prices):
    duplicates = prices.index.duplicated().sum()
    print(f"Duplicate dates found: {duplicates}")
    prices = prices[~prices.index.duplicated()]
    return prices

def missing_data(prices, critical_assets=None, mask_days_nan=False):
    # Missing Data 
    # Check overall missing
    missing_summary = prices.isna().sum()
    print("Missing values per column:\n", missing_summary)

    if missing_summary.any(): 
        # Plot NaNs timeline
        plt.figure(figsize=(15, 3))
        plt.imshow(prices.isna().T, aspect='auto', cmap='gray_r')
        plt.title('Missing Data (white = NaN)')
        plt.ylabel('Assets')
        plt.xlabel('Time')
        plt.show()
        
        # Small gaps → forward-fill (.ffill()), then back-fill if needed.
        # Forward fill up to 3 days, then backward fill up to 3 days
        prices_cleaned = prices.ffill(limit=3).bfill(limit=3)
        # Large gaps → consider dropping those assets or time ranges.
        # TODO tune this threshold 
        coverage_threshold = 0.8
        asset_coverage = prices_cleaned.notna().mean()
        assets_to_warn = asset_coverage[asset_coverage < coverage_threshold].index
        if not assets_to_warn.empty:
            print(f"Warning: The following assets have <{coverage_threshold*100}% data coverage and should be reviewed: {list(assets_to_warn)}")
            prices_cleaned = too_many_nan(prices_cleaned, critical_assets=critical_assets)
        
        # Optional: mask days with too many missing assets.
        if mask_days_nan: 
            day_coverage = prices.notna().mean(axis=1)
            prices_cleaned = prices[day_coverage >= 0.9]  # keep days with ≥90% data
        
        return prices_cleaned
    else: 
        return prices

def too_many_nan(prices, critical_assets=None): 
    """
    Handles assets with too many NaNs. Critical assets are interpolated instead of being dropped.

    Args:
        prices (pandas.DataFrame): The price data.
        critical_assets (list): List of assets that cannot be dropped.

    Returns:
        pandas.DataFrame: Cleaned price data.
    """
    if critical_assets is None:
        critical_assets = []

    # Assets with too many NaNs 
    coverage = prices.notna().mean()
    low_coverage_assets = coverage[coverage < 0.8]
    print("Assets with <80% data coverage:\n", low_coverage_assets)

    # Handle critical assets
    for asset in critical_assets:
        if asset in low_coverage_assets.index:
            print(f"Critical asset '{asset}' has low coverage. Applying interpolation.")
            prices[asset] = prices[asset].interpolate(method='linear').ffill().bfill()

    # Drop non-critical assets with low coverage
    non_critical_assets = low_coverage_assets.index.difference(critical_assets)
    if not non_critical_assets.empty:
        print(f"Dropping non-critical assets with low coverage: {list(non_critical_assets)}")
        prices = prices.drop(columns=non_critical_assets)

    return prices

def constant_prices(prices):
    # Ensure the input is not None
    if prices is None:
        raise ValueError("Input 'prices' is None. Please provide a valid DataFrame.")

    # Constant Columns / Zero Variance
    zero_var_cols = prices.columns[prices.nunique() <= 1]
    print("Constant value columns:\n", zero_var_cols.tolist())
    if not zero_var_cols.empty:
        # Drop constant columns
        prices = prices.drop(columns=zero_var_cols)
        print("Dropped constant value columns.")
    return prices

def clean_outliers(prices, threshold = 0.3, option = 0, remove_spikes = False): 
    
    # Outliers / Jumps / Spikes 
    returns = prices.pct_change()
    # TODO tune this threshold 
    spikes = returns.abs() > threshold
    
    if spikes.any().any():
        print("Price spikes detected:\n", spikes.sum())

    if option ==1: 
        # Option 1: clip outliers 
        returns_clipped = returns.clip(lower=-0.10, upper=0.10)
        prices_cleaned = (1 + returns_clipped).cumprod() * prices.iloc[0]
        
        return prices_cleaned
    elif option == 2:
        # Option 2: Flag or Exclude Outliers
        spike_mask = returns.abs() > 0.3
        spike_counts = spike_mask.sum()

        if remove_spikes: 
            # Option A: Remove entire assets with too many spikes
            assets_to_keep = spike_counts[spike_counts < 5].index  # TODO: tune this threshold
            prices_cleaned = prices[assets_to_keep]
        else: 
            # Option B: Mask individual spikes (set to NaN, then fill)
            returns_cleaned = returns.mask(spike_mask)
            prices_cleaned = (1 + returns_cleaned).cumprod() * prices.iloc[0]
            
        return prices_cleaned
    else: 
        return prices

def load_data(path = 'example_prices.csv'):
    """
    Reads and processes a CSV file containing price data.

    This function loads price data from a CSV file, parses the dates, 
    and sets the 'dates' column as the index. It returns a cleaned-up 
    DataFrame with prices over a period of time for different assets.

    Args:
        path (str): The file path to the CSV file containing price data.

    Returns:
        pandas.DataFrame: A cleaned DataFrame with indexed dates and 
        price information for various assets.
    """
    print("Reading data from:", path)
    prices = pd.read_csv(path, index_col='AsOfDate',parse_dates=True)
    
    print("Shape of data:", prices.shape)
    # Dublicated Index 
    prices = clean_dublicated_indices(prices)
    
    # Sort data 
    # Non-monotonic Time Index
    if not prices.index.is_monotonic_increasing:
        print("Dates are not sorted. Sorting now.")
        prices = prices.sort_index()
    
    # Missing data
    prices = missing_data(prices)
    
    # Constant Prices
    prices = constant_prices(prices)
    
    # Remove outliers 
    prices = clean_outliers(prices)
    
    return prices 
    
    
def clean_dublicated_indices(df):
    duplicates = df.index.duplicated().sum()
    print(f"Duplicate dates found: {duplicates}")
    df = df[~df.index.duplicated()]
    return df

def load_close_prices(path='close_prices_insample.csv'):
    print("Loading closing prices from:", path)
    df = pd.read_csv(path, index_col='AsOfDate', parse_dates=True)
    df = clean_dublicated_indices(df)
    df.sort_index(inplace=True)
    return df

def load_full_prices(path='prices_insample.csv'):
    """
    Loads prices_insample.csv, which has a two-row header:
    Row 0: Asset name repeated
    Row 1: Indicator (Open, High, Low, Close)
    Then from Row 2 onward, the data.
    
    We parse it into a multi-index for columns and then flatten the columns.
    We also parse the first column as the date index.
    """
    print("Loading multi-header full price data from:", path)
    
    # Note header=[0,1] means the first two rows are used for column names
    # index_col=0 means the first column is used as the row index (dates)
    df = pd.read_csv(path, header=[0,1], index_col=0, parse_dates=True)
    
    # Flatten the multi-index columns into single-level
    # e.g., ("ganymede_bonds", "Open") → "ganymede_bonds_Open"
    df.columns = [f"{upper}_{lower}" for upper, lower in df.columns]
    
    # Ensure the index is sorted
    df.sort_index(inplace=True)
    
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {path}")
    return df

def load_macro_data(path='macro_data_insample.csv'):
    print("Loading macro data from:", path)
    df = pd.read_csv(path, index_col='AsOfDate', parse_dates=True)
    df.sort_index(inplace=True)
    return df