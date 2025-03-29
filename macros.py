import pandas as pd

# Import data with multi-level columns
macro_data = pd.read_csv('macro_data_insample.csv', header=[0, 1], index_col=0)

# Convert index to datetime
macro_data.index = pd.to_datetime(macro_data.index)

# Optional: Create more intuitive column names using tuples
macro_data.columns = macro_data.columns.set_names(['indicator', 'moon'])

# Example of how to access specific data:
# Get all GDP data
gdp_data = macro_data['gdp']
cpi_data = macro_data['cpi']
# Get all Titan data
titan_data = macro_data.xs('titan', axis=1, level=1)

# rate of gdp growth

gdp_growth = gdp_data.pct_change()

cpi_growth = cpi_data.pct_change()

# Generate trading signals based on CPI growth
def generate_cpi_signals(cpi_growth, lookback_period=30, threshold=0.00):
    """
    Generate trading signals based on CPI growth rates
    - Bearish signal when CPI growth is above threshold (inflation concerns)
    - Bullish signal when CPI growth is below threshold (inflation moderating)
    """
    # Initialize signals DataFrame with same structure as input
    signals = pd.DataFrame(0, index=cpi_growth.index, columns=cpi_growth.columns)
    
    # Calculate rolling average and generate signals for each column
    for column in cpi_growth.columns:
        rolling_cpi = cpi_growth[column].rolling(window=lookback_period).mean()
        signals.loc[rolling_cpi > threshold, column] = -1  # Bearish signal
        signals.loc[rolling_cpi < -threshold, column] = 1  # Bullish signal
    
    return signals

# Create trading signals
cpi_signals = generate_cpi_signals(cpi_growth)

#  

def generate_gdp_signals(gdp_growth, lookback_period=30, threshold=0.00):
    """
    Generate trading signals based on GDP growth rates
    - Bullish signal when GDP growth is above threshold (economic expansion)
    - Bearish signal when GDP growth is below negative threshold (economic contraction)
    """
    # Initialize signals DataFrame with same structure as input
    signals = pd.DataFrame(0, index=gdp_growth.index, columns=gdp_growth.columns)
    
    # Calculate rolling average and generate signals for each column
    for column in gdp_growth.columns:
        rolling_gdp = gdp_growth[column].rolling(window=lookback_period).mean()
        signals.loc[rolling_gdp > threshold, column] = 1     # Bullish signal
        signals.loc[rolling_gdp < -threshold, column] = -1   # Bearish signal
    
    return signals

# Create trading signals
gdp_signals = generate_gdp_signals(gdp_growth)
