
from model_utils import generate_model_positions_pretrained, get_features_and_target
# foward looking bias check
def forward_looking_bias_model(ret, pos_func, **kwargs):
    """
    Checks for forward-looking bias by comparing positions on the full return dataset
    versus a truncated version (dropping the last 20 rows).
    Returns True if a bias is detected, False otherwise.
    """
    pos_full, model_full = pos_func(ret, **kwargs)
    pos_short, model_short = pos_func(ret.iloc[:-20], **kwargs)
    print(pos_full.shape)
    print(pos_short.shape)
    # Compare on common indices:
    common_idx = pos_short.index
    diff = (pos_full-pos_short).abs().sum()
    #diff = (pos_full.loc[common_idx] - pos_short).abs().sum().sum()
    if diff.sum() > 0:
        print(f'Forward looking bias detected in {kwargs.get("model_name", "model")}')
        return True
    else:
        print(f'No forward looking bias detected in {kwargs.get("model_name", "model")}')
        return False
 

def test_hindsight_bias_using_target(ret, pretrained_model):
    """
    Test for hindsight bias by comparing the risk-adjusted positions produced by the pretrained model 
    with those produced by the target signal (shifted by 2 days) and risk adjusted in the same way.
    
    The target is computed without any future data. If the model's positions too closely mimic 
    the shifted target, it could be a sign that the model is inadvertently using future (hindsight) information.
    """

    # Parameters for the test
    trend_window = 3   # use small windows for synthetic testing
    vol_window = 3
    risk_window = 20
    min_periods = 1
    smoothing_window = 5

    # Compute target from a single asset's return series using your helper
    # (Assuming ret has a column named "asset")
    _, target = get_features_and_target(ret['asset'], trend_window=trend_window, vol_window=vol_window)
    
    # Create the "ideal" positions: since the trade is executed 2 days later,
    # shift the target by 2 days.
    ideal_pos = target.shift(2)
    
    # Risk-adjust the ideal positions using the same method as in your model functions.
    # First, compute the rolling risk based on a two-day lag.
    risk_ideal = (ideal_pos.shift(2) * ret['asset']).dropna().rolling(window=risk_window, min_periods=min_periods).std()
    ideal_adjusted = ideal_pos.div(risk_ideal)
    
    # Now, get the risk-adjusted positions from the pretrained model function.
    # (Assuming ret is a DataFrame with a column named "asset")
    pos_adjusted = generate_model_positions_pretrained(
        ret,
        model=pretrained_model,
        trend_window=trend_window,
        vol_window=vol_window,
        risk_window=risk_window,
        smoothing_window=smoothing_window,
        min_periods=min_periods,
    )
    
    # Align indices: restrict to the common indices where both series are defined.
    common_idx = pos_adjusted.index.intersection(ideal_adjusted.index)
    model_positions = pos_adjusted.loc[common_idx, 'asset']
    ideal_positions = ideal_adjusted.loc[common_idx]
    
    # Evaluate the similarity between model and ideal positions.
    avg_abs_diff = (model_positions - ideal_positions).abs().mean()
    corr = model_positions.corr(ideal_positions)
    
    print("Average absolute difference:", avg_abs_diff)
    print("Correlation between model and shifted target positions:", corr)
    
    # Define a threshold: if the correlation is extremely high (e.g. > 0.99)
    # or the average difference is extremely low, the model might be using future data.
    if corr > 0.99 or avg_abs_diff < 1e-6:
        raise AssertionError("Hindsight bias detected: model's risk-adjusted positions too closely match the shifted target.")
    else:
        print("Hindsight bias test passed: model's positions do not match the shifted target too closely.")
        


# check if short selling is done correctly
def check_short_selling(pos, model_ret):
    """
    Check if short selling is done correctly by comparing negative positions
    with their respective returns to ensure that the sign of the return is positive.

    pos: array of positions
    model_ret: array of returns ((pos*ret).dropna(how='all')) if using returns calculated as prices.diff()
    """
    # MAKE SURE THAT RETURNS ARE SHIFTED BY 2 DAYS AHEAD, OTHERWISE APPLY
    # model_ret = model_ret.shift(2)

    num_rows = min(pos.shape[0], model_ret.shape[0])
    num_cols = min(pos.shape[1], model_ret.shape[1])

    counter = 0 

    for i in range(num_rows):
        for j in range(num_cols):
            if pos[i, j] < 0 and ret[i, j] > 0:
                print(f"Short selling is done correctly at {counter} position(s)")
                counter += 1
                if counter > 10: # just for fun, look to see that there are 
                    return





# @Jannik: 
# log returns for features 
# log_returns = np.log(prices / prices.shift(1))