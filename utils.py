
# foward looking bias check

def forward_looking_bias_model(ret, pos_func, **kwargs):
    """
    Checks for forward-looking bias by comparing positions on the full return dataset
    versus a truncated version (dropping the last 20 rows).
    Returns True if a bias is detected, False otherwise.
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
        return True
    else:
        print(f'No forward looking bias detected in {kwargs.get("model_name", "model")}')
        return False


# GARCH model 


# @Jannik: 
# log returns for features 
# log_returns = np.log(prices / prices.shift(1))