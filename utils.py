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