import numpy as np
from collab.config_params_artwork import Config as config

def train_val_split(R, val_perc= config.val_perc):
    
    """
    Splitting R into train and val
    
    input: 
    R: R matrix
    val_perc: pecentage for val data
    
    output: train data, val data
    """
    
    val = np.zeros(R.shape)
    train = R.copy()
    
    dist = {1, 2, 3, 4, 5}
    
    for user in range(R.shape[0]):
        
        user_dist = set([round(i) for i in set(R[user, :]) if i>0])
        
        if len(dist - user_dist) <= 1:

            size = round(val_perc * len(np.nonzero(R[user, :])[0]))

            val_R = np.random.choice(R[user, :].nonzero()[0], size=size, replace=False)
            train[user, val_R] = 0.
            val[user, val_R] = R[user, val_R]
        
    # val and training are truly disjoint
    assert(np.all((train * val) == 0)) 
    return train, val
