import numpy as np
import pandas as pd

def matrix_generation(upsampled_df, verbose=True):

    """
    Generates an user-item interaction array
    
    input: Upsampled dataframe
    
    output: 
    R: user-item interaction array
    data: user-item interaction pivot dataframe
    
    """
        
    if verbose: print('\n------------R Matrix Creation------------')

    data = upsampled_df[['item_id', 'user_id', 'rating']]
    data.drop_duplicates(subset=['item_id', 'user_id'])

    #creating matrix
    data = pd.pivot_table(data,values='rating',index='user_id',columns='item_id')
    data = data[data.index != 'temp_user']
    data=data.fillna(0)

    #input matrix
    R = np.array(data)
    
    #checking sparsit
    sparsity = 1.0-(np.count_nonzero(R) / float(np.prod(R.shape)) )
    if verbose: 
        print(f'\nR matrix shape: {R.shape}')
        print(f'\nR matrix sparsity: {sparsity}')

    return R ,data
