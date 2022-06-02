import pandas as pd

def preprocessing(dataframe, featured_artwork, min_user_item_thrsh, verbose=False):

    """
    Collaborative model Preprocessing
    
    input:
    dataframe: user-item-rating csv
    model: artwork/ room
    
    output: Preprocessed dataframes
    """
    
    if verbose: print('------------Data Preprocessing------------')
    
    pd.set_option('mode.chained_assignment',None)
    
    org_df = dataframe

    
    df = org_df[['item_id', 'user_id', 'rating']]
    if verbose: print('original shape',df.shape)
            
    #duplicate check
    df = df.drop_duplicates()
    if verbose: print('after dups check',df.shape)
    
    if verbose: 
        print(f'\nNumber of unique Items before filtration: {df.item_id.nunique()}')
        print(f'Number of unique users before filtration: {df.user_id.nunique()}')
        print(f'Total no.of.rows before filtration: {df.shape[0]}')

    min_user_ratings = min_user_item_thrsh[0]
    min_item_ratings = min_user_item_thrsh[1]
    
    # Filtering the Artworks that has been visited less than the given threshold
    df['item_freq'] = df.groupby('item_id')['item_id'].transform('count')
    df = df[df['item_freq'] >= min_item_ratings]
    
    # Filtering the Artworks without Artwork profile/ that has no data in Database
    if featured_artwork:
        df = df[df['item_id'].isin(featured_artwork)]
    
    
    # Filtering the User ids who visited less than the given threshold number of atrtworks
    df['user_freq'] = df.groupby('user_id')['user_id'].transform('count')
    df = df[df['user_freq'] >= min_user_ratings]
    

    if verbose: 
        print(f'\nNumber of unique Items after filtration: {df.item_id.nunique()}')
        print(f'Number of unique users after filtration: {df.user_id.nunique()}')
        print(f'Total no.of.rows after filtration: {df.shape[0]}')

    return df[['item_id', 'user_id', 'rating']]
