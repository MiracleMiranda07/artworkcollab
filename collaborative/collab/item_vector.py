import os
import numpy as np  
from collab.artwork_content_recsys import ArtworkContent

#Defining Item vectors as input for Matrix factoroization Art Colab Model
def item_vector_calc(item_list,content_df, verbose=True):
    """
    deriving the item vector for matrix factorization

    ---------- Input ---------------
    item_list: list of Artwork ids to be encoded
    content_df: Content data full dump

    ---------- output ---------------
    data: Encoded item vector dataframe

    """ 
    
    if verbose: print('\n------------Item vector Creation------------')

    content_df = content_df[content_df['ARTWORK_ID'].isin(item_list)]
    features = ['artwork_id', 'artwork_medium', 'materials', 'artwork_year', 'artwork_price']

    artwork_content_recsys = ArtworkContent(content_df)

    data, nan_values = artwork_content_recsys.preprocessing(features)
    data= artwork_content_recsys.nlp_material(data)
    data = artwork_content_recsys.bucket_price(data)
    data = artwork_content_recsys.bucket_year(data)

    data.rename(columns = {'lemmatized_materials':'materials',
                              'artwork_period':'artwork_year', 'artwork_price_range':'artwork_price'}, inplace = True)

    data = artwork_content_recsys.impute_missing(data, nan_values, impute=False)
    
    catergorical_columns = list(data.columns)
    encoded_data = artwork_content_recsys.one_hot_encoding(data, catergorical_columns)

    if verbose: 
        print(f'\nArtwork features for item vector: {catergorical_columns}')
        print('\nEncoded data shape: ', encoded_data.shape)
    
    return encoded_data, np.array(encoded_data)
