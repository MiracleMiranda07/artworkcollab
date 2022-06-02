import pandas as pd

def normalize_df(df, feature):
    """
    Normalising and scaling the values
    
    """
    
    norm_df=(df-df.min())/(df.max()-df.min())
    scaled_df = norm_df.div(norm_df.sum(axis = 1), axis = 0)
    scaled_df.columns = scaled_df.columns.str.lstrip(f'{feature}_')
    return scaled_df.T

def sort_weights(df, n):
    """
    sorting the values individually
    
    """
    
    temp_weight = [i for i in zip(df.index, df)]
    temp_weight.sort(key=lambda x: x[1], reverse = True)
    
    if len(temp_weight) > n:
        temp_weight = temp_weight[0:n]

    return temp_weight

def user_profile_enrichment(user_df, n):
    """
    Prepare the user profile enrichment data as a dictionary from the user vector

    ---------- Input ---------------
    user_df: user vector
    n: Top n values to be taken for a feature

    ---------- output ---------------
    weight_dict: user as key and his weight on particular feature(tuple) as value

    """

    features = ['artwork_medium', 'materials', 'artwork_year', 'artwork_price']

    try:
        user_df.set_index("user_id", inplace = True)
    except:
        pass
    
    users = set(user_df.index)
    encoded_columns = list(user_df.columns)

    artwork_medium_columns = [i for i in encoded_columns if features[0] in i]
    artwork_materials_columns = [i for i in encoded_columns if features[1] in i]
    artwork_year_columns = [i for i in encoded_columns if features[2] in i]
    artwork_price_columns = [i for i in encoded_columns if features[3] in i]  

    medium_df = normalize_df(user_df[artwork_medium_columns], features[0])
    material_df = normalize_df(user_df[artwork_materials_columns], features[1])
    year_df = normalize_df(user_df[artwork_year_columns], features[2])
    price_df = normalize_df(user_df[artwork_price_columns], features[3])

    weight_dict = {}
    for i in users:
        feature_weight_dict = {}

        feature_weight_dict['medium_weight'] = sort_weights(medium_df[i], n)
        feature_weight_dict['material_weight'] = sort_weights(material_df[i], n)
        feature_weight_dict['year_weight'] = sort_weights(year_df[i], n)
        feature_weight_dict['price_weight'] = sort_weights(price_df[i], n)

        weight_dict[i] = feature_weight_dict

    return weight_dict
