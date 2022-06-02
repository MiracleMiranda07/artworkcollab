# +
# Importing required packages
import re
import nltk
import time
import string
import numpy as np
import pandas as pd

from nltk.corpus import wordnet, stopwords
from scipy.spatial.distance import pdist, jaccard, squareform
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('mode.chained_assignment',None)


# -

class RoomContent:
    """
       Room Content Model training usiing Artwork features
        
    """

    def __init__(self, dataFrame):
        """
        initializing the dataframe
        
        """
        # renaming the raw column names to lower case names 
        dataFrame.rename(columns = {'ARTWORK_ID':'artwork_id', 'ARTIST_ID':'artist_id', 'ROOM_ID':'room_id', 
                                'ARTWORK_PRICE':'artwork_price', 'GALLERY_ID':'gallery_id', 'ARTWORK_MEDIUM':'artwork_medium',
                                'MATERIALS':'materials', 'ARTWORK_YEAR':'artwork_year', 'GALLERY_ESTABLISHED_YEAR':'gallery_established_year'
                           }, inplace = True)
        self.dataFrame = dataFrame
        
        
    def preprocessing(self, features):
        """
        Preprocessing the input dataframe for the training process
        
        ---------- Input ---------------
        features: The feature that needs to be considered for calculating the similarity
        
        ---------- output ---------------
        data: Preprocessed datafram
        nan_values: missing feature values
        
        """
        data = self.dataFrame[features]
        artwork_features = features
        artwork_features.remove('room_id')
        
        # finding the missing values in the features
        nan_values = {i: list(data[data[i].isna()]['room_id']) for i in artwork_features}

        # imputing the missing values
        data = self.impute_missing(data, nan_values, impute=True)
        
        # dropping null rows and duplicate rows
        data = data.dropna()
        print(f'Dataset shape: {data.shape}')

        # indexing Room id for the dataframe
        data.set_index('room_id', inplace=True)


        return data, nan_values

    def impute_missing(self, data, nan_values, impute = True):
        """
        Imputes missing values
        
        ---------- Input ---------------
        data: Preprocessed datafram
        nan_values: missing feature values
        impute: imputes value if True, remove imputed values if False
        
        ---------- output ---------------
        data: Imputed datafram
        
        """ 
        
        if impute:
            impute_values ={
            'artist_id' : 0, 'artwork_medium' : 'dummy_medium', 'materials' : 'dummy_materials',
            'gallery_established_year' : int(data.gallery_established_year.mode())
            }

            for item, value in nan_values.items():
                if len(value) > 0:
                    data.loc[data['room_id'].isin(value), item] = impute_values[item]
        else:
            data = data.reset_index()
            for item, value in nan_values.items():
                if len(value) > 0:
                    data.loc[data['room_id'].isin(value), item] = np.nan
            data.set_index('room_id', inplace=True)
        
        return data
    
    def create_full_text(self, df):
        """
        cleans the given text
        
        ---------- Input ---------------
        df: text to be cleaned
        
        ---------- output ---------------
        full_text: cleaned text
        
        """  
        
        full_text = ''
        for idx, row in df.iterrows():
            for i in row:
                if i != '':
                    full_text += ' '+ i
                    
        return full_text
        
    def create_bow(self, data):
        """
        creates a bag of words for each room
        
        ---------- Input ---------------
        data: dataframe 
        
        ---------- output ---------------
        df: dataframe with bow for every room
        
        """  
        
        df = pd.DataFrame(columns=['bow'])
        for i in set(data.index):
            temp = data.copy()
            temp = temp[temp.index == i]
            full_text = self.create_full_text(temp)
            df.loc[i] = full_text
        return df
        
    def clean_text(self, txt):
        """
        cleans the given text
        
        ---------- Input ---------------
        txt: text to be cleaned
        
        ---------- output ---------------
        temp_text: cleaned text
        
        """   
        stopwords = nltk.corpus.stopwords.words('english')
        txt = txt.lower()
        txt = "".join([c for c in txt if c not in string.punctuation])
        tokens = re.split('\W+', txt)
        txt = [word for word in tokens if word not in stopwords]
        temp_text = None
        for i, j in enumerate(txt):
            if i != 0:
                temp_text += '_' + j
            else:
                temp_text = j
        return temp_text
    
    def split_column(self, data, feature):
        """
        split a column into two or more columns
        if an room has two or more values in one column(feature)
        
        ---------- Input ---------------
        data: dataframe
        feature: artist id or gallery id
        
        ---------- output ---------------
        data: dataframe with splitted columns
        
        """ 
        data[feature] = data[feature].apply(str)

        artist_df = data[feature].str.split(',', expand=True)
        artist_df = artist_df.fillna(value=np.nan)
        artist_df = artist_df.add_prefix(feature)

        data = pd.concat([data, artist_df], axis=1)
        data = data.drop(feature, 1)
        return data
        
    def one_hot_encoding(self, data, catergorical_columns):
        """
        Encoding the categorical columns using one hot encoding method
        
        ---------- Input ---------------
        data: dataframe
        catergorical_columns: columns to be encoded
        
        ---------- output ---------------
        data: encoded dataframe
        
        """ 
        return pd.get_dummies(data, columns = catergorical_columns)
        
        
    def bow_vectorizer(self, df):
        """
        vectorizes the bow
        
        ---------- Input ---------------
        data: dataframe
        
        ---------- output ---------------
        data: tfidf_df vectors
        
        """ 
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df['bow'])
        print(f'Vectorized matrix shape: {tfidf_matrix.shape}')
        tfidf_df = pd.DataFrame(tfidf_matrix.todense(), columns=tfidf.get_feature_names(), index=df.index)
        return tfidf_df
        
    def similarity_results(self, encoded_data, metric):

        """
        Calculate the Similarity between two 
        Artwork/ Room based on the given content 
        with the given distance metric.

        Parameters
        ---------- Input ---------------
        dataFrame: Input content Dataframe

        encoded_data: Dataframe, One hot encoded dataframe of 
            the original dataframe.

        metric : str or function,
            The distance metric to use. 
            Distance metrics: 'cosine', 'dice', 'euclidean', 
            'hamming', 'jaccard'.

        ---------- Output ---------------
        results: Dictionary, A dictionary containing the Key 
        value pair of item and Similar items

        """

        sim =  1 - pdist(encoded_data, metric)

        similarity_matrix = pd.DataFrame(squareform(sim), index=encoded_data.index, columns= encoded_data.index)

        results = {}

        for idx, row in similarity_matrix.iterrows():
            item_list = row.drop(labels = [idx])
            sorted_list = item_list.sort_values(ascending = False)    
            results[idx] = [items for items in sorted_list.iteritems()]  

        return results
        
    def fit(self):
        """
        Driver function for trining the Room content model
        
        """ 
        start_time = time.time()
        print('\n------ Room content model training started ------')

        dataFrame = self.dataFrame
        min_artwork = 3
        self.dataFrame['artwork_count'] = dataFrame.groupby('room_id')['room_id'].transform('count')
        
        filtered_room = set(dataFrame[dataFrame['artwork_count'] < min_artwork]['room_id'])
        self.filtered_room_list = [(room_id, 0) for room_id in filtered_room]
        
        self.dataFrame = dataFrame[dataFrame['artwork_count'] >= min_artwork]

        features = ['room_id',  'artwork_medium', 'materials', 'gallery_established_year', 'artist_id']
        data, nan_values = self.preprocessing(features)

        data['cleaned_medium'] = data['artwork_medium'].apply(lambda x : self.clean_text(x))
        data['cleaned_material'] = data['materials'].apply(lambda x : self.clean_text(x))

        data = data.drop('materials', 1)
        data = data.drop('artwork_medium', 1)

        data.rename(columns = {'cleaned_material':'materials'}, inplace = True)
        data.rename(columns = {'cleaned_medium':'artwork_medium'}, inplace = True)

        data = self.split_column(data, 'artist_id')
        data = self.split_column(data, 'gallery_established_year')

        data = self.impute_missing(data, nan_values, impute=False)

        data = data.fillna('')
        
        df = self.create_bow(data)

        tfidf_df = self.bow_vectorizer(df)
        self.similar_room_results = self.similarity_results(tfidf_df, 'cosine')
        
        self.result = {}
        for key, value in self.similar_room_results.items():
            value.extend(self.filtered_room_list)
            self.result[key] = value
        print("Time taken to train Room content model: %s seconds" % (time.time() - start_time))
