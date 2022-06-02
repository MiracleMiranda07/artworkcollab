import os
import json
import pickle
import numpy as np
import pandas as pd
import collab.user_profile_enrichment as upe
from collab.preprocessing import preprocessing
from collab.matrix_generation import matrix_generation
from collab.item_vector import item_vector_calc
from collab.artwork_collab_recsys import MatrixFactorization
from collab.model_evaluation import ModelEvaluation
from collab.train_test_split import train_test_split
from collab.config_params_artwork import Config as config


# +
class Collaborative_Artwork():
    
    def __init__(self,utilitymatrix_artwork,all_artwork_profile,ovr_artwork_profile):
        
        self.utilitymatrix_artwork=utilitymatrix_artwork
        self.all_artwork_profile=all_artwork_profile
        self.ovr_artwork_profile=ovr_artwork_profile

    def fit(self,utilitymatrix_artwork,all_artwork_profile,ovr_artwork_profile):
        
        """
        Artwork collab model training flow

        ---------- Input ---------------
        dataframe : utility matrix

        ---------- output ---------------
        Recommendation 
        """ 

        df=utilitymatrix_artwork
        all_artwork_profile=all_artwork_profile
        ovr_artwork_profile=ovr_artwork_profile
        
#         print(df.head(5))

        all_avail_artwork_profile = list(all_artwork_profile.ARTWORK_ID.unique())
        current_ovr_artworks = list(set(ovr_artwork_profile['ARTWORK_ID']))

        #preprocessing    
        preprocessed_df = preprocessing(df, all_avail_artwork_profile, min_user_item_thrsh=config.min_user_item_thrsh, verbose=True)

        #valset creation
        val_df = train_test_split(preprocessed_df,config.test_split_thresh)
        
        try:
            val_users=val_df['user_id'].to_list()
            val_items=val_df['item_id'].to_list()
        except:
            val_users = []
            val_items = []
            print('\n No Validation set created')

        #train set creation
        filtered_df=preprocessed_df[~(preprocessed_df['user_id'].isin(val_users) & preprocessed_df['item_id'].isin(val_items))]

        existing_items = list(filtered_df.item_id.unique())
        temp_df = pd.DataFrame()

        for i in current_ovr_artworks:
            if i not in existing_items:
                temp_df = temp_df.append({'user_id': 'temp_user', 'item_id': i, 'rating': float(1)}, ignore_index=True)

        frames = [filtered_df, temp_df]
        completed_df = pd.concat(frames)

        #creation of input matrix
        R ,data = matrix_generation(completed_df, verbose=True)

        #item weights-encoding
        encoded_item_data, encoded_item_array = item_vector_calc(data.columns, all_artwork_profile, verbose=True)

        latent_features = encoded_item_array.shape[1]
        print('latent_features: ', latent_features)
        
        n_users, n_items = R.shape
        item_vecs = encoded_item_array
        user_vecs = np.random.normal(scale=1./latent_features, size=(n_users, latent_features))   
        
        reg= config.regularization
        
        #model training
        model = MatrixFactorization(R, data, user_vecs, item_vecs, \
                                item_fact_reg=reg, user_fact_reg=reg, \
                                user_bias_reg=reg, item_bias_reg=reg,val_df=val_df)

        model.fit(epoch=config.epoch, n_iter=config.iteration, learning_rate=config.learning_rate)
        
        #remapping the result
        results = model.prediction_conversion(current_ovr_artworks)
        total_results = model.prediction_conversion()

        
        
        # UPE
        user_vec_df = pd.DataFrame(model.user_vecs, columns=encoded_item_data.columns, index=data.index)
        upe_dict = upe.user_profile_enrichment(user_vec_df, config.n_upe)
        self.upe = json.dumps(upe_dict, indent = 4)
        self.user_vec_df = user_vec_df.reset_index()
        
        try:

            #evaluation
            actual_pred_df = model.actual_pred_df(val_df)
            val_set=actual_pred_df.values.tolist()

            model_eval = ModelEvaluation(actual_pred_df)
            map_k_dict = model_eval.precision_recall_at_k(val_set,config.k,config.mapk_threshold) 
            self.map_k = json.dumps(map_k_dict, indent = 4)
        except:
            self.map_k = None
            print(f'\nmodel validation failed')
        
        self.result=results

