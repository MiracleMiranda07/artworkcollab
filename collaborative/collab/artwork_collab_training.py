import os
import json
import pickle
import numpy as np
import pandas as pd
from collab.preprocessing import preprocessing
from collab.matrix_generation import matrix_generation
from collab.artwork_collab_recsys import MatrixFactorization
from collab.model_evaluation import ModelEvaluation
from collab.train_test_split import train_test_split
from collab.config_params_room import Config as config


class Collaborative_Room():
    
    def __init__(self,utilitymatrix_room,all_artwork_profile,ovr_artwork_profile):
        
        self.utilitymatrix_room=utilitymatrix_room
        self.all_artwork_profile=None
        self.ovr_artwork_profile=ovr_artwork_profile
    
    def fit(self,utilitymatrix_room,all_artwork_profile,ovr_artwork_profile):
         
        """
        Room collab model training flow

        ---------- Input ---------------
        dataframe : utility matrix-room

        ---------- output ---------------
        Recommendation 
        """ 

        
        df=utilitymatrix_room
        all_artwork_profile=None
        ovr_artwork_profile=(ovr_artwork_profile)
        
        all_avail_artwork_profile =None
        current_ovr_rooms= list(set(ovr_artwork_profile['ROOM_ID']))
        
        preprocessed_df = preprocessing(df,all_avail_artwork_profile,min_user_item_thrsh=config.min_user_item_thrsh,verbose=True)
        preprocessed_df.shape
        
        val_df = train_test_split(preprocessed_df,config.test_split_thresh)

        try:
            val_users=val_df['user_id'].to_list()
            val_items=val_df['item_id'].to_list()
        except:
            val_users = []
            val_items = []
            print('\nNo Validation set created')
        
        filtered_df=preprocessed_df[~(preprocessed_df['user_id'].isin(val_users) & preprocessed_df['item_id'].isin(val_items))]
        
        existing_items = list(filtered_df.item_id.unique())
        temp_df = pd.DataFrame()

        for i in current_ovr_rooms:
            if i not in existing_items:
                temp_df = temp_df.append({'user_id': 'temp_user', 'item_id': i, 'rating': float(1)}, ignore_index=True)
                
        frames = [filtered_df, temp_df]
        completed_df = pd.concat(frames)
        
        R ,data = matrix_generation(completed_df, verbose=True)
        
        latent_features = config.latent_features
        print('latent_features: ', latent_features)
        
        n_users, n_items = R.shape
        item_vecs = np.random.normal(scale=1./latent_features, size=(n_items, latent_features))
        user_vecs = np.random.normal(scale=1./latent_features, size=(n_users, latent_features))
        
        reg = config.regularization
        
        model = MatrixFactorization(R, data, user_vecs, item_vecs,\
                            item_fact_reg=reg, user_fact_reg=reg, \
                            user_bias_reg=reg, item_bias_reg=reg,val_df=val_df)
                            
        model.fit(epoch=config.epoch, n_iter=config.iteration, learning_rate=config.learning_rate)
        results = model.prediction_conversion(current_ovr_rooms)
        total_results = model.prediction_conversion()
        
        try:
            actual_pred_df = model.actual_pred_df(val_df)
            val_set=actual_pred_df.values.tolist()

            model_eval = ModelEvaluation(actual_pred_df)
            map_k_dict = model_eval.precision_recall_at_k(val_set,config.k,config.mapk_threshold) 
            self.map_k = json.dumps(map_k_dict, indent = 4)
        except Exception as e:
            self.map_k = None
            print(f'\nmodel validation failed with error: {e}')
        
        self.result=results
