"""

Affinity model takes user_tracking_event and collection_details as a input to 
create the user-item-rating matrix.

"""
import pickle
import pandas as pd
import datetime
from tqdm import tqdm
import os
import time
import statistics
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Useraffinityartwork():

    def __init__(self,tracking_df,collection_df):
        
        self.tracking_df = tracking_df
        self.collection_df=collection_df
        
    def useraffinity_preprocessing(self,tracking_df):
        
        """
        preprocessing the dataframe
        
        """

        print('-----------Useraffinityartwork preprocessing started-----------')
            
        df=tracking_df
        df = df.reset_index(drop=True)
              
        #filtration on needed columns
        df_fil=df[['ITEM_ID','PERFORMED_ACTION','USER_COMBINED_ID','CREATED_DATE']]
        df_fil=df_fil[~df_fil['PERFORMED_ACTION'].isin([16,17])]
        
        print(df_fil.shape)
        
        #renaming the columns
        df_fil = df_fil.rename({'ITEM_ID': 'item_id', 'PERFORMED_ACTION': 'PerformedAction','USER_COMBINED_ID':'user_id','CREATED_DATE':'timestamp'}, axis=1)
        
        #drop duplicates
        df_fil=df_fil.drop_duplicates(subset=['user_id','item_id','timestamp','PerformedAction'])
        
        print('Dataset Shape: ',df_fil.shape)
        
        #drop missing userid_records
        df_fil = df_fil[df_fil['user_id'].notna()]
        
        #initial sorting
        df_fil.sort_values(by=['user_id','timestamp','PerformedAction'],ascending=True,
                           na_position='first',inplace=True)
                           
        preprocessed_df = df_fil.drop_duplicates(subset=['timestamp','user_id'], keep="first")
        
        print('Dataset Shape after removing duplicates: ',preprocessed_df.shape)
        
        return preprocessed_df
        
    def dwelltime_visits_calculation(self,preprocessed_df):
        
        """
        deriving dwelling time for each artwork 
        
        ---------- Input ---------------
        dataframe: preprocessed dataframe
        
        ---------- output ---------------
        dataframe: dataframe with dwelltime and no of visits for each artwork
        
        """   
        
        #deriving dwelltime for each artwork by finding the time difference between two performed action
        preprocessed_df["time"] = pd.to_datetime(preprocessed_df["timestamp"])
        preprocessed_df['new_date_col'] = preprocessed_df['time'].dt.date
        preprocessed_df["dwell"] = preprocessed_df.groupby([preprocessed_df.user_id,preprocessed_df.new_date_col])["time"].diff(periods=-1)/np.timedelta64(1,'s') 
        preprocessed_df["dwell"] = preprocessed_df.groupby(['user_id','new_date_col'], sort=False)['dwell'].apply(lambda x: x.fillna(x.mean()))
        preprocessed_df["dwell"]=preprocessed_df["dwell"].fillna(0)
        preprocessed_df["dwell"]  =preprocessed_df["dwell"].abs()
        
        
        #filtration on dwelltime >3 (threshold)
        filtered_dataframe=preprocessed_df[(preprocessed_df['dwell'] > 3)]
        
        #filtration on performed action ='ENTER_ARTWORK_DETAIL(2)'
        filtered_dataframe=filtered_dataframe[filtered_dataframe['PerformedAction'].isin([2])]
        filtered_dataframe['item_id'].isnull().any()
        
        filtered_dataframe=filtered_dataframe[['user_id','item_id','PerformedAction','dwell']]
        filtered_dataframe['item_id']=filtered_dataframe['item_id'].astype(float)
        
        #aggregation on visits and dwelltime for all artworks 
        tracking_aggregated_df=filtered_dataframe.groupby(by=['user_id','item_id','PerformedAction'])['dwell'].agg(['mean','count']).reset_index().rename(columns={'mean':'dwell','count':'visits'})
        
        return tracking_aggregated_df
        
    def Merge_collections_with_tracking(self,collection_df,tracking_aggregated_df):
        
        """
        Merge tracking_data with collection_data to find whether the user has any collected items or not
        
        ---------- Input ---------------
        dataframe: aggregated tracking dataframe ,collection dataframe
        
        ---------- output ---------------
        dataframe: dataframe with is_collected column (collected(1) or notcollected(0))
        
        """   


        #Merging collections and collection_items to check the favourite items
        df_final_fil=collection_df

        #drop duplicates
        df_final_fil=df_final_fil.drop_duplicates(subset=['OWNER_USER','ENTITY_ID'])

        #entity_type filtration on artworks with artworkid--2      
        df_final_fil=df_final_fil[df_final_fil['ENTITY_TYPE'].isin(['2'])]
                        
        #creating a boolean column for collection
        df_final_fil['is_collected'] = 1

        #Merging user Tracking event and collections to connect the user with corresponding favourites
        collections=df_final_fil
        tracking=tracking_aggregated_df
         
        tracking["item_id"] = tracking["item_id"].astype(float)
        tracking["item_id"] = tracking["item_id"].astype(int)

        collection_tracking=pd.merge(tracking, collections, how='left', left_on=["user_id","item_id"], right_on=['OWNER_USER','ENTITY_ID'])

        collection_tracking['is_collected'] = collection_tracking['is_collected'].fillna(0)
        collection_tracking['is_collected']=collection_tracking['is_collected'].astype(int)

        collection_tracking_df=collection_tracking[['user_id','item_id','PerformedAction','dwell','visits','is_collected']]
        
        return collection_tracking_df
    
    def log_calculation(self,collection_tracking_df):
        
        """
        deriving the log value for dwelltime 
        
        ---------- Input ---------------
        dataframe: collection_tracking_dataframe
        
        ---------- output ---------------
        dataframe: dataframe with logtransform on dwelltime column
        
        """   
    
        #deriving log value dwell time - Non Linear transformation
        users = list(set(collection_tracking_df.user_id))
        result_df=pd.DataFrame()
        for i in users:
            temp_df = collection_tracking_df[collection_tracking_df['user_id']==i]
            temp_df.reset_index(inplace=True) 
            for idx,row in temp_df.iterrows():
                log_dwell=np.log10(row['dwell'])
                temp_df.loc[idx, "log_dwell"] =log_dwell                
            result_df = pd.concat([result_df,temp_df]) 
            
        return result_df
            
    def zscore_calculation(self,result_df):
        
        """
        deriving the zscore value to find the deviation on dwelltime for each user
        binning the no of visits between 1-5 range
        
        ---------- Input ---------------
        dataframe: collection_tracking_dataframe with logtransform
        
        ---------- output ---------------
        dataframe: dataframe with dwelltime zscore and scaledvisits
        
        """  
        

        result_df = result_df[result_df['user_id'].map(result_df['user_id'].value_counts()) > 1]
        
        #deriving zscore value using mean logvalue on dwelltime - for getting significant ratings
        users = list(set(result_df.user_id))
        resulted_df=pd.DataFrame()
        for i in users:
            temp_df = result_df[result_df['user_id']==i]
            temp_df.reset_index(inplace=True) 
            for idx,row in temp_df.iterrows():
                log_dwelllist=temp_df['log_dwell'].tolist()                  
                #dwell
                mean_value=statistics.mean(log_dwelllist)
                median_value=statistics.median(log_dwelllist)
                stdev_value=statistics.stdev(log_dwelllist)
                if stdev_value!=0:
                    dwell_zscore=(row['log_dwell']-mean_value)/stdev_value
                else:
                    dwell_zscore=stdev_value                       
                temp_df.loc[idx, "dwell_zscore"] =dwell_zscore   
                
            #binning the no.of.visits to 1-5 scale range
            no_of_visits=pd.cut(temp_df['visits'], bins=5,labels=[1,2,3,4,5],right=False,include_lowest=True).to_list()
            temp_df.insert(7, 'visit_weight', no_of_visits)
            
            resulted_df = pd.concat([resulted_df,temp_df]) 
            
        return resulted_df
        
        
    def scaling_value(self,resulted_df):
        
        """
        binning the dwelltime between 1-5 range
        
        ---------- Input ---------------
        dataframe: dataframe with dwelltime zscore and scaledvisits
        
        ---------- output ---------------
        dataframe: dataframe with scaleddwelltime and scaledvisits
        
        """  
    
        scaled_df=resulted_df[['user_id','item_id','dwell','visits','log_dwell','dwell_zscore','visit_weight','is_collected']]
        scaled_df['dwell_zscore'] = scaled_df['dwell_zscore'].round(decimals=4)
        #scaling dwelltime bewteen 1-5  range
        scaled_df.loc[(scaled_df['dwell_zscore']>=0) & (scaled_df['dwell_zscore']<=0.5), 'dwell_weight'] =3
        scaled_df.loc[(scaled_df['dwell_zscore']>0.5) & (scaled_df['dwell_zscore']<=1), 'dwell_weight'] =3.5
        scaled_df.loc[(scaled_df['dwell_zscore']>1) & (scaled_df['dwell_zscore']<=1.5), 'dwell_weight'] =4
        scaled_df.loc[(scaled_df['dwell_zscore']>1.5) & (scaled_df['dwell_zscore']<=2), 'dwell_weight'] =4.5
        scaled_df.loc[(scaled_df['dwell_zscore']>2), 'dwell_weight'] =5
        scaled_df.loc[(scaled_df['dwell_zscore']<0) & (scaled_df['dwell_zscore']>=-0.5), 'dwell_weight'] =2.5
        scaled_df.loc[(scaled_df['dwell_zscore']<-0.25) & (scaled_df['dwell_zscore']>=-0.50), 'dwell_weight'] =2
        scaled_df.loc[(scaled_df['dwell_zscore']<-0.50) & (scaled_df['dwell_zscore']>=-0.75), 'dwell_weight'] =1.5
        scaled_df.loc[(scaled_df['dwell_zscore']<-0.75) & (scaled_df['dwell_zscore']>=-1), 'dwell_weight']=1
        scaled_df.loc[(scaled_df['dwell_zscore']<-1), 'dwell_weight']=0.5
                     
        return scaled_df
        
    def rating_calculation(self,scaled_df):
        
        """
        assigining weight for both dwelltime and visits for each user
        
        ---------- Input ---------------
        dataframe: dataframe with dwelltime scaledzscore and scaledvisits
        
        ---------- output ---------------
        dataframe: dataframe with user-item-rating
        
        """  
        #assigining weight for both dwelltime and visits for each user
        users = list(set(scaled_df.user_id))
        final_df=pd.DataFrame()

        for i in users:
            temp_df = scaled_df[scaled_df['user_id']==i]
            temp_df.reset_index(inplace=True) 
            threshold=temp_df[temp_df['is_collected']==1]
            
            #weightage rule for collected user
            if (threshold.shape[0]>=3):
                temp_df.loc[temp_df['is_collected'] == 0, 'rating'] =(temp_df['dwell_weight']*0.4+temp_df['visit_weight']*0.2).round(2)

                temp_df.loc[temp_df['is_collected'] == 1, 'rating'] =(temp_df['dwell_weight']*0.2+temp_df['visit_weight']*0.2+3).round(2)


            #weightage rule for non-collected-user
            else:
                temp_df.loc[temp_df['is_collected'] == 0, 'rating'] =(temp_df['dwell_weight']*0.6+temp_df['visit_weight']*0.4).round(2)
                temp_df.loc[temp_df['is_collected'] == 1, 'rating'] =(temp_df['dwell_weight']*0.2+temp_df['visit_weight']*0.2+3).round(2)

            final_df = pd.concat([final_df,temp_df]) 
            
        
        return final_df
        
    def fit(self,tracking_df,collection_df):
    
        start_time = time.time()
        
        print()
                
        preprocessed_df=self.useraffinity_preprocessing(tracking_df)
        
        tracking_aggregated_df=self.dwelltime_visits_calculation(preprocessed_df)
        
        collection_tracking_df=self.Merge_collections_with_tracking(collection_df,tracking_aggregated_df)

        result_df=self.log_calculation(collection_tracking_df)

        resulted_df=self.zscore_calculation(result_df)

        scaled_df=self.scaling_value(resulted_df)

        self.result=self.rating_calculation(scaled_df)
        
        #self.result=self.result[['user_id','item_id','rating']]
                   
        print("Time taken to train useraffinity artwork model: %s seconds" % (time.time() - start_time))

                                   

                    

        

        
