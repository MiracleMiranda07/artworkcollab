import os
import pickle
import numpy as np
import pandas as pd
import argparse
import joblib
import datetime
from io import StringIO
import nltk
import json
from azureml.core import Run
from azureml.core import Workspace
from azureml.core.model import Model
from azure.storage.blob import BlockBlobService

from azureml.core import Workspace, Experiment, Datastore
from azureml.core import Dataset

from test_config import Config as config
from useraffinity.useraffinity_model_artwork import Useraffinityartwork
from collab.artwork_collab_training import Collaborative_Artwork


def get_latest_data(datastore, input_data_path, dataset_name, file_save_path, pick_latest):
    
    """
    
    Fetches latest data from ADL
    
    """
    
    dataset = Dataset.File.from_files(datastore.path(input_data_path))

    # Registering and loading the retrived dataset/ Files
    registered_data = dataset.register(ws, dataset_name, create_new_version=True)
    input_path =  dataset.download(target_path=file_save_path, overwrite=True)

    # Picking latest data from the dataset
    if pick_latest:

        import datetime
        file_dict = {}
        for file in dataset.to_path():
            str_date = file.rsplit(' ', 1)[0].rsplit('_', 2)[2]
            date = datetime.datetime.strptime(str_date, '%Y-%m-%d')
            file_dict[date] = file
        
        latest_data = file_dict[max(file_dict.keys())]
        data_path = ''
        
        for path in input_path:
            if latest_data in path:
                data_path =  path
    else:
        data_path = [path for path in input_path if '.csv' in path]
        
    return data_path


def read_csv(f):
    try:
        return pd.read_csv(f)
    except Exception as e:
        print(f'file :{f} failed while reading with error: {e}')
        return pd.DataFrame()


def get_data(collab_config_json):
    
    """
    
    Fetches all input_data
    
    """
    
    
    # Get Datastore and Pull dataset from the give path
    datastore = Datastore.get(ws, datastore_name=collab_config_json['datastore_name'])
    
    collection_full_load_path = get_latest_data(datastore, collab_config_json['collection_full_load']
                                , collab_config_json['collection_full_load_dataset_name'], collab_config_json['file_save_path']
                                   , pick_latest=True)
    event_full_load_path = get_latest_data(datastore, collab_config_json['event_full_load']
                                , collab_config_json['event_full_load_dataset_name'], collab_config_json['file_save_path']
                                   , pick_latest=True)
    content_full_load_path = get_latest_data(datastore, collab_config_json['content_full_load']
                                , collab_config_json['content_full_load_dataset_name'], collab_config_json['file_save_path']
                                   , pick_latest=True)
    content_OVR_load_path = get_latest_data(datastore, collab_config_json['content_OVR_load']
                                , collab_config_json['content_OVR_load_dataset_name'], collab_config_json['file_save_path']
                                   , pick_latest=True)
    
    collection_delta_load_path = get_latest_data(datastore, collab_config_json['collection_delta_load']
                                , collab_config_json['collection_delta_load_dataset_name'], collab_config_json['file_save_path']
                                   , pick_latest=False)
    event_delta_load_path = get_latest_data(datastore, collab_config_json['event_delta_load']
                                , collab_config_json['event_delta_load_dataset_name'], collab_config_json['file_save_path']
                                   , pick_latest=False)
    
    
    collection = []
    collection.append(collection_full_load_path)
    collection.extend(collection_delta_load_path)
    
    event_tracking = []
    event_tracking.append(event_full_load_path)
    event_tracking.extend(event_delta_load_path)
    
    print('\n*************Data Paths*************')
    print('collection data paths: ', collection)
    print('Event tracking data paths: ', event_tracking)
    print('Complete artwork profile data path: ', content_full_load_path)
    print('Current ovr artwork profile data path: ', content_OVR_load_path)
    
    
    collection_df = pd.concat([read_csv(f) for f in collection ])
    event_tracking_df = pd.concat([read_csv(f) for f in event_tracking ])
    
    all_artwork_profile = pd.read_csv(content_full_load_path, escapechar='\\')
    ovr_artwork_profile = pd.read_csv(content_OVR_load_path, escapechar='\\') 
    
    return event_tracking_df, collection_df, all_artwork_profile, ovr_artwork_profile

def create_utilitymatrix_artwork(tracking_df,collection_df):

    utilitymatrix_artwork = Useraffinityartwork(tracking_df,collection_df)
    utilitymatrix_artwork.fit(tracking_df,collection_df)
    return utilitymatrix_artwork.result

def train_useraffinity_model(tracking_df,collection_df):
    """
    
    Artwork useraffinity_model training
    
    """
            
    print('\n********Artwork utility matrix creation started********')
    
    try:
        utilitymatrix_artwork = create_utilitymatrix_artwork(tracking_df,collection_df)
        print('\n********Artwork utility matrix creation completed successfully********')
    except Exception as e:
        utilitymatrix_artwork = None
        print(f"\nArtwork utility matrix creation failed with Error: {e}")
        
    
    return utilitymatrix_artwork


def create_collab_artwork(utilitymatrix_artwork,all_artwork_profile,ovr_artwork_profile):

    collab_artwork = Collaborative_Artwork(utilitymatrix_artwork,all_artwork_profile,ovr_artwork_profile)
    collab_artwork.fit(utilitymatrix_artwork,all_artwork_profile,ovr_artwork_profile)
    return collab_artwork.result, collab_artwork.upe, collab_artwork.map_k, collab_artwork.user_vec_df

def train_collab_model(utilitymatrix_artwork, all_artwork_profile, ovr_artwork_profile):  
    """
    
    Artwork collab model Training
    
    """
    
    print('\n********Artwork collaborative model Training started********')
    nltk.download('all')
    try:
        collab_artwork, upe, map_k, user_vec_df = create_collab_artwork(utilitymatrix_artwork
                                                                         , all_artwork_profile, ovr_artwork_profile)
        print('\n********Artwork collaborative model Training Completed Successfully********')
    except Exception as e:
        collab_artwork = None
        upe = None
        map_k = None
        user_vec_df = None
        print(f"\nArtwork collab model training failed with Error: {e}")



    return collab_artwork, upe, map_k, user_vec_df

def upload_to_ADL(account_name, account_key, container_name, blob_name, data, ext):
    
    """
    
    Upload data to Azure data lake
    
    """

    if data is not None:
        try:
            block_blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
            blob_name = blob_name + '_' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.' + ext
            if ext=='csv':
                data = data.to_csv (index=False, encoding = "utf-8")

            block_blob_service.create_blob_from_text(container_name = container_name, blob_name = blob_name, text=data)
            print(f'\n{blob_name} data saved in {container_name}')

        except Exception as e:
            print(f"\n{blob_name} failed with Error: {e}")
    else:
        print(f"\n{blob_name} data is none")


def register_model(collab_artwork=None):
    """
    
    Register model in azure model registry 
    
    """
    
    print('\n********Artwork collaborative model registration started********')
    
    model_registered = []

    if collab_artwork is not None:
        try:
            artwork_filename = 'artwork_collab_model.pkl'       
            pickle.dump(collab_artwork, open(artwork_filename, 'wb'))        

            model = Model.register(model_path=artwork_filename,
                       model_name="artwork_collab_model.pkl",
                       tags=None,
                       description="Trained artwork Collab model",
                       workspace=ws)
            model_registered.extend([True, True])
            print('\n********Artwork collaborative model registration completed successfully********')
        except Exception as e:
            model_registered.extend([False, False])
            print(f'\nArtwork collab registration error:{e}')
    else:
        model_registered.extend([False, False])
        print(f'\nArtwork collab model training failed')
    return model_registered


def getdata_from_ADL(account_name,account_key,container_name,blob_name):   
    
    """
    
    Fetch historical data from ADLS
    
    """
    
    block_blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
    blob_name=blob_name+'.csv'
    blobstring = block_blob_service.get_blob_to_text(container_name=container_name,blob_name=blob_name).content
    df = pd.read_csv(StringIO(blobstring))    
    return df


def main(tracking_df,collection_df,all_artwork_profile,ovr_artwork_profile,collab_config_json):
    
    import datetime
    
    historical_tracking_data=getdata_from_ADL(collab_config_json['account_name']
                                              ,collab_config_json['account_key']
                                              ,collab_config_json['container_name_historical_data']
                                              ,collab_config_json['blob_name_historical_data']
                                                 )
    
    final_tracking_df=pd.concat([historical_tracking_data,tracking_df])
    
    if config.test:
        final_tracking_df = final_tracking_df[final_tracking_df['USER_COMBINED_ID'].isin(config.test_users)]
    
    # Calculate rating matrix
    utilitymatrix_artwork = train_useraffinity_model(final_tracking_df,collection_df)
    
    # Collab model training
    collab_artwork, upe, map_k, user_vec_df = train_collab_model(utilitymatrix_artwork
                                                                 , all_artwork_profile,ovr_artwork_profile)
    
    # Uploading upe
    upload_to_ADL(collab_config_json['account_name'], collab_config_json['account_key'], collab_config_json['container_name_upe']
                 , collab_config_json['blob_name_upe'], upe, collab_config_json['ext_upe'])
    
    # Uploading utility matrix
    upload_to_ADL(collab_config_json['account_name'], collab_config_json['account_key'], collab_config_json['container_name_utilitymatrix']
             , collab_config_json['blob_name_utilitymatrix_artwork'], utilitymatrix_artwork, collab_config_json['ext_utilitymatrix'])
    
    # Uploading map@k 
    upload_to_ADL(collab_config_json['account_name'],collab_config_json['account_key'],collab_config_json['container_name_mapk']
             ,collab_config_json['blob_name_mapk_artwork'],map_k,collab_config_json['ext_mapk'])
    
    # Uploading user vector dataframe 
    upload_to_ADL(collab_config_json['account_name'],collab_config_json['account_key'], 'ml-training-data\\user-vector'
            , 'user_vector', user_vec_df, 'csv')
    
    return register_model(collab_artwork)


if __name__=='__main__':    
    run = Run.get_context()
    ws = Run.get_context().experiment.workspace
    
    
    run.log("Training start time", str(datetime.datetime.now()))
    
    parser = argparse.ArgumentParser("model_registration_flag")
    parser.add_argument("--model_registration_flag", type=str)
    args = parser.parse_args()
    os.makedirs(args.model_registration_flag, exist_ok=True)
    
    # Read JSON file
    
    collab_config = run.input_datasets['collab_config_json']
        
    json_path =  collab_config.download(target_path='collab_configs/', overwrite=True)[0]
    json_file = open(json_path)
    collab_config_json = json.load(json_file)
    
    event_tracking_df, collection_df, all_artwork_profile, ovr_artwork_profile = get_data(collab_config_json)
       
    
    # train model
    model_log = main(event_tracking_df, collection_df, all_artwork_profile, ovr_artwork_profile, collab_config_json)
    np.savetxt(args.model_registration_flag+"/model_registration_flag.txt",np.array(model_log))
    
    run.log("Training end time", str(datetime.datetime.now()))
    run.complete()
