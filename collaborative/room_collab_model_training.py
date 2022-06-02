import os
import pickle
import numpy as np
import pandas as pd
import argparse
import datetime
import joblib
import json
from io import StringIO
from azureml.core import Run
from azureml.core import Workspace
from azureml.core.model import Model
from azure.storage.blob import BlockBlobService

from azureml.core import Workspace, Experiment, Datastore
from azureml.core import Dataset

from test_config import Config as config
from useraffinity.useraffinity_model_room import Useraffinityroom
from collab.room_collab_training import Collaborative_Room


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


def get_data(collab_config):
       
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

def create_utilitymatrix_room(tracking_df,collection_df):
    utilitymatrix_room = Useraffinityroom(tracking_df,collection_df)
    utilitymatrix_room.fit(tracking_df,collection_df)
    return utilitymatrix_room.result 


def train_useraffinity_model(tracking_df,collection_df):
    """
    
    Artwork useraffinity_model training
    
    """
            
    print('\n********Room Utility matrix creation started********')
    
    try:
        utilitymatrix_room= create_utilitymatrix_room(tracking_df,collection_df)
        print('\n********Room utility matrix creation completed successfully********')
    except Exception as e:
        utilitymatrix_room = None
        print(f"\nRoom utility matrix creation failed with Error: {e}")
        
    return utilitymatrix_room


def create_collab_room(utilitymatrix_room,all_artwork_profile,ovr_artwork_profile):
    collab_room = Collaborative_Room(utilitymatrix_room,all_artwork_profile,ovr_artwork_profile)
    collab_room.fit(utilitymatrix_room,all_artwork_profile,ovr_artwork_profile)
    return collab_room.result, collab_room.map_k


def train_collab_model(utilitymatrix_room, all_artwork_profile, ovr_artwork_profile):
    """
    
    Artwork collab model Training
    
    """
    
    print('\n********Room collaborative model Training started********')

    try:
        collab_room, map_k = create_collab_room(utilitymatrix_room,all_artwork_profile,ovr_artwork_profile)
        print('\n********Room collaborative model Training Completed Successfully********')
    except Exception as e:
        collab_room = None
        map_k = None
        print(f"\nRoom collab model training failed with Error: {e}")
    
    return collab_room, map_k

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


def register_model(collab_room=None):
    """
    
    Register model in azure model registry 
    
    """
    
    
    print('\n********Room collaborative model registration started********')
    
    model_registered = []
    if collab_room is not None:
        try:
            room_filename = 'room_collab_model.pkl'       
            pickle.dump(collab_room, open(room_filename, 'wb'))        

            model = Model.register(model_path=room_filename,
                       model_name="room_collab_model.pkl",
                       tags=None,
                       description="Trained Room Collab model",
                       workspace=ws)
            model_registered.extend([True, True])
            print('\n********Room collaborative model registration completed successfully********')
        except Exception as e:
            model_registered.extend([False, False])
            print(f'Room collab registration error:{e}')
    else:
        model_registered.extend([False, False])
        print(f'Room collab model training failed')
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


def main(tracking_df,collection_df,all_artwork_profile,ovr_artwork_profile):
    
    import datetime  
    historical_tracking_data = getdata_from_ADL(collab_config_json['account_name']
                                                ,collab_config_json['account_key']
                                                ,collab_config_json['container_name_historical_data']
                                                ,collab_config_json['blob_name_historical_data']
                                               )
    final_tracking_df=pd.concat([historical_tracking_data,tracking_df])
    
    if config.test:
        final_tracking_df = final_tracking_df[final_tracking_df['USER_COMBINED_ID'].isin(config.test_users)]
    
    # Calculate rating matrix
    utilitymatrix_room = train_useraffinity_model(final_tracking_df,collection_df)
    
    # Collab model training
    collab_room, map_k = train_collab_model(utilitymatrix_room, all_artwork_profile,ovr_artwork_profile)
    
    #uploading utility matrix
    upload_to_ADL(collab_config_json['account_name'],collab_config_json['account_key'],collab_config_json['container_name_utilitymatrix']
             ,collab_config_json['blob_name_utilitymatrix_room'],utilitymatrix_room,collab_config_json['ext_utilitymatrix'])
    
    # Uploading map@k 
    upload_to_ADL(collab_config_json['account_name'], collab_config_json['account_key'], collab_config_json['container_name_mapk']
             , collab_config_json['blob_name_mapk_room'], map_k, collab_config_json['ext_mapk'])
    
    return register_model(collab_room)


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
    model_log = main(event_tracking_df, collection_df, all_artwork_profile, ovr_artwork_profile)
    np.savetxt(args.model_registration_flag+"/model_registration_flag.txt",np.array(model_log))
    
    run.log("Training end time", str(datetime.datetime.now()))
    run.complete()
