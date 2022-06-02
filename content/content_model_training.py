import os
import nltk
import json
import pickle
import datetime
import argparse
import numpy as np
import pandas as pd

from azureml.core import Workspace, Experiment, Datastore
from azureml.core import Dataset

nltk.download('all')

from artwork_content_recsys import ArtworkContent
from room_content_recsys import RoomContent

from azureml.core import Run
from azureml.core import Workspace
from azureml.core.model import Model

def get_data(content_config):
    """
    
    Fetches latest data from ADL
    
    """
    json_path =  content_config.download(target_path='content_configs/', overwrite=True)[0]
    json_file = open(json_path)
    content_config_json = json.load(json_file)
    print('content_config_json :', content_config_json)
    
    # Get Datastore and Pull dataset from the give path
    datastore = Datastore.get(ws, datastore_name=content_config_json['datastore_name'])
    dataset = Dataset.File.from_files(datastore.path(content_config_json['input_data_path']))
    
    # Registering and loading the retrived dataset/ Files
    registered_data = dataset.register(ws, content_config_json['dataset_name'], create_new_version=True)
    input_path =  dataset.download(target_path=content_config_json['file_save_path'], overwrite=True)
    print('content OVR files: ', input_path)
    
    # Picking latest data from the dataset
    import datetime
    content_full_dict = {}
    for file in dataset.to_path():
        str_date = file.rsplit(' ', 1)[0].rsplit('_', 2)[2]
        date = datetime.datetime.strptime(str_date, '%Y-%m-%d')
        content_full_dict[date] = file
    
    current_ovr_data = content_full_dict[max(content_full_dict.keys())]
    print('current_ovr_data: ', current_ovr_data)
    
    current_ovr_data_path = ''
    for path in input_path:
        if current_ovr_data in path:
            current_ovr_data_path =  path
    
    print('current_ovr_data_path: ', current_ovr_data_path)
    content_df = pd.read_csv(current_ovr_data_path, escapechar='\\')
    
    return content_df

def train_artwork_content(dataFrame):
    """
    
    Artwork content model training
    
    """
    artwork_model = ArtworkContent(dataFrame)
    artwork_model.fit()
    return artwork_model.result

def train_room_content(dataFrame):
    """
    
    Room content model training
    
    """
    room_model = RoomContent(dataFrame)
    room_model.fit()
    return room_model.result

def train_model(dataFrame):
    """
    
    Content model training
    
    """
    
    print("Model Training Started")
    try:
        artwork_content_model = train_artwork_content(dataFrame)
    except Exception as e:
        artwork_content_model = None
        print(f"Artwork content model training failed with Error: {e}")
        
    try:
        room_content_model = train_room_content(dataFrame)
    except Exception as e:
        room_content_model = None
        print(f"Room content model training failed with Error: {e}")
        
    print("\nModel Training Completed")

    return artwork_content_model, room_content_model

def save_model(save_dir, artwork_content_model=None, room_content_model=None):
    """
    
    Save model locally
    
    """
    
    if artwork_content_model is not None:
        artwork_filename = os.path.join(save_dir, 'artwork_content_model.pkl')
        pickle.dump(artwork_content_model, open(artwork_filename, 'wb'))
        print(f'\nArtwork Content model saved :{artwork_filename}') 
    
    if room_content_model is not None:
        room_filename = os.path.join(save_dir, 'room_content_model.pkl')
        pickle.dump(room_content_model, open(room_filename, 'wb'))
        print(f'\nRoom Content model saved :{room_filename}') 

def register_model(artwork_content_model, room_content_model):
    """
    
    Model registration in Azure ML workspace
    
    """
#     model_path = 'x'
    model_registered = []
    
    if artwork_content_model is not None:
        try:
            artwork_filename = 'artwork_content_model.pkl'       
            pickle.dump(artwork_content_model, open(artwork_filename, 'wb'))        

            model = Model.register(model_path=artwork_filename,
                           model_name="artwork_content_model.pkl",
                           tags=None,
                           description="Trained Artwork content model",
                           workspace=ws)
            model_registered.append(True)
        except Exception as e:
            model_registered.append(False)
            print(f'Artwork content registration error:{e}')
    else:
        model_registered.append(False)
        print(f'Artwork content model training failed')
        
    if room_content_model is not None:
        try:
            room_filename = 'room_content_model.pkl'       
            pickle.dump(room_content_model, open(room_filename, 'wb'))        

            model = Model.register(model_path=room_filename,
                       model_name="room_content_model.pkl",
                       tags=None,
                       description="...",
                       workspace=ws)
            model_registered.append(True)
        except Exception as e:
            model_registered.append(False)
            print(f'Room content registration error:{e}')
    else:
        model_registered.append(False)
        print(f'Room content model training failed')

    return model_registered


def main(content_df):
    artwork_content_model, room_content_model = train_model(content_df)
    
    return register_model(artwork_content_model, room_content_model)

if __name__=='__main__':
    
    run = Run.get_context()
    ws = Run.get_context().experiment.workspace
    
    run.log("Training start time", str(datetime.datetime.now()))
    
    parser = argparse.ArgumentParser("model_registration_flag")
    parser.add_argument("--model_registration_flag", type=str, help="model_registration_flag")
    args = parser.parse_args()
    os.makedirs(args.model_registration_flag, exist_ok=True)
    
    # Read JSON file
    content_config = run.input_datasets['content_config_json']
    content_df = get_data(content_config)
    
    # train model
    print('****************Content Model Training****************')
    model_log = main(content_df)
    np.savetxt(args.model_registration_flag+"/model_registration_flag.txt",np.array(model_log))
    
    run.log("Training end time", str(datetime.datetime.now()))
    run.complete()

# <!--   content_df = pd.read_csv(r"C:\Users\Milton\Desktop\MCH\model dev\content\content_latest_OVR.csv") -->
