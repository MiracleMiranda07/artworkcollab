""" import required libraries """
import pandas as pd
import json
import os, sys
import pickle
import random
import logging
from rich.logging import RichHandler
import time
from pathlib import Path
import socket
import operator
from functools import reduce

from azure.storage.blob import BlobServiceClient, BlobClient, PublicAccess, ContainerClient
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model
from azureml.core import Workspace, Dataset, Datastore
from azure.core.exceptions import ResourceExistsError

from azure.identity import DefaultAzureCredential, ManagedIdentityCredential, ClientSecretCredential
from azure.keyvault.secrets import SecretClient

''' decorater used to disable function printing to the console '''
def disable_printing(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper


""" create respective json file for storing first time users for artwork and room grid view, and log files if not exists """
def create_file(dir_path, path):
    print('file path :', path)
    # dir_path = Path(dir_path)

    if dir_path.exists() and dir_path.is_dir():
        if not os.path.exists(path):
            path.touch(exist_ok=True)
        else:
            pass
    else:
        dir_path.mkdir(exist_ok=True)
        path.touch(exist_ok=True)


""" Called when the service is loaded """
def init():
    global artwork_collab_model_path
    global artwork_content_model_path
    global room_collab_model_path
    global room_content_model_path
    global artwork_content_model
    global room_content_model
    global artwork_collab_model
    global room_collab_model
    global artwork_known_users_lst
    global room_known_users_lst
    global file_path_room_grid_json
    global file_path_artwork_grid_json
    global file_path_prev_artwork_id_json
    global file_path_prev_room_id_json    
    global STORAGE_ACCOUNT
    global CREDENTIALS
    global EVTDATA_FOLDER
    global ANALYTICS_CONTAINER
    global logger
    global LOG_DIR_PATH
    global JSON_DIR_PATH
    global blob_service_client
    global artwork_detail_filter_value
    global artwork_grid_filter_value
    global room_detail_filter_value
    global room_grid_filter_value
    global shuffle_detail_val
    global shuffle_grid_val

    ws=None

    __file__ = '.'
    BASE_DIR = Path(__file__).parent.parent.absolute()

    subscription_id= os.environ.get("SUBSCRIPTION_ID")
    resource_group= os.environ.get("RESOURCE_GP")
    workspace_name= os.environ.get("WORKSPACE_NAME")

    tenant_id= os.environ.get("AZURE_TENANT_ID")
    pr_id= os.environ.get("AZURE_CLIENT_ID")
    pr_pwd= os.environ.get("AZURE_CLIENT_SECRET")

    key_vault_name= os.environ.get("KEY_VAULT_NAME")
       
    KEYVAULT_URI= f"https://{key_vault_name}.vault.azure.net"

    print(f"tenanT id: {tenant_id}")

    credential = ClientSecretCredential(tenant_id, pr_id, pr_pwd)
    secret_client = SecretClient(vault_url=KEYVAULT_URI, credential=credential)
        
    try:
        svc_pr = ServicePrincipalAuthentication(
                    tenant_id=tenant_id,
                    service_principal_id=pr_id,
                    service_principal_password=pr_pwd)

        ws = Workspace(subscription_id = subscription_id,
                    resource_group = resource_group,
                    workspace_name = workspace_name,
                    auth=svc_pr)
        # ws.write_config(path='./config.json')
        print('Library configuration succeeded')
    except Exception as e:
        print(f'Workspace not found. Error- {e}')
     
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.DEBUG)
    logging.getLogger("azure.storage.blob").setLevel(logging.DEBUG)


    logger = logging.getLogger("log")
    # logger.disabled = True
    
    JSON_DIR_PATH = Path(BASE_DIR, "json_data")
    
    file_path_artwork_grid_json = Path(JSON_DIR_PATH, 'artwork_grid_view.json')
    file_path_prev_artwork_id_json = Path(JSON_DIR_PATH, 'prev_artwork_id.json')
    file_path_prev_room_id_json = Path(JSON_DIR_PATH, 'prev_room_id.json')
    file_path_room_grid_json = Path(JSON_DIR_PATH, 'room_grid_view.json')
    
    create_file(JSON_DIR_PATH, file_path_room_grid_json)
    create_file(JSON_DIR_PATH, file_path_artwork_grid_json)
    create_file(JSON_DIR_PATH, file_path_prev_artwork_id_json)
    create_file(JSON_DIR_PATH, file_path_prev_room_id_json)
    

    """ path to respective Azure models """
    artwork_content_model_path = Model.get_model_path(model_name='artwork_content_model.pkl', _workspace=ws)
    artwork_collab_model_path = Model.get_model_path(model_name='artwork_collab_model.pkl', _workspace=ws)
    room_collab_model_path = Model.get_model_path(model_name='room_collab_model.pkl', _workspace=ws)
    room_content_model_path = Model.get_model_path(model_name='room_content_model.pkl', _workspace=ws)
    

    """ load all the models """
    with open(artwork_content_model_path, 'rb') as handle1:
        artwork_content_model = pickle.load(handle1)    
    # print('artwork_content_model: ', list(artwork_content_model.keys())[:20])

    with open(artwork_collab_model_path, 'rb') as handle2:
        artwork_collab_model = pickle.load(handle2)
    # print('artwork_collab_model: ', list(artwork_collab_model.keys())[:20])
    
    with open(room_content_model_path, 'rb') as handle3:
       room_content_model = pickle.load(handle3)
    # print('room_content_model: ', list(room_content_model.keys())[:20])

    with open(room_collab_model_path, 'rb') as handle4:
       room_collab_model = pickle.load(handle4)
    # print('room_collab_model: ', list(room_collab_model.keys())[:20])

    artwork_known_users_lst = list(artwork_collab_model.keys())
    # print('known_users_artwork: ', artwork_known_users_lst[:10])

    room_known_users_lst = list(room_collab_model.keys())
    # print('known_users_room: ', room_known_users_lst[:10])
    

    # Initialize the connection to Azure storage account
    STORAGE_ACCOUNT = os.environ.get("STORAGE_ACCOUNT")
    CREDENTIALS = os.environ.get("STORAGE_KEY")
    ANALYTICS_CONTAINER = os.environ.get("ANALYTICS_CONTAINER")
    EVTDATA_FOLDER = os.environ.get("EVTDATA_FOLDER")

    blob_service_client =  BlobServiceClient(f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
                                                            credential=CREDENTIALS)
                       
    ''' Filter values for details and grid page for Artworks and Rooms '''
    artwork_detail_filter_value= int(os.environ.get("artwork_detail_filter_value"))
    room_detail_filter_value= int(os.environ.get("room_detail_filter_value"))
    artwork_grid_filter_value= int(os.environ.get("artwork_grid_filter_value"))
    room_grid_filter_value= int(os.environ.get("room_grid_filter_value"))
    shuffle_grid_val = int(os.environ.get("shuffle_grid_val"))
    shuffle_detail_val = int(os.environ.get("shuffle_detail_val"))

    logger.info("\n\t\t ***** Init complete ***** \t\n")


""" Code to execute the json request and accordingly return the recommendations """
def run(request):
    global artwork_grid_view
    global room_grid_view
    global prev_artwork_id
    global prev_room_id
    global artwork_grid_view_json
    global room_grid_view_json
    global sold_hidden_artwork_combined_lst
    global featured_artworks
    global featured_rooms
    global shuffle_grid_val
    global shuffle_detail_val
          
   
    # 'artwork_grid_view' --> contains user_id and sorted artwork list for this specific user as <key:Value> pair 
    # and this will be updated based on the users activity
    artwork_grid_view = {}
    room_grid_view= {}
    prev_artwork_id = {}
    prev_room_id = {}
    artwork_grid_view_json={}
    room_grid_view_json={}
    prev_artwork_id_json = {}
    prev_room_id_json = {}
    
 
    try:
        logger.info('\n\t request: %s' % request)
        
        data = json.loads(request)

        item_id=None
        if "itemId" in data:
            if data["itemId"] in ["None", "null", None]:
                item_id = ""
            else:
                item_id = str(data["itemId"])
        else:
            item_id=None

        user_id = None
        if "userId" in data:
            if data["userId"] in ["None", "null", None]:
                user_id = ""
            else:
                user_id = str(data["userId"])
        else:
            user_id=None

        performed_action = None
        if "performedAction" in data:
            if data["performedAction"] in ["None", "null", None]:
                performed_action = ""
            else:
                performed_action = str(data["performedAction"])
        else:
            performed_action=None

        # pre-recommendation
        """ get featured artworks """
        featured_artworks = [k for (k, v) in artwork_content_model.items()]#[:200]#artwork_content_model.keys()
        # print('\n featured artworks: ', featured_artworks[:10])
        # featured_artworks.sort()
        featured_rooms = [k for (k, v) in room_content_model.items()]#[:200]

        # post-recommendation
        """ from consume API
            Accessing Azure blob storage data - CONSUME SOLD/HIDDEN DATA FROM EVTDATA --- ADLS GEN2
        """

        az_response = download_blob_to_df(blob_service_client, ANALYTICS_CONTAINER, EVTDATA_FOLDER)
        if ((az_response['status']==True) and (len(az_response['df'].index)!= 0)):
            data_df = az_response['df']
            cols = ['action', 'actionData']
            df1 = data_df.copy()
            df1 = df1[df1[cols].notnull().all(axis=1)] # taking not null values of action and actionData columns
            # print(df1)
            sold_artwork_lst = df1[df1['action'] == 'ARTWORK_SOLD']['actionData'].tolist()
            displayed_artwork_lst = df1[df1['action'] == 'ARTWORK_DISPLAYED']['actionData'].tolist()

            # print('\n Sold Artwork List: ', sold_artwork_lst)
            logger.info('\t Sold Artwork List: %s' % (sold_artwork_lst))
            # print('\n Displayed Artwork List: ', displayed_artwork_lst)
            logger.info('\t Displayed Artwork List: %s' % (displayed_artwork_lst))

            lst = [df1[df1['action'] == action]['actionData'].values.tolist() for action in ['ARTWORK_SOLD', 'ARTWORK_DISPLAYED'] if action in df1.action.unique().tolist()]
            
            removal_lst = reduce(operator.concat, lst)

            # list of artworks (SOLD/HIDDEN) to be filtered out from the recommendation by appending at the back
            sold_hidden_artwork_combined_lst = [int(i) for n, i in enumerate(removal_lst) if i not in removal_lst[:n]]

            logger.info('SOLD/HIDDEN Artwork combined Lst: %s' % (sold_hidden_artwork_combined_lst))

            msg = 'Evtdata json file successfully read'

        else:
            msg = 'May be EvtData json file is empty/format error.//' + az_response['error']


        
        """ load last artwork and room id from respective json to pass to grid view of (room & artwork) """

        try:
            if os.path.exists(file_path_prev_artwork_id_json) and os.stat(file_path_prev_artwork_id_json).st_size != 0:
                with open(file_path_prev_artwork_id_json) as file:
                    prev_artwork_id_json = json.load(file)
            else:
                prev_artwork_id_json = {}
        except Exception as e:
            print(f'Error while reading prev_artwork_id_json file: {e}')
            pass
        
        logger.info('\t Prev Artwork Id: %s' % (prev_artwork_id_json))
        print('\n Prev Artwork Id Json: ', prev_artwork_id_json)

        try:
            if os.path.exists(file_path_prev_room_id_json) and os.stat(file_path_prev_room_id_json).st_size != 0:
                with open(file_path_prev_room_id_json) as file:
                    prev_room_id_json = json.load(file)
            else:
                prev_room_id_json = {}
        except Exception as e:
            print(f'Error while reading prev_room_id_json file: {e}')
            pass

        logger.info('\t Prev Room Id: %s' % (prev_room_id_json))
        print('\n Prev Room Id Json: ', prev_room_id_json)

        
        """ Read Artwork and Room <key:value> from Json """

        try:
            if os.path.exists(file_path_artwork_grid_json) and os.stat(file_path_artwork_grid_json).st_size != 0:
                with open(file_path_artwork_grid_json) as file:
                    artwork_grid_view_json = json.load(file)
            else:
                artwork_grid_view_json = {}
        except Exception as e:
            print(f'Error while reading artwork_grid_view_json file: {e}')
            pass

        print('\n\t Artwork grid view json keys: ', artwork_grid_view_json.keys())

        try:
            if os.path.exists(file_path_room_grid_json) and os.stat(file_path_room_grid_json).st_size != 0:
                with open(file_path_room_grid_json) as file:
                    room_grid_view_json = json.load(file)
            else:
                room_grid_view_json = {}    
        except Exception as e:
            print(f'Error while reading room_grid_view_json file: {e}')
            pass

        print('\n\t Room grid view json keys: ', room_grid_view_json.keys())


        """ For ROOMS & ARTWORKS -- based on if user is known / unknown -- accordingly trigger collab & content model"""
        final_response={}
        final_result={}
        response={}
        collab_result=[]
        content_result=[]
        new_response=[]
        new_recs_lst=[]
        prev_artwork_itemId_lst=[]
        prev_room_itemId_lst=[]

        # Run inference
        if user_id:
            if performed_action == "ENTER_ARTWORK":
                """ Check based on item_id ---> provided / not provided for Artwork GridView """
                if (item_id=='') or (item_id== None):
                    filter_val=100
                    print('\n No Artwork Id Passed ..')
                    if (user_id in list(prev_artwork_id_json.keys())) and (len(prev_artwork_id_json[user_id])!=0):
                        print('\n\t Prev. Artwork Id ..')
                        prev_artwork_itemId = prev_artwork_id_json[user_id][-1]          
                        print('\n\t enter_artwork_itemId: ', prev_artwork_itemId)     
                        response = enter_artwork_action(user_id, prev_artwork_itemId, artwork_grid_filter_value)
                    elif (user_id in list(artwork_grid_view_json.keys())) and (len(artwork_grid_view_json[user_id])!= 0):
                        print('\n\t User in Artwork Grid View .. ')
                        response = {'status': True, 'result': artwork_grid_view_json[user_id][:filter_val]}
                    else:
                        print('\n\t First Time User.. ')
                        response = enter_artwork_action(user_id, item_id=None, filter_value=artwork_grid_filter_value)
                else:
                    print('\n Artwork Id Passed..')
                    if (user_id in list(prev_artwork_id_json.keys())) and (len(prev_artwork_id_json[user_id])!=0):
                        print('\n\t Prev. Artwork Id ..')
                        prev_artwork_itemId = prev_artwork_id_json[user_id][-1]          
                        print('\n\t enter_artwork_itemId: ', prev_artwork_itemId)     
                        response = enter_artwork_action(user_id, prev_artwork_itemId, artwork_grid_filter_value)
                    
                    else:
                        print('\n\t Else pass the request ArtworkId .. ')
                        input_art_id = item_id
                        response = enter_artwork_action(user_id, input_art_id, filter_value=artwork_grid_filter_value)
                
            
                if(response['status'] == True):
                    final_result = {'status': True, 'result': response['result']}
                else:
                    final_result = {'status': False, 'error': response['error']}

            elif performed_action == "ENTER_ARTWORK_DETAIL":
                """ check if item_id provided for Artwork Details Action """
                if item_id:
                    # print('type:',type(item_id))
                    if user_id in artwork_known_users_lst:
                        print('\n\t ------- Known User // Collab Model --------')
                        response = collab_recommend(user_id, artwork_collab_model)
                        if response['status'] == True:
                            if (user_id in list(prev_artwork_id_json.keys())) and (len(prev_artwork_id_json[user_id])!=0):
                                print('\n\t Prev. Artwork Id List ..')
                                prev_artwork_itemId_lst = prev_artwork_id_json[user_id]
                                if item_id not in prev_artwork_itemId_lst:
                                    prev_artwork_itemId_lst.append(item_id)
                                else:
                                    prev_artwork_itemId_lst.remove(item_id)
                                    prev_artwork_itemId_lst.append(item_id)
                                print(prev_artwork_itemId_lst)
                                new_response = [x for x in response['result'] if x not in [int(x) for x in prev_artwork_itemId_lst]]
                                new_response.extend([int(x) for x in prev_artwork_itemId_lst])
                                # print(new_response)
                            else:
                                new_response = response['result']
                                # new_response.append(item_id)

                            if (len(sold_hidden_artwork_combined_lst)!=0):
                                new_recs_lst = [rec for rec in new_response if rec not in sold_hidden_artwork_combined_lst]
                                collab_result = enter_artwork_detail_action(user_id, item_id, new_recs_lst, artwork_detail_filter_value)
                            else:
                                collab_result = enter_artwork_detail_action(user_id, item_id, new_response, artwork_detail_filter_value)
                            final_result = {'status': True, 'result': collab_result}
                        else:
                            final_result = {'status': False, 'error': response['error']}

                    else:
                        print('\n\t ------- Unknown User // Content Model ---------')
                        response = content_recommend(int(item_id), artwork_content_model)
                        if response['status'] == True:
                            if (user_id in list(prev_artwork_id_json.keys())) and (len(prev_artwork_id_json[user_id])!=0):
                                print('\n\t Prev. Artwork Id List ..')
                                prev_artwork_itemId_lst = prev_artwork_id_json[user_id]
                                if item_id not in prev_artwork_itemId_lst:
                                    prev_artwork_itemId_lst.append(item_id)
                                else:
                                    prev_artwork_itemId_lst.remove(item_id)
                                    prev_artwork_itemId_lst.append(item_id)
                                print(prev_artwork_itemId_lst)
                                new_response = [x for x in response['result'] if x not in [int(x) for x in prev_artwork_itemId_lst]]
                                new_response.extend([int(x) for x in prev_artwork_itemId_lst])
                                # print(new_response)
                            else:
                                new_response = response['result']
                                # new_response.append(item_id)

                            if (len(sold_hidden_artwork_combined_lst)!=0):
                                new_recs_lst = [rec for rec in new_response if rec not in sold_hidden_artwork_combined_lst]
                                content_result = enter_artwork_detail_action(user_id, item_id, new_recs_lst, artwork_detail_filter_value)
                            else:
                                content_result = enter_artwork_detail_action(user_id, item_id, new_response, artwork_detail_filter_value)

                            final_result = {'status': True, 'result': content_result}
                        else:
                            final_result = {'status': False, 'error': response['error']}
                else:
                    final_result = {'status': False, 'error': f"Please provide itemId for User- {user_id} with performedAction- {performed_action}"}

            elif performed_action == "ENTER_GRID_VIEW":
                """ Check based on item_id ---> provided / not provided for Room GridView """
                if (item_id=='') or (item_id==None):
                    filter_val=100
                    print('\n No Room Id Passed ..')
                    if (user_id in list(prev_room_id_json.keys())) and (len(prev_room_id_json[user_id])!=0):
                        print('\n\t Prev. Artwork Id ..')
                        prev_room_itemId = prev_room_id_json[user_id][-1]          
                        print('\n\t enter_room_itemId: ', prev_room_itemId)     
                        response = enter_grid_view_action(user_id, prev_room_itemId, room_grid_filter_value)
                    elif (user_id in list(room_grid_view_json.keys())) and (len(room_grid_view_json[user_id])!=0):
                        print('\n\t User in Room Grid View .. ')
                        response = {'status': True, 'result': room_grid_view_json[user_id][:filter_val]}
                    else:
                        print('\n\t First Time User.. ')
                        response = enter_grid_view_action(user_id, item_id=None, filter_value=room_grid_filter_value)
                else:
                    print('\n Room Id Passed..')
                    if (user_id in list(prev_room_id_json.keys())) and (len(prev_room_id_json[user_id])!=0):
                        print('\n\t Prev. Artwork Id ..')
                        prev_room_itemId = prev_room_id_json[user_id][-1]      
                        print('\n\t enter_artwork_itemId: ', prev_room_itemId)     
                        response = enter_grid_view_action(user_id, prev_room_itemId, room_grid_filter_value)
                    
                    else:
                        print('\n\t Else pass the request Room itemId .. ')
                        input_room_id = item_id
                        response = enter_grid_view_action(user_id, input_room_id, filter_value=room_grid_filter_value)


                if(response['status'] == True):
                    final_result = {'status': True, 'result': response['result']}
                else:
                    final_result = {'status': False, 'error': response['error']}
            
            elif performed_action == "ENTER_ROOM":
                if item_id:
                    if user_id in room_known_users_lst:
                        print('\n\t ------- Known User // Collab Model --------')
                        response = collab_recommend(user_id, room_collab_model)
                        if response['status'] == True:
                            if (user_id in list(prev_room_id_json.keys())) and (len(prev_room_id_json[user_id])!=0):
                                print('\n\t Prev. Room Id List ..')
                                prev_room_itemId_lst = prev_room_id_json[user_id]
                                if item_id not in prev_room_itemId_lst:
                                    prev_room_itemId_lst.append(item_id)
                                else:
                                    prev_room_itemId_lst.remove(item_id)
                                    prev_room_itemId_lst.append(item_id)   
                                print(prev_room_itemId_lst)
                                new_response = [x for x in response['result'] if x not in [int(x) for x in prev_room_itemId_lst]]
                                new_response.extend([int(x) for x in prev_room_itemId_lst])
                                # print(new_response)
                            else:
                                new_response = response['result']

                            collab_result = enter_room_action(user_id, item_id, new_response, room_detail_filter_value)
                            final_result = {'status': True, 'result': collab_result}
                        else:
                            final_result = {'status': False, 'error': response['error']}
                    else:
                        print('\n\t ------- Unknown User // Content Model ---------')
                        response = content_recommend(int(item_id), room_content_model)
                        if response['status'] == True:
                            if (user_id in list(prev_room_id_json.keys())) and (len(prev_room_id_json[user_id])!=0):
                                print('\n\t Prev. Room Id List ..')
                                prev_room_itemId_lst = prev_room_id_json[user_id]
                                if item_id not in prev_room_itemId_lst:
                                    prev_room_itemId_lst.append(item_id)
                                else:
                                    prev_room_itemId_lst.remove(item_id)
                                    prev_room_itemId_lst.append(item_id)
                                print(prev_room_itemId_lst)
                                new_response = [x for x in response['result'] if x not in [int(x) for x in prev_room_itemId_lst]]
                                new_response.extend([int(x) for x in prev_room_itemId_lst])
                                # print(new_response)
                            else:
                                new_response = response['result']

                            content_result = enter_room_action(user_id, item_id, new_response, room_detail_filter_value)
                            final_result = {'status': True, 'result': content_result}
                        else:
                            final_result = {'status': False, 'error': response['error']}
                else:
                    final_result = {'status': False, 'error': f"Please provide itemId for User- {user_id} with performedAction: {performed_action}"}
            
            else:
                final_result = {'status': False, 'error': f"Kindly perform valid action. The performedAction - {performed_action} is incorrect."}
        else:
            final_result = {'status': False, 'error': f"Kindly provide the userId for performedAction- {performed_action}"}

        final_result['msg'] = msg
        
        final_response = final_result
        logging.info(final_response)

    except Exception as e:
        final_response = {'status': False, 'error': f'Exception occurred during the Request Run. Error: {e}'}
        logger.info(final_response)
 

    return final_response


''' Function to shuffle top n from the recommendation list '''
def shuffle_recommendation_lst(a, start, stop):
    try:
        i = start
        while (i < stop-1):
            idx = random.randrange(i, stop)
            a[i], a[idx] = a[idx], a[i]
            i += 1
    except Exception as e:
        print(f"Error while shuffling the recs: {e}")
        pass


""" ArtWork GridView """
# @disable_printing
def enter_artwork_action(user_id=None, item_id=None, filter_value=None):
    print('\n\t **ENTER ARTWORK GRID VIEW ** ')
    try:
        resp={}
        error_msg=''
        recommendations=[]
        new_recs_lst=[]
        recs=[]

        if user_id in artwork_known_users_lst:
            print('\n // Known User //')
            if user_id not in list(artwork_grid_view_json.keys()):
                print('\n\t // first time user //')
                shuffled_feature_artworks = random.sample(featured_artworks, len(featured_artworks))
                if (item_id != None) and (item_id in shuffled_feature_artworks):
                    shuffled_feature_artworks.remove(item_id)
                    artwork_grid_view[user_id] = [int(art) for art in shuffled_feature_artworks]
                else:
                    artwork_grid_view[user_id] = [int(art) for art in shuffled_feature_artworks]

                recommendations = artwork_grid_view[user_id]
            else:                
                print('\n\t ------- Known User // Collab Model --------')
                resp = collab_recommend(user_id, artwork_collab_model)
                if resp['status'] == True:
                    if (item_id != None) and (int(item_id) in resp['result']):
                        resp['result'].remove(int(item_id))
                        recommendations = resp['result']
                    else:
                        recommendations = resp['result']
                else:
                    error_msg = resp['error']

        else:
            print('\n // Unknown User //')
            if user_id not in list(artwork_grid_view_json.keys()):
                print('\n\t // first time user //')
                shuffled_feature_artworks = random.sample(featured_artworks, len(featured_artworks))
                if (item_id!= None) and (item_id in shuffled_feature_artworks):
                    shuffled_feature_artworks.remove(item_id)
                    artwork_grid_view[user_id] = [int(art) for art in shuffled_feature_artworks]
                else:
                    artwork_grid_view[user_id] = [int(art) for art in shuffled_feature_artworks]

                recommendations =  artwork_grid_view[user_id]
            else:
                print('\n\t ------- UnKnown User // Content Model --------')
                resp = content_recommend(int(item_id), artwork_content_model)
                if resp['status'] == True:
                    if (item_id != None) and (int(item_id) in resp['result']):
                        resp['result'].remove(int(item_id))
                        recommendations = resp['result']
                    else:
                        recommendations = resp['result']
                else:
                    error_msg = resp['error']
                
        if os.path.exists(file_path_artwork_grid_json) and os.stat(file_path_artwork_grid_json).st_size == 0:
            with open(file_path_artwork_grid_json, 'w') as file:
                json.dump(artwork_grid_view, file, indent=4)
        else:
            with open(file_path_artwork_grid_json, 'r+') as file:
                data = json.load(file)
                data.update(artwork_grid_view) # update a dict with another dict
                file.seek(0)
                json.dump(data, file, indent=4)

        
        if len(sold_hidden_artwork_combined_lst)!=0:
            new_recs_lst = [rec for rec in recommendations if rec not in sold_hidden_artwork_combined_lst]
            recs = new_recs_lst#[:filter_value]
            shuffle_recommendation_lst(recs, 0, shuffle_grid_val)
        else:
            recs = recommendations#[:filter_value]
            shuffle_recommendation_lst(recs, 0, shuffle_grid_val)
    
        response = {'status': True, 'result': recs}
        
    except Exception as e:
        response = {'status': False, 'error': f"Exception occurred during ENTER ARTWORK action- {e}. Error- {error_msg}"}
    
    return response


""" ArtWork Details """
@disable_printing
def enter_artwork_detail_action(user_id=None, item_id=None, recommendation=None, filter_value=None):    
    print('\n ** ENTER ARTWORK DETAIL **')

    # artwork_grid_view[user_id] = recommendation
    result = recommendation#[:filter_value]
    shuffle_recommendation_lst(result, 0, shuffle_detail_val)

    prev_artwork_id[user_id] = [item_id]

    print('\nPrev Artwork Id : ', prev_artwork_id)

    if os.path.exists(file_path_prev_artwork_id_json) and os.stat(file_path_prev_artwork_id_json).st_size == 0:
        with open(file_path_prev_artwork_id_json, 'w') as file:
            json.dump(prev_artwork_id, file, indent=4)
    else:
        with open(file_path_prev_artwork_id_json, 'r+') as file:
            data = json.load(file)
            if user_id not in list(data.keys()):
                data.update(prev_artwork_id)
            else:
                if item_id not in data[user_id]:
                    data[user_id].append(item_id)
                else:
                    data[user_id].remove(item_id)
                    data[user_id].append(item_id)
            file.seek(0)
            json.dump(data, file, indent=4)

    return result


""" Room GridView """
# @disable_printing
def enter_grid_view_action(user_id=None, item_id=None, filter_value=None):
    print('\n ** ENTER ROOM GRID VIEW ** ')

    try:
        resp={}
        error_msg=''
        recommendations=[]
        recs=[]

        if user_id in room_known_users_lst:
            print('\n\t // known user // ')   
            if user_id not in list(room_grid_view_json.keys()):
                print('\n\t // first time user // ')
                shuffled_feature_rooms = random.sample(featured_rooms, len(featured_rooms))
                if (item_id != None) and (item_id in shuffled_feature_rooms):
                    shuffled_feature_rooms.remove(item_id)
                    room_grid_view[user_id] = [int(art) for art in shuffled_feature_rooms]
                else:
                    room_grid_view[user_id] = [int(art) for art in shuffled_feature_rooms]
                
                recommendations =  room_grid_view[user_id]
            else:
                print('------- Known User // Collab Model --------')
                resp = collab_recommend(user_id, room_collab_model)
                if resp['status'] == True:
                    if (item_id != None) and (int(item_id) in resp['result']):
                        resp['result'].remove(int(item_id))
                        recommendations = resp['result']
                    else:
                        recommendations = resp['result']
                else:
                    error_msg = resp['error']
        else:
            print(' // unknown user //')
            if user_id not in list(room_grid_view_json.keys()):
                print(' // first time user // ')
                shuffled_feature_rooms = random.sample(featured_rooms, len(featured_rooms))
                if (item_id != None) and (item_id in shuffled_feature_rooms):
                    shuffled_feature_rooms.remove(item_id)
                    room_grid_view[user_id] = [int(art) for art in shuffled_feature_rooms]
                else:
                    room_grid_view[user_id] = [int(art) for art in shuffled_feature_rooms]
                
                recommendations =  room_grid_view[user_id]
            else:
                print('------- UnKnown User // Content Model --------')
                resp = content_recommend(int(item_id), room_content_model)
                if resp['status'] == True:
                    if (item_id != None) and (int(item_id) in resp['result']):
                        resp['result'].remove(int(item_id))
                        recommendations = resp['result']
                    else:
                        recommendations = resp['result']
                else:
                    error_msg = resp['error']

        if os.path.exists(file_path_room_grid_json) and os.stat(file_path_room_grid_json).st_size == 0:
            with open(file_path_room_grid_json, 'w') as file:
                json.dump(room_grid_view, file, indent=4)
        else:
            with open(file_path_room_grid_json, 'r+') as file:
                data = json.load(file)
                data.update(room_grid_view)
                file.seek(0)
                json.dump(data, file, indent=4)
        
        # print('room_grid_view: ', room_grid_view)

        recs = recommendations#[:filter_value]
        shuffle_recommendation_lst(recs, 0, shuffle_grid_val)

        response = {'status': True, 'result': recs}
    except Exception as e:
        response = {'status': False, 'error': f"Exception occurred during ENTER GRID VIEW action- {e}. Error- {error_msg}"}
    
    return response


""" Room Details """
@disable_printing
def enter_room_action(user_id=None, item_id=None, recommendation=None, filter_value=None):
    print('\n ** ENTER ROOM DETAIL** ')

    # room_grid_view[user_id] = recommendation
    result = recommendation#[:filter_value]
    shuffle_recommendation_lst(result, 0, shuffle_detail_val)

    prev_room_id[user_id] = [item_id]

    print('\nPrev Room Id: ', prev_room_id)

    if os.path.exists(file_path_prev_room_id_json) and os.stat(file_path_prev_room_id_json).st_size == 0:
        with open(file_path_prev_room_id_json, 'w') as file:
            json.dump(prev_room_id, file, indent=4)
    else:
        with open(file_path_prev_room_id_json, 'r+') as file:
            data = json.load(file)
            if user_id not in list(data.keys()):
                data.update(prev_room_id)
            else:
                if item_id not in data[user_id]:
                    data[user_id].append(item_id)
                else:
                    data[user_id].remove(item_id)
                    data[user_id].append(item_id)
            file.seek(0)
            json.dump(data, file, indent=4)

    return result


''' Recommendation through Content Model '''
@disable_printing
def content_recommend(item_id=None, model=None):
    print('\n ** Content Recommendation **')
    try:        
        recommendations = []
        if item_id:
            recs = model[item_id]
            # print('\n recs: ', recs)
            recommendations = [int(rec[0]) for rec in recs]
        else:
            recs =  list(model)
            recommendations = recs

        response = {'status': True, 'result': recommendations}
    
    except Exception as e:
        response = {'status': False, 'error': f"Exception occurred during Content Recommendation. Error for {item_id}: {e}"}

    return response


''' Recommendation through Collab Model '''
@disable_printing
def collab_recommend(user_id=None, model=None):
    print('\n ** Collab Recommendation **')
    try:
        recommendations = []
        recs=[]
        if user_id:
            recs = model[user_id]
            # print(recs)
            recommendations = [int(rec[0]) for rec in recs]
            # recommendations = [int(rec) for rec in recs] # old models
        else:
            recs =  list(model)
            recommendations = recs

        response = {'status': True, 'result': recommendations}
    except Exception as e:
        response = {'status': False, 'error': f"Exception occurred during Collab Recommendation. Error for {user_id}: {e}"}    

    return response


""" Read Evtdata blob json file from AzureDLS as Pandas DataFrame  """
@disable_printing
def download_blob_to_df(blob_service_client, blob_container, dest_folder):
    print("Intializing AzureBlobFileDownloader")
 
    # Initialize the connection to Azure storage account
    container = blob_service_client.get_container_client(blob_container)
 
    try:
        print('\n *** Download started *** \n')
        blob_list = container.list_blobs(dest_folder)

        # download recently created blob file from the container
        sorted_blob_lst = sorted(blob_list, key=lambda x: x.creation_time, reverse=True)
        print(f"\n sorted file: {sorted_blob_lst[0].name} \n")
        blob_name = sorted_blob_lst[0].name
        bytes = container.get_blob_client(blob_name).download_blob()
        json_data = bytes.content_as_text()
        # print('\n Json data: ', json_data)
      
        data_df = pd.read_json(json_data, lines=True)
    
        print('\n *** Download finished *** \n')

        response = {'status': True, 'df': data_df}
        
    except Exception as e:
        print(f"Exception occurred during Azure Blob File download. Error: {e}")

        response = {'status': False, 'error': f"Exception occurred during Azure Blob File download. Error: {e}"}
        
    return response