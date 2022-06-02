# %%
# +
# #!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# %%
import os
import pickle
import argparse
import numpy as np
import azureml.core
from pathlib import Path
from azureml.core import Run
from azureml.core import Workspace
from azureml.core.environment import Environment
from azureml.core.model import Model, InferenceConfig
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.webservice import AciWebservice, AksWebservice, Webservice
from azureml.core.compute import ComputeTarget, AksCompute, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.exceptions import WebserviceException


# AzureML SDK Version
# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

def main():

    global ws

    ws=None
    try:
        run = Run.get_context()
        ws = Run.get_context().experiment.workspace
#         ws = Workspace.from_config()
        print('Library configuration succeeded')
        print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')
    except:
        print('Workspace not found')


    # Environment Config

    # create env from conda specification file
    scoring_custom_env = Environment.from_conda_specification(name='scoring-conda-env', file_path='./conda_dep_scoring.yml')

    # Get params

    ''' Read parameters from scoring_params config file '''
    def get_params(ws):
        try:
            __path__= './params'
            PARAMS_DIR = os.path.realpath(__path__)
            print(f"PARAMS dir: {PARAMS_DIR}")

            dset1 = Dataset.get_by_name(ws, name='scoring_params')
            dset1.download(target_path=PARAMS_DIR, overwrite=True)

            import json
            params_file_path = os.path.join(PARAMS_DIR, 'parameters.json')
            with open(params_file_path) as f:
                params = json.load(f)

            response = {'status': True, 'params': params}
        except Exception as e:
            response = {'status': False, 'error' : f"Exception occurred while reading parameters from scoring config file. Error- {e}"}

        return response

    params_response = get_params(ws)

    params={}
    if params_response['status'] == True:
        params = params_response['params']
    else:
        print(params_response['error'])

    AZURE_TENANT_ID = params['TENANT_ID']
    AZURE_CLIENT_ID = params['PRINCIPAL_ID']
    AZURE_CLIENT_SECRET = params['PRINCIPAL_PWD']
    SUBSCRIPTION_ID = params['SUBSCRIPTION_ID']
    RESOURCE_GP = params['RESOURCE_GP']
    WORKSPACE_NAME = params['WORKSPACE_NAME']
    KEY_VAULT_NAME = params['KEY_VAULT_NAME']
    STORAGE_ACCOUNT = params["STORAGE_ACCOUNT"]
    STORAGE_KEY = params["STORAGE_KEY"]
    ANALYTICS_CONTAINER = params["ANALYTICS_CONTAINER"]
    EVTDATA_FOLDER = params["EVTDATA_FOLDER"]
    INFERENCE_CONTAINER = params["INFERENCE_CONTAINER"]
    artwork_detail_filter_value= params["ARTWORK_DETAIL_FILTER_VALUE"]
    room_detail_filter_value= params["ROOM_DETAIL_FILTER_VALUE"]
    artwork_grid_filter_value= params["ARTWORK_GRID_FILTER_VALUE"]
    room_grid_filter_value= params["ROOM_GRID_FILTER_VALUE"]
    shuffle_grid_val = params["SHUFFLE_GRID_VAL"]
    shuffle_detail_val = params["SHUFFLE_DETAIL_VAL"]

    #Create a dictionary to set-up the env variables   
    env_variables={"AZURE_TENANT_ID": AZURE_TENANT_ID,
                   "AZURE_CLIENT_ID": AZURE_CLIENT_ID,
                   "AZURE_CLIENT_SECRET": AZURE_CLIENT_SECRET,
                   "SUBSCRIPTION_ID": SUBSCRIPTION_ID,
                    "RESOURCE_GP": RESOURCE_GP,
                    "WORKSPACE_NAME": WORKSPACE_NAME,
                    "KEY_VAULT_NAME": KEY_VAULT_NAME,
                    "STORAGE_ACCOUNT": STORAGE_ACCOUNT,
                    "STORAGE_KEY": STORAGE_KEY,
                    "ANALYTICS_CONTAINER": ANALYTICS_CONTAINER,
                    "EVTDATA_FOLDER": EVTDATA_FOLDER,
                    "INFERENCE_CONTAINER": INFERENCE_CONTAINER,
                    "artwork_detail_filter_value": artwork_detail_filter_value,
                    "room_detail_filter_value": room_detail_filter_value,
                    "artwork_grid_filter_value": artwork_grid_filter_value,
                    "room_grid_filter_value": room_grid_filter_value,
                    "shuffle_grid_val": shuffle_grid_val,
                    "shuffle_detail_val": shuffle_detail_val
                   }
        
    scoring_custom_env.environment_variables=env_variables

    # Inference Config

    global inference_config
    inference_config=None

    try:
        # env = Environment(name=env_name)
        # env.inferencing_stack_version = "latest"
        source_directory = os.path.realpath('.')
        
        inference_config = InferenceConfig(entry_script="score_latest.py",
                                        source_directory=source_directory,
                                        environment=scoring_custom_env                                    
                                        )

        print(f'\n\t Inference config successfully loaded.. ')
    except Exception as e:
        print(f'\n\t Inference config error: {e}')


    # Get respective models

    global mdl1, mdl2, mdl3, mdl4
    mdl1=None
    mdl2=None
    mdl3=None
    mdl4=None

    try:
        mdl1 = ws.models["artwork_content_model.pkl"]
        mdl2 = ws.models["artwork_collab_model.pkl"]
        mdl3 = ws.models["room_content_model.pkl"]
        mdl4 = ws.models["room_collab_model.pkl"]

        print(f'\n\t Model loaded successfully.. ') 
        print(f'\n\t {mdl1.name+"/"+str(mdl1.version)} | {mdl2.name+"/"+str(mdl2.version)} | {mdl3.name+"/"+str(mdl3.version)} | {mdl4.name+"/"+str(mdl4.version)}')
    except Exception as e:
        print(f'\n\t Model load error.. {e}')


    # Deploy Model as Webservice on Azure

    # Method -1: Using Aks Service

    # create cluster

    # Choose a name for inference cluster
    inference_cluster_name = "mch-tst-cluster"

    # Verify that the cluster does not exist already
    try:
        aks_target = ComputeTarget(workspace=ws, name=inference_cluster_name)
        
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        print(f'No such compute found, creating new')
        prov_config = AksCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2")
        prov_config.enable_ssl(leaf_domain_label = "contoso")

        # Create the cluster
        aks_target = ComputeTarget.create(workspace = ws,
                                            name = inference_cluster_name,
                                            provisioning_configuration = prov_config)

        aks_target.wait_for_completion(show_output=True)


    # run web service

    aks_service_name= 'aks-mch-recommendation-v1'

    try:
        deployment_config = AksWebservice.deploy_configuration(cpu_cores = 0.5,
                        memory_gb = 0.5,
                        auth_enabled = True,
                        autoscale_enabled = True,
                        enable_app_insights = True,
                        description = 'Endpoint fr Testing --- MCH Recommender Web Service using directly AKS')
        
        service = Model.deploy(ws,
                        aks_service_name,
                        models = [mdl1, mdl2, mdl3, mdl4],
                        inference_config = inference_config,
                        deployment_config = deployment_config, 
                        deployment_target = aks_target, overwrite = True)

        service.wait_for_deployment(show_output = True)
        print(f'\n\t AKS Deployment done...')
    except Exception as e:
        print(f'Web Service Deployment Error: {e}')

    # Deployment logs
    aks_service = AksWebservice(ws, aks_service_name)    
    print('*************** Deployment Logs **********************')
    print(f'service state: {aks_service.state}')
    print(aks_service.get_logs())

    print('*************** Deployment Details **********************')
    print(f'scoring uri: {aks_service.scoring_uri}')

    primary_key, secondary_key = aks_service.get_keys()
    print(f'Primary key:{primary_key} ; Secondary key:{secondary_key}')



# %%
if __name__=='__main__':
    parser = argparse.ArgumentParser("score")
    parser.add_argument("--model_registration_flag", type=str, help="model_registration_flag")
    args = parser.parse_args()
    
    model_registration_flag=np.loadtxt(args.model_registration_flag+"/model_registration_flag.txt")
    
    if (model_registration_flag[0] == True) and (model_registration_flag[1] == True):
        redeployment_cnt = 0
        deployment_pending = True
        
        while deployment_pending:
            try:
                main()
                deployment_pending = False
            
            except Exception as e: 
                print(f'\nDeployment failed with errro {e}')
                redeployment_cnt = redeployment_cnt + 1
                
                if redeployment_cnt > 2: 
                    deployment_pending = False
                else:
                    print(f'Attempting to redeploy model in 3mins: try {redeployment_cnt}....')
                    time.sleep(180)
    else:
        print(f'Model is not deployed, model registration status {model_registration_flag}')
