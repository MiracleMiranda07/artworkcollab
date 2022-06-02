"""
configuration parameters to train the recommendation model
"""

class Config():

    # filtration threshold for user-item
    min_user_item_thrsh = (10, 10)
    
    # no of epochs
    epoch = 15
    
    # number of iteration per epoch
    iteration = 100
    
    #learning_rate
    learning_rate = 0.01
    regularization = 0.001
    
    #latent_features
    latent_features = 10
    
    #user journey threshold for spliting the test dataset(test data has splited when only the user journey exists the threshold)
    test_split_thresh = 20
    
    # pecentage for val data
    val_perc = 0.1
    
    #check top-k recommendation for the user using k value 
    k = 5
    
    #map@k rating threshold
    mapk_threshold = 2
