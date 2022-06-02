import pandas as pd
import time

# +

"""
spliting the data for testing the model

input -user-item interaction dataframe

output - train-set 
"""
def train_test_split(dataDrame,user_journey_len):

    original_df=dataDrame

    #filter users with journey greater than the threshold to split the val_set
    df = dataDrame.groupby('user_id').filter(lambda x: len(x) >= user_journey_len)
    
    #filtering the user with decent distribution on rating
    users=list(set(df.user_id))
    df['rating']=df['rating'].round(1)
    df['rating']=df['rating'].astype(int)
    user_list=[]
    selected_user=[]
    dis_list=[1,2,3,4,5]
    
    for i in users:
        temp_df = df[df['user_id']==i]
        user_rating = temp_df['rating'].to_list()
        user_rating = set([round(i) for i in set(user_rating) if i>0])
        set_difference = list(set(dis_list) - set(user_rating))
        if len(set_difference)<=1:
            selected_user.append(i)
                         
    #fetching val_set 
    users=list(set(selected_user))   
    print('valset unique users',len(users))
    #users=users[0:20]
    final=pd.DataFrame()
    for i in users:
        temp_df = original_df[original_df['user_id']==i]
        sample_df = temp_df.groupby('rating').apply(lambda x: x.sample(frac=0.2))
        final=pd.concat([final,sample_df])         
            
    final = final.reset_index(drop=True)
    print('validation dataset shape',final.shape)
    #final=final[['user_id','item_id','rating']]
 
                         
    return final 
