import pandas as pd

def upsampling(preprocessed_df, verbose=False):
    
    """
    user-item interaction data upsampling
    
    input: Preprocessed user-item dataframe
    
    output: Upsampled dataframe
    """

    if verbose: print('------------Upsampling the data------------')

    #before upsampling the data
    df_1 = preprocessed_df[preprocessed_df['rating'].isin([0.6,0.8,1])]
    df_2 = preprocessed_df[preprocessed_df['rating'].isin([1.2,1.4,1.6,1.8,2])]
    df_3 = preprocessed_df[preprocessed_df['rating'].isin([2.2,2.4,2.6,2.8,3])]
    df_4 = preprocessed_df[preprocessed_df['rating'].isin([3.2,3.4,3.6,3.8,4])]
    df_5 = preprocessed_df[preprocessed_df['rating'].isin([4.2,4.4,4.6,4.8,5])]
    
    if verbose: print('before upsampling',len(df_1),len(df_2),len(df_3),len(df_4),len(df_5))
    defaultsize=max(df_1.shape[0],df_2.shape[0],df_3.shape[0],df_4.shape[0],df_5.shape[0])
#     print('max',defaultsize)
    
    #upsampling the ratings to equalize the distribution
    #rating1
    users_rating1=set(list(df_1.user_id))
    final1 = pd.DataFrame()
    grouped = df_1.groupby('user_id')
    upsample_user_1=grouped.filter(lambda x: len(x) > 25)

    if defaultsize==len(df_1):
        final1 = df_1.copy()
    else:
        for i in range(50000):
            if len(final1)<=defaultsize:
                temp_df = upsample_user_1.copy()
                temp_df['user_id'] = 'dummy_' + str(i) + '_' + temp_df['user_id']
                if i == 0:
                    final1 = df_1.copy()
                else:
                    final1 = pd.concat([final1,temp_df])  

    #rating2
    users_rating2=set(list(df_2.user_id))
    final2 = pd.DataFrame()
    grouped = df_2.groupby('user_id')
    upsample_user_2=grouped.filter(lambda x: len(x) > 25)
    for i in range(50000):
        if len(final2)<=defaultsize:
            temp_df = upsample_user_2.copy()
            temp_df['user_id'] = 'dummy_' + str(i) + '_' + temp_df['user_id']
            if i == 0:
                final2 = df_2.copy()
            else:
                final2 = pd.concat([final2,temp_df])

    #rating3
    users_rating3=set(list(df_3.user_id))
    final3 = pd.DataFrame()
    grouped = df_3.groupby('user_id')
    upsample_user_3=grouped.filter(lambda x: len(x) > 25)
    for i in range(50000):
        if len(final3)<=defaultsize:
            temp_df = upsample_user_3.copy()
            temp_df['user_id'] = 'dummy_' + str(i) + '_' + temp_df['user_id']
            if i == 0:
                final3 = df_3.copy()
            else:
                final3 = pd.concat([final3,temp_df])  

    #rating4
    users_rating4=set(list(df_4.user_id))
    final4 = pd.DataFrame()
    grouped = df_4.groupby('user_id')
    upsample_user_4=grouped.filter(lambda x: len(x) > 25)
    for i in range(50000):
        if len(final4)<=defaultsize:
            temp_df = upsample_user_4.copy()
            temp_df['user_id'] = 'dummy_' + str(i) + '_' + temp_df['user_id']
            if i == 0:
                final4 = df_4.copy()
            else:
                final4 = pd.concat([final4,temp_df]) 

    #rating5
    users_rating5=set(list(df_5.user_id))
    final5 = pd.DataFrame()
    grouped = df_5.groupby('user_id')
    upsample_user_5=grouped.filter(lambda x: len(x) > 25)
    for i in range(50000):
        if len(final5)<=defaultsize:
            temp_df = upsample_user_5.copy()
            temp_df['user_id'] = 'dummy_' + str(i) + '_' + temp_df['user_id']
            if i == 0:
                final5 = df_5.copy()
            else:
                final5 = pd.concat([final5,temp_df]) 

    if verbose: print('after upsampling',len(final1),len(final2),len(final3),len(final4),len(final5))
    upsampled_df=pd.concat([final1,final2,final3,final4,final5]) 


    return upsampled_df
