# Iporting required packages
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from collab.train_val_split import train_val_split

class MatrixFactorization():
    def __init__(self, 
                 rating,
                 data,
                 user_vecs,
                 item_vecs,
                 item_fact_reg=0.0, 
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 val_df=None,
                 verbose=True):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : User x Item matrix with ratings

        
        item_fact_reg : (float)
            Regularization term for item latent factors
        
        user_fact_reg : (float)
            Regularization term for user latent factors
            
        item_bias_reg : (float)
            Regularization term for item biases
        
        user_bias_reg : (float)
            Regularization term for user biases
        
        verbose : (bool)
            Whether or not to printout training progress
        """
        if verbose: print('\n------------Model Training Started------------')
            
        train, val = train_val_split(rating) 
        self.ratings = train
        self.val = val
        self.data = data
        self.n_users, self.n_items = self.ratings.shape
        self.user_vecs = user_vecs
        self.item_vecs = item_vecs
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.val_df=val_df
        
        self.best_train_rmse = 100
        self.best_epoch = None
        self.best_user_vecs = None
        self.best_item_vecs = None
        self.best_predictions = None
        
        self.sample_row, self.sample_col = self.ratings.nonzero()
        self.n_samples = len(self.sample_row)
        
        self._v = verbose
        
    def calculate_rmse(self, pred, actual):
        """
        calculates RMSE score for the given prediction
        
        ---------- Input ---------------
        actual: Actaul values
        pred: Predicted values
        
        ---------- output ---------------
        RMSE score
        
        """  
        
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual, squared=False)

    def learning_rate_tuning(self):

        iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
        learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]

        best_params = {}
        best_params['learning_rate'] = None
        best_params['n_iter'] = 0
        best_params['train_rmse'] = np.inf
        best_params['test_rmse'] = np.inf


        for rate in learning_rates:
            self.calculate_learning_curve(iter_array, test, learning_rate=rate)
            min_idx = np.argmin(self.test_rmse)
            if self.test_rmse[min_idx] < best_params['test_rmse']:
                best_params['n_iter'] = iter_array[min_idx]
                best_params['learning_rate'] = rate
                best_params['train_rmse'] = self.train_rmse[min_idx]
                best_params['test_rmse'] = self.test_rmse[min_idx]
        
        return best_params['learning_rate']
            
    def fit(self, epoch, n_iter, learning_rate):
        """
        Fits the given epoch and learning rate values and 
        starts the training process
        
        ---------- Params ---------------
        epoch: No.of epochs
        n_iter: No.of iterations
        learning_rate: rate at which the model has to learn
        
        """  

        for i in tqdm(range(1, epoch+1)):
#             if i > 20:
#                 learning_rate = 0.001
            print('\nepoch: ', i)
            self.train(n_iter=n_iter, learning_rate=learning_rate)
            prediction = self.predict_all()

            train_rmse_score = self.calculate_rmse(prediction, self.ratings)
            val_rmse_score = self.calculate_rmse(prediction, self.val) # new
            
            print('\nTrain rmse score: ', train_rmse_score)
            print('\nVal rmse score: ', val_rmse_score)
            
            if self.best_train_rmse > train_rmse_score:
                self.best_train_rmse = train_rmse_score
                self.best_epoch = i
                self.best_user_vecs = self.user_vecs
                self.best_item_vecs = self.item_vecs
                self.best_predictions = prediction
        print(f'Best train rmse value: {self.best_train_rmse}')
        print(f'Best rmse at epoch: {self.best_epoch}')
         
    def train(self, n_iter, learning_rate):
    
        """ Train model for n_iter iterations from scratch."""
        
        self.learning_rate = learning_rate
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
        self.partial_train(n_iter)    
        
       
    def partial_train(self, n_iter):
        """ 
        Train model for n_iter iterations. Can be 
        called multiple times for further training.
        """
        
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print ('\tcurrent iteration: {}'.format(ctr))
            self.training_indices = np.arange(self.n_samples)
            np.random.shuffle(self.training_indices)
            self.sgd()
            ctr += 1
            
    def sgd(self):
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u,i] - prediction) # error
            
            # Update biases
            self.user_bias[u] += self.learning_rate * (e - self.user_bias_reg * self.user_bias[u])
            
            #Update latent factors
            self.user_vecs[u, :] += self.learning_rate * (e * self.item_vecs[i, :] - self.user_fact_reg * self.user_vecs[u,:])
    
    def predict(self, u, i):
        """ Single user and item prediction."""
        
        prediction = self.global_bias + self.user_bias[u] #+ self.item_bias[i]
        prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        return prediction
    
    def predict_all(self):
        """ Predict ratings for every user and item."""
        predictions = np.zeros((self.user_vecs.shape[0], 
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
                
        return predictions
    
    def calculate_learning_curve(self, iter_array, test, learning_rate=0.1):
        """
        Keep track of RMSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).
        
        The function creates two new class attributes:
        
        train_rmse : (list)
            Training data RMSE values for each value of iter_array
        test_rmse : (list)
            Test data RMSE values for each value of iter_array
        """
        iter_array.sort()
        self.train_rmse =[]
        self.test_rmse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print ('Iteration: {}'.format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff, learning_rate)
                
            predictions = self.predict_all()

            self.train_rmse += [calculate_rmse(predictions, self.ratings)]
            self.test_rmse += [calculate_rmse(predictions, test)]
            if self._v:
                print ('Train rmse: ' + str(self.train_rmse[-1]))
                print ('Test rmse: ' + str(self.test_rmse[-1]))
            iter_diff = n_iter
            
    def prediction_remapping(self):
        """
        Remapping the predicted value with correspondig user_id and item

        """ 
        
        predicted_df = pd.DataFrame(self.best_predictions, columns=self.data.columns, index=self.data.index)
        predicted_df = predicted_df[~predicted_df.index.str.contains("dummy")]
        predicted_df = predicted_df.unstack().reset_index(name='predicted')
        predicted_df = predicted_df[['user_id', 'item_id', 'predicted']]
        self.predicted_df = predicted_df.sort_values(['user_id', 'item_id'])      
        
    def prediction_conversion(self, current_ovr_artworks=None):
        
        """
        filtering current OVR artwork and sorting the predicted rating for each user

        """ 
        
        self.prediction_remapping()
        
        if current_ovr_artworks:
            self.predicted_df = self.predicted_df[self.predicted_df['item_id'].isin(current_ovr_artworks)]

        predicted_df = self.predicted_df.copy()
        predicted_df.rename(columns={'predicted':'rating'}, inplace=True)
        self.predicted_df_sorted = predicted_df.groupby(["user_id"], sort=False).apply(lambda x: x.sort_values(["rating"],\
                                                ascending = False)).reset_index(drop=True).head(predicted_df.shape[0]) 
        self.predicted_df_sorted['rating'] = self.predicted_df_sorted.rating.round(decimals=2)
        
        self.result = self.predicted_df_sorted.groupby('user_id')[['item_id', 'rating']].\
                    apply(lambda g: list(map(tuple, g.values.tolist()))).to_dict()

        return self.result
    
    def actual_pred_df(self,val_df):
        
        """
        Mapping actual useraffinity rating with predicted rating for validation dataset

        ---------- Input ---------------
        val_df: user_id - item_id - actualrating

        ---------- output ---------------
        validation dataframe : user_id - item_id -actualrating - predictedrating
        """ 
        
        # Actaul DataFrame
        actual_df = self.val_df.copy()
        #actual_df = actual_df[~actual_df.index.str.contains("dummy")]
        #actual_df = actual_df.unstack().reset_index(name='actual')
        actual_df.rename(columns = {'rating':'actual'}, inplace = True)
        actual_df = actual_df[['user_id', 'item_id', 'actual']]
        self.actual_df = actual_df.sort_values(['user_id', 'item_id'])

        final_df = pd.merge(self.actual_df, self.predicted_df, how='inner', left_on=['user_id','item_id'], right_on = ['user_id','item_id'])
        final_df = final_df[final_df.actual > 0]
        
        return final_df    
