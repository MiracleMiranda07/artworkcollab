# +

"""
Evaluating topk-Recommendation using map@k formula

The Average Precision metric measures how well the model gives relevant recommendations based 
on the top K recommendations.An average of the AP@N metric with all users called Mean Average Precision

"""
import pandas as pd
import json
from collections import defaultdict
class ModelEvaluation():

    
    def __init__(self, actual_pred_df):
        self.final_df = actual_pred_df
        
            
    def precision_recall_at_k(self,predictions,k,threshold):
        """Return precision and recall at k metrics for each user"""

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est in predictions:
            top_n[uid].append((iid,est))

        precisions = dict()
        recalls = dict()

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:k]


        for uid, user_ratings in top_n.items():

            # Sort user ratings by estimated value
            user_ratings=user_ratings


            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)


            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0, Precision is undefined. We here set it to 0.

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0, Recall is undefined. We here set it to 0.

            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
            
        print('MAP@K',sum(prec for prec in precisions.values()) / len(precisions))
        map_k=sum(prec for prec in precisions.values()) / len(precisions)
        map_k_dict={'map@k':map_k}
    
        return map_k_dict
