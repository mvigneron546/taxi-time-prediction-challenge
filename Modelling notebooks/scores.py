import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

def get_scores(y_true,y_pred,metrics=['rmse']):
    scores_dict={}
    errors=abs(y_true-y_pred)
    print(errors)
    for metric in metrics:
        if metric=='rmse':
            scores_dict[metric]=np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric=='mae':
            scores_dict[metric]=errors.mean()
        elif metric=='r2':
            scores_dict[metric]=r2_score(y_true, y_pred)
        elif metric=='first_quartile_error':
            scores_dict[metric]=np.percentile(errors,[25,75])[0]
        elif metric=='third_quartile_error':
            scores_dict[metric]=np.percentile(errors,[25,75])[1]
        else:
            scores_dict[metric]:'Invalid metric'
    return scores_dict