import pandas as pd 
import numpy as np 

def eliminate_negatives(y_pred):
    y_pred[y_pred < 0] = 0
    return y_pred