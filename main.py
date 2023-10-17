import os
import random
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from my_knn import My_KNN
from preps import *
from features import *

df = pd.read_csv("diabetes.csv")

nan_check(df)
cat_features(df)
define_distrib(df)
show_corr_marix(df)


