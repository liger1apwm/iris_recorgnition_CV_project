
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import IrisEnhancement
import IrisFeatureExtraction
import IrisLocalization
import IrisNormalization
import IrisMatching
import IrisPerformanceEvaluation
import pickle
import statistics


number_of_classes = 108
def max_class(index_array):
    max_label=[]

    i = 0
    while i < len(index_array):
        max_label.append(statistics.mode(index_array[i:i+3]))
        i= i+3

    #assert len(max_label) == len(index_array)/4
    return max_label

with open('images_features_train.pkl', 'rb') as file:
    images_features_train = pickle.load(file)

with open('images_features_test.pkl', 'rb') as file:
    images_features_test = pickle.load(file)

# images_features_train = images_features_train[0:99]
images_features_test = images_features_train

crr_d1 = []
crr_d2 = [] 
crr_d3 = []
crr_d4 = []
dims = []


train_labels = np.repeat(np.arange(1,number_of_classes+1), 3)
test_labels = np.repeat(np.arange(1,number_of_classes+1),3) 
index = np.arange(1,len(images_features_train)+1)
data = {'train_labels': train_labels, 'index': index}
train_label_df = pd.DataFrame(data)

for i in range(20,60, 20):
    images_features_dr_train, images_features_dr_test= IrisMatching.dimension_reduction(images_features_train, images_features_test, train_labels, k = i)


    d1 = IrisMatching.match_class(images_features_dr_train,images_features_dr_test, metric = 'L1' )
    d4 = IrisMatching.nearestCentroid(images_features_dr_train,images_features_dr_test ,train_labels, metric = 'l1' ) 
    
    d1_df = pd.DataFrame(d1,columns =["index"])
    d1_df = d1_df.merge(train_label_df, on = "index", how="left")
    d1 = d1_df["train_labels"]

    d1 = max_class(d1) 
    d4 = max_class(d4) 

    test_true_labels = np.arange(1,number_of_classes+1)

    dims.append(i)
    crr_d1.append(IrisPerformanceEvaluation.CRR(test_true_labels,d1))
    crr_d4.append(IrisPerformanceEvaluation.CRR(test_true_labels,d4))

crr_data = {'dims':dims,'crr_d1':crr_d1, 'crr_d4':crr_d4}
crr_df = pd.DataFrame(crr_data)
print(crr_df)