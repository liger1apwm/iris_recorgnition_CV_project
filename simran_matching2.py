
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
images_per_train_class  = 3
images_per_test_class = 3

def max_class(index_array, images_per_test_class):
    max_label=[]

    i = 0
    while i < len(index_array):
        max_label.append(statistics.mode(index_array[i:i+images_per_test_class]))
        i= i+images_per_test_class

    #assert len(max_label) == len(index_array)/4
    return max_label

with open('images_features_train.pkl', 'rb') as file:
    images_features_train = pickle.load(file)

with open('images_features_test.pkl', 'rb') as file:
    images_features_test = pickle.load(file)

    images_features_test = images_features_train
    
   #SIMRAN TESTING
    crr_d1 = []
    crr_d2 = [] 
    crr_d3 = []
    crr_d4 = []
    dims = []

    train_labels = np.repeat(np.arange(1,number_of_classes+1),images_per_train_class)
    test_labels = np.repeat(np.arange(1,number_of_classes+1),images_per_test_class) 
    index = np.arange(1,len(images_features_train)+1)
    data = {'train_labels': train_labels, 'index': index}
    train_label_df = pd.DataFrame(data)

    assert len(train_labels) == len(images_features_train)

    for i in range(20,40, 20):
        images_features_dr_train, images_features_dr_test= IrisMatching.dimension_reduction(images_features_train, images_features_test, train_labels, k = i)

        #calculating distance of one vector test with all train vectors and append all test results 
        #output is indices
        d1 = IrisMatching.match_class(images_features_dr_train,images_features_dr_test, metric = 'L1' )
        d2 = IrisMatching.match_class(images_features_dr_train,images_features_dr_test, metric = 'L2'  )
        d3 = IrisMatching.match_class(images_features_dr_train,images_features_dr_test , metric = 'Cosine' )
        d4 = IrisMatching.nearestCentroid(images_features_dr_train,images_features_dr_test ,train_labels, metric = 'l1' )

        d1_df = pd.DataFrame(d1,columns =["index"])
        d1_df = d1_df.merge(train_label_df, on = "index", how="left")
        d1 = d1_df["train_labels"]

        d2_df = pd.DataFrame(d2,columns =["index"])
        d2_df = d2_df.merge(train_label_df, on = "index",  how="left")
        d2 = d2_df["train_labels"]

        d3_df = pd.DataFrame(d3,columns =["index"])
        d3_df = d3_df.merge(train_label_df, on = "index",  how="left")
        d3 = d3_df["train_labels"]

        #converting maximum of 4 labels associated with each test iris to one class
        
        d1 = max_class(d1,images_per_test_class) #matched class for test vector 
        d2 = max_class(d2,images_per_test_class)
        d3 = max_class(d3,images_per_test_class)
        d4 = max_class(d4,images_per_test_class)

        assert len(d1) == 108
        assert len(d2) == 108
        assert len(d3) == 108

        test_true_labels = np.arange(1,number_of_classes+1)

        #correct recognition rate 
        #print(IrisPerformanceEvaluation.CRR(test_labels,d1))
        dims.append(i)
        crr_d1.append(IrisPerformanceEvaluation.CRR(test_true_labels,d1))
        crr_d2.append(IrisPerformanceEvaluation.CRR(test_true_labels,d2))
        crr_d3.append(IrisPerformanceEvaluation.CRR(test_true_labels,d3))
        crr_d4.append(IrisPerformanceEvaluation.CRR(test_true_labels,d4))
    
    crr_data = {'dims':dims,'crr_d1':crr_d1,'crr_d2':crr_d2,'crr_d3':crr_d3, 'crr_d4':crr_d4}
    crr_df = pd.DataFrame(crr_data)
    print(crr_df)

