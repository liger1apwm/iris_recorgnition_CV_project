from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import statistics

#using LDA from sklearn to reduce dimensions of feature vector 
def dimension_reduction(feature_train,feature_test, labels, k):

    lda = LinearDiscriminantAnalysis(n_components=k)  # k is the desired number of dimensions
    feature_vector_train_lda = lda.fit_transform(feature_train, labels)  # Transformed training data
    feature_vector_test_lda = lda.transform(feature_test) 
    return feature_vector_train_lda, feature_vector_test_lda

###################################################################################################
                    # Two Approaches to obtain matching class
                    # Appraoch 2 (on line 88) is chosen as it results in better model performance 
###################################################################################################

#Approach 1: Distance between each vector from test and each vector from train is calculated; it results in 
#an index or location where test vecto got matched

#match_class function is created for easier calculation in main code
def match_class(feature_vec_train, feature_vec_test, metric):

    if metric == 'l1':
        d = calculate_L1_distance(feature_vec_train, feature_vec_test)
    elif metric == 'l2':
        d = calculate_L2_distance(feature_vec_train, feature_vec_test)
    elif metric == 'cosine':
        d = calculate_cosine_distance(feature_vec_train, feature_vec_test)
    else:
        raise ValueError("Invalid distance_metric. Supported values are 'L1', 'L2', and 'Cosine'.")
    return d 

#L1 distance
def calculate_L1_distance(feature_vec_train, feature_vec_test):
    
    feature_vec_train = np.array(feature_vec_train)
    feature_vec_test = np.array(feature_vec_test)

    #index of min distance for each test vector
    d1= [] 

    for i in feature_vec_test:
        d1.append(np.argmin(np.sum(np.abs(i - feature_vec_train), axis =1)))

    return d1

#l2 distance
def calculate_L2_distance(feature_vec_train, feature_vec_test):    
    
    feature_vec_train = np.array(feature_vec_train)
    feature_vec_test = np.array(feature_vec_test)

    #index of min distance for each test vector
    d2= [] 

    for i in feature_vec_test:
        d2.append(np.argmin(np.sum(np.square(i - feature_vec_train), axis =1)))

    return d2

#cosine distance
def calculate_cosine_distance(feature_vec_train, feature_vec_test):    
    feature_vec_train = np.array(feature_vec_train)
    feature_vec_test = np.array(feature_vec_test)

    #index of min distance for each test vector
    d3= [] 
    cosine_distance_array = []
    for idx,train_vector in enumerate(feature_vec_train):
        # print("working on index: ",idx)
        numerator = np.dot(test_vec.T,train_vector)
        # print("numerator at index : ",idx ," is : ",numerator)
        denominator = np.dot(np.linalg.norm(test_vec),np.linalg.norm(train_vector))
        # print("denominator at index : ",idx ," is : ",denominator)

        d3_unit = 1 - (numerator/denominator)
        # print("d3_unit at index : ",idx ," is : ",d3_unit)
        cosine_distance_array.append(d3_unit)

    # print("arg min at index test : ",idx_test ," is : ",np.argmin(cosine_distance_array))
    d3.append(np.argmin(cosine_distance_array))
    return d3


#Approach 2: Nearest Centroid Method: calculates centroid of three training vectors of each class and 
# finds closest distance with test vector
def nearestCentroid(feature_vec_train, feature_vec_test, labels, metric, score):

    #when score = False, it returns a class closest to nearest centroid
    if score == False:
        #for distance metric, class label is obtained for text vector using nearest centroid method 
        clf = NearestCentroid(metric) 
        clf.fit(feature_vec_train, labels)    

        d = []
        #predict for test class
        d = clf.predict(feature_vec_test)
    
    #when score = True, it returns similarity score 
    else:
        clf = NearestCentroid(metric)
        clf.fit(feature_vec_train, labels)
        # print("labels : ", labels)
        # predicted_label = clf.predict(feature_vec_test)

        # get the centroids to be able to calculate the distances
        centroids = clf.centroids_
        # class_centroid = clf.classes_

        #vector to store the min dist for each test vector
        min_cosine_test_dist = []

        # loop tru all the testing vector that need to be calculated vs the train centroids
        for test_vect in feature_vec_test:
            
            # array to save the distances for the currect test vector vs all train centroids
            cosine_dist_vs_centers = []

            #iterate tru the centroids to calculate distances
            for train_center in centroids:
                
                # Calculate the distances to each centrids and append it
                cosine_dist_vs_centers.append(1 - cosine_similarity(test_vect.reshape(1, -1),train_center.reshape(1, -1)))

            #Append the corresponding min distance as the classified class distance 
            min_cosine_test_dist.append(min(cosine_dist_vs_centers)[0][0])

        # array to be return with cosine similarities distances
        d = min_cosine_test_dist

    return d
