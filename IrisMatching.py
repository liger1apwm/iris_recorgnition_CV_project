#reduce dimensions

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

#fea_vec, label

#[[arr1], [arr2], [arr3], ..,] 
# labels?
# 108*3*1536 training

#reducing 1536 to 20 dimensions against your class

def dimension_reduction(feature_train,feature_test, labels, k):

    lda = LinearDiscriminantAnalysis(n_components=k)  # k is the desired number of dimensions
    feature_vector_train_lda = lda.fit_transform(feature_train, labels)  # Transformed training data
    feature_vector_test_lda = lda.transform(feature_test) 
    return feature_vector_train_lda, feature_vector_test_lda

def calculate_L1_distance(feature_vec_train, feature_vec_test):
    
    feature_vec_train = np.array(feature_vec_train)
    feature_vec_test = np.array(feature_vec_test)

    #index of min distance for each test vector
    d1= [] 

    for i in feature_vec_test:
        d1.append(np.argmin(np.sum(np.abs(i - feature_vec_train), axis =1)))

    return d1

def calculate_L2_distance(feature_vec_train, feature_vec_test):    
    
    feature_vec_train = np.array(feature_vec_train)
    feature_vec_test = np.array(feature_vec_test)

    #index of min distance for each test vector
    d2= [] 

    for i in feature_vec_test:
        d2.append(np.argmin(np.sum(np.square(i - feature_vec_train), axis =1)))
        #print(d2)

    return d2

def calculate_cosine_distance(feature_vec_train, feature_vec_test):    
    
    feature_vec_train = np.array(feature_vec_train)
    feature_vec_test = np.array(feature_vec_test)

    #index of min distance for each test vector
    d3= [] 
    
    
    for idx_test,test_vec in enumerate(feature_vec_test):
        #simran code
        # A = np.sqrt(np.sum(np.square(i)))
        # for m,n in enumerate(feature_vec_train):
        #     B = np.sqrt(np.sum(np.square(n)))
        #     numerator = np.sum(np.multiply(i, n))
        #     cosine = np.divide(numerator, (A*B))
        #     cosine_distance_array.append(cosine)
        # d3.append(np.argmin(cosine_distance_array))

        #print(d3)
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

def max_class(index_array):
    max_label=[]

    i = 0
    while i < len(index_array):
        j= i+4
        max_label.append(np.max(index_array[i:j]))
        i= j 
    
    assert len(max_label) == len(index_array)/4

    return max_label

#feature_vec_train = [[1,2,3,4,5], [1,2,4,6,6], [1,2,4,5,6]]
#feature_vec_test = [[1,2,3,4,4]]

#calculate_L1_distance(feature_vec_train,feature_vec_test )
#calculate_L2_distance(feature_vec_train,feature_vec_test )
# calculate_cosine_distance(feature_vec_train,feature_vec_test )