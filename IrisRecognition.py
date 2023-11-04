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

def visualize_image(image,title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pipeline(image,cropamount):
   
   return 0


def main():
    
    database_dir = "./CASIA Iris Image Database (version 1.0)/"


    #ERIC TESTING CODE
    images_features_train = []
    images_path_train = []
    images_train = []

    images_features_test = []
    images_path_test = []
    images_test = []

    #PORTION TO GET ALL FEATURES FOR THE TRAIN DATA
    for i in range(1,109,1): 
        
        current_img_path = database_dir + f"{i:03d}" + "/1/"

        filenames = os.listdir(current_img_path)

        for filename in filenames:
            
            if filename.lower().endswith(('.bmp')):
                path = os.path.join(current_img_path,filename)
                images_path_train.append(path)
                images_train.append(cv2.imread(path))
                # print(os.path.join(current_img_path,filename))
   
    boundaries = []
    centers = []

    # # Localize iris in each image
    # for image_file in images_path_train:
    #     image = cv2.imread(image_file)

    #     print(image_file)
        
    boundary, center = IrisLocalization.IrisLocalization(images_train)
    boundaries.extend(boundary)
    centers.extend(center)


    # Normalize each localized iris
    normalized_images = IrisNormalization.IrisNormalization(boundaries, centers)
    # print("the length of the normalized images is : ",len(normalized_images))

    # print(len(normalized_images))

    output_directory = "./normalized_images_train/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

     # Save each normalized image to the output directory
    for idx, normalized_img in enumerate(normalized_images):
        
        output_path = os.path.join(output_directory, f"normalized_{idx//3 +1}_{idx%3 + 1}.png")
        if not os.path.exists(output_path):
            cv2.imwrite(output_path, normalized_img)

        # print(normalized_img.shape)
        # print(f"index before crash is {idx+1}")
        normalize_image = cv2.imread(output_path)
        normalize_image_gray = cv2.cvtColor(normalize_image, cv2.COLOR_BGR2GRAY)
        crop_amount = 48
        # print(f"image number {idx +1} is ",normalized_img)
        enhanced_image = IrisEnhancement.enhacement(normalize_image_gray)
        # (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
        x1, y1, x2, y2 = 0, 0, 512, crop_amount
        # Crop the specified region
        enhanced_image_crop = enhanced_image[y1:y2, x1:x2]
        # visualize_image(enhanced_image_crop,'Histogram Equalized')

        feature_vec = IrisFeatureExtraction.feature_extraction(enhanced_image_crop,crop_amount)
        images_features_train.append(feature_vec)

    print("Number of features vec on image features train array is: ",len(images_features_train))


    #PORTION TO GET ALL FEATURES FOR THE TEST DATA
    for i in range(1,109,1): 
        
        current_img_path = database_dir + f"{i:03d}" + "/2/"

        filenames = os.listdir(current_img_path)

        for filename in filenames:
            
            if filename.lower().endswith(('.bmp')):
                path = os.path.join(current_img_path,filename)
                images_path_test.append(path)
                images_test.append(cv2.imread(path))
                # print(os.path.join(current_img_path,filename))
   
    boundaries = []
    centers = []

    # # Localize iris in each image
    # for image_file in images_path_train:
    #     image = cv2.imread(image_file)

    #     print(image_file)
        
    boundary, center = IrisLocalization.IrisLocalization(images_test)
    boundaries.extend(boundary)
    centers.extend(center)


    # Normalize each localized iris
    normalized_images = IrisNormalization.IrisNormalization(boundaries, centers)
    # print("the length of the normalized images is : ",len(normalized_images))

    # print(len(normalized_images))

    output_directory = "./normalized_images_test/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

     # Save each normalized image to the output directory
    for idx, normalized_img in enumerate(normalized_images):
        
        output_path = os.path.join(output_directory, f"normalized_{idx//4 +1}_{idx%4 + 1}.png")
        if not os.path.exists(output_path):
            cv2.imwrite(output_path, normalized_img)

        # print(normalized_img.shape)
        # print(f"index before crash is {idx+1}")
        normalize_image = cv2.imread(output_path)
        normalize_image_gray = cv2.cvtColor(normalize_image, cv2.COLOR_BGR2GRAY)
        crop_amount = 48
        # print(f"image number {idx +1} is ",normalized_img)
        enhanced_image = IrisEnhancement.enhacement(normalize_image_gray)
        # (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
        x1, y1, x2, y2 = 0, 0, 512, crop_amount
        # Crop the specified region
        enhanced_image_crop = enhanced_image[y1:y2, x1:x2]
        # visualize_image(enhanced_image_crop,'Histogram Equalized')

        feature_vec = IrisFeatureExtraction.feature_extraction(enhanced_image_crop,crop_amount)
        images_features_test.append(feature_vec)
        #print(images_features_test[0])

    print("Number of features vec on image features test array is: ",len(images_features_test))

    #SIMRAN TESTING

    train_labels = np.repeat(np.arange(1,109), 3)
    index = np.arange(1,len(images_features_train)+1)
    data = {'train_labels': train_labels, 'index': index}
    train_label_df = pd.DataFrame(data)

    assert len(train_labels) == len(images_features_train)

    crr_d1 = []
    crr_d2 = [] 
    crr_d3 = []
    dims = []

    for i in range(20,160, 20):
        images_features_dr_train= IrisMatching.dimension_reduction(images_features_train, images_features_test, train_labels, k = i)[0]
        images_features_dr_test= IrisMatching.dimension_reduction(images_features_train, images_features_test, train_labels, k = i)[1]

        #calculating distance of one vector test with all train vectors and append all test results 
        #output is indices
        d1 = IrisMatching.calculate_L1_distance(images_features_dr_train,images_features_dr_test )
        d2 = IrisMatching.calculate_L2_distance(images_features_dr_train,images_features_dr_test )
        d3 = IrisMatching.calculate_cosine_distance(images_features_dr_train,images_features_dr_test )


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
        d1 = IrisMatching.max_class(d1) #matched class for test vector 
        d2 = IrisMatching.max_class(d2)
        d3 = IrisMatching.max_class(d3)

        assert len(d1) == 108
        assert len(d2) == 108
        assert len(d3) == 108

        test_labels = np.arange(1,109)
        #print(d1)

        #correct recognition rate 
        #print(IrisPerformanceEvaluation.CRR(test_labels,d1))
        dims.append(i)
        crr_d1.append(IrisPerformanceEvaluation.CRR(test_labels,d1))
        crr_d2.append(IrisPerformanceEvaluation.CRR(test_labels,d2))
        crr_d3.append(IrisPerformanceEvaluation.CRR(test_labels,d3))
    
    crr_data = {'dims':dims,'crr_d1':crr_d1,'crr_d2':crr_d2,'crr_d3':crr_d3}
    crr_df = pd.DataFrame(crr_data)
    print(crr_df)











   










    

    # for i in range(1,5,1):

    #     new_width = 512  # New width in pixels
    #     new_height = 64  # New height in pixels
    #     crop_amount = 48
    #     image = cv2.imread(f'./testing_pictures/iris_normalized_test{i}.png')
    #     # print(image)
    #     # Resize the image to a new width and height
        
    #     resized_image = cv2.resize(image,(new_width, new_height))
    #     resized_image_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    #     print(resized_image_gray.shape)
    #     enhanced_image = IrisEnhancement.enhacement(resized_image_gray)
    #     # (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
    #     x1, y1, x2, y2 = 0, 0, 512, crop_amount
    #     # Crop the specified region
    #     enhanced_image_crop = enhanced_image[y1:y2, x1:x2]
    #     visualize_image(enhanced_image_crop,'Histogram Equalized')

    #     # feature_vec = IrisFeatureExtraction.feature_extraction(enhanced_image_crop,crop_amount)
    #     # images_features.append(feature_vec)

    #     # print(f"the len for test image {i} is : ",len(feature_vec))
    #     # print("first 10 values : ",feature_vec[0:9])

    #     # feature_filtered1,feature_filtered2 = IrisFeatureExtraction.feature_extraction(enhanced_image_crop,crop_amount)
    #     # visualize_image(feature_filtered1,'Filter1 image')
    #     # visualize_image(feature_filtered2,'Filter2 image')
    #     # cv2.imwrite('./output_image_filter1.jpg', feature_filtered1)
    #     # cv2.imwrite('./output_image_filter2.jpg', feature_filtered2)

    # # print(f"the final vector with all 4 test image features is: ",len(images_features))
    
    

 ## LYLYBELL TESTING 

# def process_dataset(input_directory, output_directory):
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)

#     # Using os.walk() to recursively fetch image files from subdirectories
#     image_files_paths = []
#     image_files = []
#     for dirpath, dirnames, filenames in os.walk(input_directory):
#         for filename in [f for f in filenames if f.lower().endswith(('.bmp'))]:
#             image_files_paths.append(os.path.join(dirpath, filename))

#     boundaries = []
#     centers = []

#     # Localize iris in each image
#     for image_file in image_files:
#         image = cv2.imread(image_file)
        
#         boundary, center = IrisLocalization([image])
#         boundaries.extend(boundary)
#         centers.extend(center)

#     # Normalize each localized iris
#     normalized_images = IrisNormalization(boundaries, centers)

#     # Save each normalized image to the output directory
#     for idx, normalized_img in enumerate(normalized_images):
#         output_path = os.path.join(output_directory, f"normalized_{idx}.png")
#         cv2.imwrite(output_path, normalized_img)

# # Example usage:
# input_dir = "CASIA Iris Image Database (version 1.0)"
# output_dir = "iris_normalized_imgs"
# process_dataset(input_dir, output_dir)

# ## END 

if __name__ == "__main__":
  main()