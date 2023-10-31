import IrisEnhancement
import IrisFeatureExtraction
import cv2

def visualize_image(image,title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    


    #ERIC TESTING CODE
    image = cv2.imread('./iris_normalized_test.png')

    # Resize the image to a new width and height
    new_width = 512  # New width in pixels
    new_height = 64  # New height in pixels
    resized_image = cv2.resize(image,(new_width, new_height))

    # print(resized_image.shape)
    # cv2.imshow('Resized image', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    enhanced_image = IrisEnhancement.enhacement(resized_image)


    crop_amount = 48
    # (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
    x1, y1, x2, y2 = 0, 0, 512, crop_amount

    # Crop the specified region
    enhanced_image_crop = enhanced_image[y1:y2, x1:x2]

    # visualize_image(enhanced_image_crop,'Histogram Equalized')

    feature_vec = IrisFeatureExtraction.feature_extraction(enhanced_image_crop,crop_amount)

    print("the len is : ",len(feature_vec))
    print("first 10 values : ",feature_vec[0:9])

    # feature_filtered1,feature_filtered2 = IrisFeatureExtraction.feature_extraction(enhanced_image_crop,crop_amount)
    # visualize_image(feature_filtered1,'Filter1 image')
    # visualize_image(feature_filtered2,'Filter2 image')
    # cv2.imwrite('./output_image_filter1.jpg', feature_filtered1)
    # cv2.imwrite('./output_image_filter2.jpg', feature_filtered2)

   

if __name__ == "__main__":
  main()