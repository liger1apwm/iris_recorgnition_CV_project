import IrisEnhancement
import IrisFeatureExtraction
import cv2



def main():
    image = cv2.imread('./iris_normalized_test.png')

    enhanced_image = IrisEnhancement.enhacement(image)

    cv2.imshow('Histogram Equalized', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    feature_vec = IrisFeatureExtraction.feature_extraction(enhanced_image)

    print("the len is : ",len(feature_vec))
    print("first 10 values : ",feature_vec[0:9])

if __name__ == "__main__":
  main()