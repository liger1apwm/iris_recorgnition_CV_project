# iris_recorgnition_CV_project


For this Iris recognition project. The whole procedure follows 6 important steps:

1) Localize the Iris: In this steps we aim to first detect the pupil to then use edge detection and find 
the region that enclose the iris using two circles

2) Iris Normalization: After we find our Iris we will have an enclosed region between our circles that 
is the iris, our goal in this step is to convert the region which is in polar coordinates to a rectangle
images in a cartesian plane.

3) Enhacement of the normalized image: In step 1, the image had to be converted to a gray image to work with 
the image better. But images tend to have low contrast and is hard to see the patterns of the iris. Therefore
we use histogram equalization to make the iris normalized images pattern easier to capture.

4) Feature extraction: After the image is enhaced, now we can quantify the patterns in the iris by using 
the proposed filters in the paper to first filter the images using two channels to obtain a 2 filtered images
, then we will get the mean and the average absolute deviation of each 8x8 block on each filtered image to create a vector
that will contain 1536 features

5) Iris matching: now that we can extract the features of a iris image, we can collect all the iris features For
our training dataset that consist on 3 train images for 108 distinct persons (classes). After gathering the array containing 
all the features vectors for our training data. We will reduce the dimensions using fisher linear discrimant to experiment with 
reducing dimensions stating from 20 up to 100 going 10 by 10. Once the features dimensions are reduced, is faster
we are using the nearest center classifier to find the means(centers) of each class to use then with matching
the test features vectors. We then use the 3 proposed distances : L1,l2,cosine as distance measures to match tests vectors to the 
closest center by returning the class that is the minimun on each of the distances.

6) Performance and Evaluations: Now that we have matched all test iris with a class in the trianing data, now we will
create a table for all the different dimensions up to 100 to check the CRR (correct recognition rate) for eacht distance
to be able to observe what dimension with what distance gives the best recognition. With this we can create a graph using the best distance
to see how is going.
Also, we can create a plot between FMR ( false match rate) and FNMR (false non-match rate) to visualized a plot between 
how many times the process incorrectly accepts a non-match pair as a match vs how many times the process incorrectly reject a match pair 
as a non-match. 

Logic of the Design: 

We first focus on collecting all the training images in our dataset by looping through the directory localization
and finding filers on each classes in folder 1 and ending in .bmp . We then locate the iris and save the images with 
the boundries and the centers of our localized image. Following to this, We use the images with boundries and the centers
and we normalize the image. The normalized images gets the enhaced for both traning and testing. Then, for each train and testing
individually we extract the features of each enhanced image and store them in a vector. after getting this vectors,
we are using the nearest centroid classifier on the train data to find the centers for each class, then calculating to what center is each 
testing feature closest too using all 3 distances described in the paper. Once each testing image is classified. We find the 
CRR (correct recognition rate) to see what images where correctly classified to the class they belong and we do this from different
dimension reduction parameters from 20 up to 100. From this we create a graph to visualize the different distances
with all the different dimensions. Also, we do a table to calculate the FMR (false match rating) and FNMR (false non match rating)
and create a graph to see their correlation.
