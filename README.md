# sign_recognition_imageclassifier
Sign recognition using image using SVM classifier.

A classifier to recongize the 5 different signs to be used later for turtlebot navigation.

- A datset of 5 different signs captured from 3 meters away is also attached. 

The classifier first crops the images to extract relevant features from the image. Cropping the image is based on color. Cropped image is flattened and sent to KNN and SVM classifier for testing the data set. 


