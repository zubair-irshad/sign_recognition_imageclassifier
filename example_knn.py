#!/usr/bin/env python

import cv2
import sys
import csv
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC

### Load training images and labels

with open('./imgs(copy)/train.txt', 'rb') as f:
	reader = csv.reader(f)
	lines = list(reader)


def feature_vector(train,lines):

	mask = np.zeros((train.shape[0],train.shape[1],train.shape[2]))


	fin_image = np.zeros((train.shape[0],100,100, train.shape[3]))

	imh= np.zeros((100,100, train.shape[3]))

	lower_red1 = np.array([0,50,10])
	upper_red1 = np.array([30,255,255])

	lower_red2 = np.array([160,50,20])
	upper_red2 = np.array([180,255,255])

	lower_green = np.array([37, 40, 25])
	upper_green = np.array([78, 255, 255]) 

	lower_blue = np.array([80,65,30]) 
	upper_blue = np.array([120,255,255]) 

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

	X_data = []

	ab=0

	for i in range(len(lines)):

		curr_image = train[i][:][:][:]

		hsv_gen = cv2.cvtColor(curr_image, cv2.COLOR_BGR2HSV)
		mask1 = cv2.inRange(hsv_gen, lower_red1, upper_red1)
		mask2 = cv2.inRange(hsv_gen, lower_red2, upper_red2)
		mask3 = cv2.inRange(hsv_gen, lower_green, upper_green)
		mask4 = cv2.inRange(hsv_gen, lower_blue, upper_blue)

		close = mask1 + mask2 + mask3 + mask4

		morph = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel)


		(contours, _) = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

		cnt = None

		areas = [] 

		if not contours:
			new_img=curr_image[0:80,0:80]
			resized = cv2.resize(new_img, (100,100), interpolation = cv2.INTER_AREA)
			X_data.append(resized)


		for c in contours:

			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c,0.001*cv2.arcLength(c,True),True)
			areas.append(cv2.contourArea(c))
			# cv2.drawContours(abc, [approx], -1, (0, 255, 0), 2)

			if len(approx) >= 4:
				screenCnt = approx
		
		if areas:

			max_index = np.argmax(areas)
			cnt=contours[max_index]
				
		if contours:
			x,y,w,h = cv2.boundingRect(cnt)

			if h >50:
				new_img=curr_image[y:y+h,x:x+w]
				resized = cv2.resize(new_img, (100,100), interpolation = cv2.INTER_AREA)
				X_data.append(resized)

				# elif h>53 and h<55:
				#     new_img=abc[y:y+2*h,x:x+w]
				#     resized = cv2.resize(new_img, (100,100), interpolation = cv2.INTER_AREA)
			elif h<23:
				new_img=curr_image[0:80,0:80]
				resized = cv2.resize(new_img, (100,100), interpolation = cv2.INTER_AREA)
				X_data.append(resized)
		area = []


	X = np.array(X_data)

	# print (X.shape)

	# for i in range(X.shape[0]):
	# 	cv2.imshow('t',X[i][:][:][:])
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows


	fin_data = []

	for i in range(X.shape[0]):

		orig = X[i][:][:][:]
		gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
		fin_data.append(gray)

	fin = np.array(fin_data)

	return fin



train = np.array([np.array(cv2.imread("./imgs(copy)/"+lines[i][0]+".png")) for i in range(len(lines))])


img_fin = feature_vector(train,lines)

train_data = img_fin.flatten().reshape(train.shape[0], 100*100)
train_data = train_data.astype(np.float32)

train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])


# knn = KNeighborsClassifier(n_neighbors=1)

knn = LinearSVC()
knn.fit(train_data, train_labels)

### Run test images


# correct = 0.0
# confusion_matrix = np.zeros((6,6))

with open('./cleanedTestingImages/test.txt', 'rb') as f1:
	reader1 = csv.reader(f1)
	lines1 = list(reader1)


test = np.array([np.array(cv2.imread("./cleanedTestingImages/"+lines1[i][0]+".png")) for i in range(len(lines1))])

print test.shape

# print(test.shape)

img_test = feature_vector(test,lines1)

# print(img_test.shape)



test_data = img_test.flatten().reshape(test.shape[0], 100*100)
test_data = test_data.astype(np.float32)


test_label = np.array([np.int32(lines1[i][1]) for i in range(len(lines1))])


y_pred = knn.predict(train_data)

y_test = knn.predict(test_data) 

# print (y_test)


print("training acuracy", accuracy_score(train_labels, y_pred))

print("test acuracy", accuracy_score(test_label, y_test))



# print("\n\nTotal accuracy: ", correct/len(lines))
# print(confusion_matrix)
