#!/usr/bin/env python

import cv2
import sys
import csv
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC

### Load training images and labels

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


		(image,contours, _) = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

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

			if h >40:
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



# with open('./imgs(copy)/train.txt', 'rb') as f:
# 	reader = csv.reader(f)
# 	lines = list(reader)

# train = np.array([np.array(cv2.imread("./imgs(copy)/"+lines[i][0]+".png")) for i in range(len(lines))])


def feature(str1,str2):
	
	with open(str1, 'rb') as f:
		reader = csv.reader(f)
		lines = list(reader)

	train = np.array([np.array(cv2.imread(str2 + lines[i][0]+".png")) for i in range(len(lines))])

	img_fin = feature_vector(train,lines)


	train_data = img_fin.flatten().reshape(train.shape[0], 100*100)
	train_data = train_data.astype(np.float32)

	train_labels = np.zeros((train.shape[0],1))

	for i in range(len(lines)):
		train_labels[i] = np.int32(lines[i][1])
	feature = np.concatenate((train_data, train_labels), axis=1)

	return feature


path1 = './vision_checkpoint_test/train2.txt'
path2 = "./vision_checkpoint_test/"


path3 = './imgs(copy)/train.txt'
path4 = "./imgs(copy)/"

path5 = './cleanedTestingImages/test.txt'
path6 = "./cleanedTestingImages/"

path7 = './imgs/test.txt'
path8 = "./imgs/"

path7 = './imgs/test.txt'
path8 = "./imgs/"

path9 = './vision_checkpoint_test/test/test2.txt'
path10 = "./vision_checkpoint_test/test/"





feature1 = feature(path1,path2)
feature2 = feature(path3,path4)
feature3 = feature(path5,path6)
feature4 = feature(path7,path8)
feature5 = feature(path9,path10)


feature_fin = np.concatenate((feature1, feature2,feature3,feature4,feature5), axis=0)

print('Saving feature vector to csv file.......')

np.savetxt("foo.csv", feature_fin, delimiter=",")

print('Done.......')


# print (train_data.shape)

# print (train_labels.shape)

print (feature1.shape)
print (feature2.shape)
print (feature3.shape)
print (feature4.shape)
print (feature_fin.shape)

