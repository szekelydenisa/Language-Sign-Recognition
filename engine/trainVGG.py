# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from cnn import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
from glob import glob
from sklearn.utils import shuffle
import cv2
import os
import sys

folder = sys.argv[1]

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
print(folder + "\*\*.png")
# grab the image paths and sort them
imagePaths = glob(folder + "\*\*.png")
imagePaths.sort()
 
# loop over the input images
for imagePath in imagePaths:
	print(imagePath)
	sys.stdout.flush()
	# get the label from the image path, read the image and resize it
	label = imagePath[imagePath.find(os.sep)+1: imagePath.rfind(os.sep)]
	img = cv2.imread(imagePath, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (50,50))
	# labels list
	labels.append((np.array(img, dtype=np.uint8), label))

#shuffle the labels, and create a tuple of type {data, label}
labels = shuffle(shuffle(shuffle(shuffle(labels))))
data, labels = zip(*labels)

# partition the data into training, testing and validating splits
train_images = np.array(data[:int(5/6*len(data))])
train_labels = np.array(labels[:int(5/6*len(labels))])

test_images = np.array(data[int(5/6*len(data)):int(11/12*len(data))])
test_labels = np.array(labels[int(5/6*len(labels)):int(11/12*len(data))])

val_images = np.array(data[int(11/12*len(data)):])
val_labels = np.array(labels[int(11/12*len(labels)):])

# convert the labels from integers to vectors
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
val_labels= lb.transform(val_labels)
print(train_images)
train_images = np.reshape(train_images, (train_images.shape[0], 50, 50, 1))
val_images = np.reshape(val_images, (val_images.shape[0], 50, 50, 1))

# initialize our VGG-like Convolutional Neural Network
model, callbacks = SmallVGGNet.build(width=50, height=50, depth=1,
	classes=len(lb.classes_))

print("[INFO] training network...")
sys.stdout.flush()
model.summary()
model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=20, batch_size=500, callbacks=callbacks)
scores = model.evaluate(val_images, val_labels, verbose=0)
 
# evaluate the network
#print("[INFO] evaluating network...")
print(scores)
 
# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
model.save(foldername + "/model.model")
f = open(foldername + "/labels.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()