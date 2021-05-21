# import the necessary packages
import argparse
import pickle
import cv2
import numpy as np
 
class Prediction:
    @staticmethod
    def predict(image, model, label, width, height, flatten):
        # load the input image and resize it to the target spatial dimensions
        output = image.copy()
        image = cv2.resize(image, (width, height))

        image = np.array(image, dtype=np.float32)
        image = image = np.reshape(image, (1, width, height, 1))
        # load the model and label binarizer
        print("[INFO] loading network and label binarizer...")
        lb = pickle.loads(open(label, "rb").read())
 
        # make a prediction on the image
        preds = model.predict(image)
    

        # find the class label index with the largest corresponding
        # probability
        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]

        # draw the class label + probability on the output image
        text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	        (0, 0, 255), 2)
 
        return label, max(preds[0])