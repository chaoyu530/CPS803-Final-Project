import os
from imutils import paths
import cv2 as cv
import numpy as np
import joblib
from scipy.cluster.vq import *


# Load the classifier, class names, scaler, number of clusters and vocabulary
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")
# Create feature extraction and keypoint detector objects
sift = cv.xfeatures2d.SIFT_create()


def predict_image(image_path):

    # List where all the descriptors are stored
    des_list = []
    im = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    im = cv.resize(im, (300, 300))
    kpts = sift.detect(im)
    kpts, des = sift.compute(im, kpts)
    des_list.append((image_path, des))

    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[0:]:
        descriptors = np.vstack((descriptors, descriptor))

    test_features = np.zeros((1, k), "float32")
    words, distance = vq(des_list[0][1], voc)
    for w in words:
        test_features[0][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((test_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Scale the features
    test_features = stdSlr.transform(test_features)

    # Perform the predictions
    predictions = [classes_names[i] for i in clf.predict(test_features)]
    return predictions


if __name__ == "__main__":
    test_path = "CPS803 Data/test/"
    testing_names = os.listdir(test_path)
    image_paths = []
    total = 0
    correct = 0
    for training_name in testing_names:
        dir = os.path.join(test_path, training_name)
        class_path = list(paths.list_images(dir))
        crt = 0;
        count = 0;
        for i in class_path:
            count += 1
            predictions = predict_image(i)
            if predictions[0] == training_name:
                crt +=1
        print(training_name, "  Precision: ", float(crt)/count)
        total += count
        correct += crt
    print("Total Accuracy: ", float(correct)/total)