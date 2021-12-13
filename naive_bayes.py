import datetime
starttime = datetime.datetime.now()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import cv2

# ----------------------------------------------------------------------------------------------------
import os
os.chdir((os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/"))
dataset=Path('../data/large')
iter = 1000
img_target = 64
# ----------------------------------------------------------------------------------------------------


X = []
Y = []

clazz = ["butterfly","cat","chicken","cow","dog","elephant","horse","sheep","spider","squittel"]
for i in clazz:
    #Traversing folders and reading images
    for f in os.listdir("./small/%s" % i):
        print("./small/%s/%s" % (i, f))
        #Open an image and gray it out
        Images = cv2.imread("./small/%s/%s" % (i, f)) 
        image=cv2.resize(Images,(256,256),interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([image], [0,1], None, [256,256], [0.0,255.0,0.0,255.0]) 
        X.append(((hist/255).flatten()))
        Y.append(i)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
#The random rate is 100% (to ensure uniqueness for comparison) 30% of which is selected as the test set

from sklearn.preprocessing import binarize 
from sklearn.preprocessing import LabelBinarizer

class ML:
    def predict(self, x):
        #predict
        X = binarize(x, threshold=self.threshold)
        #The value that maximizes the log-likelihood function also maximizes the likelihood function
        #Y_predict = np.dot(X, np.log(prob).T)+np.dot(np.ones((1,prob.shape[1]))-X, np.log(1-prob).T)
        #lnf(x)=xlnp+(1-x)ln(1-p)
        Y_predict = np.dot(X, np.log(self.prob).T)-np.dot(X, np.log(1-self.prob).T) + np.log(1-self.prob).sum(axis=1)
        
        return self.classes[np.argmax(Y_predict, axis=1)]
        
class Bayes(ML): 
    def __init__(self,threshold):
        self.threshold = threshold
        self.classes = []
        self.prob = 0.0
        
    def fit(self, X, y):
        
        #Label binarization
        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y) 
        self.classes = labelbin.classes_ #Total statistical categories, 10 categories
        Y = Y.astype(np.float64)
        
        #Convert to a binary classification problem
        X = binarize(X, threshold=self.threshold)
        feature_count = np.dot(Y.T, X) #Matrix transpose, fusion of identical features
        class_count = Y.sum(axis=0) #Count the number of occurrences of each category
        
        #Laplace smoothing to solve the problem of zero probability
        alpha = 1.0
        smoothed_fc = feature_count + alpha
        smoothed_cc = class_count + alpha * 2
        self.prob = smoothed_fc/smoothed_cc.reshape(-1, 1)
        
        return self
        
clf0 = Bayes(0.2).fit(X_train,y_train)
predictions_labels = clf0.predict(X_test)
confusion_matrix(y_test, predictions_labels)
print (classification_report(y_test, predictions_labels))
endtime = datetime.datetime.now()
print (endtime - starttime)