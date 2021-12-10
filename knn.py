import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# The first step is to split the training set and test set

X = [] #Define image name
Y = [] #Define image label
Z = [] #Define image pixels

labels = ["butterfly","cat","chicken","cow","dog","elephant","horse","sheep","spider","squittel"]
for i in labels:
    #Traverse folders, read pictures
    for f in os.listdir("E:/desktop/cps803/data/large/%s" % i):
        #Get image name
        X.append("E:/desktop/cps803/data/large//" +str(i) + "//" + str(f))
        #Take the image category label as the folder name
        Y.append(i)

X = np.array(X)
Y = np.array(Y)

#The random rate is 100%, 20% of which are selected as the test set
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2, random_state=1)

print (len(X_train), len(X_test), len(y_train), len(y_test))

# Step 2 Image reading and conversion to pixel histogram

#Training set
XX_train = []
for i in X_train:
    #Read image
    #print i
    image = cv2.imread(i)
    
    #Image pixel size is the same
    img = cv2.resize(image, (256,256),
                     interpolation=cv2.INTER_CUBIC)

    #Calculate image histogram and store to X array
    hist = cv2.calcHist([img], [0,1], None,
                            [256,256], [0.0,255.0,0.0,255.0])

    XX_train.append(((hist/255).flatten()))

#Test set
XX_test = []
for i in X_test:
    #Read image
    #print i
    image = cv2.imread(i)
    
    #Image pixel size is the same
    img = cv2.resize(image, (256,256),
                     interpolation=cv2.INTER_CUBIC)

    #Calculate image histogram and store to X array
    hist = cv2.calcHist([img], [0,1], None,
                            [256,256], [0.0,255.0,0.0,255.0])

    XX_test.append(((hist/255).flatten()))

# The third step is based on KNN image classification processing

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=11).fit(XX_train, y_train)
predictions_labels = clf.predict(XX_test)


print (u'forecast result:')
print (predictions_labels)

print (u'Algorithm evaluation:')
print (classification_report(y_test, predictions_labels))

#Output the first 10 pictures and prediction results
k = 0
while k<10:
    #read image
    print (X_test[k])
    image = cv2.imread(X_test[k])
    print (predictions_labels[k])
    #Display image
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    k = k + 1