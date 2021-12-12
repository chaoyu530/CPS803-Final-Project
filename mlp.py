# # Image Classification with MLP

# - In this project we are going to classify Pokemon.
# - Here we will use Neural Network for image classification

import os
from pathlib import Path
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from sklearn.metrics import classification_report
random.seed(10)

# ----------------------------------------------------------------------------------------------------
os.chdir((os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/"))
dataset=Path('../data/large')
iter = 1000
img_target = 64
# ----------------------------------------------------------------------------------------------------


# Import Dataset and make an array
dirs=dataset.glob("*")

image_data=[]
labels=[]

label_dict={}
label_to_animals={}
counter=0
total_img_size = 0

for i in dirs:
    label=str(i).split("\\")[-1]
    label_dict[label]=counter
    label_to_animals[counter]=label
    
    print(i)
    count=0
    
    for img_path in i.glob("*.jpg"):
        img = image.load_img(img_path,target_size=(img_target,img_target))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        labels.append(counter)
        count +=1
        
    print(count)
    counter +=1
    total_img_size += count


X=np.array(image_data)
Y=np.array(labels)


def drawimage(image,label):
    plt.style.use('seaborn')
    plt.title(label_to_animals[label])
    plt.imshow(image)
    plt.show()


drawimage(X[344]/255,Y[344])


# ### Create train and test set
                                             
from sklearn.utils import shuffle                                            #Shuffle our data
X,Y = shuffle(X,Y,random_state=2)

X = X/255.0                                                                  #Normalisation



### Create Training and Testing Set
X_ = np.array(X)
Y_ = np.array(Y)

#Training Set
training_size = int(0.7 * total_img_size)
X = X_[:training_size,:]
Y = Y_[:training_size]

#Test Set
test_size = int(0.3 * total_img_size)
XTest = X_[test_size:,:]
YTest = Y_[test_size:]

print(X.shape,Y.shape)
print(XTest.shape,YTest.shape)


print('\nBuild Out Multi Layer Perceptron model:')
print('=> => => => => => => => => => => => => =>')
print('=> => => => => => => => => => => => => =>')
class NeuralNetwork:
    
    def __init__(self,input_size,layers,output_size):
        np.random.seed(0)
        
        model = {} #Dictionary
        
        #First Layer
        model['W1'] = np.random.randn(input_size,layers[0])
        model['b1'] = np.zeros((1,layers[0]))
        
        #Second Layer
        model['W2'] = np.random.randn(layers[0],layers[1])
        model['b2'] = np.zeros((1,layers[1]))
        
        #Third Layer
        model['W3'] = np.random.randn(layers[1],layers[2])
        model['b3'] = np.zeros((1,layers[2]))
        
        #Output Layer
        model['W4'] = np.random.randn(layers[2],output_size)
        model['b4'] = np.zeros((1,output_size))
        
        self.model = model
        self.activation_outputs = None
    
    def forward(self,x):
        
        W1,W2,W3,W4 = self.model['W1'],self.model['W2'],self.model['W3'],self.model['W4']
        b1, b2, b3,b4 = self.model['b1'],self.model['b2'],self.model['b3'],self.model['b4']
        
        z1 = np.dot(x,W1) + b1
        a1 = np.tanh(z1) 
        
        z2 = np.dot(a1,W2) + b2
        a2 = np.tanh(z2)
        
        z3 = np.dot(a2,W3) + b3
        a3 = np.tanh(z3)
        
        z4 = np.dot(a3,W4) + b4
        y_ = softmax(z4)
        
        self.activation_outputs = (a1,a2,a3,y_)
        return y_
        
    def backward(self,x,y,learning_rate=0.005):
        W1,W2,W3,W4 = self.model['W1'],self.model['W2'],self.model['W3'],self.model['W4']
        b1, b2, b3,b4 = self.model['b1'],self.model['b2'],self.model['b3'],self.model['b4']
        m = x.shape[0]
        
        a1,a2,a3,y_ = self.activation_outputs
        
        delta4 = y_ - y
        dw4 = np.dot(a3.T,delta4)
        db4 = np.sum(delta4,axis=0)
        
        delta3 = (1-np.square(a3))*np.dot(delta4,W4.T)
        dw3 = np.dot(a2.T,delta3)
        db3 = np.sum(delta3,axis=0)
        
        delta2 = (1-np.square(a2))*np.dot(delta3,W3.T)
        dw2 = np.dot(a1.T,delta2)
        db2 = np.sum(delta2,axis=0)
        
        delta1 = (1-np.square(a1))*np.dot(delta2,W2.T)
        dw1 = np.dot(X.T,delta1)
        db1 = np.sum(delta1,axis=0)
        
        dw1 += 2 * 0.0001 * self.model['W1']
        dw2 += 2 * 0.0001 * self.model['W2']
        dw3 += 2 * 0.0001 * self.model['W3']
        
        #Update the Model Parameters using Gradient Descent
        self.model["W1"]  -= learning_rate*dw1
        self.model['b1']  -= learning_rate*db1
        
        self.model["W2"]  -= learning_rate*dw2
        self.model['b2']  -= learning_rate*db2
        
        self.model["W3"]  -= learning_rate*dw3
        self.model['b3']  -= learning_rate*db3
        
        self.model["W4"]  -= learning_rate*dw4
        self.model['b4']  -= learning_rate*db4
        
        # :)
        
    def predict(self,x):
        y_out = self.forward(x)
        return np.argmax(y_out,axis=1)
    
    def summary(self):
        W1,W2,W3,W4 = self.model['W1'],self.model['W2'],self.model['W3'],self.model['W4']
        a1,a2,a3,y_ = self.activation_outputs
        
        print("W1 ",W1.shape)
        print("A1 ",a1.shape)

def softmax(a):
    e_pa = np.exp(a) #Vector
    ans = e_pa/np.sum(e_pa,axis=1,keepdims=True)
    return ans        



def loss(y_oht,dataset):
    l = -np.mean(y_oht*np.log(dataset))
    return l

def one_hot(y,depth):
    m = y.shape[0]
    y_oht = np.zeros((m,depth))
    y_oht[np.arange(m),y] = 1
    return y_oht



def train(X,Y,model,epochs,learning_rate,logs=True):
    training_loss = []
    
    classes = counter
    Y_OHT = one_hot(Y,classes)
    
    for ix in range(epochs):
        
        Y_ = model.forward(X)
        l = loss(Y_OHT,Y_)
        
        model.backward(X,Y_OHT,learning_rate)
        training_loss.append(l)
        if(logs and ix%50==0):
            print("Epoch %d Loss %.4f"%(ix,l))
            
            
    
    return training_loss

model = NeuralNetwork(input_size= img_target * img_target * 3,layers=[200,50,20],output_size=counter)
print('Model Built -----------')

# Reshaping our dataset
X = X.reshape(X.shape[0],-1)
XTest = XTest.reshape(XTest.shape[0],-1)


print('\n\n\n\nTrain Model')
l = train(X,Y,model,iter,0.0005)
print('\n\nTraining Finished')

import matplotlib.pyplot as plt
plt.style.use("dark_background")
plt.title("Training Loss vs Epochs")
plt.plot(l)

plt.show()

# Accuracy 
def getAccuracy(X,Y,model):
    outputs = model.predict(X)
    acc = np.sum(outputs==Y)/Y.shape[0]
    return acc


print(  "\ntotal_img_size: ", total_img_size, "\ntraining_size: ", training_size, "\ntest_size: ", test_size, \
        "\nclass_size: ", counter, "\niter: ", iter, "\nimg_target: ", img_target)
print("Train Accuracy: %.4f  :)"%getAccuracy(X,Y,model))
print("Test Accuracy: %.4f :("%getAccuracy(XTest,YTest,model))


# ### Plot confusion matrix
label = ["butterfly","cat","chicken","cow","dog","elephant","horse","sheep","spider","squirrel"]

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
output=model.predict(XTest)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_predictions(YTest, output, display_labels=label)
plt.show()

print(classification_report(YTest, output))
