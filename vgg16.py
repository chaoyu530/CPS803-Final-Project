import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense , Dropout
from tensorflow.keras import datasets,layers,optimizers,models,regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

training_generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range=7,
                                        horizontal_flip=True,
                                        zoom_range=0.2)
training_dataset = training_generator.flow_from_directory('E:/desktop/cps803/data/train',
                                                        target_size = (224, 224),
                                                        batch_size = 32,
                                                        class_mode = 'categorical',
                                                        shuffle = True)

training_dataset.classes
training_dataset.class_indices

test_generator = ImageDataGenerator(rescale=1./255)
test_dataset = test_generator.flow_from_directory('E:/desktop/cps803/data/test',
                                                     target_size = (224, 224),
                                                     batch_size = 32,
                                                     class_mode = 'categorical',
                                                     shuffle = False)

weight_decay=0.000
model=models.Sequential()

model.add(layers.Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(256,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(256,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(256,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.5))
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))

model.build(input_shape=(None,224,224,3))

model.summary()
Adam = optimizers.Adam(lr = 0.001)
model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics = ['accuracy'])
historic = model.fit(training_dataset, validation_data = test_dataset,epochs=70)
test_dataset.class_indices
forecasts = model.predict(test_dataset)
forecasts = np.argmax(forecasts, axis = 1)
accuracy_score(test_dataset.classes, forecasts)
cm = confusion_matrix(test_dataset.classes, forecasts)
sns.heatmap(cm, annot=True)
plt.show()
print(classification_report(test_dataset.classes, forecasts))


plt.plot(historic.history['loss'], label='train loss')
plt.plot(historic.history['val_loss'], label='test loss')
plt.legend()
plt.show()
plt.savefig('Loss')

# plot the accuracy
plt.plot(historic.history['accuracy'], label='train acc')
plt.plot(historic.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('Acc')



