
# coding: utf-8

# In[26]:


#Trying out CNN on MNIST Library

#Importing the dataset
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[29]:


#Initializing the CNN as Sequential Model
classifier = Sequential()

#First Convolutional layer
classifier.add(Convolution2D(64, (3,3), input_shape=(28, 28, 3), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = 1))

#Flattening
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[36]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/trainingSet',
                                                  target_size=(28, 28),
                                                  batch_size=64,
                                                  class_mode='categorical')


# In[ ]:


classifier.fit_generator(training_set,
               steps_per_epoch=8000/64,
               epochs=25)

