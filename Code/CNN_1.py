# %%
"""
<a href="https://colab.research.google.com/github/Shivdutta/CNN/blob/master/CNN_1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""
#ipynb-py-convert CNN_1.ipynb CNN_1.py

# %%
# Convolutional Neural Network

# Installing Theano
#!pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
#!pip install tensorflow

# Installing Keras
#!pip install --upgrade keras

# %%
# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D,BatchNormalization,Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation

# %%
# Initialising the CNN
path = "C:/GIT/CNN/Code/Images/Images"
classifier1 = Sequential()

# Step 1 - Convolution
classifier1.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier1.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier1.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier1.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier1.add(Flatten())

# Step 4 - Full connection
classifier1.add(Dense(units = 128, activation = 'relu'))

classifier1.add(Dense(units = 2, activation = 'sigmoid'))

# Compiling the CNN
classifier1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(path,
                                                 target_size = (64, 64),
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory(path,
                                            target_size = (64, 64),
                                            batch_size = 32)

classifier1.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 2000)

# serialize model to JSON
model_json = classifier1.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier1.save_weights("model.h5")
print("Saved model to disk")