# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D # 2D to deal with images
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution, specify dim (3 x 3) & num (32) of feature detectors
# specify input image format (3 channels for color), size (64 x 64)
# last param = rectifier func for ReLU layer, remove negative pixel values
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Max Pooling, using a 2x2 box to reduce images to essential features
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer


# Step 3 - Flattening, all feature maps converted into single vector
# each element becomes an input node of ANN
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
# image preprocessing: random transformations of our images to prevent overfitting
from keras.preprocessing.image import ImageDataGenerator
# this object augments the images
train_datagen = ImageDataGenerator( rescale=1./255, # all pixel values between 0 - 1
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) # preprocess test set by scaling pixel values

training_set = train_datagen.flow_from_directory('/Users/michaelchen/Desktop/ML-Course/Part8-DeepLearning/40CNN/dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('/Users/michaelchen/Desktop/ML-Course/Part8-DeepLearning/40CNN/dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000, # number of images in training set
                        epochs=25, # number of total iterations to train network
                        validation_data=test_set,
                        validation_steps=2000) # images in test set
