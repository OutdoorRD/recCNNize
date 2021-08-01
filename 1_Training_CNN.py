##### Build our classifier model based on pre-trained InceptionResNetV2:
#### Fine-tune InceptionV3 on a new set of classes


# Ref:
# https://github.com/fchollet/deep-learning-with-python-notebooks


import keras
import os

import csv
import pandas as pd
import pathlib
import fnmatch



from tensorflow.keras.applications import inception_resnet_v2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt


from tensorflow.python.client import device_lib


from keras.preprocessing import image
import numpy as np


import math

default_path = '/home/alan/Dropbox/KIT/FlickrEU/deepGreen/'

os.chdir(default_path)

# split utils (@TODO reference)
import split_utils
from keras.applications import inception_resnet_v2

from collections import Counter

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import Adam
from keras import metrics

from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as k

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping




# Scaled image size
img_width, img_height = 331, 331


batch_size = 512    # the larger is faster in training. Cponsider 1) training sample size, 2) GPU memory, 3) throughput (img/sec)
val_batch_size = batch_size # validation batch
epochs = 300 # number of epochs

save_period = 5 # model saving frequency

validation_split = 0.4 # % test photos

dropout = 0.3 # % dropout layers

loadWeights = False # for continuing training


sitename = "Seattle"
train_data_dir = "../LabelledData/Seattle/Photos_iterative_Sep2019/train/"

trainedweights_name = "../TrainedWeights/InceptionResnetV2_Seattle_retrain_instabram_15classes_Weighted_Nov2019_val_acc0.87.h5"

num_layers_train = 4

learning_rate = 1e-5 # ADAM parameter



num_classes = 14
# ****************
# Class #0 = backpacking
# Class #1 = birdwatching
# Class #2 = boating
# Class #3 = camping
# Class #4 = fishing
# Class #5 = hiking
# Class #6 = horseriding
# Class #7 = mtn_biking
# Class #8 = noactivity
# Class #9 = otheractivities
# Class #10 = pplnoactivity
# Class #11 = rock climbing
# Class #12 = swimming
# Class #13 = trailrunning




# Load the base pre-trained model

# do not include the top fully-connected layer

model = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',input_tensor=None, input_shape=(img_width, img_height, 3))
# Freeze the layers which you don't want to train. Here I am freezing the all layers.
# i.e. freeze all InceptionV3 layers

x = model.output

# Adding custom Layer
x = GlobalAveragePooling2D()(x) # before dense layer
x = Dense(1024, activation='relu')(x)
# Ref: https://datascience.stackexchange.com/questions/28120/globalaveragepooling2d-in-inception-v3-example


if dropout > 0:
    # If the network is stuck at 50% accuracy, there’s no reason to do any dropout.
    # Dropout is a regularization process to avoid overfitting; when underfitting not really useful .
    x = Dropout(dropout)(x) # x% dropout

    # A Dense (fully connected) layer which generates softmax class score for each class
    predictions_new = Dense(num_classes, activation='softmax', name='softmax')(x)
    model_final = Model(inputs = model.input, outputs = predictions_new)


## load trained weights
if loadWeights:

    ## load previously trained weights (old class number)
    model_final.load_weights(trainedweights_name)



# Fine tuning (
FREEZE_LAYERS = len(model.layers) - num_layers_train # train the newly added layers and the last few layers

for layer in model_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False


# Compile the final model using an Adam optimizer, with a low learning rate (since we are 'fine-tuning')
model_final.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

# lr: float >= 0. Learning rate.
# beta_1: float, 0 < beta < 1. Generally close to 1.
# beta_2: float, 0 < beta < 1. Generally close to 1.
# epsilon: float >= 0. Fuzz factor.If None, defaults to K.epsilon().
# decay: float >= 0. Learning rate decay over each update.
# amsgrad: boolean. Whether to apply the AMSGrad variantof this algorithm from the paper

# References:
# https://keras.io/optimizers/
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# https://medium.com/@nishantnikhil/adam-optimizer-notes-ddac4fd7218


# Save the model architecture
with open('Model/InceptionResnetV2_retrain_' + sitename + '_architecture_dropout' + dropout.__str__()  + '.json', 'w') as f:
    f.write(model_final.to_json())



# all data in train_dir and val_dir which are alias to original_data. (both dir is temporary directory)
# don't clear base_dir, because this directory holds on temp directory.
base_dir, train_tmp_dir, val_tmp_dir = split_utils.train_valid_split(train_data_dir, validation_split, seed=1)


# Initiate the train and test generators with data Augmentation
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    brightness_range = [0.7,1.0],
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)


# generator for validation data
val_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    brightness_range= [0.7, 1.0],
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)


train_generator = train_datagen.flow_from_directory(
     train_tmp_dir,
     target_size = (img_height, img_width),
     batch_size = batch_size,
     class_mode = "categorical")


validation_generator = val_datagen.flow_from_directory(
    val_tmp_dir,
    target_size = (img_height, img_width),
    batch_size=val_batch_size,
    class_mode = "categorical")



# https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
# This works with a generator or standard. Your largest class will have a weight of 1 while the others will have values greater than 1 relative to the largest class. class weights accepts a dictionary type inpu



# show class indices
print('****************')
for cls, idx in train_generator.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')



nb_train_samples = train_generator.n
nb_validation_samples = validation_generator.n

print('the ratio of validation_split is {}'.format(validation_split))
print('the size of train_dir is {}'.format(nb_train_samples))
print('the size of val_dir is {}'.format(nb_validation_samples))


# Save the model according to the conditions
checkpoint = ModelCheckpoint("../TrainedWeights/InceptionResnetV2_" + sitename + "_retrain.h5", monitor='val_accuracy',
                             verbose=1, save_best_only=False, save_weights_only=False, mode='auto', save_freq=save_period)


early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

# steps per epoch depends on the batch size
steps_per_epoch = int(np.ceil(nb_train_samples / batch_size))
validation_steps_per_epoch = int(np.ceil(nb_validation_samples / batch_size))

# Tensorboard
# If printing histograms, validation_data must be provided, and cannot be a generator.
callback_tb = keras.callbacks.TensorBoard(
        log_dir = "log_dir", # tensorflow log
        histogram_freq=1,    # histogram
        # embeddings_freq=1,
        # embeddings_data=train_generator.labels,
        write_graph=True, write_images=True
    )



# Train the model

history = model_final.fit_generator(
    train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps_per_epoch,
    callbacks = [checkpoint, early]

)


# Save the model
model_final.save('../TrainedWeights/InceptionResnetV2_retrain_' + sitename + '.h5')

# save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('../TrainedWeights/InceptionResnetV2_retrain_' + sitename + '.csv')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot( acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot( loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training loss')
plt.legend()

plt.show()





















