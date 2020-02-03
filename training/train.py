import os
import shutil
import argparse
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from azureml.core.run import Run
from PIL import Image

# Get the Azure ML run object
run = Run.get_context()

class AmlLogger(Callback):
    ''' A callback class for logging metrics using Azure Machine Learning Python SDK '''

    def on_epoch_end(self, epoch, logs={}):
        run.log('val_accuracy', float(logs.get('val_accuracy')))

    def on_batch_end(self, batch, logs={}):
        run.log('accuracy', float(logs.get('accuracy')))


# Define arguments for training
parser = argparse.ArgumentParser(description='Famous athlete classifier')
parser.add_argument('--data_dir', type=str, default='data', help='Root directory of the data')
parser.add_argument('--image_dim', type=int, default=250, help='Image dimensions')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of the optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--steps_per_epoch', type=int, default=100, help='Training steps per epoch')
parser.add_argument('--num_epochs', type=int, default=25, help='Training number of epochs')
parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
args = parser.parse_args()

# Get arguments from parser
data_dir = args.data_dir
image_dim = args.image_dim
learning_rate = args.learning_rate
batch_size = args.batch_size 
steps_per_epoch = args.steps_per_epoch 
num_epochs = args.num_epochs 
output_dir = args.output_dir 

# Load classes
classes = []
f = open('classes.txt','r')
for line in f.readlines():
    classes.append(line.replace('\n',''))
    f.close()

# Create data generator to augmnent input images
datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             rescale=1./255,
                             rotation_range=90,
                             width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             shear_range=0.2, 
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=False)

# Create train dataset with generator
train_generator = datagen.flow_from_directory(os.path.join(data_dir, 'train'),
                                              target_size=(image_dim, image_dim),
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              classes=classes)

# Create valid dataset with generator
valid_generator = datagen.flow_from_directory(os.path.join(data_dir, 'valid'),
                                              target_size=(image_dim, image_dim),
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              classes=classes)

# +
model = Sequential()

# CONV => RELU => POOL
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(image_dim, image_dim, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(len(classes)))
model.add(Activation("softmax"))
# -

# Compile model with optimizer, loss function, and metrics
optimizer = Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit_generator(train_generator, 
                    steps_per_epoch=steps_per_epoch,
                    epochs=num_epochs,
                    validation_data=valid_generator, 
                    validation_steps=10,
                    callbacks=[AmlLogger()],
                    verbose=1)

# Save the output model
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model.save(os.path.join(output_dir, 'model.h5'))
shutil.copyfile('classes.txt', os.path.join(output_dir, 'classes.txt')) 
