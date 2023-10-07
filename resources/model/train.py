import numpy as np
import keras
import tensorflowjs as tfjs
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

# Preprocess
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape1 = (28,28,1)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Create the model
batch_size1 = 32
num_classes1 = 10
epochs1 = 50

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',kernel_initializer='he_uniform',input_shape=input_shape1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_classes1, activation='softmax'))


opt = SGD(learning_rate=0.01, momentum=0.9)

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])
model.summary()

# Train the model
hist = model.fit(x_train, y_train, batch_size=batch_size1, epochs=epochs1, verbose=1, validation_data=(x_test, y_test))
print("The Model has successfully trained")

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
tfjs.converters.save_keras_model(model, './')
print("saving model as json")
model.save('mnist.keras')
print('Saving the model as mnist.keras')






