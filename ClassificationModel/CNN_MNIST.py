import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

#Hyperparameters
EPOX = 10 # EPOCHs
MY_BATCH_SIZE= 500
L2_lambda = .1

#Definitions
PIXEL_MAX_INTENSITY = 255.0

# X SHAPE (60000,28,28) Y SHAPE (60000,)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# For future predictions
testX = X_test


# Reshape for max pooling
X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

# Range Normalization
X_train = X_train/PIXEL_MAX_INTENSITY
X_test = X_test/PIXEL_MAX_INTENSITY


# One hot representation for softmax
Y_train_oh = tf.keras.utils.to_categorical(Y_train)
Y_test_oh = tf.keras.utils.to_categorical(Y_test)


# Create Model
X_in = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2],X_train.shape[3]))
c1 = tf.keras.layers.Conv2D(10,kernel_size=(4,4),strides=1,padding='same',use_bias=True,
                            bias_initializer='zeros',kernel_initializer='he_normal', activation='relu' )(X_in)
p1 = tf.keras.layers.MaxPool2D(pool_size=(4,4), strides = None, padding='valid')(c1)
c2 = tf.keras.layers.Conv2D(10, kernel_size=(3,3),strides=1,padding='valid',use_bias=True,
                            bias_initializer='zeros',kernel_initializer='he_normal', activation='relu' )(p1)
x1 = tf.keras.layers.Flatten()(c2)
X_out = tf.keras.layers.Dense(10, activation='softmax',kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=L2_lambda))(x1)
mnistModel = tf.keras.Model(inputs=X_in, outputs= X_out)

# Compile & Train
mnistModel.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics =['categorical_accuracy'])
mnistModel.fit(X_train, Y_train_oh, epochs = EPOX, batch_size = MY_BATCH_SIZE, verbose = 2)


# Evaluate Model
loss, acc = mnistModel.evaluate(X_test, Y_test_oh, verbose = 0)
print('Accuracy: ', acc)


# Make a prediction
pred = np.random.randint(1,500)
yhat = mnistModel.predict (X_test)
print("Predicted Value: ", np.argmax(yhat[pred]))
print("Actual Value: ", Y_test[pred])

plt.imshow(testX[pred])
plt.show()
