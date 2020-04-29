import tensorflow as tf
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


# Dataset: https://www.kaggle.com/neuromusic/avocado-prices/data

# Acquire data
data = read_csv(r'avocado.csv')
X,Y = data.iloc[:, 2:11], data.iloc[:, 0] # Numeric columns from 2-11 in Excel, and Y as 1st col
X = X.astype('float32')
Y = Y.astype('float32')


# Normalize, Z score
Scaler = StandardScaler()
X = Scaler.fit_transform(X)


# Random example for future predictions
testIdx = np.random.randint(1, X.shape[0])
testX = [None]*X.shape[1]
testY = Y[testIdx]
temp = X[testIdx]
for i in range(len(temp)):
    testX[i] = float(temp[i])


# Train and Test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
n_features = X_train.shape[1]


# Create Model
X_in = tf.keras.Input(shape=(n_features,))
x1 = tf.keras.layers.Dense(4,activation='relu',kernel_initializer='he_normal')(X_in)
x2 = tf.keras.layers.Dense(3, activation= 'relu',kernel_initializer='he_normal')(x1)
X_out = tf.keras.layers.Dense(1,activation='relu',kernel_initializer='he_normal' )(x2)
avocadoModel = tf.keras.Model(inputs=X_in, outputs = X_out)


# Compile & Train
avocadoModel.compile(optimizer='adam', loss = 'mean_squared_error', metrics = ['mae'])
avocadoModel.fit(X_train, Y_train, epochs = 300, batch_size = 20, verbose = 2)


# Evaluate Model
loss, mae = avocadoModel.evaluate(X_test, Y_test, verbose = 0)
print('Average Mean Absolute error: ', mae)


# Observe a prediction
yhat = avocadoModel.predict([testX])
print('Prediction: %.3f', yhat, "Actual Y: ", testY)
